"""CPU-only checks for duplex scenario fixtures and trace metrics.

Run directly: ``uv run python moshi/tests/test_duplex_scenarios.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from duplex_harness import (  # noqa: E402
    FRAME_SAMPLES,
    ScenarioValidationError,
    VadConfig,
    analyze_scenario,
    detect_speech_segments,
    iter_audio_frames,
    load_manifest,
    load_pcm16_wav,
    resolve_input_wav,
    validate_manifest,
    write_artifacts,
    write_pcm16_wav,
)
from scripts.run_duplex_regression import (  # noqa: E402
    APPLIED_CONFIG_KEYS,
    _action_protocol_failures,
    _config_application_failures,
    _operational_event_failures,
    _queue_health_failures,
    _runtime_metrics,
    _scenario_exit_failure,
    _validate_scenario_timeline,
)


FIXTURES = Path(__file__).parent / "fixtures" / "duplex"


def _manifest(**updates):
    payload = {
        "schema_version": 1,
        "id": "cpu_trace",
        "description": "synthetic trace",
        "audio": None,
        "config": {},
        "actions": [],
        "expectations": [],
        "limits": {},
        "tail_ms": 1000,
    }
    payload.update(updates)
    return payload


def _tone(start_ms: int, end_ms: int, duration_ms: int, amplitude: float = 0.2):
    samples = np.zeros(duration_ms * 48, dtype=np.float32)
    samples[start_ms * 48 : end_ms * 48] = amplitude
    return samples


def test_checked_in_manifests_validate_without_bundled_audio() -> None:
    paths = sorted(FIXTURES.glob("*.json"))
    assert len(paths) >= 5
    for path in paths:
        manifest = load_manifest(path)
        assert manifest["audio"] is None
        assert manifest["input_requirements"]
        try:
            resolve_input_wav(path, manifest)
        except ScenarioValidationError as exc:
            assert "pass --input-wav" in str(exc)
        else:
            raise AssertionError(f"{path} unexpectedly resolved missing audio")


def test_manifest_rejects_ambiguous_action_schedule() -> None:
    payload = _manifest(
        actions=[
            {
                "type": "interrupt",
                "at_ms": 100,
                "when": {"kind": "assistant_active_ms", "value": 200},
            }
        ]
    )
    try:
        validate_manifest(payload)
    except ScenarioValidationError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous action schedule was accepted")


def test_runner_rejects_expectation_windows_past_capture_end() -> None:
    cases = [
        {
            "kind": "pause",
            "start_ms": 800,
            "end_ms": 1001,
        },
        {
            "kind": "turn",
            "at_ms": 700,
            "deadline_ms": 301,
        },
        {
            "kind": "event",
            "event_kind": "turn_cap",
            "at_ms": 600,
            "deadline_ms": 401,
        },
    ]
    for expectation in cases:
        manifest = validate_manifest(_manifest(expectations=[expectation]))
        try:
            _validate_scenario_timeline(manifest, 1000)
        except ScenarioValidationError as exc:
            assert "window extends past" in str(exc)
        else:
            raise AssertionError(f"accepted out-of-capture {expectation['kind']} window")

    exact_end = validate_manifest(
        _manifest(
            expectations=[
                {"kind": "turn", "at_ms": 700, "deadline_ms": 300}
            ]
        )
    )
    _validate_scenario_timeline(exact_end, 1000)


def test_wav_validation_and_twenty_ms_framing() -> None:
    samples = np.linspace(-0.5, 0.5, FRAME_SAMPLES * 2 + 17, dtype=np.float32)
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "input.wav"
        write_pcm16_wav(path, samples)
        decoded, sample_rate = load_pcm16_wav(path)
    assert sample_rate == 48_000
    frames = list(iter_audio_frames(decoded))
    assert [frame.size for frame in frames] == [FRAME_SAMPLES] * 3
    assert np.count_nonzero(frames[-1][17:]) == 0


def test_vad_attack_release_segments_speech() -> None:
    samples = _tone(100, 300, 500)
    segments = detect_speech_segments(
        samples,
        config=VadConfig(rms_threshold=0.01, attack_frames=2, release_frames=3),
    )
    assert len(segments) == 1
    assert segments[0].start_ms == 100
    assert segments[0].end_ms == 300


def test_pause_and_turn_metrics_use_annotated_windows() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "pause",
                "label": "hesitation",
                "start_ms": 150,
                "end_ms": 350,
                "max_assistant_active_ms": 80,
            },
            {
                "kind": "turn",
                "label": "yield",
                "at_ms": 400,
                "deadline_ms": 500,
                "max_latency_ms": 150,
            },
        ]
    )
    output = _tone(200, 300, 1000) + _tone(500, 700, 1000)
    metrics = analyze_scenario(manifest, output, [])
    assert metrics["pause"][0]["assistant_active_ms"] == 100
    assert metrics["turn"][0]["latency_ms"] == 100
    assert len(metrics["failures"]) == 1
    assert "pause" in metrics["failures"][0]


def test_continuous_overlap_is_not_a_zero_latency_response() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "turn",
                "label": "user end",
                "at_ms": 1500,
                "deadline_ms": 2500,
            }
        ]
    )
    output = _tone(1000, 5000, 6000)

    result = analyze_scenario(manifest, output, [])

    assert result["turn"][0]["responded"] is False
    assert result["turn"][0]["active_at_boundary"] is True
    assert result["turn"][0]["boundary_overlap_ms"] == 3500
    assert result["passed"] is False


def test_interrupt_metrics_pair_ack_and_audio_yield() -> None:
    manifest = _manifest()
    output = _tone(100, 700, 1000)
    events = [
        {
            "t_ms": 400,
            "direction": "out",
            "message": {"type": "interrupt", "reason": "manual"},
        },
        {
            "t_ms": 450,
            "direction": "in",
            "message": {"type": "interrupted", "reason": "manual"},
        },
    ]
    result = analyze_scenario(manifest, output, events)["interrupt"][0]
    assert result["ack_latency_ms"] == 50
    assert result["audio_yield_ms"] == 300
    assert result["active_after_ack_ms"] == 250


def test_event_and_stop_limits_are_enforced() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "event",
                "label": "cap",
                "event_kind": "turn_cap",
                "at_ms": 100,
                "deadline_ms": 500,
            }
        ],
        limits={
            "max_interrupt_ack_ms": 25,
            "max_interrupt_yield_ms": 100,
            "max_audio_after_ack_ms": 75,
            "max_post_interrupt_active_ms": 200,
        },
    )
    output = _tone(100, 700, 1000)
    events = [
        {
            "t_ms": 400,
            "direction": "out",
            "message": {"type": "interrupt", "reason": "manual"},
        },
        {
            "t_ms": 450,
            "direction": "in",
            "message": {"type": "interrupted", "reason": "manual"},
        },
        {
            "t_ms": 500,
            "direction": "in",
            "message": {"type": "event", "kind": "turn_cap"},
        },
    ]
    metrics = analyze_scenario(manifest, output, events)
    assert metrics["event"][0]["observed"] is True
    assert len(metrics["failures"]) == 4
    assert any("max_interrupt_ack_ms" in failure for failure in metrics["failures"])


def test_clipping_and_transcript_runaway_are_reported() -> None:
    manifest = _manifest(
        limits={"max_clipped_samples": 0, "max_identical_word_run": 5}
    )
    output = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    events = [
        {
            "t_ms": index * 100,
            "direction": "in",
            "message": {"type": "text", "v": " loop"},
        }
        for index in range(6)
    ]
    metrics = analyze_scenario(manifest, output, events)
    assert metrics["pcm"]["clipped_samples"] == 2
    assert metrics["transcript"]["max_identical_word_run"] == 6
    assert set(metrics["runaway_flags"]) >= {"clipped_pcm", "repeated_word_run"}
    assert len(metrics["failures"]) == 2


def test_required_and_threshold_failures_are_distinct() -> None:
    manifest = _manifest(
        expectations=[
            {
                "kind": "event",
                "label": "required cap",
                "event_kind": "turn_cap",
                "deadline_ms": 500,
            }
        ],
        limits={"max_clipped_samples": 0},
    )
    metrics = analyze_scenario(
        manifest, np.array([0.0, 1.0], dtype=np.float32), []
    )
    assert metrics["required_failures"] == [
        "event 'required cap': 'turn_cap' not observed within deadline"
    ]
    assert metrics["threshold_failures"] == [
        "max_clipped_samples: observed 1 > limit 0.0"
    ]
    assert metrics["failures"] == [
        *metrics["required_failures"],
        *metrics["threshold_failures"],
    ]


def test_runner_verifies_replay_config_and_concrete_seed() -> None:
    requested = {
        "reinforce_in_silences": False,
        "vision_in_transcript": False,
        "vision_feed_model": False,
        "vision_ground_user_turns": False,
        "seed": 42,
        "text_temperature": 0.7,
        "audio_temperature": 0.8,
        "text_topk": 25,
        "audio_topk": 250,
        "repetition_penalty": 1.0,
        "repetition_penalty_context": 64,
        "padding_bonus": 0.0,
        "max_turn_text_tokens": 120,
        "inject_silence_rms": 0.004,
        "inject_silence_streak": 8,
    }
    assert set(requested) == set(APPLIED_CONFIG_KEYS)
    assert _config_application_failures(requested, dict(requested)) == []

    wrong = dict(requested, seed=7, audio_temperature=0.9)
    failures = _config_application_failures(requested, wrong)
    assert any("seed" in failure for failure in failures)
    assert any("audio_temperature" in failure for failure in failures)

    random_requested = dict(requested, seed=-1)
    concrete = dict(requested, seed=2_147_483_647)
    assert _config_application_failures(random_requested, concrete) == []
    assert _config_application_failures(random_requested, dict(requested, seed=-1))


def test_runner_treats_actions_and_transport_errors_as_operational() -> None:
    manifest = validate_manifest(
        _manifest(actions=[{"type": "interrupt", "at_ms": 100}])
    )
    incomplete = [
        {
            "t_ms": 100,
            "direction": "harness",
            "message": {"type": "action_timeout", "action": "interrupt"},
        },
        {
            "t_ms": 110,
            "direction": "transport",
            "message": {"type": "candidate_post_error", "error": "boom"},
        },
    ]
    operational = _operational_event_failures(incomplete)
    assert any("action_timeout" in failure for failure in operational)
    assert any("candidate_post_error" in failure for failure in operational)
    assert _action_protocol_failures(manifest, incomplete) == [
        "scheduled 1 action(s), sent 0"
    ]

    acknowledged = [
        {
            "t_ms": 100,
            "direction": "out",
            "message": {"type": "interrupt", "reason": "regression"},
        },
        {
            "t_ms": 130,
            "direction": "in",
            "message": {"type": "interrupted", "reason": "regression"},
        },
    ]
    assert _action_protocol_failures(manifest, acknowledged) == []


def test_ignore_thresholds_never_masks_operational_failures() -> None:
    threshold_only = {
        "operational_failures": [],
        "threshold_failures": ["latency too high"],
    }
    assert _scenario_exit_failure(threshold_only, ignore_thresholds=False)
    assert not _scenario_exit_failure(threshold_only, ignore_thresholds=True)
    operational = {
        "operational_failures": ["server error"],
        "threshold_failures": [],
    }
    assert _scenario_exit_failure(operational, ignore_thresholds=True)


def test_runtime_metrics_name_sampled_ema_and_ignore_pre_scenario_stats() -> None:
    events = [
        {
            "t_ms": -20,
            "direction": "in",
            "message": {
                "type": "stat",
                "rtf": 9,
                "pcm_dropped_ms": 999,
            },
        },
        {
            "t_ms": 100,
            "direction": "in",
            "message": {
                "type": "stat",
                "rtf": 0.4,
                "pcm_queue_depth": 3,
                "pcm_queue_capacity": 10,
                "pcm_queue_high_water": 5,
                "pcm_drop_events": 0,
                "pcm_dropped_ms": 0,
                "outbound_buffer_ms": 40,
                "outbound_high_water_ms": 160,
                "outbound_drop_events": 1,
                "outbound_dropped_ms": 59.1,
                "outbound_flush_events": 1,
                "outbound_flushed_ms": 80,
            },
        },
        {
            "t_ms": 200,
            "direction": "in",
            "message": {
                "type": "stat",
                "rtf": True,
                "pcm_queue_depth": 1,
                "pcm_queue_high_water": 6,
                "outbound_buffer_ms": 20,
                "outbound_dropped_ms": 119.2,
                "outbound_flushed_ms": 120,
            },
        },
    ]
    runtime = _runtime_metrics(events)
    assert runtime["stat_samples"] == 2
    assert runtime["rtf_ema_samples"] == 1
    assert runtime["rtf_ema_p95"] == 0.4
    assert "rtf_p95" not in runtime
    assert runtime["pcm_queue_depth_max"] == 3
    assert runtime["pcm_queue_high_water"] == 6
    assert runtime["pcm_dropped_ms"] == 0
    assert runtime["outbound_buffer_ms_max"] == 40
    assert runtime["outbound_dropped_ms"] == 119.2
    assert runtime["outbound_flushed_ms"] == 120


def test_queue_health_classifies_input_loss_and_excess_output_shedding() -> None:
    operational, thresholds = _queue_health_failures(
        {
            "pcm_drop_events": 1,
            "pcm_dropped_ms": 20.0,
            "outbound_dropped_ms": 200.1,
            "outbound_flush_events": 3,
            "outbound_flushed_ms": 800.0,
        }
    )
    assert operational == [
        "inbound PCM queue dropped audio: 1 event(s), 20.0 ms"
    ]
    assert len(thresholds) == 1
    assert "200.1 ms" in thresholds[0]

    operational, thresholds = _queue_health_failures(
        {
            "pcm_drop_events": 0,
            "pcm_dropped_ms": 0.0,
            "outbound_dropped_ms": 200.0,
            "outbound_flush_events": 10,
            "outbound_flushed_ms": 5_000.0,
        }
    )
    assert operational == []
    assert thresholds == []


def test_artifact_bundle_is_replayable() -> None:
    manifest = validate_manifest(_manifest())
    samples = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    metrics = analyze_scenario(manifest, samples, [])
    with tempfile.TemporaryDirectory() as raw:
        target = write_artifacts(
            Path(raw) / "result",
            manifest=manifest,
            input_samples=samples,
            output_samples=samples,
            events=[],
            metrics=metrics,
            run={"model_revision": "test"},
        )
        assert {path.name for path in target.iterdir()} == {
            "events.jsonl",
            "input.wav",
            "metrics.json",
            "output.wav",
            "run.json",
            "scenario.json",
        }
        assert json.loads((target / "metrics.json").read_text())["scenario_id"] == "cpu_trace"


if __name__ == "__main__":
    tests = [
        test_checked_in_manifests_validate_without_bundled_audio,
        test_manifest_rejects_ambiguous_action_schedule,
        test_runner_rejects_expectation_windows_past_capture_end,
        test_wav_validation_and_twenty_ms_framing,
        test_vad_attack_release_segments_speech,
        test_pause_and_turn_metrics_use_annotated_windows,
        test_continuous_overlap_is_not_a_zero_latency_response,
        test_interrupt_metrics_pair_ack_and_audio_yield,
        test_event_and_stop_limits_are_enforced,
        test_clipping_and_transcript_runaway_are_reported,
        test_required_and_threshold_failures_are_distinct,
        test_runner_verifies_replay_config_and_concrete_seed,
        test_runner_treats_actions_and_transport_errors_as_operational,
        test_ignore_thresholds_never_masks_operational_failures,
        test_runtime_metrics_name_sampled_ema_and_ignore_pre_scenario_stats,
        test_queue_health_classifies_input_loss_and_excess_output_shedding,
        test_artifact_bundle_is_replayable,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all duplex scenario tests passed")
