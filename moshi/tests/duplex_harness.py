"""Shared fixtures and metrics for reproducible duplex speech regressions.

The CPU tests import this module directly. ``scripts/run_duplex_regression.py``
uses the same code while driving a real aiortc peer against an already-running
PersonaPlex server. The module deliberately depends only on the standard
library and NumPy so trace analysis never requires a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
WEBRTC_SAMPLE_RATE = 48_000
FRAME_MS = 20
FRAME_SAMPLES = WEBRTC_SAMPLE_RATE * FRAME_MS // 1000
_SCENARIO_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SUPPORTED_ACTIONS = {"interrupt", "update_config"}
_SUPPORTED_EXPECTATIONS = {"event", "pause", "turn"}


class ScenarioValidationError(ValueError):
    """A scenario manifest or its input WAV is not runnable."""


@dataclass(frozen=True)
class SpeechSegment:
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class VadConfig:
    rms_threshold: float = 0.008
    attack_frames: int = 2
    release_frames: int = 5
    frame_ms: int = FRAME_MS


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioValidationError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ScenarioValidationError(f"{field} must be finite and >= {minimum}")
    return value


def validate_manifest(payload: Any, *, source: str = "manifest") -> dict[str, Any]:
    """Validate and normalize one versioned scenario manifest."""
    if not isinstance(payload, dict):
        raise ScenarioValidationError(f"{source} must contain a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ScenarioValidationError(
            f"{source}.schema_version must be {SCHEMA_VERSION}"
        )
    scenario_id = payload.get("id")
    if not isinstance(scenario_id, str) or not _SCENARIO_ID.fullmatch(scenario_id):
        raise ScenarioValidationError(
            f"{source}.id must match {_SCENARIO_ID.pattern!r}"
        )
    description = payload.get("description", "")
    if not isinstance(description, str):
        raise ScenarioValidationError(f"{source}.description must be a string")
    input_requirements = payload.get("input_requirements", "")
    if not isinstance(input_requirements, str):
        raise ScenarioValidationError(
            f"{source}.input_requirements must be a string"
        )
    audio = payload.get("audio")
    if audio is not None and (not isinstance(audio, str) or not audio.strip()):
        raise ScenarioValidationError(f"{source}.audio must be a non-empty path")
    if isinstance(audio, str) and Path(audio).is_absolute():
        raise ScenarioValidationError(
            f"{source}.audio must be manifest-relative; use --input-wav "
            "for an absolute override"
        )
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise ScenarioValidationError(f"{source}.config must be an object")

    tail_ms = _number(payload.get("tail_ms", 5_000), f"{source}.tail_ms")
    expectations = payload.get("expectations", [])
    if not isinstance(expectations, list):
        raise ScenarioValidationError(f"{source}.expectations must be a list")
    normalized_expectations: list[dict[str, Any]] = []
    for index, item in enumerate(expectations):
        field = f"{source}.expectations[{index}]"
        if not isinstance(item, dict):
            raise ScenarioValidationError(f"{field} must be an object")
        kind = item.get("kind")
        if kind not in _SUPPORTED_EXPECTATIONS:
            raise ScenarioValidationError(
                f"{field}.kind must be one of {sorted(_SUPPORTED_EXPECTATIONS)}"
            )
        normalized = dict(item)
        if kind == "pause":
            start_ms = _number(item.get("start_ms"), f"{field}.start_ms")
            end_ms = _number(item.get("end_ms"), f"{field}.end_ms")
            if end_ms <= start_ms:
                raise ScenarioValidationError(f"{field}.end_ms must exceed start_ms")
            normalized.update(start_ms=start_ms, end_ms=end_ms)
            if "max_assistant_active_ms" in item:
                normalized["max_assistant_active_ms"] = _number(
                    item["max_assistant_active_ms"],
                    f"{field}.max_assistant_active_ms",
                )
        elif kind == "turn":
            at_ms = _number(item.get("at_ms"), f"{field}.at_ms")
            deadline_ms = _number(
                item.get("deadline_ms", 4_000), f"{field}.deadline_ms"
            )
            if deadline_ms <= 0:
                raise ScenarioValidationError(f"{field}.deadline_ms must be > 0")
            normalized.update(at_ms=at_ms, deadline_ms=deadline_ms)
            if "max_latency_ms" in item:
                normalized["max_latency_ms"] = _number(
                    item["max_latency_ms"], f"{field}.max_latency_ms"
                )
        else:
            event_kind = item.get("event_kind")
            if not isinstance(event_kind, str) or not _SCENARIO_ID.fullmatch(
                event_kind
            ):
                raise ScenarioValidationError(
                    f"{field}.event_kind must match {_SCENARIO_ID.pattern!r}"
                )
            at_ms = _number(item.get("at_ms", 0), f"{field}.at_ms")
            deadline_ms = _number(
                item.get("deadline_ms", 10_000), f"{field}.deadline_ms"
            )
            if deadline_ms <= 0:
                raise ScenarioValidationError(f"{field}.deadline_ms must be > 0")
            normalized.update(
                event_kind=event_kind,
                at_ms=at_ms,
                deadline_ms=deadline_ms,
            )
        normalized_expectations.append(normalized)

    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        raise ScenarioValidationError(f"{source}.actions must be a list")
    normalized_actions: list[dict[str, Any]] = []
    for index, item in enumerate(actions):
        field = f"{source}.actions[{index}]"
        if not isinstance(item, dict):
            raise ScenarioValidationError(f"{field} must be an object")
        action_type = item.get("type")
        if action_type not in _SUPPORTED_ACTIONS:
            raise ScenarioValidationError(
                f"{field}.type must be one of {sorted(_SUPPORTED_ACTIONS)}"
            )
        has_at = "at_ms" in item
        has_when = "when" in item
        if has_at == has_when:
            raise ScenarioValidationError(
                f"{field} must contain exactly one of at_ms or when"
            )
        normalized = dict(item)
        if has_at:
            normalized["at_ms"] = _number(item["at_ms"], f"{field}.at_ms")
        else:
            when = item["when"]
            if not isinstance(when, dict) or when.get("kind") != "assistant_active_ms":
                raise ScenarioValidationError(
                    f"{field}.when currently supports only assistant_active_ms"
                )
            normalized["when"] = {
                "kind": "assistant_active_ms",
                "value": _number(when.get("value"), f"{field}.when.value"),
            }
            normalized["timeout_ms"] = _number(
                item.get("timeout_ms", 10_000), f"{field}.timeout_ms"
            )
            if normalized["timeout_ms"] <= 0:
                raise ScenarioValidationError(f"{field}.timeout_ms must be > 0")
        if action_type == "update_config" and not isinstance(
            item.get("config"), dict
        ):
            raise ScenarioValidationError(
                f"{field}.config must be an object for update_config"
            )
        normalized_actions.append(normalized)

    limits = payload.get("limits", {})
    if not isinstance(limits, dict):
        raise ScenarioValidationError(f"{source}.limits must be an object")
    allowed_limits = {
        "max_clipped_samples",
        "max_continuous_speech_ms",
        "max_identical_word_run",
        "max_interrupt_ack_ms",
        "max_interrupt_yield_ms",
        "max_audio_after_ack_ms",
        "max_post_interrupt_active_ms",
        "max_words_per_second",
    }
    unknown_limits = set(limits) - allowed_limits
    if unknown_limits:
        raise ScenarioValidationError(
            f"{source}.limits has unknown keys: {sorted(unknown_limits)}"
        )
    normalized_limits = {
        key: _number(value, f"{source}.limits.{key}")
        for key, value in limits.items()
    }

    return {
        **payload,
        "description": description,
        "input_requirements": input_requirements,
        "audio": audio,
        "config": dict(config),
        "tail_ms": tail_ms,
        "actions": normalized_actions,
        "expectations": normalized_expectations,
        "limits": normalized_limits,
    }


def load_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScenarioValidationError(f"could not read {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ScenarioValidationError(f"invalid JSON in {manifest_path}: {exc}") from exc
    return validate_manifest(payload, source=str(manifest_path))


def resolve_input_wav(
    manifest_path: Path | str,
    manifest: dict[str, Any],
    override: Path | str | None = None,
) -> Path:
    """Resolve an explicit override or a manifest-relative input WAV."""
    if override is not None:
        path = Path(override).expanduser().resolve()
    else:
        audio = manifest.get("audio")
        if not audio:
            raise ScenarioValidationError(
                f"scenario {manifest['id']!r} does not bundle speech audio; "
                "pass --input-wav with a mono PCM16 48 kHz WAV matching its "
                "documented timing"
            )
        path = (Path(manifest_path).resolve().parent / audio).resolve()
    if not path.is_file():
        raise ScenarioValidationError(f"input WAV does not exist: {path}")
    return path


def load_pcm16_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Load the runner's intentionally strict WebRTC-ready WAV format."""
    wav_path = Path(path)
    try:
        with wave.open(str(wav_path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            compression = handle.getcomptype()
            frames = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise ScenarioValidationError(f"could not decode WAV {wav_path}: {exc}") from exc
    problems = []
    if channels != 1:
        problems.append(f"mono required, got {channels} channels")
    if width != 2:
        problems.append(f"PCM16 required, got {width * 8}-bit samples")
    if sample_rate != WEBRTC_SAMPLE_RATE:
        problems.append(
            f"48 kHz required, got {sample_rate} Hz (resample before running)"
        )
    if compression != "NONE":
        problems.append(f"uncompressed PCM required, got {compression}")
    if problems:
        raise ScenarioValidationError(f"invalid input WAV {wav_path}: {'; '.join(problems)}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        raise ScenarioValidationError(f"input WAV is empty: {wav_path}")
    return samples, sample_rate


def iter_audio_frames(
    samples: np.ndarray, *, frame_samples: int = FRAME_SAMPLES
) -> Iterable[np.ndarray]:
    """Yield fixed-size frames, zero-padding only the final partial frame."""
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    for offset in range(0, mono.size, frame_samples):
        frame = mono[offset : offset + frame_samples]
        if frame.size < frame_samples:
            frame = np.pad(frame, (0, frame_samples - frame.size))
        yield frame


def detect_speech_segments(
    samples: np.ndarray,
    sample_rate: int = WEBRTC_SAMPLE_RATE,
    config: VadConfig = VadConfig(),
) -> list[SpeechSegment]:
    """Segment assistant PCM with simple attack/release RMS hysteresis."""
    if sample_rate <= 0 or config.frame_ms <= 0:
        raise ValueError("sample rate and frame duration must be positive")
    frame_samples = sample_rate * config.frame_ms // 1000
    if frame_samples <= 0:
        raise ValueError("VAD frame has no samples")
    attack = max(1, int(config.attack_frames))
    release = max(1, int(config.release_frames))
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    voiced_streak = 0
    silent_streak = 0
    active_start: int | None = None
    segments: list[SpeechSegment] = []
    frame_count = math.ceil(mono.size / frame_samples)
    for frame_index in range(frame_count):
        frame = mono[frame_index * frame_samples : (frame_index + 1) * frame_samples]
        rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        voiced = rms >= config.rms_threshold
        if active_start is None:
            voiced_streak = voiced_streak + 1 if voiced else 0
            if voiced_streak >= attack:
                active_start = frame_index - attack + 1
                silent_streak = 0
        else:
            silent_streak = 0 if voiced else silent_streak + 1
            if silent_streak >= release:
                end_frame = frame_index - release + 1
                segments.append(
                    SpeechSegment(
                        start_ms=active_start * config.frame_ms,
                        end_ms=max(active_start, end_frame) * config.frame_ms,
                    )
                )
                active_start = None
                voiced_streak = 0
                silent_streak = 0
    if active_start is not None:
        segments.append(
            SpeechSegment(
                start_ms=active_start * config.frame_ms,
                end_ms=mono.size / sample_rate * 1000.0,
            )
        )
    return segments


def _overlap_ms(segment: SpeechSegment, start_ms: float, end_ms: float) -> float:
    return max(0.0, min(segment.end_ms, end_ms) - max(segment.start_ms, start_ms))


def _first_speech_at_or_after(
    segments: Sequence[SpeechSegment], at_ms: float, deadline_ms: float
) -> float | None:
    window_end = at_ms + deadline_ms
    for segment in segments:
        if at_ms <= segment.start_ms <= window_end:
            return segment.start_ms
    return None


def _message(event: dict[str, Any]) -> dict[str, Any]:
    message = event.get("message", {})
    return message if isinstance(message, dict) else {}


def _transcript_metrics(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    word_events: list[tuple[float, str]] = []
    word = ""
    word_started_at = 0.0
    for event in events:
        message = _message(event)
        if event.get("direction") != "in" or message.get("type") != "text":
            continue
        text = message.get("v", "")
        if not isinstance(text, str):
            continue
        timestamp = float(event.get("t_ms", 0.0))
        for character in text.lower():
            if character.isalnum() or character in {"_", "'"}:
                if not word:
                    word_started_at = timestamp
                word += character
            elif word:
                word_events.append((word_started_at, word))
                word = ""
    if word:
        word_events.append((word_started_at, word))
    words = [word for _, word in word_events]
    max_run = 0
    run = 0
    previous = None
    for word in words:
        run = run + 1 if word == previous else 1
        previous = word
        max_run = max(max_run, run)
    max_words_per_second = 0
    left = 0
    for right, (timestamp, _) in enumerate(word_events):
        while timestamp - word_events[left][0] > 1_000:
            left += 1
        max_words_per_second = max(max_words_per_second, right - left + 1)
    return {
        "text": " ".join(words),
        "word_count": len(words),
        "unique_word_ratio": (
            round(len(set(words)) / len(words), 4) if words else 1.0
        ),
        "max_identical_word_run": max_run,
        "max_words_per_second": max_words_per_second,
    }


def _interrupt_metrics(
    events: Sequence[dict[str, Any]], segments: Sequence[SpeechSegment]
) -> list[dict[str, Any]]:
    interrupts = [
        event
        for event in events
        if event.get("direction") == "out" and _message(event).get("type") == "interrupt"
    ]
    acks = [
        event
        for event in events
        if event.get("direction") == "in" and _message(event).get("type") == "interrupted"
    ]
    results = []
    ack_cursor = 0
    for event in interrupts:
        sent_ms = float(event.get("t_ms", 0.0))
        reason = str(_message(event).get("reason", "manual"))
        ack = None
        while ack_cursor < len(acks):
            candidate = acks[ack_cursor]
            ack_cursor += 1
            if float(candidate.get("t_ms", -1.0)) >= sent_ms:
                ack = candidate
                break
        ack_ms = float(ack["t_ms"]) if ack is not None else None
        containing = next(
            (
                segment
                for segment in segments
                if segment.start_ms <= sent_ms < segment.end_ms
            ),
            None,
        )
        yield_ms = max(0.0, containing.end_ms - sent_ms) if containing else 0.0
        ack_segment = (
            next(
                (
                    segment
                    for segment in segments
                    if segment.start_ms <= ack_ms < segment.end_ms
                ),
                None,
            )
            if ack_ms is not None
            else None
        )
        active_after_ack = (
            ack_segment.end_ms - ack_ms if ack_segment is not None else 0.0
        ) if ack_ms is not None else None
        post_ack_active = (
            sum(
                _overlap_ms(segment, ack_ms, float("inf"))
                for segment in segments
            )
            if ack_ms is not None
            else None
        )
        results.append(
            {
                "reason": reason,
                "sent_ms": round(sent_ms, 1),
                "ack_ms": round(ack_ms, 1) if ack_ms is not None else None,
                "ack_latency_ms": (
                    round(ack_ms - sent_ms, 1) if ack_ms is not None else None
                ),
                "audio_yield_ms": round(yield_ms, 1),
                "active_after_ack_ms": (
                    round(active_after_ack, 1)
                    if active_after_ack is not None
                    else None
                ),
                "post_ack_active_ms": (
                    round(post_ack_active, 1)
                    if post_ack_active is not None
                    else None
                ),
            }
        )
    return results


def analyze_scenario(
    manifest: dict[str, Any],
    output_samples: np.ndarray,
    events: Sequence[dict[str, Any]],
    *,
    sample_rate: int = WEBRTC_SAMPLE_RATE,
    vad: VadConfig = VadConfig(),
) -> dict[str, Any]:
    """Calculate model-independent interaction and stability metrics."""
    manifest = validate_manifest(manifest)
    pcm = np.asarray(output_samples, dtype=np.float32).reshape(-1)
    segments = detect_speech_segments(pcm, sample_rate, vad)
    pause_results = []
    turn_results = []
    event_results = []
    required_failures: list[str] = []
    threshold_failures: list[str] = []
    for expectation in manifest["expectations"]:
        label = str(expectation.get("label") or expectation["kind"])
        if expectation["kind"] == "pause":
            active_ms = sum(
                _overlap_ms(segment, expectation["start_ms"], expectation["end_ms"])
                for segment in segments
            )
            result = {
                "label": label,
                "start_ms": expectation["start_ms"],
                "end_ms": expectation["end_ms"],
                "assistant_active_ms": round(active_ms, 1),
            }
            maximum = expectation.get("max_assistant_active_ms")
            if maximum is not None and active_ms > maximum:
                threshold_failures.append(
                    f"pause {label!r}: assistant active {active_ms:.1f} ms > {maximum:.1f} ms"
                )
            pause_results.append(result)
        elif expectation["kind"] == "turn":
            boundary_segment = next(
                (
                    segment
                    for segment in segments
                    if segment.start_ms < expectation["at_ms"] < segment.end_ms
                ),
                None,
            )
            onset = _first_speech_at_or_after(
                segments, expectation["at_ms"], expectation["deadline_ms"]
            )
            latency = onset - expectation["at_ms"] if onset is not None else None
            result = {
                "label": label,
                "at_ms": expectation["at_ms"],
                "onset_ms": round(onset, 1) if onset is not None else None,
                "latency_ms": round(latency, 1) if latency is not None else None,
                "responded": onset is not None,
                "active_at_boundary": boundary_segment is not None,
                "boundary_overlap_ms": (
                    round(boundary_segment.end_ms - expectation["at_ms"], 1)
                    if boundary_segment is not None
                    else 0.0
                ),
            }
            if onset is None:
                required_failures.append(
                    f"turn {label!r}: no response within deadline"
                )
            maximum = expectation.get("max_latency_ms")
            if maximum is not None and latency is not None and latency > maximum:
                threshold_failures.append(
                    f"turn {label!r}: latency {latency:.1f} ms > {maximum:.1f} ms"
                )
            turn_results.append(result)
        else:
            window_start = expectation["at_ms"]
            window_end = window_start + expectation["deadline_ms"]
            match = next(
                (
                    event
                    for event in events
                    if event.get("direction") == "in"
                    and _message(event).get("type") == "event"
                    and _message(event).get("kind") == expectation["event_kind"]
                    and window_start <= float(event.get("t_ms", -1)) <= window_end
                ),
                None,
            )
            observed_ms = float(match["t_ms"]) if match is not None else None
            event_results.append(
                {
                    "label": label,
                    "event_kind": expectation["event_kind"],
                    "observed_ms": (
                        round(observed_ms, 1) if observed_ms is not None else None
                    ),
                    "observed": match is not None,
                }
            )
            if match is None:
                required_failures.append(
                    f"event {label!r}: {expectation['event_kind']!r} not observed "
                    "within deadline"
                )

    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
    clipped_samples = int(np.count_nonzero(np.abs(pcm) >= 32767.0 / 32768.0))
    transcript = _transcript_metrics(events)
    interrupt_results = _interrupt_metrics(events, segments)
    longest_speech_ms = max((segment.duration_ms for segment in segments), default=0.0)
    limits = manifest["limits"]
    observed_limits = {
        "max_clipped_samples": clipped_samples,
        "max_continuous_speech_ms": longest_speech_ms,
        "max_identical_word_run": transcript["max_identical_word_run"],
        "max_words_per_second": transcript["max_words_per_second"],
        "max_interrupt_ack_ms": max(
            (
                result["ack_latency_ms"]
                for result in interrupt_results
                if result["ack_latency_ms"] is not None
            ),
            default=0.0,
        ),
        "max_interrupt_yield_ms": max(
            (result["audio_yield_ms"] for result in interrupt_results),
            default=0.0,
        ),
        "max_audio_after_ack_ms": max(
            (
                result["active_after_ack_ms"]
                for result in interrupt_results
                if result["active_after_ack_ms"] is not None
            ),
            default=0.0,
        ),
        "max_post_interrupt_active_ms": max(
            (
                result["post_ack_active_ms"]
                for result in interrupt_results
                if result["post_ack_active_ms"] is not None
            ),
            default=0.0,
        ),
    }
    interrupt_limits = {
        "max_interrupt_ack_ms",
        "max_interrupt_yield_ms",
        "max_audio_after_ack_ms",
        "max_post_interrupt_active_ms",
    }
    if interrupt_limits.intersection(limits) and not interrupt_results:
        required_failures.append(
            "interrupt limits configured but no interrupt was sent"
        )
    elif "max_interrupt_ack_ms" in limits and any(
        result["ack_latency_ms"] is None for result in interrupt_results
    ):
        required_failures.append("interrupt acknowledgement was not observed")
    for name, maximum in limits.items():
        observed = observed_limits[name]
        if observed > maximum:
            threshold_failures.append(
                f"{name}: observed {observed} > limit {maximum}"
            )

    errors = [
        event
        for event in events
        if event.get("direction") == "in" and _message(event).get("type") == "error"
    ]
    runaway_flags = []
    if transcript["max_identical_word_run"] >= 6:
        runaway_flags.append("repeated_word_run")
    if transcript["max_words_per_second"] >= 20:
        runaway_flags.append("text_burst")
    if longest_speech_ms >= 30_000:
        runaway_flags.append("continuous_speech_30s")
    if clipped_samples:
        runaway_flags.append("clipped_pcm")

    failures = [*required_failures, *threshold_failures]

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": manifest["id"],
        "passed": not failures and not errors,
        "failures": failures,
        # Keep hard requirements distinct from numeric quality limits so the
        # live runner's --no-fail-on-thresholds switch can never hide a
        # missing turn, cap event, Stop acknowledgement, or server error.
        "required_failures": required_failures,
        "threshold_failures": threshold_failures,
        "errors": [_message(event) for event in errors],
        "duration_ms": round(pcm.size / sample_rate * 1000.0, 1),
        "pcm": {
            "peak": round(peak, 6),
            "rms": round(rms, 6),
            "clipped_samples": clipped_samples,
        },
        "speech": {
            "active_ms": round(sum(segment.duration_ms for segment in segments), 1),
            "longest_segment_ms": round(longest_speech_ms, 1),
            "segments": [asdict(segment) for segment in segments],
        },
        "pause": pause_results,
        "turn": turn_results,
        "event": event_results,
        "interrupt": interrupt_results,
        "transcript": transcript,
        "runaway_flags": runaway_flags,
    }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pcm16_wav(
    path: Path | str, samples: np.ndarray, sample_rate: int = WEBRTC_SAMPLE_RATE
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples).reshape(-1), -1.0, 1.0)
    encoded = (pcm * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(encoded.tobytes())


def write_artifacts(
    directory: Path | str,
    *,
    manifest: dict[str, Any],
    input_samples: np.ndarray,
    output_samples: np.ndarray,
    events: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    run: dict[str, Any],
) -> Path:
    """Write a self-contained replay/analysis directory for one run."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    write_pcm16_wav(target / "input.wav", input_samples)
    write_pcm16_wav(target / "output.wav", output_samples)
    (target / "scenario.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (target / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    return target
