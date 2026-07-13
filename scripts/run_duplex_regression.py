#!/usr/bin/env python3
"""Run reproducible duplex scenarios against an already-running server.

Example::

    uv run python scripts/run_duplex_regression.py \
      --base-url http://127.0.0.1:8998 \
      --input-wav /tmp/turn-taking.wav \
      moshi/tests/fixtures/duplex/turn_taking.json

The input must be mono PCM16 at 48 kHz. Each run creates input/output WAVs,
the complete control-channel event trace, applied identity/config metadata,
and CPU-computed interaction metrics under ``artifacts/duplex`` by default.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import fractions
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import aiohttp
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from av import AudioFrame
from av.audio.resampler import AudioResampler


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
ANALYZER_PATH = REPO_ROOT / "moshi" / "tests" / "duplex_harness.py"
sys.path.insert(0, str(REPO_ROOT / "moshi" / "tests"))
sys.path.insert(0, str(REPO_ROOT / "moshi"))

from duplex_harness import (  # noqa: E402
    FRAME_MS,
    FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
    ScenarioValidationError,
    VadConfig,
    analyze_scenario,
    load_manifest,
    load_pcm16_wav,
    resolve_input_wav,
    sha256_file,
    write_artifacts,
)
from moshi.rtc_session import SessionConfig, parse_session_config  # noqa: E402


STUN_FALLBACK = [
    {
        "urls": [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
        ]
    }
]
DEFAULT_REGRESSION_PROMPT = (
    "You enjoy talking with people. Speak as yourself: warm, perceptive, "
    "relaxed, and honest. Listen closely, say what you mean plainly, and "
    "keep turns short unless there is something worth unpacking."
)
APPLIED_CONFIG_KEYS = (
    "reinforce_in_silences",
    "vision_in_transcript",
    "vision_feed_model",
    "vision_ground_user_turns",
    "seed",
    "text_temperature",
    "audio_temperature",
    "text_topk",
    "audio_topk",
    "repetition_penalty",
    "repetition_penalty_context",
    "padding_bonus",
    "max_turn_text_tokens",
    "inject_silence_rms",
    "inject_silence_streak",
)
OPERATIONAL_EVENT_TYPES = {
    "action_cancelled",
    "action_error",
    "action_skipped",
    "action_timeout",
    "candidate_post_error",
    "candidate_stream_error",
}
OUTBOUND_DROPPED_MS_THRESHOLD = 200.0
STAT_COUNTER_FIELDS = (
    "pcm_queue_depth",
    "pcm_queue_capacity",
    "pcm_queue_high_water",
    "pcm_drop_events",
    "pcm_dropped_ms",
    "outbound_buffer_ms",
    "outbound_high_water_ms",
    "outbound_drop_events",
    "outbound_dropped_ms",
    "outbound_flush_events",
    "outbound_flushed_ms",
)


class RunnerError(RuntimeError):
    """A signaling, transport, or scenario execution failure."""


class EventRecorder:
    def __init__(self) -> None:
        self._origin: Optional[float] = None
        self._events: list[dict[str, Any]] = []

    def set_origin(self, origin: Optional[float] = None) -> float:
        self._origin = time.monotonic() if origin is None else origin
        return self._origin

    def record(
        self,
        direction: str,
        message: dict[str, Any],
        *,
        at: Optional[float] = None,
    ) -> None:
        self._events.append(
            {
                "at": time.monotonic() if at is None else at,
                "direction": direction,
                "message": message,
            }
        )

    def export(self) -> list[dict[str, Any]]:
        if self._origin is None:
            raise RunnerError("event timeline has no scenario origin")
        return [
            {
                "t_ms": round((event["at"] - self._origin) * 1000.0, 1),
                "direction": event["direction"],
                "message": event["message"],
            }
            for event in self._events
        ]


class ScriptedAudioTrack(MediaStreamTrack):
    """A continuously paced mic track that starts its fixture on command."""

    kind = "audio"

    def __init__(self, samples: np.ndarray) -> None:
        super().__init__()
        self._samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self._sample_offset = 0
        self._timestamp = 0
        self._clock_started_at: Optional[float] = None
        self._scenario_started = False

    def start_scenario(self) -> None:
        self._sample_offset = 0
        self._scenario_started = True

    async def recv(self) -> AudioFrame:
        loop = asyncio.get_running_loop()
        if self._clock_started_at is None:
            self._clock_started_at = loop.time()
        target = self._clock_started_at + (
            self._timestamp + FRAME_SAMPLES
        ) / WEBRTC_SAMPLE_RATE
        await asyncio.sleep(max(0.0, target - loop.time()))
        samples = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        if self._scenario_started and self._sample_offset < self._samples.size:
            end = min(self._sample_offset + FRAME_SAMPLES, self._samples.size)
            count = end - self._sample_offset
            samples[:count] = self._samples[self._sample_offset : end]
            self._sample_offset = end
        encoded = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        frame = AudioFrame.from_ndarray(
            encoded.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
        self._timestamp += FRAME_SAMPLES
        return frame


class LiveVad:
    """Online VAD used only for event-triggered actions such as Stop."""

    def __init__(self, config: VadConfig) -> None:
        self.config = config
        self._pending = np.empty(0, dtype=np.float32)
        self._voiced_streak = 0
        self._silent_streak = 0
        self._active = False
        self._active_frames = 0

    @property
    def active_duration_ms(self) -> float:
        return self._active_frames * self.config.frame_ms if self._active else 0.0

    def feed(self, samples: np.ndarray) -> None:
        frame_samples = WEBRTC_SAMPLE_RATE * self.config.frame_ms // 1000
        self._pending = np.concatenate(
            (self._pending, np.asarray(samples, dtype=np.float32).reshape(-1))
        )
        while self._pending.size >= frame_samples:
            frame = self._pending[:frame_samples]
            self._pending = self._pending[frame_samples:]
            rms = float(np.sqrt(np.mean(np.square(frame))))
            voiced = rms >= self.config.rms_threshold
            if not self._active:
                self._voiced_streak = self._voiced_streak + 1 if voiced else 0
                if self._voiced_streak >= max(1, self.config.attack_frames):
                    self._active = True
                    self._active_frames = self._voiced_streak
                    self._silent_streak = 0
            else:
                if voiced:
                    self._active_frames += 1
                    self._silent_streak = 0
                else:
                    self._silent_streak += 1
                    if self._silent_streak >= max(1, self.config.release_frames):
                        self._active = False
                        self._active_frames = 0
                        self._voiced_streak = 0
                        self._silent_streak = 0


class RemoteAudioCollector:
    """Resample and retain the actual remote WebRTC playout track."""

    def __init__(self, vad: VadConfig) -> None:
        self._resampler = AudioResampler(
            format="s16", layout="mono", rate=WEBRTC_SAMPLE_RATE
        )
        self._capturing = False
        self._chunks: list[np.ndarray] = []
        self.live_vad = LiveVad(vad)

    def start_scenario(self) -> None:
        self._chunks.clear()
        self.live_vad = LiveVad(self.live_vad.config)
        self._capturing = True

    async def consume(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                frame = await track.recv()
                for converted in self._resampler.resample(frame):
                    samples = converted.to_ndarray()
                    if samples.ndim == 2:
                        samples = samples[0]
                    f32 = samples.astype(np.float32) / 32768.0
                    if self._capturing:
                        self._chunks.append(f32.copy())
                        self.live_vad.feed(f32)
        except (MediaStreamError, asyncio.CancelledError):
            return

    def samples(self) -> np.ndarray:
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._chunks)


def _rtc_configuration(entries: list[dict[str, Any]]) -> RTCConfiguration:
    servers = []
    for entry in entries:
        urls = entry.get("urls")
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls:
            continue
        servers.append(
            RTCIceServer(
                urls=urls,
                username=entry.get("username"),
                credential=entry.get("credential"),
            )
        )
    return RTCConfiguration(iceServers=servers)


async def _json_response(response: aiohttp.ClientResponse, context: str) -> Any:
    body = await response.text()
    if response.status >= 400:
        raise RunnerError(f"{context} returned HTTP {response.status}: {body[:500]}")
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"{context} returned invalid JSON: {exc}") from exc


async def _fetch_server_info(
    http: aiohttp.ClientSession, base_url: str
) -> dict[str, Any]:
    async with http.get(f"{base_url}/api/info") as response:
        payload = await _json_response(response, "/api/info")
    return payload if isinstance(payload, dict) else {}


async def _fetch_ice_servers(
    http: aiohttp.ClientSession, base_url: str
) -> list[dict[str, Any]]:
    async with http.get(f"{base_url}/api/rtc/ice-servers") as response:
        if response.status >= 400:
            if response.status == 503:
                await _json_response(response, "/api/rtc/ice-servers")
            return STUN_FALLBACK
        payload = await _json_response(response, "/api/rtc/ice-servers")
    entries = payload.get("iceServers") if isinstance(payload, dict) else None
    return entries if isinstance(entries, list) and entries else STUN_FALLBACK


def _remote_candidate(payload: dict[str, Any]):
    candidate_sdp = payload.get("candidate")
    if not candidate_sdp:
        return None
    body = str(candidate_sdp)
    if body.startswith("candidate:"):
        body = body[len("candidate:") :]
    candidate = candidate_from_sdp(body)
    candidate.sdpMid = payload.get("sdpMid")
    candidate.sdpMLineIndex = payload.get("sdpMLineIndex")
    return candidate


async def _stream_server_candidates(
    http: aiohttp.ClientSession,
    base_url: str,
    session_id: str,
    pc: RTCPeerConnection,
    recorder: EventRecorder,
) -> None:
    url = f"{base_url}/api/rtc/candidates"
    try:
        async with http.get(url, params={"session_id": session_id}) as response:
            if response.status >= 400:
                recorder.record(
                    "transport",
                    {
                        "type": "candidate_stream_error",
                        "status": response.status,
                    },
                )
                return
            event_name = "message"
            data_lines: list[str] = []
            async for raw in response.content:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())
                elif not line:
                    if event_name == "done":
                        return
                    if data_lines:
                        payload = json.loads("\n".join(data_lines))
                        candidate = _remote_candidate(payload)
                        if candidate is not None:
                            await pc.addIceCandidate(candidate)
                    event_name = "message"
                    data_lines.clear()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        recorder.record(
            "transport",
            {"type": "candidate_stream_error", "error": f"{type(exc).__name__}: {exc}"},
        )


def _load_extra_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioValidationError(f"could not load --config-json {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScenarioValidationError("--config-json must contain a JSON object")
    return payload


def _build_exact_config(
    manifest: dict[str, Any], extra_config: dict[str, Any]
) -> dict[str, Any]:
    # Match the dashboard's fresh-session conditioning. Empty prompt/voice
    # dataclass defaults are protocol fallbacks, not a useful benchmark.
    defaults = asdict(
        SessionConfig(
            voice_prompt="NATF1.pt",
            text_prompt=DEFAULT_REGRESSION_PROMPT,
        )
    )
    known = set(defaults)
    overrides = {**manifest.get("config", {}), **extra_config}
    unknown = set(overrides) - known
    if unknown:
        raise ScenarioValidationError(f"unknown session config fields: {sorted(unknown)}")
    parsed = parse_session_config({"type": "config", **defaults, **overrides})
    return asdict(parsed)


def _validate_scenario_timeline(
    manifest: dict[str, Any], scenario_duration_ms: float
) -> None:
    """Reject actions or scoring windows that extend past captured audio."""
    for index, action in enumerate(manifest["actions"]):
        action_deadline = action.get("at_ms", action.get("timeout_ms", 0.0))
        if action_deadline >= scenario_duration_ms:
            raise ScenarioValidationError(
                f"actions[{index}] is scheduled at/after the scenario ends "
                f"({action_deadline:.0f} >= {scenario_duration_ms:.0f} ms); "
                "increase tail_ms"
            )
    for index, expectation in enumerate(manifest["expectations"]):
        if expectation["kind"] == "pause":
            window_end = expectation["end_ms"]
        else:
            window_end = expectation["at_ms"] + expectation["deadline_ms"]
        if window_end > scenario_duration_ms:
            raise ScenarioValidationError(
                f"expectations[{index}] {expectation['kind']!r} window extends "
                f"past the scenario capture ({window_end:.0f} > "
                f"{scenario_duration_ms:.0f} ms); increase tail_ms or shorten "
                "the deadline"
            )


def _valid_stat_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _runtime_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    stats = []
    for event in events:
        message = event.get("message", {})
        if (
            event.get("direction") == "in"
            and float(event.get("t_ms", -1.0)) >= 0.0
            and message.get("type") == "stat"
        ):
            stats.append(message)
    rtf = [float(stat["rtf"]) for stat in stats if _valid_stat_number(stat.get("rtf"))]
    result = {
        # The server's stat field is an EMA, so these summarize sampled EMA
        # values rather than pretending to be per-frame latency percentiles.
        "stat_samples": len(stats),
        "rtf_ema_samples": len(rtf),
        "rtf_ema_median": round(float(np.median(rtf)), 3) if rtf else None,
        "rtf_ema_p95": round(float(np.percentile(rtf, 95)), 3) if rtf else None,
        "rtf_ema_max": round(max(rtf), 3) if rtf else None,
        "connection_states": [
            event["message"].get("state")
            for event in events
            if event.get("direction") == "transport"
            and event.get("message", {}).get("type") == "connection_state"
        ],
    }
    for field in STAT_COUNTER_FIELDS:
        values = [
            float(stat[field])
            for stat in stats
            if _valid_stat_number(stat.get(field))
        ]
        metric_name = (
            f"{field}_max"
            if field in {"pcm_queue_depth", "outbound_buffer_ms"}
            else field
        )
        if not values:
            result[metric_name] = None
        elif field.endswith("events") or field.startswith("pcm_queue_"):
            result[metric_name] = int(max(values))
        else:
            result[metric_name] = round(max(values), 1)
    return result


def _queue_health_failures(
    runtime: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Classify input loss as hard and excessive output shedding as quality."""
    operational = []
    thresholds = []
    pcm_drop_events = runtime.get("pcm_drop_events")
    pcm_dropped_ms = runtime.get("pcm_dropped_ms")
    if (pcm_drop_events or 0) > 0 or (pcm_dropped_ms or 0.0) > 0.0:
        operational.append(
            "inbound PCM queue dropped audio: "
            f"{pcm_drop_events or 0} event(s), {pcm_dropped_ms or 0.0:.1f} ms"
        )
    outbound_dropped_ms = runtime.get("outbound_dropped_ms")
    if (
        outbound_dropped_ms is not None
        and outbound_dropped_ms > OUTBOUND_DROPPED_MS_THRESHOLD
    ):
        thresholds.append(
            "outbound backlog shedding: observed "
            f"{outbound_dropped_ms:.1f} ms > quality threshold "
            f"{OUTBOUND_DROPPED_MS_THRESHOLD:.1f} ms"
        )
    return operational, thresholds


def _config_values_match(expected: Any, observed: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(observed, bool) and observed is expected
    if isinstance(expected, float):
        return (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isclose(float(observed), expected, rel_tol=1e-6, abs_tol=1e-6)
        )
    return observed == expected


def _config_application_failures(
    requested: dict[str, Any], applied: dict[str, Any]
) -> list[str]:
    """Verify the replay-critical config snapshot acknowledged by the server."""
    if not applied:
        return ["server did not send a connect config_applied snapshot"]
    failures = []
    for key in APPLIED_CONFIG_KEYS:
        if key not in applied:
            failures.append(f"config_applied omitted {key!r}")
            continue
        expected = requested[key]
        observed = applied[key]
        if key == "seed" and expected in {None, -1}:
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or not 0 <= observed <= 2_147_483_647
            ):
                failures.append(
                    "config_applied did not resolve the random seed to a concrete integer"
                )
        elif not _config_values_match(expected, observed):
            failures.append(
                f"config_applied mismatch for {key!r}: "
                f"requested {expected!r}, observed {observed!r}"
            )
    return failures


def _operational_event_failures(events: list[dict[str, Any]]) -> list[str]:
    failures = []
    for event in events:
        message = event.get("message", {})
        if not isinstance(message, dict):
            continue
        event_type = message.get("type")
        if event_type in OPERATIONAL_EVENT_TYPES:
            detail = message.get("action") or message.get("error") or ""
            suffix = f": {detail}" if detail else ""
            failures.append(f"harness/transport {event_type}{suffix}")
    return failures


def _action_protocol_failures(
    manifest: dict[str, Any], events: list[dict[str, Any]]
) -> list[str]:
    """Require every scheduled action to be sent and acknowledged."""
    expected_actions = manifest["actions"]
    outgoing = [
        event
        for event in events
        if event.get("direction") == "out"
        and isinstance(event.get("message"), dict)
        and event["message"].get("type") in {"interrupt", "update_config"}
    ]
    failures = []
    if len(outgoing) != len(expected_actions):
        failures.append(
            f"scheduled {len(expected_actions)} action(s), sent {len(outgoing)}"
        )
    for index, (expected, observed) in enumerate(zip(expected_actions, outgoing)):
        if observed["message"].get("type") != expected["type"]:
            failures.append(
                f"action {index} expected {expected['type']!r}, sent "
                f"{observed['message'].get('type')!r}"
            )

    acknowledgements = [
        event
        for event in events
        if event.get("direction") == "in" and isinstance(event.get("message"), dict)
    ]
    ack_cursor = 0
    for event in outgoing:
        message = event["message"]
        sent_ms = float(event.get("t_ms", 0.0))
        expected_type = (
            "interrupted" if message["type"] == "interrupt" else "config_applied"
        )
        match = None
        while ack_cursor < len(acknowledgements):
            candidate = acknowledgements[ack_cursor]
            ack_cursor += 1
            candidate_message = candidate["message"]
            if float(candidate.get("t_ms", -1.0)) < sent_ms:
                continue
            if candidate_message.get("type") != expected_type:
                continue
            if expected_type == "config_applied" and candidate_message.get(
                "source"
            ) != "update":
                continue
            match = candidate
            break
        if match is None:
            failures.append(
                f"{message['type']} action at {sent_ms:.1f} ms was not acknowledged"
            )
            continue
        if message["type"] == "update_config":
            acknowledged = match["message"]
            applied_keys = acknowledged.get("applied", [])
            config = acknowledged.get("config", {})
            for key, expected in message.items():
                if key == "type":
                    continue
                if key not in applied_keys:
                    failures.append(
                        f"update_config acknowledgement omitted applied key {key!r}"
                    )
                elif key not in config:
                    failures.append(
                        f"update_config acknowledgement omitted config value {key!r}"
                    )
                elif not _config_values_match(expected, config[key]):
                    failures.append(
                        f"update_config mismatch for {key!r}: "
                        f"requested {expected!r}, observed {config[key]!r}"
                    )
    return failures


def _scenario_exit_failure(
    metrics: dict[str, Any], *, ignore_thresholds: bool
) -> bool:
    return bool(metrics.get("operational_failures")) or (
        bool(metrics.get("threshold_failures")) and not ignore_thresholds
    )


async def _wait_for_ready(
    ready: asyncio.Event,
    fatal: asyncio.Event,
    timeout: float,
) -> None:
    ready_task = asyncio.create_task(ready.wait())
    fatal_task = asyncio.create_task(fatal.wait())
    try:
        done, _ = await asyncio.wait(
            {ready_task, fatal_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise RunnerError(f"server did not become ready within {timeout:.0f} s")
        if fatal_task in done and fatal.is_set():
            raise RunnerError("session failed before ready; inspect recorded control events")
    finally:
        for task in (ready_task, fatal_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(ready_task, fatal_task, return_exceptions=True)


async def _run_action(
    action: dict[str, Any],
    *,
    origin: float,
    control,
    collector: RemoteAudioCollector,
    recorder: EventRecorder,
) -> None:
    loop = asyncio.get_running_loop()
    if "at_ms" in action:
        await asyncio.sleep(max(0.0, origin + action["at_ms"] / 1000.0 - loop.time()))
    else:
        deadline = loop.time() + action["timeout_ms"] / 1000.0
        target_ms = action["when"]["value"]
        while collector.live_vad.active_duration_ms < target_ms:
            if loop.time() >= deadline:
                recorder.record(
                    "harness",
                    {
                        "type": "action_timeout",
                        "action": action["type"],
                        "target_active_ms": target_ms,
                    },
                )
                return
            await asyncio.sleep(FRAME_MS / 1000.0)
    if control.readyState != "open":
        recorder.record(
            "harness",
            {"type": "action_skipped", "action": action["type"], "reason": "control_closed"},
        )
        return
    if action["type"] == "interrupt":
        message = {
            "type": "interrupt",
            "reason": str(action.get("reason") or "regression")[:64],
        }
    else:
        message = {"type": "update_config", **action["config"]}
    recorder.record("out", message)
    control.send(json.dumps(message))


async def run_scenario(
    *,
    base_url: str,
    manifest_path: Path,
    input_wav: Path,
    artifact_dir: Path,
    extra_config: dict[str, Any],
    ready_timeout: float,
    vad: VadConfig,
) -> tuple[dict[str, Any], Path]:
    manifest = load_manifest(manifest_path)
    input_samples, sample_rate = load_pcm16_wav(input_wav)
    if sample_rate != WEBRTC_SAMPLE_RATE:
        raise ScenarioValidationError("runner input must be 48 kHz")
    scenario_duration_ms = (
        input_samples.size / WEBRTC_SAMPLE_RATE * 1000.0 + manifest["tail_ms"]
    )
    _validate_scenario_timeline(manifest, scenario_duration_ms)
    exact_config = _build_exact_config(manifest, extra_config)
    recorder = EventRecorder()
    collector = RemoteAudioCollector(vad)
    mic_track = ScriptedAudioTrack(input_samples)
    run_started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_connect=20)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        server_info = await _fetch_server_info(http, base_url)
        ice_servers = await _fetch_ice_servers(http, base_url)
        pc = RTCPeerConnection(configuration=_rtc_configuration(ice_servers))
        control = pc.createDataChannel("control")
        pc.addTrack(mic_track)
        ready = asyncio.Event()
        fatal = asyncio.Event()
        ready_payload: dict[str, Any] = {}
        applied_config: dict[str, Any] = {}
        applied_config_source: Optional[str] = None
        output_task: Optional[asyncio.Task] = None
        candidate_task: Optional[asyncio.Task] = None
        pending_candidates: list[Any] = []
        session_id: Optional[str] = None

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            nonlocal output_task
            if track.kind == "audio":
                output_task = asyncio.create_task(collector.consume(track))

        @pc.on("connectionstatechange")
        async def on_connection_state() -> None:
            recorder.record(
                "transport", {"type": "connection_state", "state": pc.connectionState}
            )
            if pc.connectionState == "failed":
                fatal.set()

        async def post_candidate(candidate) -> None:
            if session_id is None:
                pending_candidates.append(candidate)
                return
            if candidate is None:
                payload = {"session_id": session_id, "candidate": None}
            else:
                payload = {
                    "session_id": session_id,
                    "candidate": "candidate:" + candidate_to_sdp(candidate),
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                }
            async with http.post(f"{base_url}/api/rtc/candidate", json=payload) as response:
                await _json_response(response, "/api/rtc/candidate")

        @pc.on("icecandidate")
        async def on_ice_candidate(candidate) -> None:
            try:
                await post_candidate(candidate)
            except Exception as exc:
                recorder.record(
                    "transport",
                    {"type": "candidate_post_error", "error": f"{type(exc).__name__}: {exc}"},
                )

        @control.on("open")
        def on_control_open() -> None:
            payload = {"type": "config", **exact_config}
            recorder.record("out", payload)
            control.send(json.dumps(payload))

        @control.on("message")
        def on_control_message(raw: Any) -> None:
            nonlocal applied_config_source
            if not isinstance(raw, str):
                return
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                message = {"type": "invalid_json", "raw": raw[:500]}
            if not isinstance(message, dict):
                message = {"type": "invalid_message", "value": message}
            recorder.record("in", message)
            if message.get("type") == "ready":
                ready_payload.update(message)
                ready.set()
            elif message.get("type") == "config_applied":
                config = message.get("config")
                source = message.get("source")
                if isinstance(config, dict) and (
                    source == "connect" or not applied_config
                ):
                    applied_config.clear()
                    applied_config.update(config)
                    applied_config_source = str(source or "unknown")
            elif message.get("type") in {"error", "end"}:
                fatal.set()

        @control.on("close")
        def on_control_close() -> None:
            recorder.record("transport", {"type": "control_state", "state": "closed"})
            fatal.set()

        try:
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            local = pc.localDescription
            if local is None:
                raise RunnerError("aiortc did not produce a local description")
            async with http.post(
                f"{base_url}/api/rtc/offer",
                json={"sdp": local.sdp, "type": local.type},
            ) as response:
                answer = await _json_response(response, "/api/rtc/offer")
            session_id = str(answer.get("session_id") or "")
            if not session_id:
                raise RunnerError("offer response omitted session_id")
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            )
            for candidate in pending_candidates:
                await post_candidate(candidate)
            pending_candidates.clear()
            candidate_task = asyncio.create_task(
                _stream_server_candidates(http, base_url, session_id, pc, recorder)
            )
            await _wait_for_ready(ready, fatal, ready_timeout)

            origin = time.monotonic()
            recorder.set_origin(origin)
            collector.start_scenario()
            mic_track.start_scenario()
            recorder.record(
                "harness",
                {
                    "type": "scenario_started",
                    "scenario_id": manifest["id"],
                    "input_samples": int(input_samples.size),
                },
                at=origin,
            )
            action_tasks = [
                asyncio.create_task(
                    _run_action(
                        action,
                        origin=origin,
                        control=control,
                        collector=collector,
                        recorder=recorder,
                    )
                )
                for action in manifest["actions"]
            ]
            input_duration = input_samples.size / WEBRTC_SAMPLE_RATE
            run_duration = input_duration + manifest["tail_ms"] / 1000.0
            run_timer = asyncio.create_task(asyncio.sleep(run_duration))
            fatal_wait = asyncio.create_task(fatal.wait())
            done, _ = await asyncio.wait(
                {run_timer, fatal_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            ended_early = fatal_wait in done and fatal.is_set()
            for task in (run_timer, fatal_wait):
                if not task.done():
                    task.cancel()
            await asyncio.gather(run_timer, fatal_wait, return_exceptions=True)
            for task in action_tasks:
                if not task.done():
                    task.cancel()
            if action_tasks:
                action_results = await asyncio.gather(
                    *action_tasks, return_exceptions=True
                )
                for action, result in zip(manifest["actions"], action_results):
                    if isinstance(result, asyncio.CancelledError):
                        recorder.record(
                            "harness",
                            {"type": "action_cancelled", "action": action["type"]},
                        )
                    elif isinstance(result, BaseException):
                        recorder.record(
                            "harness",
                            {
                                "type": "action_error",
                                "action": action["type"],
                                "error": f"{type(result).__name__}: {result}",
                            },
                        )
            recorder.record("harness", {"type": "scenario_finished"})
            if control.readyState == "open":
                goodbye = {"type": "goodbye"}
                recorder.record("out", goodbye)
                control.send(json.dumps(goodbye))
                await asyncio.sleep(0.15)
            output_samples = collector.samples()
            events = recorder.export()
            metrics = analyze_scenario(
                manifest,
                output_samples,
                events,
                sample_rate=WEBRTC_SAMPLE_RATE,
                vad=vad,
            )
            runtime_metrics = _runtime_metrics(events)
            metrics["runtime"] = runtime_metrics
            operational_failures = list(metrics.pop("required_failures", []))
            operational_failures.extend(
                "server error: "
                f"{error.get('error') or error.get('message') or repr(error)}"
                for error in metrics.get("errors", [])
            )
            operational_failures.extend(
                _config_application_failures(exact_config, applied_config)
            )
            if applied_config_source != "connect":
                operational_failures.append(
                    "initial config_applied snapshot did not have source='connect'"
                )
            operational_failures.extend(_operational_event_failures(events))
            operational_failures.extend(_action_protocol_failures(manifest, events))
            queue_operational, queue_thresholds = _queue_health_failures(
                runtime_metrics
            )
            operational_failures.extend(queue_operational)
            metrics["threshold_failures"].extend(queue_thresholds)
            if runtime_metrics["rtf_ema_samples"] == 0:
                operational_failures.append(
                    "server emitted no numeric RTF EMA samples during the scenario"
                )
            if ended_early:
                operational_failures.append(
                    "session ended before the scenario input and analysis tail completed"
                )
            # De-duplicate correlated evidence (for example a timed-out action
            # also implies an unsent action) while preserving event order.
            metrics["operational_failures"] = list(
                dict.fromkeys(operational_failures)
            )
            metrics["failures"] = [
                *metrics["operational_failures"],
                *metrics.get("threshold_failures", []),
            ]
            metrics["passed"] = not metrics["failures"]
            run_metadata = {
                "schema_version": 1,
                "started_at": run_started_at,
                "base_url": base_url,
                "session_id": session_id,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "input_path": str(input_wav.resolve()),
                "input_sha256": sha256_file(input_wav),
                "input_duration_ms": round(input_duration * 1000.0, 1),
                "requested_config": exact_config,
                "applied_config": applied_config,
                "applied_config_source": applied_config_source,
                "server_info": server_info,
                "ready": ready_payload,
                "vad": asdict(vad),
                "tooling": {
                    "runner": {
                        "path": str(RUNNER_PATH.relative_to(REPO_ROOT)),
                        "sha256": sha256_file(RUNNER_PATH),
                    },
                    "analyzer": {
                        "path": str(ANALYZER_PATH.relative_to(REPO_ROOT)),
                        "sha256": sha256_file(ANALYZER_PATH),
                    },
                },
            }
            target = write_artifacts(
                artifact_dir,
                manifest=manifest,
                input_samples=input_samples,
                output_samples=output_samples,
                events=events,
                metrics=metrics,
                run=run_metadata,
            )
            return metrics, target
        finally:
            if candidate_task is not None and not candidate_task.done():
                candidate_task.cancel()
                await asyncio.gather(candidate_task, return_exceptions=True)
            await pc.close()
            if output_task is not None and not output_task.done():
                output_task.cancel()
                await asyncio.gather(output_task, return_exceptions=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        type=Path,
        nargs="+",
        help="Versioned duplex scenario JSON files, run sequentially",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8998",
        help="Already-running PersonaPlex server",
    )
    parser.add_argument(
        "--input-wav",
        type=Path,
        help="Override a single manifest's relative audio path",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help="SessionConfig overrides applied after each manifest's config",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "duplex",
    )
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--vad-rms", type=float, default=VadConfig().rms_threshold)
    parser.add_argument("--vad-attack-frames", type=int, default=VadConfig().attack_frames)
    parser.add_argument("--vad-release-frames", type=int, default=VadConfig().release_frames)
    parser.add_argument(
        "--no-fail-on-thresholds",
        action="store_true",
        help=(
            "Return success for numeric quality-limit overruns only; transport, "
            "server, config, action, and required-expectation failures remain fatal"
        ),
    )
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    if args.input_wav is not None and len(args.manifests) != 1:
        raise ScenarioValidationError(
            "--input-wav is unambiguous only with one manifest; use manifest-relative "
            "audio paths for sequential runs"
        )
    if not math.isfinite(args.ready_timeout) or args.ready_timeout <= 0:
        raise ScenarioValidationError("--ready-timeout must be finite and positive")
    vad = VadConfig(
        rms_threshold=args.vad_rms,
        attack_frames=args.vad_attack_frames,
        release_frames=args.vad_release_frames,
    )
    if (
        not math.isfinite(vad.rms_threshold)
        or vad.rms_threshold <= 0
        or vad.attack_frames <= 0
        or vad.release_frames <= 0
    ):
        raise ScenarioValidationError("VAD settings must be finite and positive")
    extra_config = _load_extra_config(args.config_json)
    resolved = []
    for manifest_path in args.manifests:
        manifest = load_manifest(manifest_path)
        input_path = resolve_input_wav(manifest_path, manifest, args.input_wav)
        load_pcm16_wav(input_path)
        _build_exact_config(manifest, extra_config)
        resolved.append((manifest_path, manifest, input_path))

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_root = args.artifacts_dir / stamp
    exit_failures = 0
    for manifest_path, manifest, input_path in resolved:
        print(f"running {manifest['id']} against {args.base_url} ...", flush=True)
        metrics, target = await run_scenario(
            base_url=args.base_url.rstrip("/"),
            manifest_path=manifest_path,
            input_wav=input_path,
            artifact_dir=run_root / manifest["id"],
            extra_config=extra_config,
            ready_timeout=args.ready_timeout,
            vad=vad,
        )
        exit_failure = _scenario_exit_failure(
            metrics, ignore_thresholds=args.no_fail_on_thresholds
        )
        if metrics["passed"]:
            status = "PASS"
        elif not exit_failure:
            status = "THRESHOLD FAIL (ignored)"
        else:
            status = "FAIL"
        print(f"  {status}: {target}")
        if exit_failure:
            exit_failures += 1
        if not metrics["passed"]:
            for failure in metrics["failures"]:
                print(f"    - {failure}")
    return 0 if exit_failures == 0 else 1


def main() -> int:
    args = _parser().parse_args()
    try:
        return asyncio.run(_async_main(args))
    except (ScenarioValidationError, RunnerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
