"""WebRTC session for streaming audio between a browser and Moshi.

Replaces the raw-PCM-over-WebSocket transport with a standard WebRTC
peer connection. Inbound audio comes in as Opus-decoded 48 kHz mono
frames from aiortc, gets resampled to Moshi's native 24 kHz float32,
and feeds the same `_process_audio_frame` callback the WebSocket path
used. Outbound audio (Mimi-decoded TTS) goes the other way: 24 kHz
float32 chunks are pushed onto a `MimiOutputTrack`, resampled to 48 kHz,
and emitted at real-time pace for aiortc to Opus-encode.

Text and control messages travel on a single bidirectional
`RTCDataChannel` ("control"). All message shapes are JSON.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Executor
import fractions
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional

import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler


MIMI_SAMPLE_RATE = 24_000
WEBRTC_SAMPLE_RATE = 48_000
OUTBOUND_FRAME_MS = 20
OUTBOUND_FRAME_SAMPLES = WEBRTC_SAMPLE_RATE * OUTBOUND_FRAME_MS // 1000  # 960
OUTBOUND_BUFFER_CAP_SAMPLES = WEBRTC_SAMPLE_RATE * 2  # 2 seconds; sanity cap.
# Backlog level at which recv() drops stale decoded content. One Mimi
# decode lump is 80 ms (3840 samples at 48 kHz) and the healthy buffer
# sawtooths between zero and roughly one lump, so two lumps of queued
# audio means a stall left standing latency behind.
OUTBOUND_DRAIN_BACKLOG_SAMPLES = OUTBOUND_FRAME_SAMPLES * 8
# A stall can also leave a sub-threshold backlog (1-7 frames) that the level
# gate above never sees: production and consumption both advance at 1x, so
# the residue persists as permanent reply latency. The floor of the backlog
# over a window of recv() calls measures that standing latency directly --
# the healthy sawtooth touches zero every production period, so a floor of
# a frame or more means a stall left residue behind. 25 calls ~= 500 ms
# spans several 80 ms production lumps plus jitter.
OUTBOUND_STANDING_BACKLOG_WINDOW = 25
# Resampling can make a nominal 20 ms residue a few dozen samples short of a
# full RTP frame. Treat a persistent half-frame floor as standing latency;
# healthy production still reaches zero during each 80 ms sawtooth.
OUTBOUND_STANDING_BACKLOG_MIN_SAMPLES = OUTBOUND_FRAME_SAMPLES // 2
CONTROL_TASK_MAX = 128

# STUN-only fallback used when no TURN credentials are configured. Works
# only when both peers can reach each other directly over UDP, which is
# not the case behind RunPod's HTTPS-only proxy. Production deployments
# should provide TURN credentials via the env vars consumed in server.py.
DEFAULT_STUN_FALLBACK: tuple[dict, ...] = (
    {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
)


ProcessFrameFn = Callable[[np.ndarray], list[tuple[np.ndarray, Optional[str]]]]
LogFn = Callable[[str, str], None]
BackpressureStatusFn = Callable[[], str]


def ice_servers_to_aiortc(servers: list[dict]) -> list[RTCIceServer]:
    """Translate a Cloudflare-style iceServers JSON list to aiortc objects.

    Accepts entries shaped like ``{"urls": [...], "username": "...",
    "credential": "..."}`` (matching what the browser ``RTCPeerConnection``
    constructor expects), so server- and client-side configs can share
    the same on-the-wire format.
    """
    out: list[RTCIceServer] = []
    for entry in servers:
        urls = entry.get("urls")
        if isinstance(urls, str):
            urls = [urls]
        if not urls:
            continue
        out.append(
            RTCIceServer(
                urls=list(urls),
                username=entry.get("username"),
                credential=entry.get("credential"),
            )
        )
    return out


def _f32_to_s16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _s16_to_f32(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / 32768.0


def _frame_to_mono_24k_f32(frame: AudioFrame, resampler: AudioResampler) -> np.ndarray:
    """Resample one inbound aiortc audio frame to mono 24 kHz float32.

    Returns a 1D array. May be empty if the resampler is still buffering.
    """
    out_frames = resampler.resample(frame)
    if not out_frames:
        return np.empty(0, dtype=np.float32)
    pieces = []
    for out in out_frames:
        arr = out.to_ndarray()
        # AudioResampler with layout='mono' returns shape (1, N) packed
        # planar; flatten defensively.
        if arr.ndim == 2:
            arr = arr[0]
        pieces.append(_s16_to_f32(arr))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)


class MimiOutputTrack(MediaStreamTrack):
    """Outbound audio track. Pulls 48 kHz s16 mono frames from a buffer.

    The buffer is fed by `push_24k_f32`. `recv()` paces at real time so
    aiortc emits Opus packets at the steady ~20 ms cadence its sender
    expects; if the buffer is short, recv emits silence rather than
    blocking, which keeps the codec timeline alive across GPU stalls.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_lock = asyncio.Lock()
        self._timestamp = 0
        self._start_time: Optional[float] = None
        # Rolling backlog readings, one per recv(); the window minimum is
        # the standing latency the drain logic in recv() sheds.
        self._backlog_window: deque[int] = deque(
            maxlen=OUTBOUND_STANDING_BACKLOG_WINDOW
        )
        # Cumulative transport-health counters. These stay session-local and
        # are sampled by the slow diagnostics task; no per-frame control
        # messages are emitted from the audio path.
        self._buffer_high_water = 0
        self._drop_events = 0
        self._dropped_samples = 0
        self._flush_events = 0
        self._flushed_samples = 0
        # Persistent resampler so its internal anti-alias filter state
        # carries between chunks. Recreating per-call breaks continuity.
        self._resampler = AudioResampler(
            format="s16", layout="mono", rate=WEBRTC_SAMPLE_RATE
        )

    async def push_24k_f32(self, samples: np.ndarray) -> None:
        """Append Mimi-decoded mono 24 kHz float32 samples to the buffer."""
        if samples.size == 0:
            return
        # Resample 24k -> 48k via PyAV. The resampler wants an AudioFrame.
        s16 = _f32_to_s16(samples)
        # AudioFrame.from_ndarray expects shape (channels, N) for planar.
        in_frame = AudioFrame.from_ndarray(
            s16.reshape(1, -1), format="s16", layout="mono"
        )
        in_frame.sample_rate = MIMI_SAMPLE_RATE
        out_frames = self._resampler.resample(in_frame)
        chunks: list[np.ndarray] = []
        for out in out_frames:
            arr = out.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            chunks.append(_s16_to_f32(arr))
        if not chunks:
            return
        upsampled = np.concatenate(chunks)
        async with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, upsampled])
            self._buffer_high_water = max(
                self._buffer_high_water, self._buffer.size
            )
            # Sanity cap: if recv() falls way behind, drop oldest.
            if self._buffer.size > OUTBOUND_BUFFER_CAP_SAMPLES:
                dropped = self._buffer.size - OUTBOUND_BUFFER_CAP_SAMPLES
                self._buffer = self._buffer[-OUTBOUND_BUFFER_CAP_SAMPLES:]
                self._drop_events += 1
                self._dropped_samples += dropped

    async def clear_buffer(self) -> None:
        """Drop queued assistant audio immediately."""
        async with self._buffer_lock:
            if self._buffer.size:
                self._flush_events += 1
                self._flushed_samples += self._buffer.size
            self._buffer = np.empty(0, dtype=np.float32)
            # Pre-flush readings would keep the standing-backlog floor high
            # for another window; the flush already shed that latency.
            self._backlog_window.clear()
            self._resampler = AudioResampler(
                format="s16", layout="mono", rate=WEBRTC_SAMPLE_RATE
            )

    async def _pop_chunk(self) -> np.ndarray:
        async with self._buffer_lock:
            if self._buffer.size >= OUTBOUND_FRAME_SAMPLES:
                chunk = self._buffer[:OUTBOUND_FRAME_SAMPLES]
                self._buffer = self._buffer[OUTBOUND_FRAME_SAMPLES:]
                return chunk
            return np.zeros(OUTBOUND_FRAME_SAMPLES, dtype=np.float32)

    async def recv(self) -> AudioFrame:
        loop = asyncio.get_event_loop()
        if self._start_time is None:
            self._start_time = loop.time()
        frame_duration = OUTBOUND_FRAME_SAMPLES / WEBRTC_SAMPLE_RATE
        # Pace real-time: target the next frame boundary based on cumulative
        # timestamp, sleep if we're early. If the sender was stalled for more
        # than one frame, rebase the wall-clock origin instead of bursting all
        # overdue RTP frames in a tight loop to catch up.
        target = self._start_time + (
            self._timestamp + OUTBOUND_FRAME_SAMPLES
        ) / WEBRTC_SAMPLE_RATE
        now = loop.time()
        rebased = False
        if now - target >= frame_duration:
            self._start_time = now - (
                (self._timestamp + OUTBOUND_FRAME_SAMPLES) / WEBRTC_SAMPLE_RATE
            )
            target = now
            rebased = True
        delay = target - now
        # A rebase fixes sender pacing but can strand decoded speech in this
        # buffer. Sending RTP packets at 2x does not make a receiver play them
        # at 2x (timestamps still advance at the audio clock); it only grows
        # the remote jitter buffer. Shed stale *content* here instead, then
        # keep a steady 20 ms RTP cadence.
        async with self._buffer_lock:
            backlog = self._buffer.size
            self._backlog_window.append(backlog)
            standing = (
                len(self._backlog_window) == self._backlog_window.maxlen
                and min(self._backlog_window)
                >= OUTBOUND_STANDING_BACKLOG_MIN_SAMPLES
            )
            if (rebased and backlog >= OUTBOUND_DRAIN_BACKLOG_SAMPLES) or (
                standing and backlog >= OUTBOUND_FRAME_SAMPLES * 2
            ):
                target = (
                    OUTBOUND_FRAME_SAMPLES * 4
                    if backlog >= OUTBOUND_DRAIN_BACKLOG_SAMPLES
                    else OUTBOUND_FRAME_SAMPLES
                )
                drop = max(0, backlog - target)
                if drop > 0:
                    self._buffer = self._buffer[drop:]
                    self._drop_events += 1
                    self._dropped_samples += drop
                    self._backlog_window.clear()
                    self._backlog_window.append(self._buffer.size)
        if delay > 0:
            await asyncio.sleep(delay)

        chunk = await self._pop_chunk()
        s16 = _f32_to_s16(chunk).reshape(1, -1)
        frame = AudioFrame.from_ndarray(s16, format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
        self._timestamp += OUTBOUND_FRAME_SAMPLES
        return frame

    async def diagnostics_snapshot(self) -> dict[str, int | float]:
        """Return cumulative outbound-buffer health without exposing audio."""
        async with self._buffer_lock:
            return {
                "outbound_buffer_ms": round(
                    self._buffer.size / WEBRTC_SAMPLE_RATE * 1000.0, 1
                ),
                "outbound_high_water_ms": round(
                    self._buffer_high_water / WEBRTC_SAMPLE_RATE * 1000.0, 1
                ),
                "outbound_drop_events": self._drop_events,
                "outbound_dropped_ms": round(
                    self._dropped_samples / WEBRTC_SAMPLE_RATE * 1000.0, 1
                ),
                "outbound_flush_events": self._flush_events,
                "outbound_flushed_ms": round(
                    self._flushed_samples / WEBRTC_SAMPLE_RATE * 1000.0, 1
                ),
            }


# Numeric config bounds are shared by connect-time parsing and the server's
# live update path. Finite out-of-range values clamp; malformed and non-finite
# values reject the update before they can reach sampling or accounting state.
TEMPERATURE_MIN = 0.1
TEMPERATURE_MAX = 1.5
TEXT_TOPK_MIN = 1
TEXT_TOPK_MAX = 500
AUDIO_TOPK_MIN = 1
AUDIO_TOPK_MAX = 2048
REPETITION_PENALTY_MIN = 1.0
REPETITION_PENALTY_MAX = 2.0
REPETITION_PENALTY_CONTEXT_MIN = 0
REPETITION_PENALTY_CONTEXT_MAX = 256
PADDING_BONUS_MIN = 0.0
PADDING_BONUS_MAX = 6.0
# Keep the circuit breaker and its auto-rewind signal available for every
# user-supplied configuration. LMGen still supports 0 internally so manual
# interrupt force windows remain independently testable.
MAX_TURN_TEXT_TOKENS_MIN = 40
MAX_TURN_TEXT_TOKENS_MAX = 2000
SESSION_TIMEOUT_SEC_MIN = 0
SESSION_TIMEOUT_SEC_MAX = 3600
VISION_COST_LIMIT_USD_MIN = 0.0
VISION_COST_LIMIT_USD_MAX = 10.0
VISION_COST_PER_CALL_USD_MIN = 0.0
VISION_COST_PER_CALL_USD_MAX = 10.0
SEED_RANDOM = -1
SEED_MIN = 0
SEED_MAX = 2_147_483_647
VOICE_BLEND_MIX_MIN = 0.0
VOICE_BLEND_MIX_MAX = 1.0
CLONE_STRENGTH_MIN = 0.0
CLONE_STRENGTH_MAX = 1.0


def _coerce_finite_float(value, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return out


def _clamp_float(value, minimum: float, maximum: float, field: str) -> float:
    out = _coerce_finite_float(value, field)
    return min(maximum, max(minimum, out))


def _clamp_int(value, minimum: int, maximum: int, field: str) -> int:
    out = _coerce_finite_float(value, field)
    return min(maximum, max(minimum, int(out)))


def clamp_temperature(value) -> float:
    """Coerce ``value`` to a finite temperature within slider bounds.

    Raises ``ValueError`` for non-finite input: ``float()`` accepts the
    strings "nan"/"inf", and ``json.loads`` accepts bare NaN/Infinity
    literals, so JSON parsing alone does not keep these out.
    """
    return _clamp_float(value, TEMPERATURE_MIN, TEMPERATURE_MAX, "temperature")


def clamp_text_topk(value) -> int:
    return _clamp_int(value, TEXT_TOPK_MIN, TEXT_TOPK_MAX, "text_topk")


def clamp_audio_topk(value) -> int:
    return _clamp_int(value, AUDIO_TOPK_MIN, AUDIO_TOPK_MAX, "audio_topk")


def clamp_repetition_penalty(value) -> float:
    return _clamp_float(
        value,
        REPETITION_PENALTY_MIN,
        REPETITION_PENALTY_MAX,
        "repetition_penalty",
    )


def clamp_repetition_penalty_context(value) -> int:
    return _clamp_int(
        value,
        REPETITION_PENALTY_CONTEXT_MIN,
        REPETITION_PENALTY_CONTEXT_MAX,
        "repetition_penalty_context",
    )


def clamp_padding_bonus(value) -> float:
    return _clamp_float(value, PADDING_BONUS_MIN, PADDING_BONUS_MAX, "padding_bonus")


def clamp_max_turn_text_tokens(value) -> int:
    return _clamp_int(
        value,
        MAX_TURN_TEXT_TOKENS_MIN,
        MAX_TURN_TEXT_TOKENS_MAX,
        "max_turn_text_tokens",
    )


def clamp_session_timeout_sec(value) -> int:
    return _clamp_int(
        value,
        SESSION_TIMEOUT_SEC_MIN,
        SESSION_TIMEOUT_SEC_MAX,
        "session_timeout_sec",
    )


def clamp_vision_cost_limit_usd(value) -> float:
    return _clamp_float(
        value,
        VISION_COST_LIMIT_USD_MIN,
        VISION_COST_LIMIT_USD_MAX,
        "vision_cost_limit_usd",
    )


def clamp_vision_cost_per_call_usd(value) -> float:
    return _clamp_float(
        value,
        VISION_COST_PER_CALL_USD_MIN,
        VISION_COST_PER_CALL_USD_MAX,
        "vision_cost_per_call_usd",
    )


def clamp_seed(value) -> Optional[int]:
    if value is None:
        return None
    out = _coerce_finite_float(value, "seed")
    if out == SEED_RANDOM:
        return SEED_RANDOM
    return min(SEED_MAX, max(SEED_MIN, int(out)))


def clamp_voice_blend_mix(value) -> float:
    return _clamp_float(
        value,
        VOICE_BLEND_MIX_MIN,
        VOICE_BLEND_MIX_MAX,
        "voice_blend_mix",
    )


def clamp_clone_strength(value) -> float:
    return _clamp_float(
        value,
        CLONE_STRENGTH_MIN,
        CLONE_STRENGTH_MAX,
        "clone_strength",
    )


# End-of-thought inject-gate bounds, shared by the connect-time parse and
# the live update_config path. A queued vision caption (or persona
# reinforcement) is only injected once the model's decoded audio has stayed
# below inject_silence_rms for inject_silence_streak consecutive frames,
# i.e. the current utterance has finished, so the context lands in the
# trailing silence instead of cutting speech. The RMS floor is measured on
# the model's decoded output (float PCM ~[-1, 1]); its useful range sits
# near the input-side ASR silence threshold (0.005). The streak is in
# ~80 ms frames (12.5 Hz).
INJECT_SILENCE_RMS_MIN = 0.001
INJECT_SILENCE_RMS_MAX = 0.02
INJECT_SILENCE_RMS_DEFAULT = 0.01
INJECT_SILENCE_STREAK_MIN = 4
INJECT_SILENCE_STREAK_MAX = 20
INJECT_SILENCE_STREAK_DEFAULT = 6


def clamp_inject_silence_rms(value) -> float:
    """Coerce ``value`` to a finite RMS floor within bounds.

    Rejects non-finite input like ``clamp_temperature``: a NaN floor would
    make every frame compare as non-silent (``x < nan`` is always False),
    silently disabling the whole inject path.
    """
    return _clamp_float(
        value,
        INJECT_SILENCE_RMS_MIN,
        INJECT_SILENCE_RMS_MAX,
        "inject_silence_rms",
    )


def clamp_inject_silence_streak(value) -> int:
    """Coerce ``value`` to an int silence streak within bounds.

    Parses through ``float`` first so a non-finite input raises ValueError
    rather than ``int()`` raising OverflowError on infinity.
    """
    return _clamp_int(
        value,
        INJECT_SILENCE_STREAK_MIN,
        INJECT_SILENCE_STREAK_MAX,
        "inject_silence_streak",
    )


# Ceiling for one inbound vision frame, in base64 characters. Real frames
# at /2 downscale + JPEG 0.55 are well under 100 KB; native + 0.8 detail
# mode stays under ~400 KB. 600 KB headroom catches both without exposing
# the server to a runaway client. Applies to single vision_frame messages
# and to the combined size of a chunked frame.
VISION_FRAME_MAX_CHARS = 600_000

# Chunked vision-frame reassembly bounds. The control channel's SCTP
# messages are capped at 64 KB, so the client splits larger frames into
# ordered vision_frame_chunk parts keyed by frame_id. At most this many
# half-built frames are held at once (the newest capture evicts the
# oldest), each frame carries at most this many parts, and a partial
# older than the stale window is dropped (its sender has moved on).
VISION_CHUNK_MAX_PARTIALS = 4
VISION_CHUNK_MAX_PARTS = 16
VISION_CHUNK_STALE_SEC = 10.0


def reassemble_vision_chunk(
    partials: dict,
    msg: dict,
    now_mono: float,
    log: Callable[[str, str], None],
) -> Optional[dict]:
    """Fold one vision_frame_chunk message into ``partials``.

    ``partials`` maps frame_id to an accumulator dict and is mutated in
    place; insertion order doubles as eviction order. Returns the completed
    vision_frame message once every part of a frame has arrived, else None.
    A malformed part drops the whole partial: its sender is confused, and a
    half-trusted frame reaching Gemini is worse than a missed capture.
    """
    frame_id = str(msg.get("frame_id") or "")[:128]
    data = msg.get("data", "")
    try:
        seq = int(msg.get("seq"))
        total = int(msg.get("total"))
        source_generation = _clamp_int(
            msg.get("source_generation", 0),
            0,
            SEED_MAX,
            "source_generation",
        )
    except (TypeError, ValueError, OverflowError):
        log("warning", "vision_frame_chunk: bad metadata; dropping")
        partials.pop(frame_id, None)
        return None
    if (
        not frame_id
        or not isinstance(data, str)
        or not data
        or not 2 <= total <= VISION_CHUNK_MAX_PARTS
        or not 0 <= seq < total
    ):
        log("warning", f"vision_frame_chunk: bad part {seq}/{total}; dropping")
        partials.pop(frame_id, None)
        return None
    for stale_id in [
        fid
        for fid, part in partials.items()
        if fid != frame_id and now_mono - part["started"] > VISION_CHUNK_STALE_SEC
    ]:
        partials.pop(stale_id, None)
        log("warning", "vision_frame_chunk: dropped stale partial")
    partial = partials.get(frame_id)
    if partial is None:
        while len(partials) >= VISION_CHUNK_MAX_PARTIALS:
            partials.pop(next(iter(partials)), None)
            log("warning", "vision_frame_chunk: partial evicted by newer frame")
        partial = {
            "parts": [None] * total,
            "detail": bool(msg.get("detail", False)),
            "historical_detail": bool(msg.get("historical_detail", False)),
            "source_generation": source_generation,
            "chars": 0,
            "started": now_mono,
        }
        partials[frame_id] = partial
    if (
        len(partial["parts"]) != total
        or partial["parts"][seq] is not None
        or partial["detail"] != bool(msg.get("detail", False))
        or partial["historical_detail"]
        != bool(msg.get("historical_detail", False))
        or partial["source_generation"]
        != source_generation
    ):
        log("warning", "vision_frame_chunk: inconsistent sequence; dropping frame")
        partials.pop(frame_id, None)
        return None
    partial["chars"] += len(data)
    # Same combined ceiling as a single vision_frame message.
    if partial["chars"] > VISION_FRAME_MAX_CHARS:
        log(
            "warning",
            f"vision_frame_chunk: combined frame too large: "
            f"{partial['chars']} chars; dropping",
        )
        partials.pop(frame_id, None)
        return None
    partial["parts"][seq] = data
    if any(part is None for part in partial["parts"]):
        return None
    partials.pop(frame_id, None)
    return {
        "type": "vision_frame",
        "frame_id": frame_id,
        "data": "".join(partial["parts"]),
        "detail": partial["detail"],
        "historical_detail": partial["historical_detail"],
        "source_generation": partial["source_generation"],
    }


@dataclass
class SessionConfig:
    """Per-session settings the browser sends over the control channel.

    Mirrors what the old WebSocket query string carried.
    """

    voice_prompt: str = ""
    # Optional second voice and the secondary mix share in 0.0..1.0 for a
    # blended voice prefix. Blend is active only when voice_prompt_b is set,
    # differs from voice_prompt, and voice_blend_mix > 0.0; otherwise the
    # server loads the single primary voice. Connect-time only, like the rest
    # of the voice prefix: a blended prompt re-primes the stream, so it is
    # fixed for the session and never updated live.
    voice_prompt_b: str = ""
    voice_blend_mix: float = 0.0
    # How strongly an uploaded clip conditions the timbre, in 0.0..1.0:
    # the fraction of the clip's prefix replayed during priming, taken from
    # the tail. 1.0 replays the whole clip (current behavior); lower values
    # condition less; 0.0 leaves the model's own voice. Only the raw-audio
    # upload path uses it; preset and blend prompts ignore it. Re-priming the
    # stream is reset-required, so this is connect-only like the rest of the
    # voice prefix and never updated live.
    clone_strength: float = 1.0
    text_prompt: str = ""
    vision_prompt: str = ""
    vision_in_transcript: bool = False
    # When true, live Gemini captions are drip-fed into Moshi's text
    # channel during silence windows. Off by default so vision capture is a
    # passive perception/UI feature unless the user explicitly allows the
    # voice model to use it.
    vision_feed_model: bool = False
    # When true, each detected user-audio turn can receive one fresh scene note
    # for the next answer. This is separate from ambient vision_feed_model and
    # does not require ASR/transcription, but it is intentionally opt-in because
    # it may add visual context to non-visual turns.
    vision_ground_user_turns: bool = False
    # Connect-time toggle: when set, the server periodically re-asserts the
    # persona body into the model's text channel during pad/silence windows
    # to counter long-session drift. Conditioning-adjacent, so it is fixed
    # for the session like the rest of the persona block.
    reinforce_in_silences: bool = False
    seed: Optional[int] = None
    audio_temperature: float = 0.8
    text_temperature: float = 0.7
    text_topk: int = 25
    audio_topk: int = 250
    repetition_penalty: float = 1.0
    repetition_penalty_context: int = 64
    # Keep these aligned with the embedded client's advanced slider defaults.
    # Padding bonus defaults off: it taxes response onset every frame (PAD
    # competes directly with EPAD at the moment the model would start
    # speaking) and truncates turns mid-thought. The max-turn cap remains a
    # circuit breaker; repetition penalty is available as an opt-in fallback.
    padding_bonus: float = 0.0
    max_turn_text_tokens: int = 120
    # Session length cap in seconds; 0 disables the watchdog (no time bound).
    # The client sends minutes converted to seconds, so the server stores and
    # compares seconds directly. Named for duration, not idle, so a future
    # reset-on-activity timer can reuse the field without a wire change.
    session_timeout_sec: int = 0
    # Per-session external-vision spend guard. 0 (or absent) means no
    # server-side ceiling; the browser may still enforce its own cutoff.
    vision_cost_limit_usd: float = 0.0
    # Per-frame cost estimate used to convert dispatched frames into an
    # estimated dollar spend. Kept in sync with the client's estimate so the
    # two ceilings agree. 0 disables the dollar conversion.
    vision_cost_per_call_usd: float = 0.0
    # End-of-thought inject gate (see the clamp helpers above). Live-tunable;
    # the model's decoded audio must stay below inject_silence_rms for
    # inject_silence_streak frames before a queued caption is dripped in.
    inject_silence_rms: float = INJECT_SILENCE_RMS_DEFAULT
    inject_silence_streak: int = INJECT_SILENCE_STREAK_DEFAULT


def parse_session_config(payload: dict) -> SessionConfig:
    """Parse one wire config using the shared bounded numeric contract."""
    defaults = SessionConfig()
    return SessionConfig(
        voice_prompt=str(payload.get("voice_prompt", defaults.voice_prompt)),
        voice_prompt_b=str(payload.get("voice_prompt_b", defaults.voice_prompt_b)),
        voice_blend_mix=clamp_voice_blend_mix(
            payload.get("voice_blend_mix", defaults.voice_blend_mix)
        ),
        clone_strength=clamp_clone_strength(
            payload.get("clone_strength", defaults.clone_strength)
        ),
        text_prompt=str(payload.get("text_prompt", defaults.text_prompt)),
        vision_prompt=str(payload.get("vision_prompt", defaults.vision_prompt)),
        vision_in_transcript=bool(
            payload.get("vision_in_transcript", defaults.vision_in_transcript)
        ),
        vision_feed_model=bool(
            payload.get("vision_feed_model", defaults.vision_feed_model)
        ),
        vision_ground_user_turns=bool(
            payload.get(
                "vision_ground_user_turns",
                defaults.vision_ground_user_turns,
            )
        ),
        reinforce_in_silences=bool(
            payload.get("reinforce_in_silences", defaults.reinforce_in_silences)
        ),
        seed=clamp_seed(payload.get("seed", defaults.seed)),
        audio_temperature=clamp_temperature(
            payload.get("audio_temperature", defaults.audio_temperature)
        ),
        text_temperature=clamp_temperature(
            payload.get("text_temperature", defaults.text_temperature)
        ),
        text_topk=clamp_text_topk(payload.get("text_topk", defaults.text_topk)),
        audio_topk=clamp_audio_topk(payload.get("audio_topk", defaults.audio_topk)),
        repetition_penalty=clamp_repetition_penalty(
            payload.get("repetition_penalty", defaults.repetition_penalty)
        ),
        repetition_penalty_context=clamp_repetition_penalty_context(
            payload.get(
                "repetition_penalty_context",
                defaults.repetition_penalty_context,
            )
        ),
        padding_bonus=clamp_padding_bonus(
            payload.get("padding_bonus", defaults.padding_bonus)
        ),
        max_turn_text_tokens=clamp_max_turn_text_tokens(
            payload.get("max_turn_text_tokens", defaults.max_turn_text_tokens)
        ),
        session_timeout_sec=clamp_session_timeout_sec(
            payload.get("session_timeout_sec", defaults.session_timeout_sec)
        ),
        vision_cost_limit_usd=clamp_vision_cost_limit_usd(
            payload.get("vision_cost_limit_usd", defaults.vision_cost_limit_usd)
        ),
        vision_cost_per_call_usd=clamp_vision_cost_per_call_usd(
            payload.get(
                "vision_cost_per_call_usd",
                defaults.vision_cost_per_call_usd,
            )
        ),
        inject_silence_rms=clamp_inject_silence_rms(
            payload.get("inject_silence_rms", defaults.inject_silence_rms)
        ),
        inject_silence_streak=clamp_inject_silence_streak(
            payload.get("inject_silence_streak", defaults.inject_silence_streak)
        ),
    )


class RTCSession:
    """One peer connection. Owns its track loops and DataChannel."""

    def __init__(
        self,
        *,
        frame_size: int,
        process_fn: ProcessFrameFn,
        log: LogFn,
        ice_servers: Optional[list[dict]] = None,
        backpressure_status: Optional[BackpressureStatusFn] = None,
        process_executor: Optional[Executor] = None,
    ) -> None:
        """Create a peer-connection session.

        ``ice_servers`` is a Cloudflare-shaped iceServers list (entries of
        ``{"urls": [...], "username": "...", "credential": "..."}``).
        ``None`` falls back to STUN-only, which won't traverse the RunPod
        HTTPS proxy and is intended for local LAN dev.
        """
        self._frame_size = frame_size
        self._process_fn = process_fn
        self._log = log
        self._backpressure_status = backpressure_status
        self._process_executor = process_executor

        configured = ice_servers if ice_servers else list(DEFAULT_STUN_FALLBACK)
        self._pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=ice_servers_to_aiortc(configured)
            )
        )
        self._output_track = MimiOutputTrack()
        self._pc.addTrack(self._output_track)

        # Buffered queues. Match the existing 200 ms cap on inbound PCM
        # so GPU stalls shed stale mic audio rather than ballooning latency.
        self._pcm_queue: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(
            maxsize=10
        )
        # Inbound audio is dropped silently until start_processing() runs.
        # Otherwise the warmup phase (~10 s for raw-audio voice prompts)
        # spams ~50 "pcm queue full" warnings per second while the model
        # is not yet listening. Once processing starts, queue-full
        # warnings are rate-limited to one per second so a sustained
        # overrun logs once, not on every dropped chunk.
        self._processing_started = False
        self._processing_paused = False
        self._pipeline_generation = 0
        self._pending_pcm: Optional[np.ndarray] = None
        self._process_idle = asyncio.Event()
        self._process_idle.set()
        self._last_drop_warn_at = 0.0
        self._pcm_queue_high_water = 0
        self._pcm_drop_events = 0
        self._pcm_dropped_ms = 0.0
        self._control: Optional[RTCDataChannel] = None
        self._inbound_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None
        # Strong refs to short-lived control-message handler tasks so the
        # event loop doesn't garbage-collect them mid-execution.
        self._control_tasks: set[asyncio.Task] = set()
        # SCTP delivers this channel in order, but spawning one coroutine per
        # message would otherwise let commands overtake at their first await
        # (for example, rewind racing the bookmark immediately before it, or
        # two CUDA-graph recaptures running concurrently).  Pings stay on the
        # fast path; state-changing commands cross this lock in wire order.
        self._control_message_lock = asyncio.Lock()
        self._accept_control = True
        self._active_control_task: Optional[asyncio.Task] = None
        self._last_control_overflow_warn_at = 0.0
        self._closed = asyncio.Event()
        self._on_config: Optional[Callable[[SessionConfig], Awaitable[None]]] = None
        self._on_message: Optional[Callable[[dict], Awaitable[None]]] = None
        # Optional synchronous observer for each assistant PCM frame as it is
        # pushed to the outbound track. Kept generic: the session has no
        # knowledge of what consumes the audio. None means no observer.
        self._on_pcm: Optional[Callable[[np.ndarray], None]] = None
        self._ready_sent = False
        # Why the session closed, when known. "error" marks an internal
        # failure (inbound or process loop raised), which callers use to
        # distinguish a broken transport (model state still trustworthy)
        # from broken state.
        self.close_reason: Optional[str] = None

        @self._pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            if track.kind != "audio":
                self._log("warning", f"ignoring non-audio track {track.kind}")
                return
            self._log("info", "remote audio track received, starting receiver loop")
            self._inbound_task = asyncio.create_task(self._inbound_loop(track))

        @self._pc.on("datachannel")
        def _on_datachannel(channel: RTCDataChannel) -> None:
            if channel.label != "control":
                self._log("warning", f"ignoring datachannel with label {channel.label}")
                return
            self._control = channel
            self._wire_control_channel(channel)

        @self._pc.on("connectionstatechange")
        def _on_state() -> None:
            state = self._pc.connectionState
            self._log("info", f"connection state -> {state}")
            # 'disconnected' is transient per spec; ICE may recover. Only
            # 'failed' and 'closed' are terminal.
            if state in ("closed", "failed"):
                self._closed.set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_config_handler(
        self, handler: Callable[[SessionConfig], Awaitable[None]]
    ) -> None:
        self._on_config = handler

    def set_message_handler(
        self, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        self._on_message = handler

    def set_pcm_observer(
        self, observer: Optional[Callable[[np.ndarray], None]]
    ) -> None:
        """Register a synchronous per-frame observer for outbound PCM.

        Called on the event loop with each assistant frame as it is pushed
        to the track. The observer must be cheap and non-blocking; it sees
        the same post-gate PCM the listener hears.
        """
        self._on_pcm = observer

    async def negotiate(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        """Set remote offer, build answer, return immediately.

        Does NOT wait for ICE gathering, so the returned SDP only carries
        candidates that aiortc had ready synchronously (typically host
        candidates). The caller is expected to stream remaining server
        candidates to the peer via ``iter_local_candidates`` and pump
        peer candidates back via ``add_remote_candidate``.

        Does NOT start the GPU-touching process loop. Inbound audio
        frames will be received and queued (with the same 200 ms
        drop-newest cap as the old WS path), but no model inference
        runs until ``start_processing()`` is called.
        """
        await self._pc.setRemoteDescription(offer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        return self._pc.localDescription

    async def add_remote_candidate(
        self,
        candidate_sdp: Optional[str],
        sdp_mid: Optional[str],
        sdp_mline_index: Optional[int],
    ) -> None:
        """Apply a peer-trickled ICE candidate.

        ``candidate_sdp`` is the value of ``RTCIceCandidate.candidate``
        from the browser, e.g. ``"candidate:842163049 1 udp ..."``.
        ``None`` (or an empty string) is the end-of-candidates marker
        from the browser; aiortc 1.10 manages gathering-complete on its
        own and rejects ``addIceCandidate(None)``, so we just no-op.
        """
        if not candidate_sdp:
            return
        # Browser sends the full string with a leading "candidate:" token
        # that aiortc's parser does not want.
        body = candidate_sdp[len("candidate:"):] if candidate_sdp.startswith("candidate:") else candidate_sdp
        candidate = candidate_from_sdp(body)
        candidate.sdpMid = sdp_mid
        candidate.sdpMLineIndex = sdp_mline_index
        await self._pc.addIceCandidate(candidate)

    async def iter_local_candidates(
        self, poll_interval: float = 0.1
    ) -> "AsyncIterator[dict]":
        """Yield local ICE candidates as gathering produces them.

        aiortc does not emit per-candidate events, so we poll
        ``getLocalCandidates()`` at ``poll_interval`` (default 100 ms)
        and yield any newly-seen candidate. The generator terminates
        when ``iceGatheringState`` reaches ``"complete"`` or the
        session closes; the caller is responsible for forwarding the
        events on a side channel (SSE, WebSocket, etc.).

        Each yielded item is a dict shaped like the browser's
        ``RTCIceCandidate.toJSON()`` so the client can pass it straight
        into ``new RTCIceCandidate({...})``.
        """
        # aiortc 1.10 stores iceTransports as a Set on a name-mangled
        # private attribute. With BUNDLE, the set collapses to a single
        # transport, so any element owns every candidate we will gather.
        ice_transports = getattr(
            self._pc, "_RTCPeerConnection__iceTransports", None
        )
        transport = next(iter(ice_transports), None) if ice_transports else None
        if transport is None:
            return
        gatherer = transport.iceGatherer
        seen: set[tuple] = set()
        # Best-effort: if there is exactly one m-line (one audio track
        # plus a data channel multiplexed on it), sdpMLineIndex 0 covers
        # all candidates. aiortc's bundle policy collapses to a single
        # m-line per session in our setup.
        sdp_mid = "0"
        sdp_mline_index = 0
        while not self._closed.is_set():
            for cand in gatherer.getLocalCandidates():
                key = (
                    cand.foundation,
                    cand.component,
                    cand.protocol,
                    cand.ip,
                    cand.port,
                    cand.type,
                )
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "candidate": "candidate:" + candidate_to_sdp(cand),
                    "sdpMid": sdp_mid,
                    "sdpMLineIndex": sdp_mline_index,
                }
            if self._pc.iceGatheringState == "complete":
                return
            await asyncio.sleep(poll_interval)

    def start_processing(self) -> None:
        """Start the GPU-side process loop. Call once, after system prompts.

        Drains any pre-ready audio first; whatever the client streamed
        before the model finished warming up is stale and would otherwise
        be the first thing fed into freshly-reset mimi state. Flips
        ``_processing_started`` so ``_inbound_loop`` stops silently
        dropping new audio.
        """
        while True:
            try:
                self._pcm_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._processing_started = True
        self._processing_paused = False
        self._pending_pcm = None
        self._pcm_queue_high_water = 0
        self._pcm_drop_events = 0
        self._pcm_dropped_ms = 0.0
        if self._process_task is None:
            self._process_task = asyncio.create_task(self._process_loop())

    def is_alive(self) -> bool:
        return not self._closed.is_set()

    async def wait_for_close(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        """Idempotent shutdown.

        First freezes model-facing work, then closes the peer connection.
        """
        await self.stop_processing()
        try:
            await self._pc.close()
        except Exception as exc:
            self._log("warning", f"pc.close raised: {type(exc).__name__}: {exc}")

    async def stop_processing(self) -> None:
        """Freeze and drain all model-facing work without closing the PC.

        A runner calls this before deciding whether resident state is safe to
        offer for resume. A control handler or audio frame may be awaiting
        ``run_in_executor``; cancelling its asyncio Future does not stop the
        worker thread, so every active task is drained before returning.
        Keeping the PC open briefly lets teardown send a recording-ready
        notification after the model state has already been frozen.
        """
        self._accept_control = False
        self._processing_paused = True
        self._closed.set()
        pending: list[asyncio.Task] = []
        for task in (self._inbound_task, self._process_task):
            if task is not None and not task.done():
                task.cancel()
                pending.append(task)
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._log(
                    "warning",
                    f"task drain raised: {type(exc).__name__}: {exc}",
                )
        # Cancel commands that are merely queued behind the serial lock; they
        # have submitted no executor work and must not run against a closed
        # session.  Only the active handler is allowed to finish so any worker
        # it already launched is fully drained before the session lock drops.
        current = asyncio.current_task()
        active = self._active_control_task
        control_waiters = [
            task
            for task in tuple(self._control_tasks)
            if task is not active and task is not current and not task.done()
        ]
        for task in control_waiters:
            task.cancel()
        if control_waiters:
            await asyncio.gather(*control_waiters, return_exceptions=True)
        if active is not None and active is not current and not active.done():
            await asyncio.gather(active, return_exceptions=True)

    async def clear_output_audio(self) -> None:
        await self._output_track.clear_buffer()

    @staticmethod
    def _drain_queue(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _trim_standing_inbound_backlog(self) -> float:
        """Keep at most one fresh model frame of queued microphone audio.

        After a one-off GPU stall, an 8-9 chunk queue never reaches the
        full-queue replacement path again: capture and inference both resume
        at 1x, preserving ~160 ms of stale latency forever.  Trim oldest
        chunks at each completed frame while preserving wire order.

        Returns dropped audio duration in milliseconds.
        """
        queued: list[tuple[int, np.ndarray]] = []
        while True:
            try:
                queued.append(self._pcm_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not queued:
            return 0.0

        kept_reversed: list[tuple[int, np.ndarray]] = []
        kept_samples = 0
        dropped_samples = 0
        for generation, samples in reversed(queued):
            if generation != self._pipeline_generation:
                dropped_samples += samples.size
                continue
            if (
                kept_reversed
                and kept_samples + samples.size > self._frame_size
            ):
                dropped_samples += samples.size
                continue
            kept_reversed.append((generation, samples))
            kept_samples += samples.size
        for item in reversed(kept_reversed):
            self._pcm_queue.put_nowait(item)
        if dropped_samples <= 0:
            return 0.0
        self._pcm_drop_events += 1
        dropped_ms = dropped_samples / MIMI_SAMPLE_RATE * 1000.0
        self._pcm_dropped_ms += dropped_ms
        return dropped_ms

    async def pause_and_flush_audio(self) -> int:
        """Pause inference and invalidate queued or in-flight audio."""
        self._processing_paused = True
        self._pipeline_generation += 1
        generation = self._pipeline_generation
        self._drain_queue(self._pcm_queue)
        self._pending_pcm = None
        await self._process_idle.wait()
        self._drain_queue(self._pcm_queue)
        self._pending_pcm = None
        await self._output_track.clear_buffer()
        return generation

    def resume_audio(self, generation: int) -> None:
        if generation == self._pipeline_generation:
            self._processing_paused = False

    def send_text(self, text: str) -> None:
        if self._control and self._control.readyState == "open":
            self._control.send(json.dumps({"type": "text", "v": text}))

    def send_user_text(self, text: str, final: bool = False) -> None:
        """Push recognized user-side speech to the client transcript.

        Optional feature: only the server's ASR path calls this. Mirrors
        send_text's shape with an added `final` flag so the client can tell
        a growing partial from a closed turn. Like every send helper it is a
        plain method; the ASR worker runs off the event loop, so callers
        there marshal it back via loop.call_soon_threadsafe (DataChannel.send
        is not thread-safe)."""
        if self._control and self._control.readyState == "open":
            self._control.send(
                json.dumps({"type": "user_text", "v": text, "final": bool(final)})
            )

    def send_vision_caption(
        self,
        text: str,
        frame_id: str = "",
        feed: Optional[dict] = None,
        source_generation: int = 0,
        historical_detail: bool = False,
    ) -> None:
        """Push the latest vision-side scene description to the client UI."""
        if self._control and self._control.readyState == "open":
            payload = {
                "type": "vision_caption",
                "text": text,
                "source_generation": int(source_generation),
                "historical_detail": bool(historical_detail),
            }
            if frame_id:
                payload["frame_id"] = frame_id
            if feed is not None:
                payload["feed"] = feed
            self._control.send(json.dumps(payload))

    def send_vision_status(self, enabled: bool) -> None:
        """Tell the client whether vision is available server-side."""
        if self._control and self._control.readyState == "open":
            self._control.send(
                json.dumps({"type": "vision_status", "enabled": bool(enabled)})
            )

    def send_inject_status(self, active: bool) -> None:
        """Tell the client whether a vision-context inject window is open.
        Lets the UI show the user why audio briefly drops."""
        if self._control and self._control.readyState == "open":
            self._control.send(
                json.dumps({"type": "vision_inject", "active": bool(active)})
            )

    def send_request_vision_frame(
        self, *, force: bool = False, reason: str = "cadence"
    ) -> None:
        """Ask the client to capture and send a fresh vision frame now."""
        if self._control and self._control.readyState == "open":
            self._control.send(
                json.dumps(
                    {
                        "type": "request_vision_frame",
                        "force": bool(force),
                        "reason": str(reason)[:64],
                    }
                )
            )

    def send_notice(self, text: str) -> None:
        """Server-side notice surfaced as a transient toast in the client UI."""
        if self._control and self._control.readyState == "open":
            self._control.send(
                json.dumps({"type": "notice", "text": text})
            )

    def send_event(
        self,
        kind: str,
        text: str,
        level: str = "info",
        data: Optional[dict] = None,
    ) -> None:
        """Structured event for the dashboard diagnostics rail."""
        if self._control and self._control.readyState == "open":
            payload = {"type": "event", "kind": kind, "text": text, "level": level}
            if data is not None:
                payload["data"] = data
            self._control.send(json.dumps(payload))

    def send_config_applied(
        self,
        config: dict,
        source: str = "connect",
        applied: Optional[list[str]] = None,
    ) -> None:
        """Push the server-applied session config snapshot to the client."""
        if self._control and self._control.readyState == "open":
            self._control.send(
                json.dumps(
                    {
                        "type": "config_applied",
                        "source": source,
                        "applied": applied or [],
                        "config": config,
                    }
                )
            )

    def send_context_status(
        self,
        status: str,
        data: Optional[dict] = None,
    ) -> None:
        """Push context-queue/inject metadata to the dashboard."""
        if self._control and self._control.readyState == "open":
            payload = {"type": "context_status", "status": status}
            if data is not None:
                payload["data"] = data
            self._control.send(json.dumps(payload))

    def send_interrupted(self, reason: str) -> None:
        if self._control and self._control.readyState == "open":
            self._control.send(
                json.dumps({"type": "interrupted", "reason": reason})
            )

    def send_ready(self, identity: Optional[dict] = None) -> None:
        """Signal the client that warmup is done and the session is live.

        Optional accelerator/build identity is folded into the one-shot
        payload; absent fields are simply not sent so an older client keeps
        working.
        """
        if self._ready_sent:
            return
        if self._control and self._control.readyState == "open":
            payload = {"type": "ready"}
            if identity:
                payload.update(identity)
            self._control.send(json.dumps(payload))
            self._ready_sent = True

    def send_stat(
        self,
        vram_used: Optional[int] = None,
        gpu_util: Optional[int] = None,
        rtf: Optional[float] = None,
        idle_rms: Optional[float] = None,
        silence_streak: Optional[int] = None,
        diagnostics: Optional[dict[str, int | float]] = None,
    ) -> None:
        """Periodic accelerator-memory / utilization / inference-health readout.

        Fields that could not be sampled are omitted; sending an empty stat
        is a no-op so the timer never floods the channel with bare frames.
        ``rtf`` is the server-measured real-time factor (compute time per
        audio frame / that frame's audio duration); ``idle_rms`` and
        ``silence_streak`` expose the inject gate's observed decoded idle
        floor and its current silent-frame count so the Silence floor
        slider can be tuned against reality. The client reads only the
        fields it knows, so consumers compose without ordering.
        """
        if not (self._control and self._control.readyState == "open"):
            return
        payload = {"type": "stat"}
        if vram_used is not None:
            payload["vram_used"] = int(vram_used)
        if gpu_util is not None:
            payload["gpu_util"] = int(gpu_util)
        if rtf is not None:
            payload["rtf"] = round(float(rtf), 3)
        if idle_rms is not None:
            payload["idle_rms"] = round(float(idle_rms), 4)
        if silence_streak is not None:
            payload["silence_streak"] = int(silence_streak)
        if diagnostics:
            # The snapshot is generated locally from numeric counters. Keep
            # this allowlist explicit so future internal diagnostics cannot
            # accidentally put free-form paths or messages on the wire.
            integer_fields = {
                "pcm_queue_depth",
                "pcm_queue_capacity",
                "pcm_queue_high_water",
                "pcm_drop_events",
                "outbound_drop_events",
                "outbound_flush_events",
            }
            float_fields = {
                "pcm_dropped_ms",
                "outbound_buffer_ms",
                "outbound_high_water_ms",
                "outbound_dropped_ms",
                "outbound_flushed_ms",
            }
            for field in integer_fields:
                value = diagnostics.get(field)
                if isinstance(value, (int, float)):
                    payload[field] = int(value)
            for field in float_fields:
                value = diagnostics.get(field)
                if isinstance(value, (int, float)):
                    payload[field] = round(float(value), 1)
        if len(payload) == 1:
            return
        self._control.send(json.dumps(payload))

    async def diagnostics_snapshot(self) -> dict[str, int | float]:
        """Return queue health for the low-frequency stat envelope."""
        snapshot: dict[str, int | float] = {
            "pcm_queue_depth": self._pcm_queue.qsize(),
            "pcm_queue_capacity": self._pcm_queue.maxsize,
            "pcm_queue_high_water": self._pcm_queue_high_water,
            "pcm_drop_events": self._pcm_drop_events,
            "pcm_dropped_ms": round(self._pcm_dropped_ms, 1),
        }
        snapshot.update(await self._output_track.diagnostics_snapshot())
        return snapshot

    def send_error(self, reason: str) -> None:
        if self._control and self._control.readyState == "open":
            self._control.send(json.dumps({"type": "error", "reason": reason}))

    def send_end(self, reason: str) -> None:
        """Announce a server-initiated session end before the pc closes.

        Gives the client a chance to take its graceful shutdown path
        (keeping the local recording) rather than reading the imminent
        peer-connection close as a transport failure.
        """
        if self._control and self._control.readyState == "open":
            self._control.send(json.dumps({"type": "end", "reason": reason}))

    def send_pong(self, t: object, seq: Optional[int] = None) -> None:
        """Echo a heartbeat ping so the client can measure app-level RTT.

        `t` is the client's opaque send timestamp, returned verbatim; `seq`
        lets the client match replies to sends and count drops.
        """
        if self._control and self._control.readyState == "open":
            payload = {"type": "pong", "t": t}
            if seq is not None:
                payload["seq"] = seq
            self._control.send(json.dumps(payload))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _wire_control_channel(self, channel: RTCDataChannel) -> None:
        @channel.on("open")
        def _on_open() -> None:
            self._log("info", "control channel open")

        @channel.on("message")
        def _on_message(message: object) -> None:
            if not self._accept_control or self._closed.is_set():
                return
            if len(self._control_tasks) >= CONTROL_TASK_MAX:
                now = asyncio.get_event_loop().time()
                if now - self._last_control_overflow_warn_at >= 1.0:
                    self._log(
                        "warning",
                        "control queue full; dropping excess message",
                    )
                    self._last_control_overflow_warn_at = now
                return
            # Hold a strong ref so the event loop's weak set can't GC the
            # task mid-handler.
            task = asyncio.create_task(self._handle_control_message(message))
            self._control_tasks.add(task)
            task.add_done_callback(self._control_task_done)

    def _control_task_done(self, task: asyncio.Task) -> None:
        """Retrieve handler failures and terminate a potentially torn session."""
        self._control_tasks.discard(task)
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is None:
            return
        self.close_reason = "error"
        self._log(
            "error",
            f"control handler: {type(exc).__name__}: {exc}",
        )
        try:
            reason = f"control_error: {type(exc).__name__}: {exc}"
            self.send_error(reason[:512])
        except Exception as send_exc:
            self._log(
                "warning",
                "control error notify failed: "
                f"{type(send_exc).__name__}: {send_exc}",
            )
        # A failed mutation may have partially changed model state.  Wake the
        # runner so it tears down without issuing a resume grant.
        self._closed.set()

    async def _handle_control_message(self, message: object) -> None:
        if not self._accept_control or self._closed.is_set():
            return
        if not isinstance(message, str):
            self._log(
                "warning",
                f"control: ignoring non-string message {type(message).__name__}",
            )
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError as exc:
            self._log("warning", f"control: bad JSON: {exc}")
            return
        if not isinstance(payload, dict):
            self._log(
                "warning",
                f"control: payload is not an object: {type(payload).__name__}",
            )
            return
        kind = payload.get("type")
        if kind == "ping":
            t = payload.get("t")
            if not isinstance(t, (int, float)) or isinstance(t, bool):
                self._log("warning", f"control: ping missing/bad 't': {t!r}")
                return
            seq_raw = payload.get("seq")
            seq = None
            if seq_raw is not None:
                if isinstance(seq_raw, int) and not isinstance(seq_raw, bool):
                    seq = seq_raw
                else:
                    self._log("warning", f"control: ping bad 'seq': {seq_raw!r}")
            self.send_pong(t, seq)
            return

        async with self._control_message_lock:
            if not self._accept_control or self._closed.is_set():
                return
            task = asyncio.current_task()
            self._active_control_task = task
            try:
                if kind == "config":
                    try:
                        cfg = parse_session_config(payload)
                    except (TypeError, ValueError, OverflowError) as exc:
                        self._log("warning", f"control: bad config: {exc}")
                        self.send_error(f"bad_config: {exc}")
                        return
                    if self._on_config is not None:
                        await self._on_config(cfg)
                elif self._on_message is not None:
                    await self._on_message(payload)
                else:
                    self._log("warning", f"control: unknown type {kind!r}")
            finally:
                if self._active_control_task is task:
                    self._active_control_task = None

    async def _inbound_loop(self, track: MediaStreamTrack) -> None:
        # Persistent resampler keeps anti-alias filter state across frames.
        resampler = AudioResampler(
            format="s16", layout="mono", rate=MIMI_SAMPLE_RATE
        )
        try:
            while True:
                try:
                    frame = await track.recv()
                except MediaStreamError:
                    break
                samples = _frame_to_mono_24k_f32(frame, resampler)
                if samples.size == 0:
                    continue
                if not self._processing_started or self._processing_paused:
                    # Warmup phase: model is not consuming yet. Dropping
                    # this chunk is the right thing; do it silently so
                    # the log is not flooded by ~10 s of warning spam.
                    continue
                queued = (self._pipeline_generation, samples)
                try:
                    self._pcm_queue.put_nowait(queued)
                    self._pcm_queue_high_water = max(
                        self._pcm_queue_high_water,
                        self._pcm_queue.qsize(),
                    )
                except asyncio.QueueFull:
                    dropped_samples = samples
                    try:
                        _, dropped_samples = self._pcm_queue.get_nowait()
                        self._pcm_queue.put_nowait(queued)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass
                    self._pcm_queue_high_water = max(
                        self._pcm_queue_high_water,
                        self._pcm_queue.qsize(),
                    )
                    self._pcm_drop_events += 1
                    self._pcm_dropped_ms += (
                        dropped_samples.size / MIMI_SAMPLE_RATE * 1000.0
                    )
                    now = asyncio.get_event_loop().time()
                    if now - self._last_drop_warn_at >= 1.0:
                        diagnostics = ""
                        if self._backpressure_status is not None:
                            try:
                                diagnostics = self._backpressure_status()
                            except Exception as exc:
                                diagnostics = (
                                    f"diagnostics_error={type(exc).__name__}: {exc}"
                                )
                        diagnostics_suffix = (
                            f"; {diagnostics}" if diagnostics else ""
                        )
                        incoming_ms = samples.size / MIMI_SAMPLE_RATE * 1000.0
                        self._log(
                            "warning",
                            "pcm queue full "
                            f"q={self._pcm_queue.qsize()}/{self._pcm_queue.maxsize} "
                            f"high_water={self._pcm_queue_high_water} "
                            f"drop_events={self._pcm_drop_events} "
                            f"dropped_ms_total={self._pcm_dropped_ms:.1f} "
                            f"incoming_ms={incoming_ms:.1f}, "
                            "dropping stale inbound audio"
                            f"{diagnostics_suffix}",
                        )
                        self._last_drop_warn_at = now
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.close_reason = "error"
            self._log("error", f"inbound_loop: {type(exc).__name__}: {exc}")
        finally:
            self._closed.set()

    async def _process_loop(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            while not self._closed.is_set():
                try:
                    pcm_generation, pcm = await asyncio.wait_for(
                        self._pcm_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                if (
                    self._processing_paused
                    or pcm_generation != self._pipeline_generation
                ):
                    continue
                if pcm.shape[-1] == 0:
                    continue
                self._pending_pcm = (
                    pcm
                    if self._pending_pcm is None
                    else np.concatenate((self._pending_pcm, pcm))
                )
                while (
                    not self._processing_paused
                    and self._pending_pcm is not None
                    and self._pending_pcm.shape[-1] >= self._frame_size
                ):
                    chunk = self._pending_pcm[: self._frame_size]
                    self._pending_pcm = self._pending_pcm[self._frame_size :]
                    generation = self._pipeline_generation
                    self._process_idle.clear()
                    in_flight = asyncio.ensure_future(
                        loop.run_in_executor(
                            self._process_executor, self._process_fn, chunk
                        )
                    )
                    try:
                        results = await asyncio.shield(in_flight)
                    except asyncio.CancelledError:
                        try:
                            await in_flight
                        except BaseException:
                            pass
                        raise
                    finally:
                        self._process_idle.set()
                    trimmed_ms = self._trim_standing_inbound_backlog()
                    if trimmed_ms > 0:
                        now = loop.time()
                        if now - self._last_drop_warn_at >= 1.0:
                            self._log(
                                "warning",
                                "trimmed standing inbound audio backlog "
                                f"dropped_ms={trimmed_ms:.1f} "
                                f"q={self._pcm_queue.qsize()}/{self._pcm_queue.maxsize}",
                            )
                            self._last_drop_warn_at = now
                    if (
                        self._processing_paused
                        or generation != self._pipeline_generation
                    ):
                        continue
                    for pcm_data, text in results:
                        frame_f32 = pcm_data.astype(np.float32)
                        await self._output_track.push_24k_f32(frame_f32)
                        if self._on_pcm is not None:
                            self._on_pcm(frame_f32)
                        if text is not None:
                            self.send_text(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.close_reason = "error"
            self._log("error", f"process_loop: {type(exc).__name__}: {exc}")
        finally:
            self._closed.set()
