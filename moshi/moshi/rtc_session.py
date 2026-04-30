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
import fractions
import json
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

# STUN-only fallback used when no TURN credentials are configured. Works
# only when both peers can reach each other directly over UDP, which is
# not the case behind RunPod's HTTPS-only proxy. Production deployments
# should provide TURN credentials via the env vars consumed in server.py.
DEFAULT_STUN_FALLBACK: tuple[dict, ...] = (
    {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
)


ProcessFrameFn = Callable[[np.ndarray], list[tuple[np.ndarray, Optional[str]]]]
LogFn = Callable[[str, str], None]


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
            # Sanity cap: if recv() falls way behind, drop oldest.
            if self._buffer.size > OUTBOUND_BUFFER_CAP_SAMPLES:
                self._buffer = self._buffer[-OUTBOUND_BUFFER_CAP_SAMPLES:]

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
        # Pace real-time: target the next frame boundary based on cumulative
        # timestamp, sleep if we're early.
        target = self._start_time + (
            self._timestamp + OUTBOUND_FRAME_SAMPLES
        ) / WEBRTC_SAMPLE_RATE
        delay = target - loop.time()
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


@dataclass
class SessionConfig:
    """Per-session settings the browser sends over the control channel.

    Mirrors what the old WebSocket query string carried.
    """

    voice_prompt: str = ""
    text_prompt: str = ""
    seed: Optional[int] = None
    audio_temperature: float = 0.7
    text_temperature: float = 0.7
    text_topk: int = 25
    audio_topk: int = 250
    repetition_penalty: float = 1.2
    repetition_penalty_context: int = 64
    padding_bonus: float = 0.0
    max_turn_text_tokens: int = 0


class RTCSession:
    """One peer connection. Owns its track loops and DataChannel."""

    def __init__(
        self,
        *,
        frame_size: int,
        process_fn: ProcessFrameFn,
        log: LogFn,
        ice_servers: Optional[list[dict]] = None,
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

        configured = ice_servers if ice_servers else list(DEFAULT_STUN_FALLBACK)
        self._pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=ice_servers_to_aiortc(configured)
            )
        )
        self._output_track = MimiOutputTrack()
        self._pc.addTrack(self._output_track)

        # Buffered queues. Match the existing 200 ms cap on inbound PCM
        # so GPU stalls drop the newest chunk rather than ballooning.
        self._pcm_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=10)
        self._control: Optional[RTCDataChannel] = None
        self._inbound_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None
        # Strong refs to short-lived control-message handler tasks so the
        # event loop doesn't garbage-collect them mid-execution.
        self._control_tasks: set[asyncio.Task] = set()
        self._closed = asyncio.Event()
        self._on_config: Optional[Callable[[SessionConfig], Awaitable[None]]] = None
        self._ready_sent = False

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
        be the first thing fed into freshly-reset mimi state.
        """
        while True:
            try:
                self._pcm_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._process_task is None:
            self._process_task = asyncio.create_task(self._process_loop())

    def is_alive(self) -> bool:
        return not self._closed.is_set()

    async def wait_for_close(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        """Idempotent shutdown.

        Awaits the cancelled inbound/process tasks before returning so that
        any in-flight GPU work (shielded inside ``_process_loop``) finishes
        and releases its grip on shared lm_gen / mimi state. Without this
        await, the caller's ``self.lock.release()`` could fire while the
        GPU thread is still mutating those tensors, racing the next
        session's ``reset_streaming()``.
        """
        self._closed.set()
        pending: list[asyncio.Task] = []
        for task in (self._inbound_task, self._process_task, *self._control_tasks):
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
        try:
            await self._pc.close()
        except Exception as exc:
            self._log("warning", f"pc.close raised: {type(exc).__name__}: {exc}")

    def send_text(self, text: str) -> None:
        if self._control and self._control.readyState == "open":
            self._control.send(json.dumps({"type": "text", "v": text}))

    def send_ready(self) -> None:
        if self._ready_sent:
            return
        if self._control and self._control.readyState == "open":
            self._control.send(json.dumps({"type": "ready"}))
            self._ready_sent = True

    def send_error(self, reason: str) -> None:
        if self._control and self._control.readyState == "open":
            self._control.send(json.dumps({"type": "error", "reason": reason}))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _wire_control_channel(self, channel: RTCDataChannel) -> None:
        @channel.on("open")
        def _on_open() -> None:
            self._log("info", "control channel open")

        @channel.on("message")
        def _on_message(message: object) -> None:
            # Hold a strong ref so the event loop's weak set can't GC the
            # task mid-handler.
            task = asyncio.create_task(self._handle_control_message(message))
            self._control_tasks.add(task)
            task.add_done_callback(self._control_tasks.discard)

    async def _handle_control_message(self, message: object) -> None:
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
        if kind == "config":
            try:
                seed_raw = payload.get("seed")
                seed = None if seed_raw is None else int(seed_raw)
                cfg = SessionConfig(
                    voice_prompt=str(payload.get("voice_prompt", "")),
                    text_prompt=str(payload.get("text_prompt", "")),
                    seed=seed,
                    audio_temperature=float(payload.get("audio_temperature", 0.7)),
                    text_temperature=float(payload.get("text_temperature", 0.7)),
                    text_topk=int(payload.get("text_topk", 25)),
                    audio_topk=int(payload.get("audio_topk", 250)),
                    repetition_penalty=float(payload.get("repetition_penalty", 1.2)),
                    repetition_penalty_context=int(
                        payload.get("repetition_penalty_context", 64)
                    ),
                    padding_bonus=float(payload.get("padding_bonus", 0.0)),
                    max_turn_text_tokens=int(payload.get("max_turn_text_tokens", 0)),
                )
            except (TypeError, ValueError) as exc:
                self._log("warning", f"control: bad config: {exc}")
                self.send_error(f"bad_config: {exc}")
                return
            if self._on_config is not None:
                await self._on_config(cfg)
        else:
            # 'reset' was intentionally not implemented: mid-session resets
            # racing the executor's GPU thread is the same hazard the
            # original WebSocket handler's lock + shield carefully avoided
            # at session boundaries. If/when we want it, route through the
            # process loop between frames so the reset shares a thread
            # with the inference work.
            self._log("warning", f"control: unknown type {kind!r}")

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
                try:
                    self._pcm_queue.put_nowait(samples)
                except asyncio.QueueFull:
                    self._log(
                        "warning",
                        f"pcm queue full ({self._pcm_queue.qsize()}), dropping chunk",
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log("error", f"inbound_loop: {type(exc).__name__}: {exc}")
        finally:
            self._closed.set()

    async def _process_loop(self) -> None:
        loop = asyncio.get_event_loop()
        all_pcm_data: Optional[np.ndarray] = None
        try:
            while not self._closed.is_set():
                try:
                    pcm = await asyncio.wait_for(
                        self._pcm_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                if pcm.shape[-1] == 0:
                    continue
                all_pcm_data = (
                    pcm if all_pcm_data is None
                    else np.concatenate((all_pcm_data, pcm))
                )
                while all_pcm_data.shape[-1] >= self._frame_size:
                    chunk = all_pcm_data[: self._frame_size]
                    all_pcm_data = all_pcm_data[self._frame_size :]
                    in_flight = asyncio.ensure_future(
                        loop.run_in_executor(None, self._process_fn, chunk)
                    )
                    try:
                        results = await asyncio.shield(in_flight)
                    except asyncio.CancelledError:
                        try:
                            await in_flight
                        except BaseException:
                            pass
                        raise
                    for pcm_data, text in results:
                        await self._output_track.push_24k_f32(
                            pcm_data.astype(np.float32)
                        )
                        if text is not None:
                            self.send_text(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log("error", f"process_loop: {type(exc).__name__}: {exc}")
        finally:
            self._closed.set()
