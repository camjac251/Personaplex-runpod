# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from collections import deque
import datetime
import json
import random
import os
from pathlib import Path
import tarfile
import secrets
import sys
import threading
import time
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

from aiortc import RTCSessionDescription

from .models import loaders, MimiModel, LMGen
from .models.lm import MAX_REPETITION_CONTEXT
from .rtc_session import DEFAULT_STUN_FALLBACK, RTCSession, SessionConfig
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuning for better performance


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


UPLOAD_PREFIX = "upload:"
UPLOAD_MAX_BYTES = 20 * 1024 * 1024
UPLOAD_ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
# Voice cloning timbre captures inside ~10 s; longer references add
# prosody and emotional range. 60 s is the comfortable upper bound:
# enough room for a self-introduction or read-aloud paragraph, while
# keeping the warmup bounded since each prompt-second turns into ~12.5
# sequential GPU encode steps that hold the session lock. A 5-minute
# MP3 (allowed by the 20 MB byte cap) would be a self-DoS.
UPLOAD_MAX_VOICE_PROMPT_SECONDS = 60.0

# Default system prompt for the vision side. Generic by design: the user
# can override it via the SessionConfig.vision_prompt field (surfaced as a
# textarea in the embedded UI).
DEFAULT_VISION_SYSTEM_PROMPT = (
    "You are an observer. Describe exactly what is happening in this scene "
    "in one short sentence. Keep it brief and factual. You have memory of "
    "prior frames in this session; use them to track movement and changes."
)

# How many recent assistant text fragments to keep around for the Gemini
# transcript-context window. ~80 fragments is roughly the last 6-8 seconds
# of model speech.
TRANSCRIPT_BUFFER_MAX = 80

# Vision-context tokens are pushed into _vision_pending and drained
# one per audio frame (Mimi runs at ~12.5 Hz) only while the model is
# in a pad streak. Cap the queue so a steady Gemini stream cannot let
# context lag arbitrarily far behind reality.
VISION_QUEUE_MAX = 64

# Wait for N consecutive PAD text tokens before starting a vision
# inject. Ensures we interrupt during natural silence, not mid-word.
# Pulled from NVIDIA/personaplex PR #69's `LIVE_PROMPT_BOUNDARY_STREAK`.
LIVE_PROMPT_BOUNDARY_STREAK = 2

# Hard cap on how many tokens we'll inject in one window before forcing
# a return to normal generation. ~4 s at 12.5 Hz.
LIVE_PROMPT_MAX_STEPS = 48

# Re-arm the vision-frame request flag every N pad frames of sustained
# silence. Without this, the cadence task fires once when the model
# enters a pad streak and never again until the streak breaks. Users
# observing a static scene with no audio activity then see one frame
# every fallback-timer interval. ~62 frames is ~5 s at 12.5 Hz.
PAD_STREAK_REREQUEST_EVERY = 62

# Auto-rewind: if the LM safety net (max_turn_text_tokens) triggers this
# many times within COLLAPSE_WINDOW_SEC, treat that as a sign the model
# is wobbling and restore the latest snapshot in-place. The thresholds
# are conservative; healthy sessions never get close.
COLLAPSE_TRIGGER_THRESHOLD = 3
COLLAPSE_WINDOW_SEC = 30.0

# Cooldown between auto-rewinds. Without this, a wobbling model state
# can re-trigger pad-force right after a restore (the snapshotted state
# is itself the wobbling state) and produce a rewind storm.
AUTO_REWIND_MIN_INTERVAL_SEC = 60.0

# If Gemini returns N consecutive non-2xx responses, auto-disable vision
# for the rest of the session and tell the client. Stops the server from
# silently retrying a broken schema for the full session lifetime.
VISION_AUTO_DISABLE_THRESHOLD = 3


class ServerState:
    """Per-process state: models, locks, vision pipeline, session bookkeeping.

    Single-session by design: ``self.lock`` (asyncio.Lock) gates concurrent
    connect attempts; ``self._infer_lock`` (threading.Lock) guards lm_gen
    state against concurrent mutation from the executor thread and from
    event-loop coroutines.
    """

    def __init__(self, mimi: MimiModel, lm_gen: LMGen, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 device: str | torch.device, voice_prompt_dir: str | None = None,
                 uploads_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False):
        self.mimi = mimi
        self.lm_gen = lm_gen
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.uploads_dir = uploads_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        # Session gate: one RTC session at a time. asyncio.Lock so
        # negotiation and teardown can await without blocking the loop.
        self.lock = asyncio.Lock()
        # Guards lm_gen state against concurrent mutation. Held by the
        # executor thread inside _process_audio_frame, and by the rewind,
        # snapshot, and vision-injection paths (which dispatch to the
        # executor before acquiring) so they cannot interleave with an
        # in-flight step().
        self._infer_lock = threading.Lock()
        # Set in _run_rtc_session for the lifetime of an active session.
        # Lets vision-side coroutines push captions back to the client
        # without plumbing a session reference through every call site.
        self._active_session: Optional["RTCSession"] = None
        # Stashed asyncio loop reference. Set in _run_rtc_session once the
        # loop is known; cleared in finally. Used by the executor thread
        # (which doesn't own the loop) to schedule DataChannel sends via
        # call_soon_threadsafe rather than touching aiortc directly.
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Strong refs to long-running session tasks. asyncio holds only
        # weak references to tasks created via create_task; without this
        # set, the runner task that owns the lock can be garbage-collected
        # mid-session, leaving the lock permanently held.
        self._session_tasks: set[asyncio.Task] = set()
        # Cloudflare TURN credentials (optional). When both are set we mint
        # ephemeral creds via their API per session; otherwise STUN-only.
        # Read from env so the values never enter the repo.
        self._turn_key_id = os.environ.get("TURN_KEY_ID", "").strip() or None
        self._turn_api_token = os.environ.get("TURN_KEY_API_TOKEN", "").strip() or None
        self._ice_cache: Optional[list[dict]] = None
        self._ice_cache_expires_at: float = 0.0
        self._ice_cache_lock = asyncio.Lock()
        # Gemini state. _interaction_ids chains turns via the Interactions
        # API's previous_interaction_id; _vision_in_flight prevents
        # overlapping calls from corrupting that chain.
        self._gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip() or None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._interaction_ids: dict[str, str] = {}
        self._vision_in_flight: set[str] = set()
        # Per-session vision dispatch tasks. Each handle_vision_frame call
        # is fire-and-forget from on_message; we hold strong refs here so
        # session teardown can cancel + drain stragglers before another
        # session opens. Otherwise a late Gemini response from session A
        # can overwrite session B's _vision_pending under _infer_lock.
        self._vision_tasks: dict[str, set[asyncio.Task]] = {}
        # Vision-context inject state. _vision_pending holds tokens waiting
        # to be drip-fed into the model's text channel during pad streaks.
        # _vision_pad_streak counts how many recent natural text emissions
        # have been PAD; once it crosses LIVE_PROMPT_BOUNDARY_STREAK we
        # start consuming the queue one token per outer audio frame, with
        # outbound audio gated to silence for those frames. Reset on each
        # new session in _run_rtc_session.
        self._vision_pending: deque[int] = deque()
        self._vision_pad_streak: int = 0
        self._vision_inject_steps: int = 0
        # Per-session system prompt for Gemini. Set in _run_rtc_session
        # from cfg.vision_prompt (or DEFAULT_VISION_SYSTEM_PROMPT if blank).
        self._vision_system_prompt: str = DEFAULT_VISION_SYSTEM_PROMPT
        # Rolling buffer of recent assistant text fragments. Included in
        # every Gemini call so the vision side knows what the model has
        # been saying, and can prioritize / not contradict it.
        self._transcript_recent: deque[str] = deque(maxlen=TRANSCRIPT_BUFFER_MAX)
        # Active session id for the single live session; lets executor-
        # side code (collapse detection / rewind) reach into per-session
        # state without plumbing through.
        self._active_session_id: Optional[str] = None
        # Collapse detection: timestamps of recent _pad_force_remaining
        # transitions (i.e. max_turn_text_tokens safety net firings). When
        # the count in the last COLLAPSE_WINDOW_SEC crosses the threshold,
        # we auto-rewind to the latest snapshot.
        self._collapse_triggers: deque[float] = deque(maxlen=16)
        self._prev_pad_force_remaining: int = 0
        # Flag set by _process_audio_frame (executor thread) when the
        # model just entered a natural pad streak; a cadence task on the
        # event loop drains it and asks the client for a fresh vision
        # frame. A plain bool is safe here under CPython's GIL: writes
        # and reads are atomic at the bytecode boundary, and the worst
        # outcome of a missed flip is one skipped vision request (next
        # pad streak will set it again). If we ever move to no-GIL
        # Python, swap for threading.Event for explicit memory ordering.
        self._vision_request_pending: bool = False
        # Inject-window edge detection: track transitions so we can
        # notify the client ("Injecting context...") and log on open/close.
        self._inject_active: bool = False
        # Auto-rewind cooldown bookkeeping. Updated on a successful
        # rewind; checked before the next would fire.
        self._last_rewind_at: float = 0.0
        # Gemini consecutive-error counter for the auto-disable path.
        # Reset on every 2xx success and on session start.
        self._gemini_consecutive_errors: int = 0
        self._vision_force_disabled: bool = False
        # Per-session toggle: when set by cfg.vision_in_transcript the
        # server echoes each Gemini description into the main transcript
        # with a [vision] prefix for debugging context-injection.
        self._vision_in_transcript: bool = False
        # Live sessions awaiting trickled candidates. Keyed by the
        # opaque session_id returned in the offer response. Entries are
        # cleared in _run_rtc_session's finally block.
        self._candidate_sessions: dict[str, "RTCSession"] = {}
        # Rewind history: session_id -> [(monotonic_ts, flattened_state_dict)].
        # State dicts hold tensor clones so the snapshot doesn't follow the
        # live model.
        self._session_snapshots: dict[str, list[tuple[float, dict]]] = {}
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
    
    def warmup(self):
        # More warmup iterations for CUDA graphs to stabilize
        for _ in range(8):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            # Clear CUDA cache after warmup to free any fragmented memory
            torch.cuda.empty_cache()

    def _resolve_upload_path(self, name: str) -> Optional[str]:
        """Return an absolute path inside uploads_dir, or None if unsafe/missing.
        Blocks path traversal and ensures the resolved path stays under uploads_dir."""
        if self.uploads_dir is None or not name:
            return None
        if os.sep in name or (os.altsep and os.altsep in name) or name.startswith("."):
            return None
        base = os.path.realpath(self.uploads_dir)
        candidate = os.path.realpath(os.path.join(base, name))
        try:
            if os.path.commonpath([base, candidate]) != base:
                return None
        except ValueError:
            return None
        return candidate

    async def handle_voice_upload(self, request):
        """Accept a multipart upload of an audio file for voice prompting.
        Returns JSON {filename: "upload:<name>"} on success."""
        if self.uploads_dir is None:
            return web.json_response({"error": "uploads disabled on this server"}, status=503)
        if request.content_length is not None and request.content_length > UPLOAD_MAX_BYTES:
            return web.json_response({"error": "file too large"}, status=413)
        try:
            reader = await request.multipart()
        except Exception as e:
            return web.json_response({"error": f"invalid multipart body: {e}"}, status=400)

        field = await reader.next()
        while field is not None and field.name != "file":
            field = await reader.next()
        if field is None:
            return web.json_response({"error": "missing 'file' field"}, status=400)

        original = field.filename or "upload"
        ext = Path(original).suffix.lower()
        if ext not in UPLOAD_ALLOWED_EXT:
            return web.json_response(
                {"error": f"unsupported extension {ext or '(none)'}; allowed: {sorted(UPLOAD_ALLOWED_EXT)}"},
                status=400,
            )

        safe_name = f"upload_{secrets.token_urlsafe(8)}{ext}"
        out_path = Path(self.uploads_dir) / safe_name
        total = 0
        try:
            with open(out_path, "wb") as f:
                while True:
                    chunk = await field.read_chunk(size=64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > UPLOAD_MAX_BYTES:
                        f.close()
                        out_path.unlink(missing_ok=True)
                        return web.json_response({"error": "file too large"}, status=413)
                    f.write(chunk)
        except Exception as e:
            out_path.unlink(missing_ok=True)
            return web.json_response({"error": f"failed to save file: {e}"}, status=500)

        # Validate it decodes. sphn.read is CPU-bound; run in executor so we do not
        # block the event loop on large files.
        loop = asyncio.get_event_loop()
        try:
            sample_pcm, sample_sr = await loop.run_in_executor(
                None, sphn.read, str(out_path)
            )
        except Exception as e:
            out_path.unlink(missing_ok=True)
            return web.json_response({"error": f"could not decode audio: {e}"}, status=400)

        # Reject long uploads early. sphn.read returns shape (C, T); duration
        # in seconds is T / sample_sr. See UPLOAD_MAX_VOICE_PROMPT_SECONDS.
        try:
            duration = float(sample_pcm.shape[-1]) / float(sample_sr)
        except (TypeError, ZeroDivisionError, AttributeError):
            duration = 0.0
        if duration > UPLOAD_MAX_VOICE_PROMPT_SECONDS:
            out_path.unlink(missing_ok=True)
            return web.json_response(
                {
                    "error": (
                        f"audio too long ({duration:.1f}s); voice prompts "
                        f"are capped at {UPLOAD_MAX_VOICE_PROMPT_SECONDS:.0f}s "
                        "for clone quality and warmup time"
                    )
                },
                status=400,
            )

        logger.info(
            f"voice upload saved: {safe_name} ({total} bytes, "
            f"{duration:.1f}s, original={original!r})"
        )
        return web.json_response({"filename": f"{UPLOAD_PREFIX}{safe_name}", "bytes": total})

    @torch.no_grad()
    def _process_audio_frame(self, chunk_np):
        """Run GPU inference for one audio frame. Called from thread executor
        so the asyncio event loop stays responsive during GPU work.

        Also runs the vision-context inject state machine: when the queue
        is non-empty and the model has been in a PAD streak for at least
        LIVE_PROMPT_BOUNDARY_STREAK frames, force one queued token onto
        the text channel and zero the outbound audio for that frame.
        Drip cadence is one token per outer call to match Mimi's 12.5 Hz.
        """
        chunk = torch.from_numpy(chunk_np).to(device=self.device)[None, None]
        codes = self.mimi.encode(chunk)
        results = []
        pad_id = self.lm_gen.lm_model.text_padding_token_id

        with self._infer_lock:
            prev_pad_streak = self._vision_pad_streak
            # Decide once per outer call whether to inject this frame.
            inject_token: Optional[int] = None
            if (
                self._vision_pending
                and self._vision_pad_streak >= LIVE_PROMPT_BOUNDARY_STREAK
            ):
                if self._vision_inject_steps < LIVE_PROMPT_MAX_STEPS:
                    inject_token = self._vision_pending.popleft()
                    self._vision_inject_steps += 1
                else:
                    # Cap hit mid-window; drop the rest as stale and let
                    # the next Gemini response repopulate fresh.
                    self._vision_pending.clear()
                    self._vision_inject_steps = 0
            else:
                # Idle: no inject this frame.
                self._vision_inject_steps = 0

            for c in range(codes.shape[-1]):
                # Only force a token on the first inner iteration so the
                # drip cadence stays at one per outer call regardless of
                # how many Mimi codes a chunk emits.
                forced_text = None
                if inject_token is not None and c == 0:
                    forced_text = torch.tensor(
                        [[inject_token]], device=self.device, dtype=torch.long
                    )

                tokens = self.lm_gen.step(codes[:, :, c: c + 1], text_token=forced_text)
                if tokens is None:
                    continue
                assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                main_pcm = self.mimi.decode(tokens[:, 1:9])
                main_pcm = main_pcm.cpu()
                pcm_np = main_pcm[0, 0].numpy()

                # Audio gate: silence outbound PCM while we're injecting
                # so the listener never hears the model trying to speak
                # the forced text.
                if forced_text is not None:
                    pcm_np = np.zeros_like(pcm_np)

                text_token = tokens[0, 0, 0].item()

                # Track pad streak on natural emissions only. Forced
                # tokens don't represent the model's intent to be silent.
                if forced_text is None:
                    if text_token == pad_id:
                        self._vision_pad_streak += 1
                    else:
                        self._vision_pad_streak = 0

                text = None
                # Don't surface forced tokens in the visible transcript.
                if forced_text is None and text_token not in (0, 3):
                    _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                    text = _text.replace("▁", " ")
                    # Keep a short rolling tail of natural text for the
                    # vision-side transcript-context window.
                    if text:
                        self._transcript_recent.append(text)
                results.append((pcm_np, text))

            # --- collapse detection ----------------------------------
            # _pad_force_remaining transitions 0 -> >0 when the LM safety
            # net (max_turn_text_tokens) kicks in. Three of those inside
            # a short window means the model is wobbling; restore the
            # latest snapshot in place. Cheap, runs in the lock we
            # already hold.
            pad_force = self.lm_gen._pad_force_remaining
            if pad_force > 0 and self._prev_pad_force_remaining == 0:
                now = time.monotonic()
                cutoff = now - COLLAPSE_WINDOW_SEC
                while self._collapse_triggers and self._collapse_triggers[0] < cutoff:
                    self._collapse_triggers.popleft()
                self._collapse_triggers.append(now)
                if len(self._collapse_triggers) >= COLLAPSE_TRIGGER_THRESHOLD:
                    # Cooldown: the snapshotted state itself is often the
                    # wobbling state, so back-to-back rewinds would storm.
                    cooldown_left = AUTO_REWIND_MIN_INTERVAL_SEC - (now - self._last_rewind_at)
                    if cooldown_left > 0:
                        logger.warning(
                            "auto-rewind suppressed by cooldown (%.0f s remaining)",
                            cooldown_left,
                        )
                        self._collapse_triggers.clear()
                    else:
                        sid = self._active_session_id
                        snapshots = self._session_snapshots.get(sid, []) if sid else []
                        if snapshots:
                            _, state_dict = snapshots[-1]
                            # set_streaming_state_inplace pops entries from
                            # the dict it's given. Pass a fresh shallow copy
                            # so subsequent rewinds still find the keys.
                            self.lm_gen.set_streaming_state_inplace(
                                dict(state_dict)
                            )
                            # Clear the safety-net state too. Otherwise
                            # _pad_force_remaining (12 frames of forced pad)
                            # carries over and the rewound state immediately
                            # re-triggers the streak.
                            self.lm_gen._pad_force_remaining = 0
                            self.lm_gen._non_pad_streak = 0
                            self._last_rewind_at = now
                            logger.warning(
                                "auto-rewind: %d pad-force triggers in %.0fs, "
                                "restored latest snapshot",
                                len(self._collapse_triggers),
                                COLLAPSE_WINDOW_SEC,
                            )
                            self._collapse_triggers.clear()
                            sess = self._active_session
                            loop = self._main_loop
                            if sess is not None and loop is not None:
                                # DataChannel sends touch the asyncio loop's
                                # SCTP transport; aiortc is not thread-safe.
                                # Schedule the send back on the loop thread.
                                try:
                                    loop.call_soon_threadsafe(
                                        sess.send_notice,
                                        "Auto-rewind: model wobbled, "
                                        "restored recent snapshot",
                                    )
                                except Exception as exc:
                                    logger.warning(
                                        "auto-rewind notice scheduling failed: %s: %s",
                                        type(exc).__name__,
                                        exc,
                                    )
            self._prev_pad_force_remaining = pad_force

            # --- inject window edge detection ------------------------
            # Surface inject-window open/close so the client can label
            # the brief audio gating ("Injecting context...") and so the
            # server log records what the user is hearing.
            now_inject_active = self._vision_inject_steps > 0
            if now_inject_active != self._inject_active:
                self._inject_active = now_inject_active
                if now_inject_active:
                    logger.info(
                        "vision inject window opened (%d tokens queued)",
                        len(self._vision_pending),
                    )
                else:
                    logger.info("vision inject window closed")
                sess = self._active_session
                loop = self._main_loop
                if sess is not None and loop is not None:
                    try:
                        loop.call_soon_threadsafe(
                            sess.send_inject_status, now_inject_active
                        )
                    except Exception as exc:
                        logger.warning(
                            "send_inject_status scheduling failed: %s: %s",
                            type(exc).__name__,
                            exc,
                        )

            # --- server-driven vision cadence ------------------------
            # When the model just entered a pad streak (silence), ask the
            # client to send a fresh frame. Also re-arm periodically
            # during sustained silence so observation-only sessions
            # (user moving around without speaking) still get frames at
            # a steady cadence; otherwise the transition fires once and
            # the user sees nothing until the fallback timer.
            entered_streak = (
                prev_pad_streak < LIVE_PROMPT_BOUNDARY_STREAK
                and self._vision_pad_streak >= LIVE_PROMPT_BOUNDARY_STREAK
            )
            sustained_rearm = (
                self._vision_pad_streak >= LIVE_PROMPT_BOUNDARY_STREAK
                and self._vision_pad_streak > prev_pad_streak
                and self._vision_pad_streak % PAD_STREAK_REREQUEST_EVERY == 0
            )
            if entered_streak or sustained_rearm:
                self._vision_request_pending = True
        return results

    def _take_snapshot(self) -> dict:
        """Capture the current streaming state of all modules.
        Uses flattening to produce a state dict that can be restored in-place
        to preserve CUDA graph memory addresses.

        Clone happens inside the lock. An earlier optimization tried to
        detach inside the lock and clone outside, but tensor.detach()
        shares storage with the original; the clone then copies whatever
        the executor thread is computing at clone time, producing a torn
        snapshot. Rewinds from such a snapshot restore corrupted state.
        """
        from .modules.streaming import _flatten_streaming_state
        with self._infer_lock:
            state = self.lm_gen.get_streaming_state()
            state_dict: dict = {}
            metadata: dict = {}
            _flatten_streaming_state(state_dict, metadata, state, prefix="")
            snapshot = {k: v.detach().clone() for k, v in state_dict.items()}
        snapshot.update(metadata)
        return snapshot

    async def handle_vision_frame(
        self,
        session_id: str,
        base64_data: str,
        clog: ColorizedLog,
        detail: bool = False,
    ):
        """Send a frame to Gemini using the stateful Interactions API.

        ``detail`` is set when the user explicitly requested this frame
        (UI "Capture Now" button). The frame itself is encoded at higher
        resolution on the client; we log it here for visibility.
        """
        if not self._gemini_api_key:
            return
        # Auto-disable kicks in after VISION_AUTO_DISABLE_THRESHOLD
        # consecutive non-2xx responses (handled below). Once tripped,
        # short-circuit until the next session starts.
        if self._vision_force_disabled:
            return

        # In-flight guard to prevent overlapping calls from corrupting the chain
        if session_id in self._vision_in_flight:
            return
        self._vision_in_flight.add(session_id)

        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

        try:
            prev_id = self._interaction_ids.get(session_id)
            url = f"https://generativelanguage.googleapis.com/v1beta/interactions?key={self._gemini_api_key}"

            input_parts = []
            if not prev_id:
                input_parts.append({
                    "type": "text",
                    "text": self._vision_system_prompt,
                })

            # Pull a snapshot of recent assistant text and feed it to
            # Gemini so the vision side knows what the model is currently
            # talking about. Keeps the scene description aligned with the
            # conversation. Skipped if empty or on the very first call
            # (the system prompt already covers the cold-start case).
            recent_snippet = "".join(list(self._transcript_recent)).strip()
            if recent_snippet:
                input_parts.append({
                    "type": "text",
                    "text": f"Recent assistant speech: {recent_snippet}",
                })

            if detail:
                clog.log("info", "vision: detail frame (user-requested)")
            
            input_parts.append({
                "type": "image",
                "mime_type": "image/jpeg",
                "data": base64_data
            })

            payload = {
                "model": "gemini-3.5-flash",
                "input": input_parts,
                "generation_config": {
                    "max_output_tokens": 50,
                    # Gemini 3.5 Flash defaults to medium thinking which
                    # adds 1-3 s TTFT. `thinking_level` lives directly
                    # in generation_config (NOT nested under a `thinking`
                    # object; that was the earlier 400 with "Unknown
                    # parameter 'thinking'"). If this also 400s the
                    # consecutive-error counter below will auto-disable
                    # vision and surface the error to the client.
                    "thinking_level": "minimal",
                },
            }
            if prev_id:
                payload["previous_interaction_id"] = prev_id

            headers = {"Api-Revision": "2026-05-20"}
            async with self._http_session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    self._gemini_consecutive_errors = 0
                    data = await resp.json()
                    new_id = data.get("id")
                    if new_id:
                        # Drop the write if the session rolled over while
                        # we were awaiting Gemini. Otherwise teardown's
                        # _interaction_ids.pop already ran (no-op then)
                        # and this would resurrect a dead entry.
                        if session_id == self._active_session_id:
                            self._interaction_ids[session_id] = new_id
                        else:
                            clog.log(
                                "warning",
                                "vision: late response after session close; dropping chain id",
                            )
                    else:
                        # Missing id means the Interactions chain is
                        # effectively reset; drop the prior id so the
                        # next call starts fresh with the system prompt.
                        clog.log(
                            "warning",
                            "Gemini response missing 'id'; dropping chain",
                        )
                        self._interaction_ids.pop(session_id, None)
                    
                    # Interactions API "new schema" (opt-in today via
                    # Api-Revision: 2026-05-20, default 2026-05-26).
                    # Shape: {"id": "...", "steps": [{"type":
                    #   "model_output", "content": [{"type": "text",
                    #   "text": "..."}]}, ...]}
                    # Earlier steps may be thoughts or tool calls;
                    # concatenate text from every model_output step's
                    # text-typed content blocks. This is what the SDK
                    # `interaction.output_text` convenience surfaces.
                    steps = data.get("steps") or []
                    text_parts: list[str] = []
                    for step in steps:
                        if step.get("type") != "model_output":
                            continue
                        for block in step.get("content") or []:
                            if block.get("type") == "text":
                                text_parts.append(block.get("text") or "")
                    text = "".join(text_parts).strip()
                    if not text:
                        clog.log(
                            "warning",
                            f"Gemini returned no text (steps={len(steps)})",
                        )
                    else:
                        clog.log("info", f"vision: {text}")
                        # Surface the description to the client UI.
                        # Non-blocking; failure is non-fatal but log it.
                        try:
                            sess = self._active_session
                            if sess is not None:
                                sess.send_vision_caption(text)
                                # Optional: echo the description into the
                                # main transcript with a [vision] prefix
                                # so the user can see what context the
                                # model is getting fed.
                                if self._vision_in_transcript:
                                    sess.send_text(f" [vision] {text} ")
                        except Exception as exc:
                            clog.log(
                                "warning",
                                f"send_vision_caption failed: {type(exc).__name__}: {exc}",
                            )
                        # Inject the raw description. No `<system>` wrap:
                        # PersonaPlex was trained with `<system>` only at
                        # t=0, so embedding it mid-stream is the most
                        # off-distribution part of the path. The empirical
                        # community recipe (VAOS gist, jmanhype 2026-02)
                        # drip-feeds the bare text at Mimi cadence and the
                        # state machine in _process_audio_frame gates the
                        # outbound audio while it does.
                        tokens = self.text_tokenizer.encode(f" {text}")

                        def _set_vision_context() -> None:
                            with self._infer_lock:
                                # asyncio.create_task can't cancel executor
                                # work mid-flight, so the per-session drain
                                # in _run_rtc_session.finally may return
                                # before this function runs. Gate the
                                # mutation on the active session id so a
                                # late Gemini response from a closed
                                # session can't clobber the next session's
                                # pending queue.
                                if session_id != self._active_session_id:
                                    return
                                # Replace pending queue: latest scene wins.
                                # An in-flight inject finishes its already-
                                # popped tokens; the next window picks up
                                # the fresh context.
                                self._vision_pending.clear()
                                self._vision_pending.extend(
                                    tokens[:VISION_QUEUE_MAX]
                                )

                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, _set_vision_context)
                else:
                    err_text = await resp.text()
                    clog.log("warning", f"Gemini Interactions error ({resp.status}): {err_text}")
                    self._interaction_ids.pop(session_id, None)
                    self._gemini_consecutive_errors += 1
                    if self._gemini_consecutive_errors >= VISION_AUTO_DISABLE_THRESHOLD:
                        self._vision_force_disabled = True
                        clog.log(
                            "warning",
                            f"vision auto-disabled after {self._gemini_consecutive_errors} consecutive errors",
                        )
                        try:
                            sess = self._active_session
                            if sess is not None:
                                sess.send_vision_status(False)
                                sess.send_notice(
                                    f"Vision auto-disabled after {VISION_AUTO_DISABLE_THRESHOLD} consecutive errors: {err_text[:120]}"
                                )
                        except Exception:
                            pass
        except Exception as exc:
            clog.log(
                "warning",
                f"vision processing failed: {type(exc).__name__}: {exc}",
            )
        finally:
            self._vision_in_flight.discard(session_id)

    def _resolve_voice_prompt_path(self, voice_prompt_filename: str) -> tuple[Optional[str], Optional[str]]:
        """Resolve the on-disk path for a voice prompt name.

        Returns (resolved_path, requested_path). resolved_path is None
        when no prompt was requested. Raises FileNotFoundError when a
        named prompt is missing or escapes the uploads dir.
        """
        if not voice_prompt_filename:
            return None, None
        if voice_prompt_filename.startswith(UPLOAD_PREFIX):
            upload_name = voice_prompt_filename[len(UPLOAD_PREFIX):]
            requested = self._resolve_upload_path(upload_name)
            if requested is None or not os.path.exists(requested):
                raise FileNotFoundError(
                    f"Uploaded voice prompt '{upload_name}' not found"
                )
            return requested, requested
        if self.voice_prompt_dir is None:
            return None, None
        requested = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
        if not os.path.exists(requested):
            raise FileNotFoundError(
                f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
            )
        return requested, requested

    async def _fetch_ice_servers(self) -> tuple[list[dict], bool]:
        """Return ``(iceServers, turn_failed)`` for the current session.

        With ``TURN_KEY_ID`` and ``TURN_KEY_API_TOKEN`` set, mints a fresh
        24-hour credential pack from Cloudflare Realtime and caches it for
        12 hours. Otherwise returns the STUN-only fallback, which only
        works when both peers can reach each other directly over UDP
        (i.e. on LAN; not through RunPod's HTTPS proxy).

        ``turn_failed`` is ``True`` only when TURN was configured but
        provisioning failed (4xx, non-JSON, network error, empty list).
        Callers facing the network use it to fail the session fast with
        503 instead of silently handing the client a STUN-only config
        that cannot traverse RunPod NAT. ``False`` for both healthy
        TURN and the no-TURN-configured LAN dev case.
        """
        if not (self._turn_key_id and self._turn_api_token):
            return [dict(s) for s in DEFAULT_STUN_FALLBACK], False

        async with self._ice_cache_lock:
            now = time.monotonic()
            if self._ice_cache is not None and now < self._ice_cache_expires_at:
                return self._ice_cache, False

            ttl_seconds = 86400  # Cloudflare's documented max.
            url = (
                "https://rtc.live.cloudflare.com/v1/turn/keys/"
                f"{self._turn_key_id}/credentials/generate"
            )
            stun_fallback = [dict(s) for s in DEFAULT_STUN_FALLBACK]
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {self._turn_api_token}",
                            "Content-Type": "application/json",
                        },
                        json={"ttl": ttl_seconds},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        body_text = await resp.text()
                        if resp.status >= 400:
                            logger.warning(
                                "Cloudflare TURN creds fetch failed: "
                                f"{resp.status} {body_text[:200]}"
                            )
                            return stun_fallback, True
                        try:
                            data = json.loads(body_text)
                        except ValueError as exc:
                            logger.warning(
                                f"Cloudflare TURN creds fetch returned non-JSON: {exc}"
                            )
                            return stun_fallback, True
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(f"Cloudflare TURN creds fetch error: {exc}")
                return stun_fallback, True

            servers = data.get("iceServers")
            if isinstance(servers, dict):
                # Cloudflare currently returns a single object; spec also
                # allows an array. Accept both.
                servers = [servers]
            if not isinstance(servers, list) or not servers:
                logger.warning(
                    "Cloudflare returned no iceServers; falling back to STUN"
                )
                return stun_fallback, True

            self._ice_cache = servers
            # Refresh halfway through the TTL so we never serve creds that
            # are about to expire mid-session.
            self._ice_cache_expires_at = now + ttl_seconds / 2
            refresh_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=ttl_seconds // 2)
            logger.info(
                "Cloudflare TURN creds minted (ttl=%ds, refresh at %s)",
                ttl_seconds,
                refresh_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            return servers, False

    async def handle_ice_servers(self, _request):
        servers, turn_failed = await self._fetch_ice_servers()
        if turn_failed:
            return web.json_response(
                {
                    "error": "turn_unavailable",
                    "detail": (
                        "TURN provisioning failed; the server cannot mint "
                        "Cloudflare credentials. Connections behind NAT "
                        "(including RunPod's HTTPS proxy) will not work. "
                        "Check TURN_KEY_ID / TURN_KEY_API_TOKEN and the "
                        "Cloudflare Realtime dashboard."
                    ),
                },
                status=503,
            )
        return web.json_response({"iceServers": servers})

    async def handle_rtc_candidate(self, request):
        """Accept a peer-trickled ICE candidate.

        Body: ``{"session_id": str, "candidate": str | null,
        "sdpMid": str | null, "sdpMLineIndex": int | null}``.
        ``candidate=null`` (or omitted) means the peer has finished
        gathering and we forward that as ``addIceCandidate(None)``.
        """
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"error": f"invalid json: {exc}"}, status=400)

        session_id = body.get("session_id")
        if not session_id or session_id not in self._candidate_sessions:
            return web.json_response({"error": "unknown session_id"}, status=404)

        session = self._candidate_sessions[session_id]
        try:
            await session.add_remote_candidate(
                body.get("candidate"),
                body.get("sdpMid"),
                body.get("sdpMLineIndex"),
            )
        except Exception as exc:
            logger.warning(f"addIceCandidate failed: {exc}")
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({"ok": True})

    async def handle_rtc_candidates_stream(self, request):
        """Stream local ICE candidates to the peer over SSE.

        Polls aiortc's gatherer at ~100 ms cadence and emits each new
        candidate as a server-sent event. Closes when gathering reports
        ``complete`` or the session goes away. Single-shot per session;
        the client opens this once after receiving the answer.
        """
        session_id = request.query.get("session_id", "")
        session = self._candidate_sessions.get(session_id)
        if session is None:
            return web.json_response({"error": "unknown session_id"}, status=404)

        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Disable nginx-style proxy buffering so the
                # Cloudflare/RunPod edge actually streams.
                "X-Accel-Buffering": "no",
            },
        )
        await resp.prepare(request)
        try:
            async for cand in session.iter_local_candidates():
                payload = json.dumps(cand)
                await resp.write(f"data: {payload}\n\n".encode("utf-8"))
            # Final sentinel so the client can close the EventSource
            # cleanly without waiting for a connection error.
            await resp.write(b"event: done\ndata: {}\n\n")
        except (asyncio.CancelledError, ConnectionResetError):
            raise
        except Exception as exc:
            logger.warning(f"candidate stream error: {exc}")
        return resp

    async def _try_acquire_session_lock(self, timeout: float) -> bool:
        """Acquire ``self.lock`` with a timeout, safe against the known
        ``asyncio.wait_for(lock.acquire())`` race.

        ``asyncio.wait_for`` cancels the inner coroutine on timeout, but
        ``Lock.acquire`` can complete the acquisition in the same tick the
        cancellation arrives. Older asyncio versions then leak the lock
        (cancellation propagates to the caller while the locked flag stays
        set). We work around it by shielding the acquire task and, on
        timeout, releasing the lock if the task in fact succeeded.
        """
        waiter = asyncio.create_task(self.lock.acquire())
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            # If the cancellation arrived after acquire() returned True,
            # the task is done with no exception and the lock is held by
            # this coroutine. Release it so future offers can proceed.
            if (
                waiter.done()
                and not waiter.cancelled()
                and waiter.exception() is None
            ):
                try:
                    self.lock.release()
                except RuntimeError:
                    pass
            return False

    async def handle_rtc_offer(self, request):
        """WebRTC signaling: accept SDP offer, return SDP answer.

        Lifecycle:
          1. Try to acquire ``self.lock`` with a short timeout. Return 409
             ``session_busy`` if a session is already live.
          2. Negotiate the peer connection (no GPU work yet) and return
             the answer. The browser opens its 'control' DataChannel.
          3. A background task waits for a ``config`` DataChannel
             message, applies it, runs system prompts under the lock,
             sends ``ready``, then starts the GPU process loop and
             holds the lock until the peer connection closes.
        """
        try:
            body = await request.json()
            offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])
        except (ValueError, KeyError) as exc:
            return web.json_response({"error": f"invalid offer: {exc}"}, status=400)

        if not await self._try_acquire_session_lock(timeout=0.25):
            return web.json_response({"error": "session_busy"}, status=409)

        # Once the lock is acquired, every failure path below MUST release
        # it. asyncio.CancelledError does not inherit from Exception in
        # Python 3.8+, so we use a bare ``except`` and re-raise after
        # cleanup. Without this, a client closing the HTTP connection
        # mid-negotiation, or any unexpected exception in RTCSession
        # construction, leaves the lock permanently held and every future
        # offer wedged on HTTP 409 until restart.
        session: Optional[RTCSession] = None
        session_id: Optional[str] = None
        owns_lock = True
        try:
            clog = ColorizedLog.randomize()
            peer = request.remote
            peer_port = (
                request.transport.get_extra_info("peername")[1]
                if request.transport is not None else "?"
            )
            clog.log("info", f"Incoming RTC offer from {peer}:{peer_port}")

            config_event: asyncio.Event = asyncio.Event()
            config_holder: dict = {"cfg": None}

            async def on_config(cfg: SessionConfig) -> None:
                if config_event.is_set():
                    clog.log("warning", "ignoring duplicate config message")
                    return
                config_holder["cfg"] = cfg
                config_event.set()

            t_ice = time.monotonic()
            ice_servers, turn_failed = await self._fetch_ice_servers()
            clog.log(
                "info",
                f"timing: ice_servers fetched in {(time.monotonic() - t_ice) * 1000:.0f} ms",
            )
            if turn_failed:
                # Refuse the session rather than hand the client a
                # STUN-only config that will fail to traverse NAT 30 s
                # later with no actionable signal.
                clog.log(
                    "error",
                    "TURN unavailable; refusing offer to avoid silent NAT failure",
                )
                self.lock.release()
                owns_lock = False
                return web.json_response(
                    {"error": "turn_unavailable"}, status=503
                )
            session = RTCSession(
                frame_size=self.frame_size,
                process_fn=self._process_audio_frame,
                log=clog.log,
                ice_servers=ice_servers,
            )
            session.set_config_handler(on_config)

            try:
                t_neg = time.monotonic()
                answer = await session.negotiate(offer)
                clog.log(
                    "info",
                    f"timing: negotiate (no ICE wait) {(time.monotonic() - t_neg) * 1000:.0f} ms",
                )
            except Exception as exc:
                clog.log("error", f"negotiate failed: {type(exc).__name__}: {exc}")
                await session.close()
                self.lock.release()
                owns_lock = False
                return web.json_response(
                    {"error": f"negotiate failed: {exc}"}, status=500
                )

            session_id = secrets.token_urlsafe(16)
            self._candidate_sessions[session_id] = session

            # Spawn the long-running session runner. It owns the lock from
            # this point on. Strong-ref the task so the event loop's weak
            # set cannot garbage-collect it. The runner is also the one
            # that removes session_id from _candidate_sessions on close,
            # so the trickle endpoints stay live for the full negotiation
            # window and a tick beyond.
            task = asyncio.create_task(
                self._run_rtc_session(
                    session, config_event, config_holder, clog, session_id
                )
            )
            self._session_tasks.add(task)
            task.add_done_callback(self._session_tasks.discard)
            owns_lock = False  # ownership transferred to the runner
            return web.json_response(
                {
                    "sdp": answer.sdp,
                    "type": answer.type,
                    "session_id": session_id,
                }
            )
        except BaseException:
            # Anything from a torn transport (peer_port lookup) to
            # RTCPeerConnection construction failures, including
            # asyncio.CancelledError if the client drops the request.
            if session is not None:
                try:
                    await session.close()
                except Exception:
                    pass
            # If we'd already registered the session for ICE trickle but
            # never handed ownership to the runner (create_task raised),
            # the runner's finally will never run; drop the entry here so
            # the candidate endpoints don't point at a closed session.
            if session_id is not None:
                self._candidate_sessions.pop(session_id, None)
            if owns_lock:
                try:
                    self.lock.release()
                except RuntimeError:
                    pass
            raise

    async def _run_rtc_session(
        self,
        session: "RTCSession",
        config_event: asyncio.Event,
        config_holder: dict,
        clog: ColorizedLog,
        session_id: Optional[str] = None,
    ) -> None:
        try:
            try:
                await asyncio.wait_for(config_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                clog.log("error", "no config received within 30 s, closing")
                session.send_error("config_timeout")
                return

            cfg: SessionConfig = config_holder["cfg"]
            clog.log("info", f"config: voice_prompt={cfg.voice_prompt!r}")

            try:
                voice_prompt_path, requested = self._resolve_voice_prompt_path(
                    cfg.voice_prompt
                )
            except FileNotFoundError as exc:
                clog.log("error", str(exc))
                session.send_error(f"voice_prompt_not_found: {exc}")
                return

            if voice_prompt_path is not None and self.lm_gen.voice_prompt != voice_prompt_path:
                t_vp = time.monotonic()
                if voice_prompt_path.endswith(".pt"):
                    self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
                else:
                    self.lm_gen.load_voice_prompt(voice_prompt_path)
                clog.log(
                    "info",
                    f"timing: voice prompt load {(time.monotonic() - t_vp) * 1000:.0f} ms ({voice_prompt_path})",
                )

            # Empty list (not None) so _step_text_prompt_core iterates as a
            # no-op when the user clears the textarea. Iterating None raises
            # TypeError inside the executor and tears the session down.
            self.lm_gen.text_prompt_tokens = (
                self.text_tokenizer.encode(wrap_with_system_tags(cfg.text_prompt))
                if cfg.text_prompt else []
            )
            if cfg.seed is not None and cfg.seed != -1:
                seed_all(cfg.seed)

            self.lm_gen.temp = cfg.audio_temperature
            self.lm_gen.temp_text = cfg.text_temperature
            self.lm_gen.top_k_text = max(1, cfg.text_topk)
            self.lm_gen.top_k = max(1, cfg.audio_topk)
            self.lm_gen.repetition_penalty = max(1.0, cfg.repetition_penalty)
            self.lm_gen.repetition_penalty_context = max(
                0, min(cfg.repetition_penalty_context, MAX_REPETITION_CONTEXT)
            )
            self.lm_gen.padding_bonus = max(0.0, cfg.padding_bonus)
            self.lm_gen.max_turn_text_tokens = max(0, cfg.max_turn_text_tokens)
            self.lm_gen._non_pad_streak = 0
            self.lm_gen._pad_force_remaining = 0

            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            # Reset the vision-inject state machine and the transcript
            # buffer. Leftover state from a previous session would
            # otherwise leak into this one.
            with self._infer_lock:
                self._vision_pending.clear()
                self._vision_pad_streak = 0
                self._vision_inject_steps = 0
                self._transcript_recent.clear()
            # Apply the per-session vision system prompt. Falls back to
            # the generic default when the client didn't supply one.
            self._vision_system_prompt = (
                cfg.vision_prompt.strip() or DEFAULT_VISION_SYSTEM_PROMPT
            )
            self._vision_in_transcript = bool(cfg.vision_in_transcript)
            # Expose the session and id so vision-side coroutines can push
            # captions back to the client, and so the executor-side
            # collapse detector can find the right snapshot list.
            self._active_session = session
            self._active_session_id = session_id
            # Stash the loop so the executor thread can schedule sends.
            self._main_loop = asyncio.get_event_loop()
            # Reset collapse-detection state for the new session.
            self._collapse_triggers.clear()
            self._prev_pad_force_remaining = 0
            self._vision_request_pending = False
            self._inject_active = False
            self._last_rewind_at = 0.0
            # Reset auto-disable so a previous session's vision failures
            # don't carry over and silently block this session's calls.
            self._gemini_consecutive_errors = 0
            self._vision_force_disabled = False

            # System prompts are 10-25 s of synchronous Mimi+LM steps
            # (longer for raw-audio voice prompts because every prompt
            # frame goes through Mimi.encode). Running them inline on
            # the asyncio main thread starves aiortc's RTCP keepalive
            # and ICE consent-freshness tasks; after ~30 s with no
            # outbound packets, the peer connection state flips to
            # 'failed' and the client sees "Connection failed". Pushing
            # the work into the default thread executor (the same path
            # _process_audio_frame already uses) keeps the loop free.
            #
            # Run the 4 phases sequentially in their own executor calls
            # so we can check session.is_alive() between phases and
            # bail early if the peer dropped mid-warmup. Each phase is
            # shield+drained on cancel so the GPU thread cannot keep
            # mutating lm_gen / mimi streaming state past the lock
            # release in finally.
            t_sp = time.monotonic()
            loop = asyncio.get_event_loop()

            self._vision_tasks[session_id] = set()

            async def on_message(msg: dict):
                mtype = msg.get("type")
                if mtype == "rewind":
                    snapshots = self._session_snapshots.get(session_id, [])
                    if not snapshots:
                        clog.log("warning", "rewind requested but no snapshots available")
                        try:
                            session.send_notice("Rewind: no snapshot yet (wait a few seconds)")
                        except Exception:
                            pass
                        return
                    snap_ts, state_dict = snapshots[-1]
                    age_sec = max(0.0, time.monotonic() - snap_ts)
                    clog.log("info", f"rewinding to snapshot from {age_sec:.0f} s ago")

                    def _do_rewind():
                        with self._infer_lock:
                            # set_streaming_state_inplace consumes the dict it's given.
                            # Pass a shallow copy so the snapshot stays reusable on the
                            # next rewind.
                            self.lm_gen.set_streaming_state_inplace(dict(state_dict))

                    await loop.run_in_executor(None, _do_rewind)
                    try:
                        session.send_notice(f"Rewound to snapshot from {age_sec:.0f} s ago")
                    except Exception:
                        pass
                elif mtype == "vision_frame":
                    base64_data = msg.get("data", "")
                    if base64_data:
                        # Cap inbound frame size. Real frames at /2
                        # downscale + JPEG 0.55 are well under 100 KB;
                        # native + 0.8 detail mode stays under ~400 KB.
                        # 600 KB headroom catches both without exposing
                        # the server to a runaway client.
                        if len(base64_data) > 600_000:
                            clog.log(
                                "warning",
                                f"vision_frame too large: {len(base64_data)} chars; dropping",
                            )
                            return
                        detail = bool(msg.get("detail", False))
                        task = asyncio.create_task(
                            self.handle_vision_frame(
                                session_id, base64_data, clog, detail=detail
                            )
                        )
                        tasks = self._vision_tasks.get(session_id)
                        if tasks is not None:
                            tasks.add(task)
                            task.add_done_callback(tasks.discard)

            session.set_message_handler(on_message)

            # Block the first snapshot until warmup completes. Warmup runs
            # in an executor without holding _infer_lock; snapshot_task
            # acquires the lock, so without this gate the first +30 s
            # snapshot can race a long voice-prompt load and read a torn
            # _streaming_state.
            warmup_done = asyncio.Event()

            async def snapshot_task():
                try:
                    await warmup_done.wait()
                    while session.is_alive():
                        # Snapshots cost a brief audio-frame stall (lock
                        # held during tensor clone). 60 s keeps that hit
                        # to once per minute; rewinds still target a
                        # state from within the last minute.
                        await asyncio.sleep(60.0)
                        if not session.is_alive():
                            break
                        clog.log("info", "taking session snapshot")
                        snap = await loop.run_in_executor(None, self._take_snapshot)
                        # Teardown can pop the bucket while the executor is
                        # cloning. setdefault here would resurrect a stale
                        # entry that lives forever.
                        if not session.is_alive():
                            break
                        history = self._session_snapshots.get(session_id)
                        if history is None:
                            break
                        history.append((time.monotonic(), snap))
                        # Keep only last 5 snapshots
                        if len(history) > 5:
                            history.pop(0)
                except asyncio.CancelledError:
                    pass

            _snap_t = asyncio.create_task(snapshot_task())
            self._session_tasks.add(_snap_t)
            _snap_t.add_done_callback(self._session_tasks.discard)

            async def cadence_task():
                """Drain _vision_request_pending and ping the client.

                The executor thread sets the flag when the model enters
                a fresh pad streak. We poll at 5 Hz and dispatch.
                """
                try:
                    while session.is_alive():
                        await asyncio.sleep(0.2)
                        if self._vision_request_pending and session.is_alive():
                            self._vision_request_pending = False
                            try:
                                session.send_request_vision_frame()
                            except Exception as exc:
                                clog.log(
                                    "warning",
                                    f"send_request_vision_frame failed: "
                                    f"{type(exc).__name__}: {exc}",
                                )
                except asyncio.CancelledError:
                    pass

            _cad_t = asyncio.create_task(cadence_task())
            self._session_tasks.add(_cad_t)
            _cad_t.add_done_callback(self._session_tasks.discard)
            phases = (
                ("voice_prompt", self.lm_gen._step_voice_prompt, (self.mimi,)),
                ("audio_silence_a", self.lm_gen._step_audio_silence, ()),
                ("text_prompt", self.lm_gen._step_text_prompt, ()),
                ("audio_silence_b", self.lm_gen._step_audio_silence, ()),
            )
            warmup_aborted = False
            for phase_name, phase_fn, phase_args in phases:
                if not session.is_alive():
                    clog.log(
                        "info",
                        f"client disconnected during warmup before {phase_name}; aborting",
                    )
                    warmup_aborted = True
                    break
                in_flight = asyncio.ensure_future(
                    loop.run_in_executor(None, phase_fn, *phase_args)
                )
                try:
                    await asyncio.shield(in_flight)
                except asyncio.CancelledError:
                    try:
                        await in_flight
                    except BaseException:
                        pass
                    raise
            if warmup_aborted:
                return
            self.mimi.reset_streaming()
            clog.log(
                "info",
                f"timing: system prompts {(time.monotonic() - t_sp) * 1000:.0f} ms",
            )
            warmup_done.set()

            if not session.is_alive():
                clog.log("info", "client disconnected during warmup")
                return

            # Capture a baseline snapshot before the user can interact, so the
            # Rewind button always has something to restore even in the first
            # 30 s of the session (snapshot_task otherwise only fires at +30 s).
            try:
                baseline = await loop.run_in_executor(None, self._take_snapshot)
                self._session_snapshots.setdefault(session_id, []).append(
                    (time.monotonic(), baseline)
                )
                clog.log("info", "baseline snapshot captured")
            except Exception as exc:
                clog.log(
                    "warning",
                    f"baseline snapshot failed: {type(exc).__name__}: {exc}",
                )

            session.send_ready()
            # Tell the client whether the vision pipeline is reachable so
            # it can disable the Add Vision button (or warn the user) when
            # the server has no GEMINI_API_KEY configured.
            try:
                session.send_vision_status(bool(self._gemini_api_key))
            except Exception as exc:
                clog.log(
                    "warning",
                    f"send_vision_status failed: {type(exc).__name__}: {exc}",
                )
            session.start_processing()
            await session.wait_for_close()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            clog.log("error", f"_run_rtc_session: {type(exc).__name__}: {exc}")
            try:
                session.send_error(f"server_error: {exc}")
            except Exception:
                pass
        finally:
            if session_id is not None:
                self._candidate_sessions.pop(session_id, None)
                self._interaction_ids.pop(session_id, None)
                self._session_snapshots.pop(session_id, None)
                self._vision_in_flight.discard(session_id)
                # Drain in-flight Gemini calls before the next session can
                # acquire the lock. A stale handle_vision_frame still
                # awaiting a response would otherwise overwrite the next
                # session's _vision_pending under _infer_lock.
                pending_vision = self._vision_tasks.pop(session_id, set())
                for vt in pending_vision:
                    vt.cancel()
                if pending_vision:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*pending_vision, return_exceptions=True),
                            timeout=2.0,
                        )
                    except asyncio.TimeoutError:
                        clog.log("warning", "vision tasks did not drain within 2 s")
            self._active_session = None
            self._active_session_id = None
            self._main_loop = None
            try:
                await session.close()
            finally:
                # Return cached-but-freed GPU blocks to the allocator so
                # external observers (nvidia-smi, RunPod metrics) see VRAM
                # drop back to baseline between sessions. The model
                # weights and KV cache buffer stay resident; only the
                # snapshot clones and transient allocations are released.
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                self.lock.release()
                clog.log("info", "session closed, lock released")


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - try to download voices.tgz from HF
      - extract it once
      - return extracted directory (or None if not available)
    If voice_prompt_dir is provided:
      - just return it
    """
    def _resolve_voice_dir(candidate: Path) -> Optional[Path]:
        if any(candidate.glob("*.pt")):
            return candidate
        nested = candidate / "voices"
        if any(nested.glob("*.pt")):
            logger.info(f"Found nested voices directory: {nested}")
            return nested
        return None

    if voice_prompt_dir is not None:
        resolved_dir = _resolve_voice_dir(Path(voice_prompt_dir))
        return str(resolved_dir) if resolved_dir is not None else voice_prompt_dir

    logger.info("retrieving voice prompts")

    # Get HF_TOKEN from environment or cache
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub.utils import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            pass

    # Try to download voices.tgz, but it's optional
    try:
        voices_tgz = hf_hub_download(hf_repo, "voices.tgz", token=hf_token)
        voices_tgz = Path(voices_tgz)
        voices_dir = voices_tgz.parent / "voices"

        if not voices_dir.exists():
            logger.info(f"extracting {voices_tgz} to {voices_tgz.parent}")
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)

        resolved_dir = _resolve_voice_dir(voices_dir)
        if resolved_dir is None:
            logger.info("voices directory exists but no .pt files found; re-extracting")
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
            resolved_dir = _resolve_voice_dir(voices_dir)

        if resolved_dir is None:
            logger.warning("voices.tgz did not contain a usable voices directory")
            return None

        return str(resolved_dir)
    except Exception as e:
        logger.info(f"Voice prompts not available from repository (this is normal): {e}")
        logger.info("Server will run without custom voice prompts")
        return None


def _get_static_path(static: Optional[str]) -> Optional[str]:
    """Resolve the static-content directory.

    None or "none": return None so the embedded WebRTC HTML is served.
    Any other value: a user-supplied directory of static files to serve.
    """
    if static is None or static == "none":
        return None
    return static


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--uploads-dir",
        type=str,
        help=(
            "Directory where user-uploaded voice prompt audio files are stored. "
            "Defaults to '<voice-prompt-dir>/uploads' when voice-prompt-dir is set, "
            "otherwise disables the upload endpoint. Pass an explicit path to enable "
            "uploads even without a preset voice directory."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    # Resolve uploads_dir. Default: <voice_prompt_dir>/uploads if the preset dir
    # exists; otherwise None (upload endpoint disabled unless user passes
    # --uploads-dir explicitly).
    if args.uploads_dir is None and args.voice_prompt_dir is not None:
        args.uploads_dir = os.path.join(args.voice_prompt_dir, "uploads")
    if args.uploads_dir is not None:
        os.makedirs(args.uploads_dir, exist_ok=True)
    logger.info(f"uploads_dir = {args.uploads_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)
    logger.info(
        "torch=%s cuda_available=%s device=%s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Get HF_TOKEN from environment
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("HF_TOKEN found in environment")
    else:
        logger.warning("HF_TOKEN not found in environment. Downloads may fail if authentication is required.")
        # Try to get token from huggingface_hub cache
        try:
            from huggingface_hub.utils import HfFolder
            cached_token = HfFolder.get_token()
            if cached_token:
                hf_token = cached_token
                logger.info("Using token from HuggingFace cache")
        except Exception:
            pass
    
    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME, token=hf_token)
    t = time.monotonic()
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded in %.1f s", time.monotonic() - t)

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME, token=hf_token)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME, token=hf_token)
    t = time.monotonic()
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded in %.1f s", time.monotonic() - t)
    # Surface the inner-monologue yield token so a mismatch with the
    # checkpoint's actual padding semantics is obvious at boot. If
    # padding_bonus silently does nothing, it's usually because this piece is
    # not what the fine-tune emits during silence.
    try:
        _pad_id = lm.text_padding_token_id
        _pad_piece = text_tokenizer.id_to_piece(_pad_id)
        logger.info(f"text_padding_token_id={_pad_id} piece={_pad_piece!r} (target of padding_bonus)")
    except Exception as e:
        logger.warning(f"could not resolve text_padding_token_id: {e}")
    
    lm_gen = LMGen(lm,
                        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                        sample_rate=mimi.sample_rate,
                        device=args.device,
                        frame_rate=mimi.frame_rate,
                        save_voice_prompt_embeddings=False)

    state = ServerState(
        mimi=mimi,
        lm_gen=lm_gen,
        text_tokenizer=text_tokenizer,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        uploads_dir=args.uploads_dir,
        save_voice_prompt_embeddings=False
    )
    logger.info("warming up the model")
    t = time.monotonic()
    state.warmup()
    logger.info("warmup complete in %.1f s", time.monotonic() - t)
    logger.info(
        "vision: %s",
        "enabled" if state._gemini_api_key else "disabled (no GEMINI_API_KEY)",
    )

    # Pre-warm Cloudflare TURN credentials so the very first session
    # after boot does not pay the credential mint round-trip. The
    # creds are cached in-process for 12 h after this call. No-op
    # when TURN_KEY_ID / TURN_KEY_API_TOKEN are unset.
    if state._turn_key_id and state._turn_api_token:
        try:
            t0 = time.monotonic()
            _, turn_failed = asyncio.run(state._fetch_ice_servers())
            if turn_failed:
                logger.warning(
                    "TURN pre-warm failed; sessions will be refused with 503 "
                    "until creds mint successfully on demand"
                )
            else:
                logger.info(
                    f"TURN creds pre-warmed in {(time.monotonic() - t0) * 1000:.0f} ms"
                )
        except Exception as exc:  # never block startup on a TURN hiccup
            logger.warning(f"TURN pre-warm failed (will mint on demand): {exc}")

    app = web.Application(client_max_size=UPLOAD_MAX_BYTES + 1024 * 1024)
    app.router.add_post("/api/rtc/offer", state.handle_rtc_offer)
    app.router.add_post("/api/rtc/candidate", state.handle_rtc_candidate)
    app.router.add_get("/api/rtc/candidates", state.handle_rtc_candidates_stream)
    app.router.add_get("/api/rtc/ice-servers", state.handle_ice_servers)
    app.router.add_post("/api/voice-upload", state.handle_voice_upload)

    async def handle_favicon(_):
        # Browser auto-requests /favicon.ico on every page; without a
        # route the server logs a 404 noise line on every visit.
        return web.Response(status=204)
    app.router.add_get("/favicon.ico", handle_favicon)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    else:
        # Serve embedded web client when no built static content is available
        async def handle_embedded_client(_):
            html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PersonaPlex - SurAiverse Edition</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300..700&family=Source+Serif+4:opsz,wght@8..60,300..700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Source Serif 4', serif;
            background:
                radial-gradient(1200px 600px at 10% -10%, rgba(198, 161, 91, 0.22), transparent 60%),
                radial-gradient(900px 500px at 90% 10%, rgba(47, 93, 80, 0.18), transparent 55%),
                linear-gradient(180deg, #f7f2ea 0%, #efe7d8 45%, #e8dcc8 100%);
            color: #1c1a17; min-height: 100vh; display: flex; flex-direction: column;
        }
        .header { padding: 24px 20px; text-align: center; border-bottom: 1px solid rgba(154, 122, 58, 0.35); background: rgba(244, 239, 230, 0.85); }
        .header h1 { color: #1c1a17; font-size: 2.4em; margin-bottom: 6px; font-family: 'Fraunces', serif; letter-spacing: 0.03em; }
        .header .brand-tagline { color: #3a3329; font-size: 0.95em; }
        .header .brand-subtag { color: #9a7a3a; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.2em; margin-top: 6px; }
        .main { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 26px 20px; }
        .chat-container { width: 100%; max-width: 700px; }

        .status-strip { background: rgba(255, 255, 255, 0.7); border: 1px solid rgba(154, 122, 58, 0.35); 
                        border-radius: 14px; padding: 12px 16px; margin: 20px 0 24px; box-shadow: 0 6px 18px rgba(26, 20, 12, 0.12); }
        .status-row { display: flex; justify-content: space-between; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.18em; color: #3a3329; margin-bottom: 8px; }
        .progress-track { height: 8px; border-radius: 999px; background: rgba(47, 93, 80, 0.15); overflow: hidden; }
        .progress-bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #c6a15b 0%, #e1c48a 55%, #9a7a3a 100%); transition: width 0.3s ease; }
        .progress-steps { display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.7em; color: rgba(58, 51, 41, 0.6); }
        .progress-steps span.active { color: #2f5d50; font-weight: 600; }
        
        /* Homepage / Setup View */
        .setup-view { display: block; }
        .conversation-view { display: none; }
        .setup-view.hidden { display: none; }
        .conversation-view.active { display: block; }
        
        /* Form styling for light theme */
        .form-section { background: rgba(250, 246, 239, 0.92); border-radius: 16px; padding: 24px; margin-bottom: 20px; 
                        border: 1px solid rgba(156, 131, 84, 0.3); box-shadow: 0 6px 18px rgba(26, 20, 12, 0.12); }
        .form-section-title { font-size: 0.95em; font-weight: 600; color: #3a3329; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.16em; }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; font-size: 0.9em; font-weight: 500; color: #555; margin-bottom: 8px; }
        .form-group textarea, .form-group select { 
            width: 100%; padding: 12px; border-radius: 12px; border: 1px solid rgba(156, 131, 84, 0.4);
            background: rgba(255, 255, 255, 0.9); color: #1c1a17; font-size: 0.95em; transition: border-color 0.2s; }
        .form-group textarea:focus, .form-group select:focus { 
            outline: none; border-color: #9a7a3a; box-shadow: 0 0 0 3px rgba(198,161,91,0.2); }
        .form-group textarea { min-height: 100px; resize: vertical; }
        .char-count { text-align: right; font-size: 0.8em; color: #888; margin-top: 4px; }
        
        /* Preset buttons */
        .presets-container { background: rgba(255, 255, 255, 0.6); border-radius: 12px; padding: 12px; margin-bottom: 12px; border: 1px solid rgba(156, 131, 84, 0.2); }
        .presets-label { font-size: 0.75em; font-weight: 500; color: #8a7a5a; margin-bottom: 8px; display: block; text-transform: uppercase; letter-spacing: 0.18em; }
        .presets { display: flex; flex-wrap: wrap; gap: 8px; }
        .preset-btn { padding: 6px 14px; font-size: 0.82em; background: rgba(255,255,255,0.9); color: #5f5136; 
                      border: 1px solid rgba(156, 131, 84, 0.4); border-radius: 20px; cursor: pointer; transition: all 0.2s; }
        .preset-btn:hover { background: #2f5d50; color: #f7f1e6; border-color: #2f5d50; }
        
        /* Status badge */
        .status-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; 
                        border-radius: 20px; background: rgba(255,255,255,0.8); box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; }
        .status-dot.connected { background: #76b900; box-shadow: 0 0 10px rgba(118,185,0,0.5); }
        .status-dot.connecting { background: #f0ad4e; animation: pulse 1s infinite; }
        .status-dot.disconnected { background: #dc3545; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        /* Buttons */
        .btn { padding: 14px 32px; border-radius: 30px; border: none; font-size: 0.95em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
               cursor: pointer; transition: all 0.3s; display: inline-flex; align-items: center; gap: 8px; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: #2f5d50; color: #f7f1e6; }
        .btn-primary:hover:not(:disabled) { background: #24463b; transform: translateY(-2px); 
                                            box-shadow: 0 5px 20px rgba(47,93,80,0.35); }
        .btn-danger { background: #9a3b3b; color: #fff; }
        .btn-danger:hover:not(:disabled) { background: #7f2f2f; transform: translateY(-2px); }
        .btn-container { text-align: center; margin-top: 24px; }

        /* Advanced sliders */
        .advanced-toggle { display: flex; align-items: center; justify-content: space-between; cursor: pointer;
                           padding: 4px 0; user-select: none; }
        .advanced-toggle .arrow { transition: transform 0.2s; font-size: 0.8em; color: #6e5d3b; }
        .advanced-toggle.open .arrow { transform: rotate(90deg); }
        .advanced-body { display: none; margin-top: 16px; }
        .advanced-body.open { display: block; }
        .slider-group-title { font-size: 0.75em; font-weight: 600; color: #2f5d50;
                              text-transform: uppercase; letter-spacing: 0.14em;
                              margin: 18px 0 10px; padding-bottom: 6px;
                              border-bottom: 1px solid rgba(47, 93, 80, 0.2); }
        .slider-group-title:first-child { margin-top: 0; }
        .toggle-row { display: flex; align-items: center; justify-content: space-between;
                      padding: 8px 0; font-size: 0.88em; color: #3a3329; }
        .toggle-row label { display: flex; align-items: center; gap: 10px; cursor: pointer;
                            user-select: none; font-weight: 500; }
        .toggle-row input[type="checkbox"] { accent-color: #2f5d50; cursor: pointer;
                                              width: 16px; height: 16px; }
        .toggle-row-hint { font-size: 0.72em; color: #8a7a5a; margin: -2px 0 10px 26px;
                            line-height: 1.4; }
        .slider-row { margin-bottom: 14px; }
        .slider-row .slider-label { display: flex; justify-content: space-between; font-size: 0.82em;
                                    color: #3a3329; margin-bottom: 6px; font-weight: 500; }
        .slider-row .slider-label .slider-value { color: #2f5d50; font-variant-numeric: tabular-nums; font-weight: 600; }
        .slider-row input[type="range"] { width: 100%; accent-color: #2f5d50; }
        .slider-row .slider-hint { font-size: 0.72em; color: #8a7a5a; margin-top: 4px; line-height: 1.4; }
        .slider-actions { display: flex; gap: 8px; margin-top: 8px; }
        .slider-reset { background: rgba(255,255,255,0.85); color: #5f5136; border: 1px solid rgba(156, 131, 84, 0.4);
                        border-radius: 6px; padding: 6px 12px; font-size: 0.78em; cursor: pointer; }
        .slider-reset:hover { background: #2f5d50; color: #f7f1e6; border-color: #2f5d50; }
        .seed-row { margin-bottom: 14px; }
        .seed-row .seed-toggle { display: inline-flex; align-items: center; gap: 6px; font-size: 0.78em;
                                  color: #5f5136; font-weight: 500; cursor: pointer; user-select: none; }
        .seed-row .seed-toggle input { accent-color: #2f5d50; cursor: pointer; }
        .seed-row input[type="number"] { width: 100%; padding: 8px 10px; border-radius: 8px;
                                          border: 1px solid rgba(156, 131, 84, 0.4); background: rgba(255, 255, 255, 0.9);
                                          color: #1c1a17; font-size: 0.9em; font-family: inherit;
                                          font-variant-numeric: tabular-nums; }
        .seed-row input[type="number"]:focus { outline: none; border-color: #9a7a3a;
                                                box-shadow: 0 0 0 3px rgba(198,161,91,0.2); }
        .seed-row input[type="number"]:disabled { opacity: 0.5; cursor: not-allowed;
                                                   background: rgba(245, 240, 228, 0.6); }
        
        /* Conversation view */
        .visualizer-container { display: flex; gap: 30px; justify-content: center; margin: 30px 0; }
        .visualizer { width: 140px; height: 140px; border-radius: 50%; display: flex; align-items: center; 
                      justify-content: center; position: relative; background: rgba(255,255,255,0.85); 
                      box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
        .visualizer.ai { border: 3px solid #00a8cc; }
        .visualizer.user { border: 3px solid #76b900; }
        .visualizer-label { position: absolute; bottom: -30px; font-size: 0.9em; color: #666; font-weight: 500; }
        .visualizer-canvas { position: absolute; inset: 0; width: 100%; height: 100%;
                             border-radius: 50%; pointer-events: none; }
        
        .transcript { background: rgba(255,255,255,0.9); border-radius: 12px; padding: 20px; min-height: 100px; 
                      max-height: 200px; overflow-y: auto; margin-bottom: 24px; 
                      box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .transcript-label { font-size: 0.85em; color: #888; margin-bottom: 10px; font-weight: 500; }
        .transcript-text { font-size: 1.05em; line-height: 1.7; color: #333; }
        
        .controls { display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; }

        .download-row { display: none; align-items: center; justify-content: space-between; gap: 12px;
                        background: rgba(255,255,255,0.85); border: 1px solid rgba(156, 131, 84, 0.35);
                        border-radius: 14px; padding: 14px 16px; margin-top: 18px; }
        .download-title { font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.18em; color: #6e5d3b; }
        .download-sub { font-size: 0.8em; color: #7b6a4a; }
        
        .footer { padding: 20px; text-align: center; border-top: 1px solid rgba(154, 122, 58, 0.35); background: rgba(244, 239, 230, 0.85); }
        .footer a { color: #2f5d50; text-decoration: none; }
        .footer a:hover { text-decoration: underline; }
        
        .error-msg { background: #fff5f5; border: 1px solid #dc3545; color: #dc3545; padding: 15px;
                     border-radius: 8px; margin-bottom: 20px; display: none; }
        .mic-icon { width: 24px; height: 24px; }

        /* Voice upload */
        .upload-row { margin-top: 14px; padding-top: 14px; border-top: 1px dashed rgba(154, 122, 58, 0.35); }
        .upload-toggle-btn { background: rgba(255,255,255,0.85); color: #5f5136;
                             border: 1px solid rgba(156, 131, 84, 0.4); border-radius: 8px;
                             padding: 8px 14px; font-size: 0.85em; cursor: pointer;
                             font-family: inherit; transition: all 0.2s; }
        .upload-toggle-btn:hover { background: #2f5d50; color: #f7f1e6; border-color: #2f5d50; }
        .upload-toggle-btn .arrow { display: inline-block; margin-right: 6px; transition: transform 0.2s; }
        .upload-toggle-btn.open .arrow { transform: rotate(90deg); }
        .upload-area { display: none; margin-top: 12px; padding: 14px;
                       background: rgba(250, 246, 239, 0.7); border-radius: 10px;
                       border: 1px dashed rgba(156, 131, 84, 0.5); }
        .upload-area.open { display: block; }
        .upload-area .hint { font-size: 0.78em; color: #8a7a5a; margin-bottom: 10px; line-height: 1.45; }
        .upload-area input[type="file"] { font-family: inherit; font-size: 0.88em; color: #3a3329; }
        .upload-area input[type="file"]::file-selector-button {
            margin-right: 10px; padding: 6px 12px; border-radius: 6px;
            border: 1px solid rgba(156, 131, 84, 0.4); background: #fff; color: #5f5136;
            font-family: inherit; font-size: 0.85em; cursor: pointer;
        }
        .upload-area input[type="file"]::file-selector-button:hover {
            background: #2f5d50; color: #f7f1e6; border-color: #2f5d50;
        }
        .upload-status { margin-top: 10px; font-size: 0.85em; min-height: 1.2em; }
        .upload-status.uploading { color: #6e5d3b; }
        .upload-status.success { color: #2f5d50; font-weight: 600; }
        .upload-status.error { color: #9a3b3b; }
        .upload-clear { display: none; margin-top: 8px; padding: 5px 12px; font-size: 0.78em;
                        background: rgba(255,255,255,0.85); color: #9a3b3b;
                        border: 1px solid rgba(154, 59, 59, 0.4); border-radius: 6px; cursor: pointer; }
        .upload-clear:hover { background: #9a3b3b; color: #fff; border-color: #9a3b3b; }
        .upload-clear.visible { display: inline-block; }
        select:disabled { opacity: 0.55; cursor: not-allowed; }
        
        /* Vision Preview */
        .vision-container { display: none; width: 100%; max-width: 500px; margin: 0 auto 20px; border-radius: 16px; overflow: hidden; 
                           border: 1px solid rgba(154, 122, 58, 0.35); box-shadow: 0 8px 24px rgba(0,0,0,0.15); background: #000; position: relative; }
        .vision-container.active { display: block; }
        .vision-video { width: 100%; display: block; transform: none !important; } /* Explicitly disable mirroring */
        .vision-label { position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.6); color: #fff; padding: 4px 10px; 
                         border-radius: 6px; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.1em; backdrop-filter: blur(4px); }
        .vision-status { position: absolute; bottom: 12px; right: 12px; color: #fff; font-size: 0.7em; opacity: 0.8; }
        .vision-caption { position: absolute; bottom: 12px; left: 12px; right: 90px; color: #fff; font-size: 0.85em;
                          line-height: 1.3; text-shadow: 0 1px 3px rgba(0,0,0,0.85); opacity: 0;
                          transition: opacity 0.25s ease; pointer-events: none; }
        .vision-caption.visible { opacity: 1; }
        .vision-meta { display: none; max-width: 500px; margin: 4px auto 8px; padding: 0 4px;
                       font-size: 0.78em; color: #6a5a40; gap: 12px; align-items: center;
                       justify-content: center; flex-wrap: wrap; }
        .vision-meta.visible { display: flex; }
        .vision-meta select { font-size: 0.92em; padding: 2px 6px; }
        .captions-log { display: none; max-width: 500px; margin: 0 auto 12px; padding: 10px 14px;
                        background: rgba(154, 122, 58, 0.06); border: 1px solid rgba(154, 122, 58, 0.18);
                        border-radius: 10px; font-size: 0.8em; line-height: 1.5; max-height: 160px;
                        overflow-y: auto; }
        .captions-log.visible { display: block; }
        .captions-log-title { font-weight: 600; color: #5a4a32; margin-bottom: 6px;
                              text-transform: uppercase; font-size: 0.75em; letter-spacing: 0.08em; }
        .captions-log-entry { color: #3a3329; }
        .captions-log-entry .ts { color: #9a8a6a; margin-right: 6px; }
        .notice-toast { position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
                        background: rgba(58, 51, 41, 0.95); color: #efe7d8; padding: 10px 18px;
                        border-radius: 10px; font-size: 0.85em; box-shadow: 0 6px 18px rgba(0,0,0,0.25);
                        opacity: 0; transition: opacity 0.25s ease; pointer-events: none; z-index: 9999; }
        .notice-toast.visible { opacity: 1; }
        
        .btn-vision { background: #2f5d50; color: #fff; }
        .btn-vision.active { background: #9a3b3b; }
        .btn-rewind { background: rgba(255,255,255,0.9); color: #3a3329; border: 1px solid rgba(154, 122, 58, 0.4); }
        .btn-rewind:hover { background: #efe7d8; }
        
        /* Responsive */
        @media (max-width: 600px) {
            .chat-container { padding: 0 10px; }
            .form-section { padding: 16px; }
            .visualizer { width: 100px; height: 100px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>PersonaPlex</h1>
        <div class="brand-tagline">Simplified &amp; one-click install by SurAiverse</div>
        <div class="brand-subtag">Based on NVIDIA PersonaPlex 7B</div>
    </div>
    
    <div class="main">
        <div class="chat-container">
            <div class="status-strip">
                <div class="status-row">
                    <span>Session</span>
                    <span id="progressLabel">Ready</span>
                </div>
                <div class="progress-track">
                    <div class="progress-bar" id="progressBar" style="width: 20%;"></div>
                </div>
                <div class="progress-steps">
                    <span id="stepReady" class="active">Ready</span>
                    <span id="stepConnecting">Connecting</span>
                    <span id="stepLive">Live</span>
                    <span id="stepComplete">Complete</span>
                </div>
            </div>
            <!-- Setup View (Homepage) -->
            <div class="setup-view" id="setupView">
                <div class="form-section">
                    <div class="form-section-title">Text Prompt</div>
                    <div class="presets-container">
                        <span class="presets-label">Examples:</span>
                        <div class="presets">
                            <button class="preset-btn" onclick="setPreset('assistant')">Assistant (default)</button>
                            <button class="preset-btn" onclick="setPreset('medical')">Medical office (service)</button>
                            <button class="preset-btn" onclick="setPreset('bank')">Bank (service)</button>
                            <button class="preset-btn" onclick="setPreset('astronaut')">Astronaut (fun)</button>
                        </div>
                    </div>
                    <div class="form-group">
                        <textarea id="textPrompt" maxlength="2000" placeholder="Enter your text prompt...">You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.</textarea>
                        <div class="char-count"><span id="charCount">0</span>/2000</div>
                    </div>
                </div>
                
                <div class="form-section">
                    <div class="form-section-title">Vision Prompt</div>
                    <div class="form-group">
                        <textarea id="visionPrompt" maxlength="1000" placeholder="Prompt sent to the vision model alongside each captured frame.">You are an observer. Describe exactly what is happening in this scene in one short sentence. Keep it brief and factual. You have memory of prior frames in this session; use them to track movement and changes.</textarea>
                        <div class="char-count"><span id="visionCharCount">0</span>/1000</div>
                    </div>
                </div>

                <div class="form-section">
                    <div class="form-section-title">Voice</div>
                    <div class="form-group">
                        <select id="voicePrompt">
                            <option value="NATF0.pt">NATURAL_F0</option>
                            <option value="NATF1.pt">NATURAL_F1</option>
                            <option value="NATF2.pt">NATURAL_F2</option>
                            <option value="NATF3.pt">NATURAL_F3</option>
                            <option value="NATM0.pt">NATURAL_M0</option>
                            <option value="NATM1.pt">NATURAL_M1</option>
                            <option value="NATM2.pt">NATURAL_M2</option>
                            <option value="NATM3.pt">NATURAL_M3</option>
                            <option value="VARF0.pt">VARIETY_F0</option>
                            <option value="VARF1.pt">VARIETY_F1</option>
                            <option value="VARF2.pt">VARIETY_F2</option>
                            <option value="VARF3.pt">VARIETY_F3</option>
                            <option value="VARF4.pt">VARIETY_F4</option>
                            <option value="VARM0.pt">VARIETY_M0</option>
                            <option value="VARM1.pt">VARIETY_M1</option>
                            <option value="VARM2.pt">VARIETY_M2</option>
                            <option value="VARM3.pt">VARIETY_M3</option>
                            <option value="VARM4.pt">VARIETY_M4</option>
                        </select>
                    </div>
                    <div class="upload-row">
                        <button type="button" class="upload-toggle-btn" id="uploadToggle" onclick="toggleUploadArea()">
                            <span class="arrow">&#9656;</span>Clone a voice (upload 10-30s of clean audio)
                        </button>
                        <div class="upload-area" id="uploadArea">
                            <div class="hint">
                                Mono or stereo, any common format (wav, mp3, flac, ogg, m4a, opus). 10-30s of one
                                clear speaker works best. Uploaded audio is normalized and fed through Mimi as a
                                voice prefix - the model continues in that timbre. Not zero-shot perfect, but
                                recognizable.
                            </div>
                            <input type="file" id="voiceUploadInput" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.opus,.aac">
                            <div class="upload-status" id="uploadStatus"></div>
                            <button type="button" class="upload-clear" id="uploadClear" onclick="clearUploadedVoice()">
                                Remove (use preset above instead)
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="form-section">
                    <div class="advanced-toggle" id="advancedToggle" onclick="toggleAdvanced()">
                        <div class="form-section-title" style="margin-bottom: 0;">Sampling &amp; Repetition</div>
                        <span class="arrow">&#9656;</span>
                    </div>
                    <div class="advanced-body" id="advancedBody">
                        <div class="slider-group-title">Text sampling</div>
                        <div class="slider-row">
                            <div class="slider-label">
                                <span>Text temperature</span>
                                <span class="slider-value" id="textTempValue">0.70</span>
                            </div>
                            <input type="range" id="textTempSlider" min="0.1" max="1.5" step="0.05" value="0.7">
                            <div class="slider-hint">Higher = more varied word choice. Lower = more focused.</div>
                        </div>
                        <div class="slider-row">
                            <div class="slider-label">
                                <span>Text top-k</span>
                                <span class="slider-value" id="textTopkValue">25</span>
                            </div>
                            <input type="range" id="textTopkSlider" min="1" max="500" step="1" value="25">
                            <div class="slider-hint">Number of word candidates considered each step.</div>
                        </div>
                        <div class="slider-group-title">Audio sampling</div>
                        <div class="slider-row">
                            <div class="slider-label">
                                <span>Audio temperature</span>
                                <span class="slider-value" id="audioTempValue">0.70</span>
                            </div>
                            <input type="range" id="audioTempSlider" min="0.1" max="1.5" step="0.05" value="0.7">
                            <div class="slider-hint">Higher = more expressive prosody. Lower = flatter delivery.</div>
                        </div>
                        <div class="slider-row">
                            <div class="slider-label">
                                <span>Audio top-k</span>
                                <span class="slider-value" id="audioTopkValue">250</span>
                            </div>
                            <input type="range" id="audioTopkSlider" min="1" max="2048" step="1" value="250">
                            <div class="slider-hint">Number of audio token candidates considered each step.</div>
                        </div>
                        <div class="slider-group-title">Behavior</div>
                        <div class="slider-row">
                            <div class="slider-label">
                                <span>Repetition penalty</span>
                                <span class="slider-value" id="repPenaltyValue">1.20</span>
                            </div>
                            <input type="range" id="repPenaltySlider" min="1.0" max="2.0" step="0.05" value="1.2">
                            <div class="slider-hint">1.0 = off. 1.1-1.3 = gentle. 1.5+ = aggressive. Stops the model from looping.</div>
                        </div>
                        <div class="slider-row">
                            <div class="slider-label">
                                <span>Repetition context window</span>
                                <span class="slider-value" id="repContextValue">64</span>
                            </div>
                            <input type="range" id="repContextSlider" min="0" max="256" step="8" value="64">
                            <div class="slider-hint">How many recent text tokens the penalty considers.</div>
                        </div>
                        <div class="slider-row">
                            <div class="slider-label">
                                <span>Padding bonus</span>
                                <span class="slider-value" id="padBonusValue">0.0</span>
                            </div>
                            <input type="range" id="padBonusSlider" min="0" max="6" step="0.1" value="0">
                            <div class="slider-hint">Biases the model toward silence tokens. 0 = off. 2-4 stops rambling by making it yield the turn sooner.</div>
                        </div>
                        <div class="slider-row">
                            <div class="slider-label">
                                <span>Max turn length (tokens)</span>
                                <span class="slider-value" id="maxTurnValue">0</span>
                            </div>
                            <input type="range" id="maxTurnSlider" min="0" max="2000" step="50" value="0">
                            <div class="slider-hint">Hard cap: after N consecutive non-silence text tokens, force pad for ~1 s. 0 = off. 500 ≈ 40 s sustained talk. Safety net under padding_bonus.</div>
                        </div>
                        <div class="slider-group-title">Microphone input</div>
                        <div class="toggle-row">
                            <label><input type="checkbox" id="echoCancelToggle" checked> Echo cancellation</label>
                        </div>
                        <div class="toggle-row-hint">Keep on. Off = speaker bleed reaches the model and can start a feedback loop.</div>
                        <div class="toggle-row">
                            <label><input type="checkbox" id="noiseSuppToggle" checked> Noise suppression</label>
                        </div>
                        <div class="toggle-row-hint">On suppresses keyboard / fan / room hiss before the model hears it.</div>
                        <div class="toggle-row">
                            <label><input type="checkbox" id="autoGainToggle"> Auto gain control</label>
                        </div>
                        <div class="toggle-row-hint">Off by default. Browser AGC can cause amplitude swings that confuse Moshi at 24 kHz.</div>
                        <div class="slider-group-title">Vision</div>
                        <div class="toggle-row">
                            <label><input type="checkbox" id="visionInTranscriptToggle"> Echo vision context in transcript</label>
                        </div>
                        <div class="toggle-row-hint">Adds Gemini's scene descriptions inline in the AI response with a [vision] prefix. Useful for debugging whether vision context is shaping replies.</div>
                        <div class="slider-group-title">Reproducibility</div>
                        <div class="seed-row">
                            <div class="slider-label">
                                <span>Random seed</span>
                                <label class="seed-toggle">
                                    <input type="checkbox" id="seedRandomToggle" checked>
                                    <span>Use random</span>
                                </label>
                            </div>
                            <input type="number" id="seedInput" min="0" max="2147483647" step="1" value="42" disabled>
                            <div class="slider-hint">Set a fixed seed to reproduce a take. Uncheck "Use random" to enable.</div>
                        </div>
                        <div class="slider-actions">
                            <button class="slider-reset" type="button" onclick="resetAdvanced()">Reset to defaults</button>
                        </div>
                    </div>
                </div>

                <div class="error-msg" id="errorMsg"></div>

                <div class="btn-container">
                    <button class="btn btn-primary" id="connectBtn" onclick="startConversation()">
                        <svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/>
                        </svg>
                        Connect
                    </button>
                </div>
            </div>
            
            <!-- Conversation View -->
            <div class="conversation-view" id="conversationView">
                <div style="text-align: center;">
                    <div class="status-badge">
                        <span class="status-dot disconnected" id="statusDot"></span>
                        <span id="statusText">Disconnected</span>
                    </div>
                </div>
            
            <div class="error-msg" id="convErrorMsg"></div>
            
            <div class="vision-container" id="visionContainer">
                <video class="vision-video" id="visionVideo" autoplay playsinline muted></video>
                <div class="vision-label" id="visionLabel">Vision Off</div>
                <div class="vision-status" id="visionStatus">Idle</div>
                <div class="vision-caption" id="visionCaption"></div>
            </div>
            <div class="vision-meta" id="visionMeta">
                <span id="visionCostDisplay">0 frames · ~$0.0000</span>
                <span>·</span>
                <label>
                    Fallback every
                    <select id="visionIntervalSelect">
                        <option value="1000">1 s</option>
                        <option value="3000">3 s</option>
                        <option value="5000" selected>5 s</option>
                        <option value="10000">10 s</option>
                        <option value="15000">15 s</option>
                        <option value="30000">30 s</option>
                    </select>
                </label>
            </div>
            <div class="captions-log" id="captionsLog">
                <div class="captions-log-title">Vision history</div>
                <div id="captionsLogEntries"></div>
            </div>
            <div class="notice-toast" id="noticeToast"></div>
            
            <div class="visualizer-container">
                <div class="visualizer ai" id="aiVisualizer">
                    <canvas class="visualizer-canvas" id="aiCanvas"></canvas>
                    <span class="visualizer-label">AI</span>
                </div>
                <div class="visualizer user" id="userVisualizer">
                    <canvas class="visualizer-canvas" id="userCanvas"></canvas>
                    <span class="visualizer-label">You</span>
                </div>
            </div>
            
            <div class="transcript">
                <div class="transcript-label">AI Response</div>
                <div class="transcript-text" id="transcript">Speak into your microphone...</div>
            </div>
            
            <div class="controls">
                <button class="btn btn-danger" id="stopBtn" onclick="stopConversation()">
                    Disconnect
                </button>
                <button class="btn btn-vision" id="visionBtn" onclick="toggleVision()" title="Send screen/camera context to the model">
                    Add Vision
                </button>
                <button class="btn btn-vision" id="captureNowBtn" onclick="forceCapture()" title="Force a high-detail frame send right now" style="display:none;">
                    Capture Now
                </button>
                <button class="btn btn-rewind" id="visionPauseBtn" onclick="toggleVisionPause()" title="Pause automatic frame capture without releasing the stream" style="display:none;">
                    Pause Vision
                </button>
                <button class="btn btn-rewind" id="rewindBtn" onclick="sendRewind()" title="Un-stick the model if it starts looping">
                    Rewind
                </button>
                <button class="btn btn-primary" id="newConvBtn" onclick="newConversation()" style="display:none;">
                    New Conversation
                </button>
            </div>
            <div class="download-row" id="downloadRow">
                <div>
                    <div class="download-title">Session Complete</div>
                    <div class="download-sub">Download your conversation audio</div>
                </div>
                <a class="btn btn-primary" id="downloadLink" download="personaplex_conversation.webm">Download Audio</a>
            </div>
            <audio id="aiAudio" autoplay playsinline style="display:none;"></audio>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Created by <a href="https://www.youtube.com/@suraiverse" target="_blank">Suresh Pydikondala (SurAiverse)</a> | 
           <a href="https://huggingface.co/nvidia/personaplex-7b-v1" target="_blank">NVIDIA PersonaPlex</a></p>
    </div>

    <script>
        // Text prompt presets
        const PRESETS = {
            assistant: "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
            medical: "You work for Dr. Jones's medical office, and you are receiving calls to record information for new patients. Information: Record full name, date of birth, any medication allergies, tobacco smoking history, alcohol consumption history, and any prior medical conditions. Assure the patient that this information will be confidential, if they ask.",
            bank: "You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity. The transaction was flagged due to unusual location (transaction attempted in Miami, FL; customer normally transacts in Seattle, WA).",
            astronaut: "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor."
        };

        // STUN-only fallback if /api/rtc/ice-servers fails. Will not work
        // through the RunPod HTTPS proxy; only useful on a LAN dev box.
        const ICE_SERVERS_FALLBACK = [
            { urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] },
        ];

        async function fetchIceServers() {
            const res = await fetch('/api/rtc/ice-servers', { method: 'GET' });
            if (res.status === 503) {
                // Server has TURN configured but cannot mint creds. Don't
                // silently fall back to STUN. That would leave the user
                // staring at a stuck connect for ~30 s before iceConnectionState
                // times out, with no actionable signal.
                let detail = 'TURN unavailable on the server. Connections behind NAT will fail.';
                try {
                    const data = await res.json();
                    if (data && data.detail) detail = data.detail;
                } catch (e) {}
                throw new Error(detail);
            }
            if (!res.ok) {
                console.warn('ice-servers fetch failed, falling back to STUN: HTTP ' + res.status);
                return ICE_SERVERS_FALLBACK;
            }
            try {
                const data = await res.json();
                if (Array.isArray(data.iceServers) && data.iceServers.length > 0) {
                    return data.iceServers;
                }
            } catch (err) {
                console.warn('ice-servers fetch parse failed, falling back to STUN:', err);
            }
            return ICE_SERVERS_FALLBACK;
        }
        const CONNECT_BTN_HTML = '<svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Connect';

        // Peer connection + track refs.
        let pc = null;
        let controlChannel = null;
        let micStream = null;       // local MediaStream from getUserMedia
        let aiStream = null;        // remote MediaStream from pc.ontrack
        let isReady = false;        // server has signalled 'ready'
        let connectT0 = 0;          // performance.now() at Connect click
        let connectTimings = null;  // per-phase elapsed ms, logged at 'ready'
        let sessionId = null;       // server-issued session id for trickle ICE
        let candidateStream = null; // EventSource for server-trickled candidates
        let pendingCandidates = []; // local candidates gathered before sessionId arrived
        
        // Vision / Rewind refs and state.
        let visionStream = null;
        let visionInterval = null;
        let visionBtn = null;
        let visionVideo = null;
        let visionContainer = null;
        let visionLabel = null;
        let visionPaused = false;
        let visionEnabledFromServer = true;  // assumed until server says otherwise
        let visionFramesSent = 0;
        let visionFrameIntervalMs = 5000;  // fallback only; server drives most frames
        let visionLastSentAt = 0;
        let visionStatusTickTimer = null;
        let visionInjecting = false;
        let lastRewindClickAt = 0;

        // Per-call cost estimate for Gemini 3.5 Flash with our payload
        // shape (~500 input tokens including transcript context, 50
        // output tokens, thinking minimal). Pricing reference May 2026.
        const VISION_PER_CALL_USD = 0.0012;
        function markConnect(name) {
            if (!connectTimings) return;
            connectTimings[name] = Math.round(performance.now() - connectT0);
        }
        async function postCandidate(cand) {
            if (!sessionId) return;
            try {
                await fetch('/api/rtc/candidate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        candidate: cand ? cand.candidate : null,
                        sdpMid: cand ? cand.sdpMid : null,
                        sdpMLineIndex: cand ? cand.sdpMLineIndex : null,
                    }),
                });
            } catch (err) {
                console.warn('candidate POST failed:', err);
            }
        }
        async function flushPendingCandidates() {
            const buf = pendingCandidates;
            pendingCandidates = [];
            for (const c of buf) await postCandidate(c);
        }

        // AudioContext is used only to host AnalyserNodes for the visualizer
        // and to mux mic + AI streams into a single MediaRecorder destination
        // for the optional session download. It does not touch the realtime
        // audio path; WebRTC owns capture and playback.
        let audioContext = null;
        let aiSourceNode = null;
        let userSourceNode = null;
        let aiAnalyser = null;
        let userAnalyser = null;
        let recordingDestination = null;
        let mediaRecorder = null;
        let recordedChunks = [];
        let shouldShowDownload = false;
        let visualizerRAF = null;
        
        // View elements
        const setupView = document.getElementById('setupView');
        const conversationView = document.getElementById('conversationView');
        const textPromptInput = document.getElementById('textPrompt');
        const visionPromptInput = document.getElementById('visionPrompt');
        const visionCharCount = document.getElementById('visionCharCount');

        // Persist text and vision prompts across sessions. Restore first,
        // then bind input handlers that write back to localStorage.
        try {
            const savedTextPrompt = localStorage.getItem('pp_textPrompt');
            if (savedTextPrompt !== null) textPromptInput.value = savedTextPrompt;
            const savedVisionPrompt = localStorage.getItem('pp_visionPrompt');
            if (savedVisionPrompt !== null && visionPromptInput) {
                visionPromptInput.value = savedVisionPrompt;
            }
        } catch (e) {}
        textPromptInput.addEventListener('input', () => {
            try { localStorage.setItem('pp_textPrompt', textPromptInput.value); } catch (e) {}
        });
        if (visionPromptInput) {
            visionPromptInput.addEventListener('input', () => {
                try { localStorage.setItem('pp_visionPrompt', visionPromptInput.value); } catch (e) {}
            });
        }
        const voicePromptSelect = document.getElementById('voicePrompt');
        const charCount = document.getElementById('charCount');
        const connectBtn = document.getElementById('connectBtn');
        const errorMsg = document.getElementById('errorMsg');
        const downloadRow = document.getElementById('downloadRow');
        const downloadLink = document.getElementById('downloadLink');
        const progressBar = document.getElementById('progressBar');
        const progressLabel = document.getElementById('progressLabel');
        const stepReady = document.getElementById('stepReady');
        const stepConnecting = document.getElementById('stepConnecting');
        const stepLive = document.getElementById('stepLive');
        const stepComplete = document.getElementById('stepComplete');
        
        // Conversation view elements
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const stopBtn = document.getElementById('stopBtn');
        const newConvBtn = document.getElementById('newConvBtn');
        const transcript = document.getElementById('transcript');
        const convErrorMsg = document.getElementById('convErrorMsg');
        const aiVisualizer = document.getElementById('aiVisualizer');
        const userVisualizer = document.getElementById('userVisualizer');
        
        // Initialize character count
        function updateCharCount() {
            charCount.textContent = textPromptInput.value.length;
            if (visionCharCount) visionCharCount.textContent = visionPromptInput.value.length;
        }
        textPromptInput.addEventListener('input', updateCharCount);
        if (visionPromptInput) visionPromptInput.addEventListener('input', updateCharCount);
        updateCharCount();

        // Advanced sampling sliders
        const ADVANCED_DEFAULTS = {
            textTemp: 0.7, textTopk: 25,
            audioTemp: 0.7, audioTopk: 250,
            repPenalty: 1.2, repContext: 64,
            padBonus: 0.0,
            maxTurn: 0,
        };
        const advancedToggle = document.getElementById('advancedToggle');
        const advancedBody = document.getElementById('advancedBody');
        const textTempSlider = document.getElementById('textTempSlider');
        const textTempValue = document.getElementById('textTempValue');
        const textTopkSlider = document.getElementById('textTopkSlider');
        const textTopkValue = document.getElementById('textTopkValue');
        const audioTempSlider = document.getElementById('audioTempSlider');
        const audioTempValue = document.getElementById('audioTempValue');
        const audioTopkSlider = document.getElementById('audioTopkSlider');
        const audioTopkValue = document.getElementById('audioTopkValue');
        const repPenaltySlider = document.getElementById('repPenaltySlider');
        const repPenaltyValue = document.getElementById('repPenaltyValue');
        const repContextSlider = document.getElementById('repContextSlider');
        const repContextValue = document.getElementById('repContextValue');
        const padBonusSlider = document.getElementById('padBonusSlider');
        const padBonusValue = document.getElementById('padBonusValue');
        const maxTurnSlider = document.getElementById('maxTurnSlider');
        const maxTurnValue = document.getElementById('maxTurnValue');
        const seedRandomToggle = document.getElementById('seedRandomToggle');
        const seedInput = document.getElementById('seedInput');
        const echoCancelToggle = document.getElementById('echoCancelToggle');
        const noiseSuppToggle = document.getElementById('noiseSuppToggle');
        const autoGainToggle = document.getElementById('autoGainToggle');
        const visionInTranscriptToggle = document.getElementById('visionInTranscriptToggle');

        const MIC_DEFAULTS = { echoCancel: true, noiseSupp: true, autoGain: false };
        try {
            const e = localStorage.getItem('pp_echoCancel');
            if (e !== null) echoCancelToggle.checked = e === '1';
            const n = localStorage.getItem('pp_noiseSupp');
            if (n !== null) noiseSuppToggle.checked = n === '1';
            const g = localStorage.getItem('pp_autoGain');
            if (g !== null) autoGainToggle.checked = g === '1';
            const vt = localStorage.getItem('pp_visionInTranscript');
            if (vt !== null && visionInTranscriptToggle) visionInTranscriptToggle.checked = vt === '1';
        } catch (e) {}
        if (visionInTranscriptToggle) {
            visionInTranscriptToggle.addEventListener('change', () => {
                try { localStorage.setItem('pp_visionInTranscript', visionInTranscriptToggle.checked ? '1' : '0'); } catch (e) {}
            });
        }

        function getMicConstraints() {
            return {
                echoCancellation: echoCancelToggle.checked,
                noiseSuppression: noiseSuppToggle.checked,
                autoGainControl: autoGainToggle.checked,
            };
        }

        function applyMicConstraintsLive() {
            // Live-apply to the running capture. Browser decides whether it
            // can honor the change without reconnecting the track; if not,
            // the new setting takes effect on the next getUserMedia (next
            // session).
            if (!micStream) return;
            const track = micStream.getAudioTracks()[0];
            if (!track) return;
            track.applyConstraints(getMicConstraints()).catch((err) => {
                console.warn('applyConstraints failed (takes effect next session):', err);
            });
        }

        [
            [echoCancelToggle, 'pp_echoCancel'],
            [noiseSuppToggle, 'pp_noiseSupp'],
            [autoGainToggle, 'pp_autoGain'],
        ].forEach(([tog, key]) => {
            tog.addEventListener('change', () => {
                try { localStorage.setItem(key, tog.checked ? '1' : '0'); } catch (e) {}
                applyMicConstraintsLive();
            });
        });

        function bindSlider(slider, label, decimals) {
            const update = () => {
                const v = parseFloat(slider.value);
                label.textContent = decimals > 0 ? v.toFixed(decimals) : String(v | 0);
                try { localStorage.setItem('pp_' + slider.id, slider.value); } catch (e) {}
            };
            slider.addEventListener('input', update);
            try {
                const saved = localStorage.getItem('pp_' + slider.id);
                if (saved !== null) slider.value = saved;
            } catch (e) {}
            update();
        }
        bindSlider(textTempSlider, textTempValue, 2);
        bindSlider(textTopkSlider, textTopkValue, 0);
        bindSlider(audioTempSlider, audioTempValue, 2);
        bindSlider(audioTopkSlider, audioTopkValue, 0);
        bindSlider(repPenaltySlider, repPenaltyValue, 2);
        bindSlider(repContextSlider, repContextValue, 0);
        bindSlider(padBonusSlider, padBonusValue, 1);
        bindSlider(maxTurnSlider, maxTurnValue, 0);

        // Seed control: persisted to localStorage. When "Use random" is checked, no seed
        // query param is sent; the server picks one. Otherwise the value in seedInput is used.
        function syncSeedDisabled() {
            seedInput.disabled = seedRandomToggle.checked;
        }
        try {
            const savedRandom = localStorage.getItem('pp_seedRandom');
            if (savedRandom !== null) seedRandomToggle.checked = savedRandom === '1';
            const savedSeed = localStorage.getItem('pp_seedValue');
            if (savedSeed !== null) seedInput.value = savedSeed;
        } catch (e) {}
        syncSeedDisabled();
        seedRandomToggle.addEventListener('change', () => {
            syncSeedDisabled();
            try { localStorage.setItem('pp_seedRandom', seedRandomToggle.checked ? '1' : '0'); } catch (e) {}
        });
        seedInput.addEventListener('input', () => {
            try { localStorage.setItem('pp_seedValue', seedInput.value); } catch (e) {}
        });

        function toggleAdvanced() {
            advancedToggle.classList.toggle('open');
            advancedBody.classList.toggle('open');
            try { localStorage.setItem('pp_advancedOpen', advancedBody.classList.contains('open') ? '1' : '0'); } catch (e) {}
        }
        try {
            if (localStorage.getItem('pp_advancedOpen') === '1') {
                advancedToggle.classList.add('open');
                advancedBody.classList.add('open');
            }
        } catch (e) {}

        function resetAdvanced() {
            textTempSlider.value = ADVANCED_DEFAULTS.textTemp;
            textTopkSlider.value = ADVANCED_DEFAULTS.textTopk;
            audioTempSlider.value = ADVANCED_DEFAULTS.audioTemp;
            audioTopkSlider.value = ADVANCED_DEFAULTS.audioTopk;
            repPenaltySlider.value = ADVANCED_DEFAULTS.repPenalty;
            repContextSlider.value = ADVANCED_DEFAULTS.repContext;
            padBonusSlider.value = ADVANCED_DEFAULTS.padBonus;
            maxTurnSlider.value = ADVANCED_DEFAULTS.maxTurn;
            [textTempSlider, textTopkSlider, audioTempSlider, audioTopkSlider, repPenaltySlider, repContextSlider, padBonusSlider, maxTurnSlider]
                .forEach(s => s.dispatchEvent(new Event('input')));
            echoCancelToggle.checked = MIC_DEFAULTS.echoCancel;
            noiseSuppToggle.checked = MIC_DEFAULTS.noiseSupp;
            autoGainToggle.checked = MIC_DEFAULTS.autoGain;
            [echoCancelToggle, noiseSuppToggle, autoGainToggle]
                .forEach(t => t.dispatchEvent(new Event('change')));
            seedRandomToggle.checked = true;
            seedInput.value = '42';
            seedRandomToggle.dispatchEvent(new Event('change'));
            seedInput.dispatchEvent(new Event('input'));
        }
        
        // Set preset text
        function setPreset(presetName) {
            if (PRESETS[presetName]) {
                textPromptInput.value = PRESETS[presetName];
                updateCharCount();
            }
        }

        // Voice upload (clone)
        let uploadedVoiceFilename = null;
        const uploadToggleBtn = document.getElementById('uploadToggle');
        const uploadArea = document.getElementById('uploadArea');
        const voiceUploadInput = document.getElementById('voiceUploadInput');
        const uploadStatus = document.getElementById('uploadStatus');
        const uploadClearBtn = document.getElementById('uploadClear');

        function toggleUploadArea() {
            uploadArea.classList.toggle('open');
            uploadToggleBtn.classList.toggle('open');
        }

        function setUploadStatus(text, kind) {
            uploadStatus.textContent = text || '';
            uploadStatus.className = 'upload-status' + (kind ? ' ' + kind : '');
        }

        async function uploadVoiceFile(file) {
            if (!file) return;
            // 20 MB cap matches server.
            if (file.size > 20 * 1024 * 1024) {
                setUploadStatus('File too large (max 20 MB)', 'error');
                return;
            }
            setUploadStatus('Uploading ' + file.name + '...', 'uploading');
            uploadClearBtn.classList.remove('visible');
            try {
                const form = new FormData();
                form.append('file', file);
                const res = await fetch('/api/voice-upload', { method: 'POST', body: form });
                let json = null;
                try { json = await res.json(); } catch (e) { json = null; }
                if (!res.ok) {
                    const msg = (json && json.error) || ('upload failed (' + res.status + ')');
                    throw new Error(msg);
                }
                if (!json || !json.filename) {
                    throw new Error('server returned no filename');
                }
                uploadedVoiceFilename = json.filename;
                setUploadStatus('Using uploaded voice: ' + file.name, 'success');
                uploadClearBtn.classList.add('visible');
                voicePromptSelect.disabled = true;
            } catch (err) {
                uploadedVoiceFilename = null;
                setUploadStatus('Upload failed: ' + (err.message || err), 'error');
                uploadClearBtn.classList.remove('visible');
                voicePromptSelect.disabled = false;
            }
        }

        voiceUploadInput.addEventListener('change', (ev) => {
            const file = ev.target.files && ev.target.files[0];
            if (file) uploadVoiceFile(file);
        });

        function clearUploadedVoice() {
            uploadedVoiceFilename = null;
            voiceUploadInput.value = '';
            setUploadStatus('', '');
            uploadClearBtn.classList.remove('visible');
            voicePromptSelect.disabled = false;
        }
        
        function showSetupView() {
            setupView.classList.remove('hidden');
            conversationView.classList.remove('active');
        }
        
        function showConversationView() {
            setupView.classList.add('hidden');
            conversationView.classList.add('active');
        }

        function setProgress(value, label, complete = false) {
            progressBar.style.width = value + '%';
            progressLabel.textContent = label;
            stepReady.classList.add('active');
            stepConnecting.classList.toggle('active', value >= 60);
            stepLive.classList.toggle('active', value >= 100 && !complete);
            stepComplete.classList.toggle('active', complete);
        }
        
        function setStatus(status, text) {
            statusDot.className = 'status-dot ' + status;
            statusText.textContent = text;
            if (status === 'connecting') {
                setProgress(60, 'Connecting');
            } else if (status === 'connected') {
                setProgress(100, 'Live');
            } else {
                setProgress(20, 'Ready');
            }
        }
        
        function showError(msg, inConversation = false) {
            const el = inConversation ? convErrorMsg : errorMsg;
            el.textContent = msg;
            el.style.display = 'block';
            setTimeout(() => { el.style.display = 'none'; }, 8000);
        }
        
        // ============================================================
        // Audio context. Used only to host AnalyserNodes for the visualizers
        // and to mux mic + AI streams into one MediaStream for MediaRecorder.
        // The realtime audio path lives entirely on the WebRTC peer
        // connection: getUserMedia -> pc.addTrack on the way out, and
        // pc.ontrack -> <audio>.srcObject on the way back.
        // ============================================================

        const aiAudioElement = document.getElementById('aiAudio');

        async function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                // Device switch (headphones unplugged, default output
                // changes) suspends the context. Auto-resume keeps the
                // visualizer + recording graph alive across system events.
                audioContext.addEventListener('statechange', () => {
                    if (audioContext.state === 'suspended' || audioContext.state === 'interrupted') {
                        audioContext.resume().catch(() => {});
                    }
                });
            }
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
        }

        function attachAudioGraph() {
            // Wires the AI remote stream and the local mic stream into
            // analysers (for visualizers) and a MediaStream destination
            // (for MediaRecorder). Idempotent.
            if (!audioContext) return;
            if (!recordingDestination) {
                recordingDestination = audioContext.createMediaStreamDestination();
            }
            if (aiStream && !aiSourceNode) {
                aiSourceNode = audioContext.createMediaStreamSource(aiStream);
                aiAnalyser = audioContext.createAnalyser();
                aiAnalyser.fftSize = 256;
                aiAnalyser.smoothingTimeConstant = 0.85;
                aiSourceNode.connect(aiAnalyser);
                aiSourceNode.connect(recordingDestination);
            }
            if (micStream && !userSourceNode) {
                userSourceNode = audioContext.createMediaStreamSource(micStream);
                userAnalyser = audioContext.createAnalyser();
                userAnalyser.fftSize = 256;
                userAnalyser.smoothingTimeConstant = 0.85;
                userSourceNode.connect(userAnalyser);
                userSourceNode.connect(recordingDestination);
            }
        }

        function startSessionRecording() {
            shouldShowDownload = false;
            recordedChunks = [];
            downloadRow.style.display = 'none';
            if (!recordingDestination) return;
            try {
                mediaRecorder = new MediaRecorder(recordingDestination.stream);
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data && event.data.size > 0) recordedChunks.push(event.data);
                };
                mediaRecorder.onstop = function () {
                    // Use `this` (the MediaRecorder) instead of the closure
                    // variable: cleanup() nulls `mediaRecorder` before this
                    // async handler fires, which otherwise crashes on
                    // `mediaRecorder.mimeType` after a fast disconnect.
                    if (!shouldShowDownload || recordedChunks.length === 0) return;
                    const blob = new Blob(recordedChunks, { type: this.mimeType || 'audio/webm' });
                    const url = URL.createObjectURL(blob);
                    downloadLink.href = url;
                    downloadRow.style.display = 'flex';
                };
                mediaRecorder.start();
            } catch (err) {
                console.warn('Session recording unavailable:', err);
            }
        }

        function stopSessionRecording(showDownload = null) {
            if (showDownload !== null) shouldShowDownload = showDownload;
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                try { mediaRecorder.stop(); } catch (err) {}
            }
        }

        // ============================================================
        // Visualizer
        // ============================================================
        const aiCanvas = document.getElementById('aiCanvas');
        const userCanvas = document.getElementById('userCanvas');
        const VIZ_AI_COLOR = '#00a8cc';
        const VIZ_USER_COLOR = '#76b900';
        let vizBuffer = null;

        function fitCanvas(canvas) {
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            const w = Math.max(1, Math.floor(rect.width * dpr));
            const h = Math.max(1, Math.floor(rect.height * dpr));
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }
            return dpr;
        }

        function drawVisualizer(canvas, analyser, color, isLive) {
            const ctx = canvas.getContext('2d');
            fitCanvas(canvas);
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            const cx = w / 2;
            const cy = h / 2;
            const maxR = Math.min(w, h) * 0.46;
            let intensity = 0;
            if (analyser && isLive) {
                if (!vizBuffer || vizBuffer.length !== analyser.frequencyBinCount) {
                    vizBuffer = new Uint8Array(analyser.frequencyBinCount);
                }
                analyser.getByteFrequencyData(vizBuffer);
                let sumSq = 0;
                for (let i = 0; i < vizBuffer.length; i++) sumSq += vizBuffer[i] * vizBuffer[i];
                intensity = Math.min(1, Math.sqrt(sumSq / vizBuffer.length) / 110);
            }
            const baseR = maxR * 0.35;
            const r = baseR + (maxR - baseR) * intensity;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.85;
            ctx.fill();
            ctx.globalAlpha = 1;
            if (isLive) {
                ctx.beginPath();
                ctx.arc(cx, cy, maxR * 0.18, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
            }
        }

        function isRtcLive() {
            return !!(pc && (pc.connectionState === 'connected' || pc.connectionState === 'connecting'));
        }

        function startVisualizers() {
            stopVisualizers();
            const tick = () => {
                const live = isRtcLive() && isReady;
                drawVisualizer(aiCanvas, aiAnalyser, VIZ_AI_COLOR, live);
                drawVisualizer(userCanvas, userAnalyser, VIZ_USER_COLOR, live);
                visualizerRAF = requestAnimationFrame(tick);
            };
            visualizerRAF = requestAnimationFrame(tick);
        }

        function stopVisualizers() {
            if (visualizerRAF != null) {
                cancelAnimationFrame(visualizerRAF);
                visualizerRAF = null;
            }
        }

        // ============================================================
        // WebRTC connection
        // ============================================================

        function buildConfigPayload() {
            const voiceParam = uploadedVoiceFilename || voicePromptSelect.value || '';
            const useSeed = !seedRandomToggle.checked && seedInput.value !== '';
            return {
                voice_prompt: voiceParam,
                text_prompt: textPromptInput.value || '',
                vision_prompt: (visionPromptInput && visionPromptInput.value) || '',
                vision_in_transcript: !!(visionInTranscriptToggle && visionInTranscriptToggle.checked),
                audio_temperature: parseFloat(audioTempSlider.value),
                text_temperature: parseFloat(textTempSlider.value),
                text_topk: parseInt(textTopkSlider.value, 10),
                audio_topk: parseInt(audioTopkSlider.value, 10),
                repetition_penalty: parseFloat(repPenaltySlider.value),
                repetition_penalty_context: parseInt(repContextSlider.value, 10),
                padding_bonus: parseFloat(padBonusSlider.value),
                max_turn_text_tokens: parseInt(maxTurnSlider.value, 10),
                seed: useSeed ? parseInt(seedInput.value, 10) : -1,
            };
        }

        function claimMediaSession() {
            // Hint the browser to treat this like an active media session.
            // Helps reduce background-tab throttling on the audio element.
            if (!('mediaSession' in navigator)) return;
            try {
                navigator.mediaSession.metadata = new MediaMetadata({
                    title: 'PersonaPlex Conversation',
                    artist: 'PersonaPlex',
                });
                navigator.mediaSession.playbackState = 'playing';
            } catch (err) {
                // Some Firefox builds reject MediaMetadata; non-fatal.
            }
        }

        function releaseMediaSession() {
            if (!('mediaSession' in navigator)) return;
            try { navigator.mediaSession.playbackState = 'none'; } catch (err) {}
        }

        function handleControlMessage(msg) {
            if (msg.type === 'ready') {
                isReady = true;
                markConnect('ready');
                if (connectTimings) {
                    console.groupCollapsed('connect timings (ms from Connect click)');
                    Object.entries(connectTimings).forEach(([k, v]) => console.log(k.padEnd(22), v));
                    console.groupEnd();
                }
                console.log('Server ready');
                setStatus('connected', 'Connected - Speak now!');
                stopBtn.disabled = false;
                transcript.textContent = '';
                claimMediaSession();
                attachAudioGraph();
                startSessionRecording();
                startVisualizers();
            } else if (msg.type === 'text') {
                transcript.textContent += msg.v || '';
                transcript.scrollTop = transcript.scrollHeight;
            } else if (msg.type === 'vision_caption') {
                showVisionCaption(msg.text || '');
                addCaptionToLog(msg.text || '');
                const el = document.getElementById('visionStatus');
                if (el) {
                    el.textContent = 'Response received';
                    setTimeout(updateVisionStatus, 1500);
                }
            } else if (msg.type === 'vision_status') {
                visionEnabledFromServer = !!msg.enabled;
                if (!visionEnabledFromServer) {
                    const btn = document.getElementById('visionBtn');
                    if (btn) {
                        btn.disabled = true;
                        btn.title = 'Vision unavailable: server has no GEMINI_API_KEY';
                    }
                }
            } else if (msg.type === 'request_vision_frame') {
                // Server is asking for a fresh frame (model just went
                // quiet). The model is bored; honor the request even if
                // the scene barely changed. Motion-gate would otherwise
                // suppress every server-driven capture for static views.
                if (visionStream && !visionPaused) captureFrame(false, true);
            } else if (msg.type === 'vision_inject') {
                // Server has begun (or finished) drip-feeding a Gemini
                // description into the model's text channel. Surface it
                // so the user knows why audio briefly stops.
                visionInjecting = !!msg.active;
                updateVisionStatus();
            } else if (msg.type === 'notice') {
                showNoticeToast(msg.text || '');
            } else if (msg.type === 'error') {
                console.warn('server error:', msg.reason);
                showError('Server error: ' + (msg.reason || 'unknown'), true);
                cleanup();
            } else if (msg.type === 'end') {
                setStatus('disconnected', 'Disconnected');
                cleanup();
            } else {
                console.warn('unknown control message:', msg);
            }
        }

        async function startConversation() {
            try {
                connectBtn.disabled = true;
                connectBtn.textContent = 'Connecting...';
                downloadRow.style.display = 'none';
                downloadLink.removeAttribute('href');
                isReady = false;
                showConversationView();
                setStatus('connecting', 'Setting up microphone...');
                transcript.textContent = '';

                // Per-phase timing for diagnosing slow session starts.
                // Logged to the console as a grouped breakdown when the
                // server signals 'ready'.
                connectT0 = performance.now();
                connectTimings = {};
                sessionId = null;
                pendingCandidates = [];
                if (candidateStream) {
                    try { candidateStream.close(); } catch (e) {}
                    candidateStream = null;
                }

                // Mic permission + capture. With browser defaults (AEC, NS,
                // AGC governed by the toggles) this is the only audio
                // capture in the pipeline.
                micStream = await navigator.mediaDevices.getUserMedia({
                    audio: getMicConstraints(),
                });
                markConnect('getUserMedia');

                setStatus('connecting', 'Fetching network config...');
                const iceServers = await fetchIceServers();
                markConnect('fetchIceServers');
                // iceCandidatePoolSize triggers candidate gathering as
                // soon as the PeerConnection is created, overlapping it
                // with track add and createOffer. Without this the
                // gather only starts AFTER setLocalDescription, which
                // serializes ~1-3 s of TURN allocation before signaling.
                pc = new RTCPeerConnection({ iceServers, iceCandidatePoolSize: 1 });

                pc.ontrack = (event) => {
                    aiStream = event.streams && event.streams[0]
                        ? event.streams[0]
                        : new MediaStream([event.track]);
                    aiAudioElement.srcObject = aiStream;
                    const playPromise = aiAudioElement.play();
                    if (playPromise && playPromise.catch) {
                        playPromise.catch((err) => {
                            console.warn('AI audio autoplay blocked:', err);
                        });
                    }
                    // Wire the analyser/recording graph if 'ready' has
                    // already fired and we missed it earlier.
                    if (audioContext) attachAudioGraph();
                };

                pc.onconnectionstatechange = () => {
                    console.log('pc state:', pc && pc.connectionState);
                    if (!pc) return;
                    // 'disconnected' is transient per spec; ICE may
                    // recover. Only 'failed' and 'closed' are terminal.
                    if (pc.connectionState === 'failed') {
                        showError('Connection failed. Network or NAT may be blocking media.', true);
                        cleanup();
                    } else if (pc.connectionState === 'closed') {
                        if (isReady) setStatus('disconnected', 'Disconnected');
                        cleanup();
                    } else if (pc.connectionState === 'disconnected') {
                        // Show a soft-warning status; do not tear down.
                        setStatus('connecting', 'Reconnecting...');
                    }
                };

                pc.oniceconnectionstatechange = () => {
                    if (!pc || isReady) return;
                    const s = pc.iceConnectionState;
                    console.log('ice state:', s);
                    if (s === 'checking') {
                        setStatus('connecting', 'Connecting peers...');
                    } else if (s === 'connected' || s === 'completed') {
                        markConnect('iceConnected');
                        // Brief: we are about to open the DataChannel.
                        setStatus('connecting', 'Securing channel...');
                    } else if (s === 'failed') {
                        showError('ICE failed: could not establish a media path. TURN may be unreachable.', true);
                        cleanup();
                    }
                };

                // Data channel must be created BEFORE createOffer to appear
                // in the SDP. The server side wires its handler on
                // pc.on('datachannel') by label.
                controlChannel = pc.createDataChannel('control');
                controlChannel.onopen = () => {
                    markConnect('dataChannelOpen');
                    const cfg = buildConfigPayload();
                    controlChannel.send(JSON.stringify({ type: 'config', ...cfg }));
                    setStatus('connecting', 'Loading AI model (this may take a moment)...');
                };
                controlChannel.onmessage = (e) => {
                    if (typeof e.data !== 'string') return;
                    let msg;
                    try { msg = JSON.parse(e.data); }
                    catch (err) { console.warn('bad control JSON:', err); return; }
                    handleControlMessage(msg);
                };

                // Trickle ICE: ship each local candidate to the server as it
                // is gathered, instead of waiting for full gathering before
                // signaling. Candidates that arrive before sessionId is set
                // are buffered and flushed once the offer response lands.
                pc.onicecandidate = (e) => {
                    if (sessionId) postCandidate(e.candidate);
                    else pendingCandidates.push(e.candidate);
                };

                // Add the mic track. Pass the stream so the remote side sees
                // it as part of one MediaStream group (cleaner ontrack
                // semantics on the server, though aiortc tolerates either).
                micStream.getAudioTracks().forEach((track) => {
                    pc.addTrack(track, micStream);
                });

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                markConnect('setLocalDescription');

                setStatus('connecting', 'Negotiating session...');
                const res = await fetch('/api/rtc/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type,
                    }),
                });
                if (res.status === 409) {
                    throw new Error('Another session is already active. Please wait for it to end.');
                }
                if (!res.ok) {
                    let detail = '';
                    try { detail = (await res.json()).error || ''; } catch (_) {}
                    throw new Error('Server returned ' + res.status + (detail ? (': ' + detail) : ''));
                }
                const answer = await res.json();
                markConnect('serverAnswer');
                sessionId = answer.session_id || null;
                await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
                markConnect('setRemoteDescription');

                // Open the server->client candidate stream and flush any
                // local candidates that the browser already gathered while
                // we were waiting for the answer.
                if (sessionId) {
                    candidateStream = new EventSource(
                        '/api/rtc/candidates?session_id=' + encodeURIComponent(sessionId)
                    );
                    candidateStream.onmessage = (e) => {
                        try {
                            const cand = JSON.parse(e.data);
                            pc.addIceCandidate(cand).catch((err) => {
                                console.warn('addIceCandidate failed:', err);
                            });
                        } catch (err) {
                            console.warn('bad candidate JSON:', err);
                        }
                    };
                    candidateStream.addEventListener('done', () => {
                        if (candidateStream) {
                            candidateStream.close();
                            candidateStream = null;
                        }
                    });
                    candidateStream.onerror = () => {
                        // Browser closes EventSource on server hangup; that
                        // is the normal path after gathering completes. The
                        // 'done' event handler nulls candidateStream first,
                        // so this onerror only fires for real network errors.
                        if (candidateStream) {
                            candidateStream.close();
                            candidateStream = null;
                        }
                    };
                    flushPendingCandidates();
                }

                // Init the AudioContext now (after the user-gesture-driven
                // Connect click) so the analyser graph is ready when the
                // server signals 'ready'.
                await initAudioContext();
            } catch (err) {
                console.error('Error:', err);
                if (err.name === 'NotAllowedError') {
                    showError('Microphone access denied. Please allow microphone access and try again.');
                } else {
                    showError(err.message || 'Failed to start conversation');
                }
                cleanup();
                connectBtn.disabled = false;
                connectBtn.innerHTML = CONNECT_BTN_HTML;
                showSetupView();
            }
        }

        function stopConversation() {
            stopSessionRecording(true);
            cleanup();
            setStatus('disconnected', 'Disconnected');
            setProgress(100, 'Complete', true);
            transcript.textContent += '\\n\\n[Conversation ended]';
            stopBtn.style.display = 'none';
            newConvBtn.style.display = 'inline-flex';
        }

        function newConversation() {
            showSetupView();
            connectBtn.disabled = false;
            connectBtn.innerHTML = CONNECT_BTN_HTML;
            stopBtn.style.display = 'inline-flex';
            newConvBtn.style.display = 'none';
            setProgress(20, 'Ready');
            downloadRow.style.display = 'none';
            downloadLink.removeAttribute('href');
        }

        function cleanup() {
            stopSessionRecording(null);
            isReady = false;
            if (candidateStream) {
                try { candidateStream.close(); } catch (e) {}
                candidateStream = null;
            }
            sessionId = null;
            pendingCandidates = [];
            if (controlChannel) {
                try { controlChannel.close(); } catch (e) {}
                controlChannel = null;
            }
            if (pc) {
                try { pc.ontrack = null; } catch (e) {}
                try { pc.onconnectionstatechange = null; } catch (e) {}
                try { pc.onicecandidate = null; } catch (e) {}
                try { pc.close(); } catch (e) {}
                pc = null;
            }
            if (aiAudioElement) {
                try { aiAudioElement.pause(); } catch (e) {}
                try { aiAudioElement.srcObject = null; } catch (e) {}
            }
            if (aiSourceNode) {
                try { aiSourceNode.disconnect(); } catch (e) {}
                aiSourceNode = null;
            }
            if (userSourceNode) {
                try { userSourceNode.disconnect(); } catch (e) {}
                userSourceNode = null;
            }
            if (aiAnalyser) {
                try { aiAnalyser.disconnect(); } catch (e) {}
                aiAnalyser = null;
            }
            if (userAnalyser) {
                try { userAnalyser.disconnect(); } catch (e) {}
                userAnalyser = null;
            }
            if (recordingDestination) {
                try { recordingDestination.disconnect(); } catch (e) {}
                recordingDestination = null;
            }
            mediaRecorder = null;
            aiStream = null;
            if (micStream) {
                try { micStream.getTracks().forEach((t) => t.stop()); } catch (e) {}
                micStream = null;
            }
            connectBtn.disabled = false;
            connectBtn.innerHTML = CONNECT_BTN_HTML;
            stopVisualizers();
            stopVision();
            releaseMediaSession();
        }

        async function toggleVision() {
            if (visionStream) {
                stopVision();
                return;
            }
            if (!visionEnabledFromServer) {
                showNoticeToast('Vision unavailable: server has no GEMINI_API_KEY');
                return;
            }
            try {
                // Two paths: a virtual-camera source via getUserMedia, or
                // native screen sharing via getDisplayMedia. The OK/Cancel
                // dialog lets the user pick without us having to enumerate
                // devices and guess what's plugged in.
                const useWebcam = confirm("Click 'OK' for Webcam / Virtual Camera, or 'Cancel' for Native Screen Sharing.");

                if (useWebcam) {
                    visionStream = await navigator.mediaDevices.getUserMedia({ video: true });
                } else {
                    visionStream = await navigator.mediaDevices.getDisplayMedia({ video: true });
                }

                visionVideo = document.getElementById('visionVideo');
                visionContainer = document.getElementById('visionContainer');
                visionLabel = document.getElementById('visionLabel');
                visionBtn = document.getElementById('visionBtn');

                visionVideo.srcObject = visionStream;
                visionContainer.classList.add('active');
                visionBtn.classList.add('active');
                visionBtn.textContent = 'Stop Vision';
                visionLabel.textContent = 'Vision Active';

                // Reveal the manual Capture Now and Pause buttons, plus
                // the per-session meta row (cost meter + interval select).
                const captureBtn = document.getElementById('captureNowBtn');
                if (captureBtn) captureBtn.style.display = '';
                const pauseBtn = document.getElementById('visionPauseBtn');
                if (pauseBtn) {
                    pauseBtn.style.display = '';
                    pauseBtn.textContent = 'Pause Vision';
                }
                visionPaused = false;
                visionInjecting = false;
                const meta = document.getElementById('visionMeta');
                if (meta) meta.classList.add('visible');
                // Reveal the captions history panel up-front with a
                // placeholder so users see *something* before the first
                // Gemini response lands (which can take several seconds).
                const log = document.getElementById('captionsLog');
                const entries = document.getElementById('captionsLogEntries');
                if (log && entries) {
                    log.classList.add('visible');
                    if (entries.children.length === 0) {
                        const placeholder = document.createElement('div');
                        placeholder.className = 'captions-log-entry placeholder';
                        placeholder.innerHTML = '<span class="text" style="color: #9a8a6a; font-style: italic;"></span>';
                        placeholder.querySelector('.text').textContent = 'Awaiting first scene description...';
                        entries.appendChild(placeholder);
                    }
                }
                visionFramesSent = 0;
                updateVisionCost();

                // Most frames come from server-side requests; this is the
                // fallback in case the server is silent for too long.
                visionInterval = setInterval(() => {
                    if (!visionPaused) captureFrame(false);
                }, visionFrameIntervalMs);
                // 1 Hz refresh of the "last X s ago" status pill so the
                // user sees a live counter, not a frozen value.
                if (visionStatusTickTimer) clearInterval(visionStatusTickTimer);
                visionStatusTickTimer = setInterval(updateVisionStatus, 1000);
                updateVisionStatus();
            } catch (err) {
                console.error('Vision access denied:', err);
                showError('Could not start vision: ' + err.message);
            }
        }

        function stopVision() {
            if (visionInterval) {
                clearInterval(visionInterval);
                visionInterval = null;
            }
            if (visionStream) {
                visionStream.getTracks().forEach(t => t.stop());
                visionStream = null;
            }
            // Drop the motion-gate baseline so the next session starts
            // from scratch. Without this, the first auto-capture after
            // a reconnect is compared against the previous session's
            // last frame and gets dropped when the scene looks similar.
            visionLastFrameData = null;
            if (visionContainer) visionContainer.classList.remove('active');
            if (visionBtn) {
                visionBtn.classList.remove('active');
                visionBtn.textContent = 'Add Vision';
            }
            if (visionLabel) visionLabel.textContent = 'Vision Off';
            const captureBtn = document.getElementById('captureNowBtn');
            if (captureBtn) captureBtn.style.display = 'none';
            const pauseBtn = document.getElementById('visionPauseBtn');
            if (pauseBtn) pauseBtn.style.display = 'none';
            const meta = document.getElementById('visionMeta');
            if (meta) meta.classList.remove('visible');
            const log = document.getElementById('captionsLog');
            const entries = document.getElementById('captionsLogEntries');
            if (log) log.classList.remove('visible');
            if (entries) entries.innerHTML = '';
            visionPaused = false;
            visionInjecting = false;
            if (visionStatusTickTimer) {
                clearInterval(visionStatusTickTimer);
                visionStatusTickTimer = null;
            }
            visionLastSentAt = 0;
        }

        // Track the last sent frame so we can skip ones that haven't
        // changed enough to warrant a vision-model call (motion-gating).
        let visionLastFrameData = null;
        const VISION_MOTION_THRESHOLD = 0.04; // mean abs delta on 0..1 scale

        // Two-tier capture settings.
        //   Automatic frames: /2 downscale, JPEG 0.55. Keeps the payload
        //     small for cheap Gemini calls.
        //   Detail frames (user-requested via Capture Now): native
        //     resolution, JPEG 0.8. For when text or fine detail needs
        //     to be readable.
        // `force` bypasses the motion gate. Server-driven captures and
        // Capture Now both set it; the fallback interval does not, so
        // an idle observer doesn't burn Gemini calls on a frozen scene.
        async function captureFrame(detail, force) {
            if (!visionStream || !controlChannel || controlChannel.readyState !== 'open') return;

            const divisor = detail ? 1 : 2;
            const quality = detail ? 0.8 : 0.55;

            const canvas = document.createElement('canvas');
            canvas.width = Math.max(160, Math.floor(visionVideo.videoWidth / divisor));
            canvas.height = Math.max(90, Math.floor(visionVideo.videoHeight / divisor));
            const ctx = canvas.getContext('2d');
            ctx.drawImage(visionVideo, 0, 0, canvas.width, canvas.height);

            // Motion gate: subsampled mean abs pixel delta. Skipped when
            // `detail` or `force` is set (user/server explicitly asked).
            if (!detail && !force) {
                const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
                if (visionLastFrameData && visionLastFrameData.length === frame.data.length) {
                    let diff = 0;
                    for (let i = 0; i < frame.data.length; i += 16) {
                        diff += Math.abs(frame.data[i] - visionLastFrameData[i]);
                    }
                    const meanDelta = diff / (frame.data.length / 16) / 255;
                    if (meanDelta < VISION_MOTION_THRESHOLD) {
                        console.debug('vision: motion gate suppressed frame (delta=' + meanDelta.toFixed(4) + ')');
                        return;
                    }
                }
                visionLastFrameData = new Uint8ClampedArray(frame.data);
            }

            const dataUrl = canvas.toDataURL('image/jpeg', quality);
            const base64 = dataUrl.split(',')[1];

            controlChannel.send(JSON.stringify({
                type: 'vision_frame',
                data: base64,
                detail: !!detail,
            }));
            visionFramesSent += 1;
            visionLastSentAt = performance.now();
            updateVisionCost();
            updateVisionStatus();
        }

        // Live status pill at the bottom of the vision preview. Replaces
        // the static "Idle" so users moving around with no audio still see
        // confirmation that frames are flowing.
        function updateVisionStatus() {
            const el = document.getElementById('visionStatus');
            if (!el) return;
            if (visionInjecting) { el.textContent = 'Injecting context...'; return; }
            if (visionPaused) { el.textContent = 'Paused'; return; }
            if (!visionLastSentAt) { el.textContent = 'Idle'; return; }
            const age = Math.max(0, Math.round((performance.now() - visionLastSentAt) / 1000));
            el.textContent = visionFramesSent + ' frame' + (visionFramesSent === 1 ? '' : 's') +
                             ' · last ' + age + ' s ago';
        }

        function forceCapture() {
            // Manual trigger: bypass motion gate and pause state, send a
            // high-detail frame.
            captureFrame(true);
        }

        // Bind the fallback-interval selector. Server-driven cadence
        // does most of the work; this only fires when the server has
        // been silent for too long.
        (function bindVisionInterval() {
            const sel = document.getElementById('visionIntervalSelect');
            if (!sel) return;
            try {
                const saved = localStorage.getItem('pp_visionIntervalMs');
                if (saved) {
                    sel.value = saved;
                    visionFrameIntervalMs = parseInt(saved, 10) || visionFrameIntervalMs;
                }
            } catch (e) {}
            sel.addEventListener('change', () => {
                visionFrameIntervalMs = parseInt(sel.value, 10) || 15000;
                try { localStorage.setItem('pp_visionIntervalMs', String(visionFrameIntervalMs)); } catch (e) {}
                // Restart the timer at the new cadence if vision is active.
                if (visionInterval) {
                    clearInterval(visionInterval);
                    visionInterval = setInterval(() => {
                        if (!visionPaused) captureFrame(false);
                    }, visionFrameIntervalMs);
                }
            });
        })();

        // Fade-in a fresh vision caption, fade-out after a few seconds.
        let visionCaptionTimer = null;
        function showVisionCaption(text) {
            const el = document.getElementById('visionCaption');
            if (!el) return;
            el.textContent = text;
            el.classList.add('visible');
            if (visionCaptionTimer) clearTimeout(visionCaptionTimer);
            visionCaptionTimer = setTimeout(() => {
                el.classList.remove('visible');
            }, 8000);
        }

        // Rolling history of recent captions. Last 10 entries visible.
        function addCaptionToLog(text) {
            if (!text) return;
            const entries = document.getElementById('captionsLogEntries');
            const log = document.getElementById('captionsLog');
            if (!entries || !log) return;
            // Drop the "Awaiting first scene description..." placeholder
            // that was seeded in toggleVision so users had something to
            // look at before the first Gemini response arrived.
            const first = entries.firstChild;
            if (first && first.classList && first.classList.contains('placeholder')) {
                entries.removeChild(first);
            }
            const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            const div = document.createElement('div');
            div.className = 'captions-log-entry';
            div.innerHTML = '<span class="ts"></span><span class="text"></span>';
            div.querySelector('.ts').textContent = ts;
            div.querySelector('.text').textContent = text;
            entries.prepend(div);
            while (entries.children.length > 10) {
                entries.removeChild(entries.lastChild);
            }
            log.classList.add('visible');
        }

        // Toast notification for transient server-side notices
        // (auto-rewind, vision unavailable, etc.).
        let noticeToastTimer = null;
        function showNoticeToast(text) {
            const el = document.getElementById('noticeToast');
            if (!el || !text) return;
            el.textContent = text;
            el.classList.add('visible');
            if (noticeToastTimer) clearTimeout(noticeToastTimer);
            noticeToastTimer = setTimeout(() => {
                el.classList.remove('visible');
            }, 4500);
        }

        function updateVisionCost() {
            const el = document.getElementById('visionCostDisplay');
            if (!el) return;
            const total = visionFramesSent * VISION_PER_CALL_USD;
            el.textContent = visionFramesSent + ' frames · ~$' + total.toFixed(4);
        }

        function toggleVisionPause() {
            visionPaused = !visionPaused;
            const btn = document.getElementById('visionPauseBtn');
            if (btn) btn.textContent = visionPaused ? 'Resume Vision' : 'Pause Vision';
            updateVisionStatus();
        }

        function sendRewind() {
            // 1 s debounce so a frustrated click-storm can't saturate
            // _infer_lock on the server (each rewind acquires the lock
            // to apply set_streaming_state_inplace).
            const now = performance.now();
            if (now - lastRewindClickAt < 1000) return;
            lastRewindClickAt = now;
            if (controlChannel && controlChannel.readyState === 'open') {
                controlChannel.send(JSON.stringify({ type: 'rewind' }));
                // Outcome arrives as a notice toast (success or "no snapshot
                // yet"). No optimistic transcript scribble; it lied when the
                // rewind didn't actually fire.
            }
        }

        // Handle page unload
        window.addEventListener('beforeunload', cleanup);
    </script>
</body>
</html>"""
            return web.Response(text=html, content_type='text/html')
        
        logger.info("Serving embedded web client (no build required)")
        app.router.add_get("/", handle_embedded_client)
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")

    # Cloudflare's TURN returns a bare 401 on CHANNEL_BIND that aioice
    # cannot retry, leaving "Task exception was never retrieved" stack
    # traces in the log even though the WebRTC connection succeeds via
    # plain Send-Indication. Filter THAT specific symptom out at the
    # asyncio exception handler. Other aioice failures (network outage,
    # DNS failure, real TURN auth issues, malformed STUN responses)
    # must keep surfacing or operators have no diagnostic when TURN is
    # genuinely broken.
    async def _install_aioice_noise_filter(_app):
        def _handler(loop, context):
            exc = context.get("exception")
            if isinstance(exc, Exception):
                cls = type(exc)
                mod = cls.__module__ or ""
                if (
                    mod.startswith("aioice.")
                    and cls.__name__ == "TransactionFailed"
                ):
                    msg = str(exc)
                    if (
                        "CHANNEL_BIND" in msg
                        or "401" in msg
                        or "Unauthorized" in msg
                    ):
                        return
            loop.default_exception_handler(context)
        asyncio.get_event_loop().set_exception_handler(_handler)
    app.on_startup.append(_install_aioice_noise_filter)

    async def _close_http_session(_app):
        """Close the lazily-created Gemini HTTP client on shutdown so
        aiohttp doesn't emit ResourceWarning at process exit."""
        sess = getattr(state, "_http_session", None)
        if sess is not None and not sess.closed:
            try:
                await sess.close()
            except Exception as exc:
                logger.warning(
                    "closing gemini http session raised: %s: %s",
                    type(exc).__name__,
                    exc,
                )
    app.on_cleanup.append(_close_http_session)

    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
