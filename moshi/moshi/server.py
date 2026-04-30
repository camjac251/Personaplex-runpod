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
from dataclasses import dataclass
import json
import random
import os
from pathlib import Path
import tarfile
import secrets
import sys
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

from .models import loaders, MimiModel, LMModel, LMGen
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


@dataclass
class ServerState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 uploads_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.uploads_dir = uploads_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        self.lock = asyncio.Lock()
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
            await loop.run_in_executor(None, sphn.read, str(out_path))
        except Exception as e:
            out_path.unlink(missing_ok=True)
            return web.json_response({"error": f"could not decode audio: {e}"}, status=400)

        logger.info(f"voice upload saved: {safe_name} ({total} bytes, original={original!r})")
        return web.json_response({"filename": f"{UPLOAD_PREFIX}{safe_name}", "bytes": total})

    @torch.no_grad()
    def _process_audio_frame(self, chunk_np):
        """Run GPU inference for one audio frame. Called from thread executor
        so the asyncio event loop stays responsive during GPU work."""
        chunk = torch.from_numpy(chunk_np).to(device=self.device)[None, None]
        codes = self.mimi.encode(chunk)
        results = []
        for c in range(codes.shape[-1]):
            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
            if tokens is None:
                continue
            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
            main_pcm = self.mimi.decode(tokens[:, 1:9])
            main_pcm = main_pcm.cpu()
            pcm_np = main_pcm[0, 0].numpy()
            text_token = tokens[0, 0, 0].item()
            text = None
            if text_token not in (0, 3):
                _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                text = _text.replace("▁", " ")
            results.append((pcm_np, text))
        return results

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

    async def _fetch_ice_servers(self) -> list[dict]:
        """Return iceServers config for the current session.

        With ``TURN_KEY_ID`` and ``TURN_KEY_API_TOKEN`` set, mints a fresh
        24-hour credential pack from Cloudflare Realtime and caches it for
        12 hours. Otherwise returns the STUN-only fallback, which only
        works when both peers can reach each other directly over UDP
        (i.e. on LAN; not through RunPod's HTTPS proxy).
        """
        if not (self._turn_key_id and self._turn_api_token):
            return [dict(s) for s in DEFAULT_STUN_FALLBACK]

        async with self._ice_cache_lock:
            now = time.monotonic()
            if self._ice_cache is not None and now < self._ice_cache_expires_at:
                return self._ice_cache

            ttl_seconds = 86400  # Cloudflare's documented max.
            url = (
                "https://rtc.live.cloudflare.com/v1/turn/keys/"
                f"{self._turn_key_id}/credentials/generate"
            )
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
                            return [dict(s) for s in DEFAULT_STUN_FALLBACK]
                        try:
                            data = json.loads(body_text)
                        except ValueError as exc:
                            logger.warning(
                                f"Cloudflare TURN creds fetch returned non-JSON: {exc}"
                            )
                            return [dict(s) for s in DEFAULT_STUN_FALLBACK]
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning(f"Cloudflare TURN creds fetch error: {exc}")
                return [dict(s) for s in DEFAULT_STUN_FALLBACK]

            servers = data.get("iceServers")
            if isinstance(servers, dict):
                # Cloudflare currently returns a single object; spec also
                # allows an array. Accept both.
                servers = [servers]
            if not isinstance(servers, list) or not servers:
                logger.warning(
                    "Cloudflare returned no iceServers; falling back to STUN"
                )
                return [dict(s) for s in DEFAULT_STUN_FALLBACK]

            self._ice_cache = servers
            # Refresh halfway through the TTL so we never serve creds that
            # are about to expire mid-session.
            self._ice_cache_expires_at = now + ttl_seconds / 2
            logger.info(
                f"Cloudflare TURN creds minted (ttl={ttl_seconds}s, "
                f"refresh at +{ttl_seconds // 2}s)"
            )
            return servers

    async def handle_ice_servers(self, _request):
        servers = await self._fetch_ice_servers()
        return web.json_response({"iceServers": servers})

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

            ice_servers = await self._fetch_ice_servers()
            session = RTCSession(
                frame_size=self.frame_size,
                process_fn=self._process_audio_frame,
                log=clog.log,
                ice_servers=ice_servers,
            )
            session.set_config_handler(on_config)

            try:
                answer = await session.negotiate(offer)
            except Exception as exc:
                clog.log("error", f"negotiate failed: {type(exc).__name__}: {exc}")
                await session.close()
                self.lock.release()
                owns_lock = False
                return web.json_response(
                    {"error": f"negotiate failed: {exc}"}, status=500
                )

            # Spawn the long-running session runner. It owns the lock from
            # this point on. Strong-ref the task so the event loop's weak
            # set cannot garbage-collect it.
            task = asyncio.create_task(
                self._run_rtc_session(session, config_event, config_holder, clog)
            )
            self._session_tasks.add(task)
            task.add_done_callback(self._session_tasks.discard)
            owns_lock = False  # ownership transferred to the runner
            return web.json_response({"sdp": answer.sdp, "type": answer.type})
        except BaseException:
            # Anything from a torn transport (peer_port lookup) to
            # RTCPeerConnection construction failures, including
            # asyncio.CancelledError if the client drops the request.
            if session is not None:
                try:
                    await session.close()
                except Exception:
                    pass
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
                if voice_prompt_path.endswith(".pt"):
                    self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
                else:
                    self.lm_gen.load_voice_prompt(voice_prompt_path)
                clog.log("info", f"loaded voice prompt: {voice_prompt_path} (requested: {requested})")

            self.lm_gen.text_prompt_tokens = (
                self.text_tokenizer.encode(wrap_with_system_tags(cfg.text_prompt))
                if cfg.text_prompt else None
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

            async def is_alive():
                return session.is_alive()

            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "system prompts done")

            if not session.is_alive():
                clog.log("info", "client disconnected during warmup")
                return

            session.send_ready()
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
            try:
                await session.close()
            finally:
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


def _get_static_path(static: Optional[str], hf_repo: str) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        # Get HF_TOKEN from environment or cache
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            try:
                from huggingface_hub.utils import HfFolder
                hf_token = HfFolder.get_token()
            except Exception:
                pass
        
        # Try to download dist.tgz from HuggingFace
        try:
            dist_tgz = hf_hub_download(hf_repo, "dist.tgz", token=hf_token)
            dist_tgz = Path(dist_tgz)
            dist = dist_tgz.parent / "dist"
            if not dist.exists():
                with tarfile.open(dist_tgz, "r:gz") as tar:
                    tar.extractall(path=dist_tgz.parent)
            return str(dist)
        except Exception as e:
            logger.warning(f"Could not download static content from HuggingFace: {e}")
            # Try to find local client/dist folder
            script_dir = Path(__file__).parent.parent.parent
            local_dist = script_dir / "client" / "dist"
            if local_dist.exists():
                logger.info(f"Using local client dist: {local_dist}")
                return str(local_dist)
            logger.warning("No static content available. Web UI will not be served.")
            logger.warning("To build the client, run: cd client && npm install && npm run build")
            return None
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


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

    static_path: None | str = _get_static_path(args.static, args.hf_repo)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

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
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME, token=hf_token)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME, token=hf_token)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
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
    state = ServerState(
        mimi=mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        uploads_dir=args.uploads_dir,
        save_voice_prompt_embeddings=False,
    )
    logger.info("warming up the model")
    state.warmup()
    app = web.Application(client_max_size=UPLOAD_MAX_BYTES + 1024 * 1024)
    app.router.add_post("/api/rtc/offer", state.handle_rtc_offer)
    app.router.add_get("/api/rtc/ice-servers", state.handle_ice_servers)
    app.router.add_post("/api/voice-upload", state.handle_voice_upload)
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

    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/bundle.min.js"></script>
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
            try {
                const res = await fetch('/api/rtc/ice-servers', { method: 'GET' });
                if (!res.ok) throw new Error('HTTP ' + res.status);
                const data = await res.json();
                if (Array.isArray(data.iceServers) && data.iceServers.length > 0) {
                    return data.iceServers;
                }
            } catch (err) {
                console.warn('ice-servers fetch failed, falling back to STUN:', err);
            }
            return ICE_SERVERS_FALLBACK;
        }
        const CONNECT_BTN_HTML = '<svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Connect';

        // Peer connection + track refs.
        let pc = null;
        let controlChannel = null;
        let micStream = null;       // local MediaStream from getUserMedia
        let aiStream = null;        // remote MediaStream from pc.ontrack
        let micVad = null;
        let userSpeaking = false;
        let isReady = false;        // server has signalled 'ready'

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
        }
        textPromptInput.addEventListener('input', updateCharCount);
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

        const MIC_DEFAULTS = { echoCancel: true, noiseSupp: true, autoGain: false };
        try {
            const e = localStorage.getItem('pp_echoCancel');
            if (e !== null) echoCancelToggle.checked = e === '1';
            const n = localStorage.getItem('pp_noiseSupp');
            if (n !== null) noiseSuppToggle.checked = n === '1';
            const g = localStorage.getItem('pp_autoGain');
            if (g !== null) autoGainToggle.checked = g === '1';
        } catch (e) {}

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
                mediaRecorder.onstop = () => {
                    if (!shouldShowDownload || recordedChunks.length === 0) return;
                    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
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

        function waitForIceComplete(p) {
            if (p.iceGatheringState === 'complete') return Promise.resolve();
            return new Promise((resolve) => {
                const check = () => {
                    if (p.iceGatheringState === 'complete') {
                        p.removeEventListener('icegatheringstatechange', check);
                        resolve();
                    }
                };
                p.addEventListener('icegatheringstatechange', check);
            });
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
                console.log('Server ready');
                setStatus('connected', 'Connected - Speak now!');
                stopBtn.disabled = false;
                transcript.textContent = '';
                claimMediaSession();
                attachAudioGraph();
                startSessionRecording();
                startVisualizers();
                initVADForUI();
            } else if (msg.type === 'text') {
                transcript.textContent += msg.v || '';
                transcript.scrollTop = transcript.scrollHeight;
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

                // Mic permission + capture. With browser defaults (AEC, NS,
                // AGC governed by the toggles) this is the only audio
                // capture in the pipeline.
                micStream = await navigator.mediaDevices.getUserMedia({
                    audio: getMicConstraints(),
                });

                showConversationView();
                setStatus('connecting', 'Negotiating...');
                transcript.textContent = 'Connecting to server...';

                const iceServers = await fetchIceServers();
                pc = new RTCPeerConnection({ iceServers });

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

                // Data channel must be created BEFORE createOffer to appear
                // in the SDP. The server side wires its handler on
                // pc.on('datachannel') by label.
                controlChannel = pc.createDataChannel('control');
                controlChannel.onopen = () => {
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

                // Add the mic track. Pass the stream so the remote side sees
                // it as part of one MediaStream group (cleaner ontrack
                // semantics on the server, though aiortc tolerates either).
                micStream.getAudioTracks().forEach((track) => {
                    pc.addTrack(track, micStream);
                });

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                await waitForIceComplete(pc);

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
                await pc.setRemoteDescription(answer);

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

        async function initVADForUI() {
            // Client-side VAD only drives UI feedback now (the visualizer's
            // per-track 'isLive' check is binary on/off; userSpeaking can
            // gate richer state in the future). Server-side echo handling
            // lives in the browser AEC stack, which the toggles control.
            if (!micStream || !window.vad || !window.vad.MicVAD) return;
            try {
                micVad = await window.vad.MicVAD.new({
                    stream: micStream,
                    onSpeechStart: () => { userSpeaking = true; },
                    onSpeechEnd: () => { userSpeaking = false; },
                    onVADMisfire: () => { userSpeaking = false; },
                    positiveSpeechThreshold: 0.75,
                    minSpeechFrames: 6,
                });
                micVad.start();
            } catch (err) {
                console.warn('VAD init failed:', err);
                micVad = null;
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
            userSpeaking = false;
            if (micVad) {
                try { micVad.pause(); } catch (e) {}
                try { micVad.destroy && micVad.destroy(); } catch (e) {}
                micVad = null;
            }
            if (controlChannel) {
                try { controlChannel.close(); } catch (e) {}
                controlChannel = null;
            }
            if (pc) {
                try { pc.ontrack = null; } catch (e) {}
                try { pc.onconnectionstatechange = null; } catch (e) {}
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
            releaseMediaSession();
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
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
