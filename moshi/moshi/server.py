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
import random
import os
from pathlib import Path
import tarfile
import secrets
import sys
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

from .models import loaders, MimiModel, LMModel, LMGen
from .models.lm import MAX_REPETITION_CONTEXT
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

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        def _qfloat(name, default):
            raw = request.query.get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                clog.log("warning", f"bad float for {name}={raw!r}, using default {default}")
                return default

        def _qint(name, default):
            raw = request.query.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                clog.log("warning", f"bad int for {name}={raw!r}, using default {default}")
                return default

        # Parse sampling params here, but apply them inside the session lock below
        # to avoid races between concurrent connections mutating shared lm_gen state.
        sampling_params = {
            "temp": _qfloat("audio_temperature", 0.7),
            "temp_text": _qfloat("text_temperature", 0.7),
            "top_k_text": max(1, _qint("text_topk", 25)),
            "top_k": max(1, _qint("audio_topk", 250)),
            "repetition_penalty": max(1.0, _qfloat("repetition_penalty", 1.2)),
            "repetition_penalty_context": max(
                0, min(_qint("repetition_penalty_context", 64), MAX_REPETITION_CONTEXT)
            ),
            "padding_bonus": max(0.0, _qfloat("padding_bonus", 0.0)),
            "max_turn_text_tokens": max(0, _qint("max_turn_text_tokens", 0)),
        }
        
        # Construct full voice prompt path. Two sources:
        #   - "upload:<name>" -> uploads_dir/<name> (user-provided wav/mp3/etc.)
        #   - "<name>"        -> voice_prompt_dir/<name> (preset .pt or audio)
        requested_voice_prompt_path = None
        voice_prompt_path = None
        voice_prompt_filename = request.query.get("voice_prompt", "") or ""
        if voice_prompt_filename.startswith(UPLOAD_PREFIX):
            upload_name = voice_prompt_filename[len(UPLOAD_PREFIX):]
            requested_voice_prompt_path = self._resolve_upload_path(upload_name)
            if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Uploaded voice prompt '{upload_name}' not found"
                )
            voice_prompt_path = requested_voice_prompt_path
        elif self.voice_prompt_dir is not None and voice_prompt_filename:
            requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            if not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            voice_prompt_path = requested_voice_prompt_path

        if voice_prompt_path is not None and self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(request.query["text_prompt"])) if len(request.query["text_prompt"]) > 0 else None
        seed = _qint("seed", -1) if "seed" in request.query else None

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio, raw float32 PCM at mimi.sample_rate
                        payload = message[1:]
                        if len(payload) == 0 or len(payload) % 4 != 0:
                            clog.log("warning", f"bad audio payload length {len(payload)}")
                            continue
                        try:
                            pcm = np.frombuffer(payload, dtype=np.float32)
                        except Exception as e:
                            clog.log("warning", f"pcm decode failed: {type(e).__name__}: {e}")
                            continue
                        try:
                            pcm_queue.put_nowait(pcm)
                        except asyncio.QueueFull:
                            # GPU is falling behind. Drop the newest chunk so
                            # the existing timeline survives rather than
                            # sliding forward with stale audio.
                            clog.log("warning", f"pcm queue full ({pcm_queue.qsize()}), dropping chunk")
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            except Exception as e:
                clog.log("error", f"recv_loop: {type(e).__name__}: {e}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def process_loop():
            loop = asyncio.get_event_loop()
            all_pcm_data = None

            try:
                while not close:
                    try:
                        pcm = await asyncio.wait_for(pcm_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    if pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]

                        # Offload GPU inference to a thread so the event loop
                        # stays responsive for recv_loop. Wrap with shield so
                        # that if process_loop gets cancelled (e.g. the session
                        # is torn down on disconnect) the GPU thread finishes
                        # its current frame before cancellation propagates.
                        # Otherwise the outer code releases self.lock while
                        # this thread is still mutating mimi / lm_gen state,
                        # and the next session's reset_streaming() races with
                        # in-flight CUDA work.
                        in_flight = asyncio.ensure_future(
                            loop.run_in_executor(
                                None, self._process_audio_frame, chunk
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

                        for pcm_data, text in results:
                            # Send raw PCM float32 directly.
                            await ws.send_bytes(b"\x01" + pcm_data.astype(np.float32).tobytes())
                            if text is not None:
                                msg = b"\x02" + bytes(text, encoding="utf8")
                                await ws.send_bytes(msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                clog.log("error", f"process_loop: {type(e).__name__}: {e}")

        clog.log("info", "accepted connection")
        if len(request.query["text_prompt"]) > 0:
            clog.log("info", f"text prompt: {request.query['text_prompt']}")
        if len(request.query["voice_prompt"]) > 0:
            clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            # Apply per-connection sampling params under the lock so concurrent
            # connections cannot interleave their settings on the shared lm_gen.
            for k, v in sampling_params.items():
                setattr(self.lm_gen, k, v)
            # Reset the max-turn cap's internal counters so a prior session's
            # state doesn't carry into this one.
            self.lm_gen._non_pad_streak = 0
            self.lm_gen._pad_force_remaining = 0

            # ~200 ms ceiling on backlog at 20 ms client-side chunks. Overflow
            # drops newest in recv_loop so GPU stalls don't balloon memory or
            # desync the timeline by serving stale audio.
            pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(process_loop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
        clog.log("info", "done with connection")
        return ws


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
    app.router.add_get("/api/chat", state.handle_chat)
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
        
        let socket = null;
        let micCaptureNode = null;
        let micWorkletStream = null;
        let micWorkletSource = null;
        let micVad = null;
        let userSpeaking = false;
        // Ground-truth "model is playing audio right now" signal, posted by
        // the pcm-player AudioWorklet on its empty<->non-empty edges. This
        // replaces a 400 ms trailing timer that could lapse during GPU
        // stalls or persist past the actual end of a response.
        let modelPlaying = false;
        const SILENT_CHUNK = new Float32Array(480);  // 20 ms of zeros at 24 kHz
        let audioContext = null;
        let recordingDestination = null;
        let mediaRecorder = null;
        let recordedChunks = [];
        let micStream = null;
        let micSource = null;
        let shouldShowDownload = false;
        let playerNode = null;
        let aiAnalyser = null;
        let userAnalyser = null;
        let visualizerRAF = null;
        const SAMPLE_RATE = 24000;
        
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
        
        async function initAudio() {
            if (!audioContext) {
                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
                } catch (err) {
                    // Some Safari / Android / Bluetooth configs reject an
                    // explicit 24 kHz context. Fall back to hardware rate;
                    // the resample path to Mimi still works.
                    console.warn('AudioContext at 24 kHz rejected, falling back to hardware rate:', err);
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }
                // Device switch (headphones unplugged, default output
                // changes) pauses the context; without this, mic capture
                // silently stops until the user clicks Stop. Auto-resume
                // keeps the session alive across common system events.
                audioContext.addEventListener('statechange', () => {
                    console.log('AudioContext state:', audioContext.state);
                    if (audioContext.state === 'suspended' || audioContext.state === 'interrupted') {
                        audioContext.resume().catch((err) => {
                            console.warn('AudioContext auto-resume failed:', err);
                        });
                    }
                });
            }
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }
            if (!playerNode) {
                const code = `
class P extends AudioWorkletProcessor {
    constructor() {
        super();
        this.chunks = [];
        this.offset = 0;
        this.playing = false;
        this.port.onmessage = (e) => {
            if (e.data === 'flush') {
                this.chunks = []; this.offset = 0;
                if (this.playing) { this.playing = false; this.port.postMessage('idle'); }
                return;
            }
            this.chunks.push(e.data);
        };
    }
    process(inputs, outputs) {
        const out = outputs[0][0];
        let w = 0;
        while (w < out.length && this.chunks.length > 0) {
            const c = this.chunks[0];
            const n = Math.min(out.length - w, c.length - this.offset);
            out.set(c.subarray(this.offset, this.offset + n), w);
            w += n;
            this.offset += n;
            if (this.offset >= c.length) { this.chunks.shift(); this.offset = 0; }
        }
        // Ground-truth "model is audibly playing" signal: notify main thread
        // on the empty<->non-empty edge. Replaces a 400 ms timer heuristic
        // with the actual state of the output buffer.
        const nowPlaying = this.chunks.length > 0;
        if (nowPlaying !== this.playing) {
            this.playing = nowPlaying;
            this.port.postMessage(nowPlaying ? 'playing' : 'idle');
        }
        return true;
    }
}
registerProcessor('pcm-player', P);

class MicCapture extends AudioWorkletProcessor {
    constructor() {
        super();
        // 20 ms at 24 kHz = 480 samples per outbound chunk.
        this.buffer = new Float32Array(480);
        this.idx = 0;
    }
    process(inputs) {
        const input = inputs[0] && inputs[0][0];
        if (!input) return true;
        let i = 0;
        while (i < input.length) {
            const n = Math.min(input.length - i, this.buffer.length - this.idx);
            this.buffer.set(input.subarray(i, i + n), this.idx);
            i += n;
            this.idx += n;
            if (this.idx === this.buffer.length) {
                // Copy so we can reuse the ring buffer without racing the consumer.
                this.port.postMessage(this.buffer.slice());
                this.idx = 0;
            }
        }
        return true;
    }
}
registerProcessor('mic-capture', MicCapture);`;
                const blob = new Blob([code], { type: 'application/javascript' });
                await audioContext.audioWorklet.addModule(URL.createObjectURL(blob));
                playerNode = new AudioWorkletNode(audioContext, 'pcm-player');
                playerNode.port.onmessage = (e) => {
                    if (e.data === 'playing') modelPlaying = true;
                    else if (e.data === 'idle') modelPlaying = false;
                };
                aiAnalyser = audioContext.createAnalyser();
                aiAnalyser.fftSize = 256;
                aiAnalyser.smoothingTimeConstant = 0.85;
                playerNode.connect(aiAnalyser);
                aiAnalyser.connect(audioContext.destination);
            }
        }

        async function startSessionRecording() {
            try {
                shouldShowDownload = false;
                recordedChunks = [];
                downloadRow.style.display = 'none';
                if (!audioContext) {
                    return;
                }
                if (!recordingDestination) {
                    recordingDestination = audioContext.createMediaStreamDestination();
                }
                if (playerNode) {
                    playerNode.connect(recordingDestination);
                }
                try {
                    // Reuse the mic stream acquired by startMicRecording if it's
                    // up already; otherwise fall back to a fresh getUserMedia.
                    // Two independent captures of the same mic diverge on AEC/NS
                    // state and can produce different effective audio, so prefer
                    // sharing one track across worklet + recording + analyser.
                    if (micWorkletStream) {
                        micStream = micWorkletStream;
                    } else {
                        micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false } });
                    }
                    micSource = audioContext.createMediaStreamSource(micStream);
                    micSource.connect(recordingDestination);
                    userAnalyser = audioContext.createAnalyser();
                    userAnalyser.fftSize = 256;
                    userAnalyser.smoothingTimeConstant = 0.85;
                    micSource.connect(userAnalyser);
                } catch (err) {
                    console.warn('Could not attach mic stream to recording:', err);
                }
                // Start the visualizer loop regardless of mic outcome so the AI side animates.
                startVisualizers();

                mediaRecorder = new MediaRecorder(recordingDestination.stream);
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data && event.data.size > 0) {
                        recordedChunks.push(event.data);
                    }
                };
                mediaRecorder.onstop = () => {
                    if (!shouldShowDownload || recordedChunks.length === 0) {
                        return;
                    }
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
            if (showDownload !== null) {
                shouldShowDownload = showDownload;
            }
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                try { mediaRecorder.stop(); } catch (err) {}
            }
            if (micSource) {
                try { micSource.disconnect(); } catch (err) {}
                micSource = null;
            }
            // Do not stop micStream tracks here when it aliases micWorkletStream;
            // cleanup() stops micWorkletStream explicitly. Dropping the shared
            // track twice fires onended on a stream the worklet is still
            // consuming, producing spurious warnings.
            if (micStream && micStream !== micWorkletStream) {
                micStream.getTracks().forEach(track => track.stop());
            }
            micStream = null;
        }
        
        function playDecodedAudio(pcmData) {
            if (!playerNode || !pcmData || pcmData.length === 0) return;
            playerNode.port.postMessage(pcmData);
        }

        // Canvas visualizers driven by AnalyserNode RMS. One RAF loop draws both circles.
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
            const dpr = fitCanvas(canvas);
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
            // Inner solid dot when audio is active.
            if (isLive) {
                ctx.beginPath();
                ctx.arc(cx, cy, maxR * 0.18, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
            }
        }

        function startVisualizers() {
            stopVisualizers();
            const tick = () => {
                const live = !!(socket && socket.readyState === WebSocket.OPEN);
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
        
        
        async function startConversation() {
            try {
                connectBtn.disabled = true;
                connectBtn.textContent = 'Connecting...';
                downloadRow.style.display = 'none';
                downloadLink.removeAttribute('href');
                
                await initAudio();
                
                // Check microphone permission
                const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false } });
                stream.getTracks().forEach(track => track.stop());
                
                // Switch to conversation view
                showConversationView();
                setStatus('connecting', 'Connecting...');
                transcript.textContent = 'Connecting to server...';
                
                // Build WebSocket URL
                const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
                const wsUrl = new URL(wsProtocol + '://' + window.location.host + '/api/chat');
                wsUrl.searchParams.set('text_prompt', textPromptInput.value || '');
                // Uploaded voice wins over preset when both are present.
                const voiceParam = uploadedVoiceFilename || voicePromptSelect.value || '';
                wsUrl.searchParams.set('voice_prompt', voiceParam);
                wsUrl.searchParams.set('text_temperature', textTempSlider.value);
                wsUrl.searchParams.set('text_topk', textTopkSlider.value);
                wsUrl.searchParams.set('audio_temperature', audioTempSlider.value);
                wsUrl.searchParams.set('audio_topk', audioTopkSlider.value);
                wsUrl.searchParams.set('repetition_penalty', repPenaltySlider.value);
                wsUrl.searchParams.set('repetition_penalty_context', repContextSlider.value);
                wsUrl.searchParams.set('padding_bonus', padBonusSlider.value);
                wsUrl.searchParams.set('max_turn_text_tokens', maxTurnSlider.value);
                if (!seedRandomToggle.checked && seedInput.value !== '') {
                    wsUrl.searchParams.set('seed', seedInput.value);
                }
                
                socket = new WebSocket(wsUrl.toString());
                socket.binaryType = 'arraybuffer';
                
                socket.onopen = () => {
                    console.log('WebSocket connected, waiting for handshake...');
                    setStatus('connecting', 'Loading AI model (this may take a moment)...');
                };
                
                socket.onmessage = (event) => {
                    const data = new Uint8Array(event.data);
                    const msgType = data[0];
                    const payload = data.slice(1);
                    
                    if (msgType === 0x00) {
                        console.log('Handshake received, starting recording...');
                        setStatus('connected', 'Connected - Speak now!');
                        stopBtn.disabled = false;
                        transcript.textContent = '';
                        // Await mic worklet before session recording so they
                        // share one MediaStream (one set of AEC/NS state).
                        startMicRecording().then(() => startSessionRecording());
                    } else if (msgType === 0x01) {
                        const pcm = new Float32Array(payload.buffer, payload.byteOffset, payload.byteLength / 4);
                        playDecodedAudio(pcm);
                    } else if (msgType === 0x02) {
                        const text = new TextDecoder().decode(payload);
                        transcript.textContent += text;
                        transcript.scrollTop = transcript.scrollHeight;
                    }
                };
                
                socket.onerror = (err) => {
                    console.error('WebSocket error:', err);
                    showError('Connection error. Make sure you accepted the security certificate.', true);
                    cleanup();
                };
                
                socket.onclose = (event) => {
                    console.log('WebSocket closed:', event.code, event.reason);
                    if (event.code !== 1000) {
                        showError('Connection closed unexpectedly. The server may still be loading the model.', true);
                    }
                    setStatus('disconnected', 'Disconnected');
                    cleanup();
                };
                
            } catch (err) {
                console.error('Error:', err);
                if (err.name === 'NotAllowedError') {
                    showError('Microphone access denied. Please allow microphone access and try again.');
                } else {
                    showError(err.message || 'Failed to start conversation');
                }
                connectBtn.disabled = false;
                connectBtn.innerHTML = '<svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Connect';
                showSetupView();
            }
        }
        
        async function initVAD() {
            if (!window.vad || !window.vad.MicVAD) {
                console.warn('VAD library not loaded; barge-in still works, mic-mute during model speech will not');
                return;
            }
            try {
                // Tighter thresholds than the library defaults: 0.75 and 6
                // speech frames (~200 ms at 32 ms hop) reduce false-positive
                // speechStart on speaker bleed when browser AEC isn't fully
                // subtracting the model's own output. Cost: ~75 ms extra
                // latency on barge-in onset, acceptable for voice convos.
                micVad = await window.vad.MicVAD.new({
                    stream: micWorkletStream,
                    onSpeechStart: () => { userSpeaking = true; },
                    onSpeechEnd: () => { userSpeaking = false; },
                    onVADMisfire: () => { userSpeaking = false; },
                    positiveSpeechThreshold: 0.75,
                    minSpeechFrames: 6,
                });
                micVad.start();
                console.log('VAD running, feedback guard active');
            } catch (err) {
                console.warn('VAD init failed; continuing without it:', err);
                micVad = null;
            }
        }

        async function startMicRecording() {
            try {
                micWorkletStream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false },
                });
                micWorkletSource = audioContext.createMediaStreamSource(micWorkletStream);
                micCaptureNode = new AudioWorkletNode(audioContext, 'mic-capture');
                // Deliver captured PCM chunks to the server as raw float32 at
                // mimi.sample_rate (24 kHz). No encoder in the path means no
                // encode/decode latency and no CDN dependency.
                micCaptureNode.port.onmessage = (e) => {
                    if (!socket || socket.readyState !== WebSocket.OPEN) return;
                    // Swap in silence when the model is actively speaking and
                    // the user isn't. Moshi still gets a continuous stream
                    // (so turn-taking state stays aligned), but mic bleed
                    // from the speakers doesn't reach the encoder. VAD keeps
                    // barge-in working: once the user starts speaking, real
                    // PCM flows even mid-model-response.
                    let pcm = e.data;  // Float32Array, 480 samples (20 ms)
                    if (modelPlaying && !userSpeaking) {
                        pcm = SILENT_CHUNK;
                    }
                    const msg = new Uint8Array(1 + pcm.byteLength);
                    msg[0] = 0x01;
                    msg.set(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength), 1);
                    socket.send(msg);
                };
                // No speaker connection; we only need the worklet to run on
                // captured audio, not emit anything to the output graph.
                micWorkletSource.connect(micCaptureNode);
                await initVAD();
                console.log('Microphone capture started (raw PCM)');
            } catch (err) {
                console.error('Microphone error:', err);
                showError('Microphone error: ' + (err.message || 'Could not start recording'), true);
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
            connectBtn.innerHTML = '<svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Connect';
            stopBtn.style.display = 'inline-flex';
            newConvBtn.style.display = 'none';
            setProgress(20, 'Ready');
            downloadRow.style.display = 'none';
            downloadLink.removeAttribute('href');
        }
        
        function cleanup() {
            stopSessionRecording(null);
            if (micVad) {
                try { micVad.pause(); } catch(e) {}
                try { micVad.destroy && micVad.destroy(); } catch(e) {}
                micVad = null;
            }
            userSpeaking = false;
            modelPlaying = false;
            if (micCaptureNode) {
                try { micCaptureNode.port.onmessage = null; } catch(e) {}
                try { micCaptureNode.disconnect(); } catch(e) {}
                micCaptureNode = null;
            }
            if (micWorkletSource) {
                try { micWorkletSource.disconnect(); } catch(e) {}
                micWorkletSource = null;
            }
            if (micWorkletStream) {
                try { micWorkletStream.getTracks().forEach(t => t.stop()); } catch(e) {}
                micWorkletStream = null;
            }
            if (socket) {
                try { socket.close(); } catch(e) {}
                socket = null;
            }
            connectBtn.disabled = false;
            connectBtn.innerHTML = '<svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Connect';
            stopVisualizers();
            if (aiAnalyser) {
                try { aiAnalyser.disconnect(); } catch(e) {}
                aiAnalyser = null;
            }
            if (userAnalyser) {
                try { userAnalyser.disconnect(); } catch(e) {}
                userAnalyser = null;
            }
            if (playerNode) {
                playerNode.port.postMessage('flush');
                playerNode.disconnect();
                playerNode = null;
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
        
        # Serve decoder files from client/public/assets if they exist
        script_dir = Path(__file__).parent.parent.parent
        decoder_path = script_dir / "client" / "public" / "assets"
        if decoder_path.exists():
            async def serve_decoder_js(_):
                file_path = decoder_path / "decoderWorker.min.js"
                if file_path.exists():
                    return web.FileResponse(file_path)
                return web.Response(status=404)
            
            async def serve_decoder_wasm(_):
                file_path = decoder_path / "decoderWorker.min.wasm"
                if file_path.exists():
                    return web.FileResponse(file_path, headers={'Content-Type': 'application/wasm'})
                return web.Response(status=404)
            
            app.router.add_get("/assets/decoderWorker.min.js", serve_decoder_js)
            app.router.add_get("/assets/decoderWorker.min.wasm", serve_decoder_wasm)
            logger.info(f"Serving decoder files from {decoder_path}")
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
