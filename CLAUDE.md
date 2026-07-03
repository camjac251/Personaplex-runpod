# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

RunPod single-template deployment of NVIDIA PersonaPlex (a Moshi finetune) over WebRTC, with an optional vision side and a snapshot/rewind safety net. Single-session by design.

## Commands

- `uv sync --frozen` - install deps.
- `uv run moshi-server --host 127.0.0.1 --port 8998 --static none --voice-prompt-dir voices` - run locally.
- `ruff check moshi/server.py moshi/rtc_session.py moshi/models/lm.py` - lint after Python edits.
- `bun run frontend:build` - build the dashboard from `frontend/` into `moshi/moshi/web_client` (the lefthook pre-commit hook runs this and stages `web_client` on every commit).
- `bunx biome check frontend/src` - lint the dashboard after JS edits.
- `uv run python moshi/tests/test_rtc_resampler.py` - resampler smoke test.
- `uv run python moshi/tests/test_session_config.py` - SessionConfig clamp tests.

## Key files

- `moshi/moshi/server.py` - server lifecycle (`ServerState`), HTTP/WebRTC routes, vision inject state machine, resume grants; serves the built dashboard from `moshi/moshi/web_client` (plain-text 503 when unbuilt).
- `moshi/moshi/rtc_session.py` - WebRTC peer connection, DataChannel protocol, `SessionConfig` (parse + clamp).
- `moshi/moshi/models/lm.py` - `LMGen` (sampling, anti-collapse, voice prompt loading).
- `moshi/moshi/modules/streaming.py` - state flatten/restore used by snapshot/rewind.
- `frontend/src/App.jsx` - the React dashboard: realtime state machine, tuning console, transcript, vision panel. `frontend/src/data/dashboardData.jsx` holds slider defs and tooltips.

## Architecture invariants

- `ServerState.lock` (asyncio.Lock) gates the single live session. Second connect returns HTTP 409.
- `ServerState._infer_lock` (threading.Lock) guards `lm_gen` state. Event-loop mutations dispatch to executor first; never sync-acquire from a coroutine.
- Inference and warmup run via `loop.run_in_executor`. Inline GPU work starves aiortc keepalives.
- Vision inject drips one token per outer frame, only when `_vision_pad_streak >= LIVE_PROMPT_BOUNDARY_STREAK`. Outbound PCM zeroed during inject. Cap at `LIVE_PROMPT_MAX_STEPS`.
- Auto-rewind: `_pad_force_remaining` tripping `COLLAPSE_TRIGGER_THRESHOLD` times in `COLLAPSE_WINDOW_SEC` restores the latest snapshot via `set_streaming_state_inplace`. Always pass `dict(state_dict)` so subsequent rewinds still find the keys.
- Anti-collapse slider defaults: `padding_bonus=1.0`, `max_turn_text_tokens=120`, `repetition_penalty=1.15`. Verify `buildConfigPayload()` is actually sending nonzero values before debugging "model rambling" symptoms.
- `torch.cuda.empty_cache()` in `_run_rtc_session.finally`. Model weights and KV cache buffer stay resident across sessions.
- Transport recovery is fresh-pc resume, not ICE restart (aiortc can't restart a live transport): unexpected transport death records a 25 s `_resume_grant`; a new offer with `resume_session_id` skips reset/warmup and continues from resident state. Server-initiated ends (`send_end`) and client `goodbye` must NOT record a grant.
- Oversized vision frames travel as `vision_frame_chunk` sequences (48 KB chunks under the 64 KB SCTP message cap) reassembled server-side into the normal `vision_frame` path.

## Gotchas

- `LMGen.text_prompt_tokens` is startup-only. Mid-stream context goes in `_vision_pending`.
- Mid-stream text injection is off-distribution. Boundary-streak + audio-gate + drip-feed is empirical; burst inject causes degenerate single-token loops.
- `<system>` wrap is t=0 only. Strip from mid-stream injected text.
- `set_streaming_state_inplace` pops the dict it's given. Pass a shallow copy.
- Aiortc `DataChannel.send` is not thread-safe. Schedule cross-thread sends via `loop.call_soon_threadsafe`.

## Pointers

- Moshi paper §3.4.4 (inner monologue, EPAD forcing): https://arxiv.org/abs/2410.00037
- NVIDIA/personaplex PR #69 (closed; origin of `LIVE_PROMPT_BOUNDARY_STREAK`): https://github.com/NVIDIA/personaplex/pull/69
