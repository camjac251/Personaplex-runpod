# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PersonaPlex served over WebRTC on any CUDA GPU host with a public IP, defaulting to Kyutai's interactivity-aligned RL checkpoint with an NVIDIA-base rollback, optional vision, and snapshot/rewind safety. Single-session by design. WebRTC connects peers directly over UDP with no TURN relay.

## Commands

- `uv sync --frozen` - install deps.
- `uv run moshi-server --host 127.0.0.1 --port 8998 --static none --voice-prompt-dir voices` - run locally.
- `uvx ruff check moshi/moshi/server.py moshi/moshi/rtc_session.py moshi/moshi/models/lm.py` - lint after Python edits.
- `bun run frontend:build` - build the dashboard from `frontend/` into `moshi/moshi/web_client` (the lefthook pre-commit hook runs this and stages `web_client` on every commit).
- `bunx biome check frontend/src` - lint the dashboard after JS edits.
- `uv run python moshi/tests/test_rtc_resampler.py` - resampler smoke test.
- `uv run python moshi/tests/test_session_config.py` - SessionConfig clamp tests.
- `uv run python moshi/tests/test_vision_chunk.py` - vision chunk reassembly tests.
- `uv run python moshi/tests/test_rtc_pipeline.py` - control ordering, teardown, and audio pipeline tests.
- `uv run python moshi/tests/test_lm_controls.py` - sampling and anti-collapse tests.
- `uv run python moshi/tests/test_hang_watchdog.py` - GPU-hang stall detector and phase-write invariant tests.
- `uv run python moshi/tests/test_duplex_scenarios.py` - CPU checks for duplex manifests, VAD, Stop/cap scoring, and artifact bundles.
- `uv run python scripts/run_duplex_regression.py --base-url http://127.0.0.1:8998 --input-wav <mono-48k-pcm16.wav> moshi/tests/fixtures/duplex/turn_taking.json` - drive an actual running WebRTC/GPU session and write replayable artifacts.
- `uv run python moshi/tests/test_cuda_dynamic_topk.py` - optional CUDA graph/top-k smoke test.

## Key files

- `moshi/moshi/server.py` - server lifecycle (`ServerState`), HTTP/WebRTC routes, vision inject state machine, resume grants; serves the built dashboard from `moshi/moshi/web_client` (plain-text 503 when unbuilt).
- `moshi/moshi/rtc_session.py` - WebRTC peer connection, DataChannel protocol, `SessionConfig` (parse + clamp).
- `moshi/moshi/models/lm.py` - `LMGen` (sampling, anti-collapse, voice prompt loading).
- `moshi/moshi/modules/streaming.py` - state flatten/restore used by snapshot/rewind.
- `frontend/src/App.jsx` - the React dashboard: realtime state machine, tuning console, transcript, vision panel. `frontend/src/data/dashboardData.jsx` holds slider defs and tooltips.
- `frontend/src/utils/sessionTrace.js` - bounded privacy-safe browser bug reports. It must never export audio/images, network or session identifiers, device IDs, credentials, or raw prompts/transcripts by default.
- `moshi/tests/duplex_harness.py` and `scripts/run_duplex_regression.py` - shared scenario validation/scoring and the real aiortc regression client.

## Architecture invariants

- `ServerState.lock` (asyncio.Lock) gates the single live session. Second connect returns HTTP 409.
- `ServerState._infer_lock` (threading.Lock) guards `lm_gen` state. Event-loop mutations dispatch to `ServerState._infer_executor` first; never sync-acquire from a coroutine.
- All model work, including startup warmup, stays on the one persistent `_infer_executor` worker. Arbitrary default-pool CUDA workers pay multi-second cold-context costs; inline GPU work starves aiortc keepalives.
- Vision inject drips one token per outer frame, only when `_vision_pad_streak >= LIVE_PROMPT_BOUNDARY_STREAK` and never within `POST_USER_TURN_INJECT_HOLDOFF_FRAMES` (~2 s) of a completed user turn, so a caption cannot displace reply formation. Outbound PCM zeroed during inject. Cap at `LIVE_PROMPT_MAX_STEPS`.
- Auto-rewind: `_pad_force_remaining` tripping `COLLAPSE_TRIGGER_THRESHOLD` times in `COLLAPSE_WINDOW_SEC` restores the latest snapshot via `set_streaming_state_inplace` and resets sampling/anti-collapse controls to the active model's safe defaults. Manual/bookmark rewind preserves tuning. Always pass `dict(state_dict)` so subsequent rewinds still find the keys. It only accepts snapshots younger than `AUTO_REWIND_SNAPSHOT_MAX_AGE_SEC` (90 s), so it depends on periodic snapshots (60 s cadence). The server defaults periodic refreshes on (`PERSONAPLEX_PERIODIC_SNAPSHOTS=1`); set it to `0` to keep only the session-start baseline. Snapshot capture defers while a context drip is active.
- Audio top-k is a scalar CUDA tensor consumed by a fixed-cardinality masked sampler. Live tuning must never reset or recapture the depformer CUDA graph.
- The default checkpoint is `kyutai/personaplex-rl-seamless` at its pinned revision; `PERSONAPLEX_MODEL=base` selects the pinned NVIDIA rollback. Repository and revision are one startup-time identity and voice markers include both.
- Native-duplex defaults: `audio_temperature=0.8`, `padding_bonus=0.0`, `repetition_penalty=1.0`, `max_turn_text_tokens=120`. User configuration cannot lower max-turn below 40, and only caps at or above `COLLAPSE_SIGNAL_MIN_TURN_TOKENS` (120) count toward auto-rewind; lower caps truncate without feeding collapse detection. Max-turn counts text across brief PAD gaps and resets only after `REPETITION_TURN_BREAK_FRAMES` natural PAD/EPAD frames. Native mode never turns client-detected overlap into an automatic interrupt; Assisted mode does. The repetition ring uses the same natural turn boundary.
- Stop is latched: `_stop_response_latched` forces assistant PAD and zero PCM until the next valid user turn completes. A fixed one-second gate alone is insufficient because the model can resume the abandoned answer. Rewind/new-session reset the latch; transport resume preserves it.
- The slow `stat` envelope exposes cumulative inbound queue and outbound-buffer pressure. Keep those fields numeric and explicitly allowlisted; never put PCM, paths, SDP/candidates, or free-form secrets in telemetry.
- Vision captions are safe/UI-only by default. After-speech and on-demand injection are retired because GPU traces proved they are neither reliable next-reply grounding nor a private context channel. Unsafe Ambient react remains explicit, injects at most once per 8 s, and may speak without a user prompt. Live Gemini captions use an 80-token budget and schema-valid JSON is accepted even if the provider marks an interaction incomplete.
- `torch.cuda.empty_cache()` in `_run_rtc_session.finally`. Model weights and KV cache buffer stay resident across sessions.
- Transport recovery is fresh-pc resume, not ICE restart (aiortc can't restart a live transport): unexpected transport death records a 25 s `_resume_grant`; a new offer with `resume_session_id` skips reset/warmup and continues from resident state. Server-initiated ends (`send_end`) and client `goodbye` must NOT record a grant.
- WebRTC connectivity is direct: `handle_ice_servers` serves an empty `iceServers` list by default and aiortc advertises its own host candidate, so the server must run where that candidate carries the public IP (on the host, or a container with host networking). `WEBRTC_STUN_URLS` (comma-separated) is an optional escape hatch for NAT'd hosts; there is no TURN relay.
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
