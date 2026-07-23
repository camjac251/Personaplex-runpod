# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

PersonaPlex served over WebRTC on any CUDA GPU host with a public IP, defaulting to Kyutai's interactivity-aligned RL checkpoint with an NVIDIA-base rollback, optional vision, and snapshot/rewind safety. Single-session by design. WebRTC connects peers directly over UDP with no TURN relay.

## Commands

- `uv sync --frozen` - install deps.
- `uv run moshi-server --host 127.0.0.1 --port 8998 --static none --voice-prompt-dir voices` - run locally.
- `uvx ruff check moshi/moshi/server.py moshi/moshi/rtc_session.py moshi/moshi/models/lm.py` - lint after Python edits.
- `bun run frontend:build` - build the dashboard from `frontend/` into `moshi/moshi/web_client` (the lefthook pre-commit hook runs this and stages `web_client` on every commit).
- `bunx biome check frontend/src` - lint the dashboard after JS edits.
- `uv run python moshi/tests/test_rtc_resampler.py` - resampler, mono Opus encoder, and soft-limiter tests.
- `uv run python moshi/tests/test_streaming_restore.py` - snapshot/restore schema-validation tests.
- `uv run python moshi/tests/test_session_config.py` - SessionConfig clamp tests.
- `uv run python moshi/tests/test_vision_chunk.py` - vision chunk reassembly tests.
- `uv run python moshi/tests/test_rtc_pipeline.py` - control ordering, teardown, and audio pipeline tests.
- `uv run python moshi/tests/test_lm_controls.py` - sampling and anti-collapse tests.
- `uv run python moshi/tests/test_hang_watchdog.py` - GPU-hang stall detector and phase-write invariant tests.
- `uv run python moshi/tests/test_duplex_scenarios.py` - CPU checks for duplex manifests, VAD, Stop/cap scoring, and artifact bundles.
- `uv run python moshi/tests/test_inject_ritual.py` - context-drip conditioning ritual and completion-seal tests.
- `uv run python moshi/tests/test_caption_cfg.py` - caption-CFG guided sampling and branch bookkeeping tests.
- `uv run python moshi/tests/test_asr_generation.py` - rewind-safe ASR generation capture and callback-publication tests.
- `uv run python moshi/tests/test_sink_kv.py` - attention-sink ring KV cache: sink=0 parity, anchor preservation across wraps, absolute positions, restore round-trip.
- `uv run python moshi/tests/test_voice_select.py` - best-of-N voice reference window selection math.
- `uv run python scripts/run_duplex_regression.py --base-url http://127.0.0.1:8998 --input-wav <mono-48k-pcm16.wav> moshi/tests/fixtures/duplex/turn_taking.json` - drive an actual running WebRTC/GPU session and write replayable artifacts.
- `uv run python moshi/tests/test_cuda_dynamic_topk.py` - optional CUDA graph/top-k smoke test.

## Key files

- `moshi/moshi/server.py` - server lifecycle (`ServerState`), HTTP/WebRTC routes, vision inject state machine, resume grants; serves the built dashboard from `moshi/moshi/web_client` (plain-text 503 when unbuilt).
- `moshi/moshi/rtc_session.py` - WebRTC peer connection, DataChannel protocol, `SessionConfig` (parse + clamp).
- `moshi/moshi/models/lm.py` - `LMGen` (sampling, anti-collapse, voice prompt loading). Voice prompt clips are boundary-silence-trimmed (100 ms guard) before LUFS normalization and tail slicing.
- `moshi/moshi/rtc_opus.py` - PyAV mono Opus encoder patched over aiortc's stock stereo encoder at import; falls back to the stock encoder if libopus/PyAV setup fails.
- `moshi/moshi/modules/streaming.py` - state flatten/restore used by snapshot/rewind.
- `frontend/src/App.jsx` - the React dashboard: realtime state machine, tuning console, transcript, vision panel. `frontend/src/data/dashboardData.jsx` holds slider defs and tooltips.
- `frontend/src/utils/sessionTrace.js` - bounded privacy-safe browser bug reports. It must never export audio/images, network or session identifiers, device IDs, credentials, or raw prompts/transcripts by default.
- `moshi/tests/duplex_harness.py` and `scripts/run_duplex_regression.py` - shared scenario validation/scoring and the real aiortc regression client.

## Architecture invariants

- `ServerState.lock` (asyncio.Lock) gates the single live session. Second connect returns HTTP 409.
- `ServerState._infer_lock` (threading.Lock) guards `lm_gen` state. Event-loop mutations dispatch to `ServerState._infer_executor` first; never sync-acquire from a coroutine.
- All model work, including startup warmup, stays on the one persistent `_infer_executor` worker. Arbitrary default-pool CUDA workers pay multi-second cold-context costs; inline GPU work starves aiortc keepalives.
- Vision inject drips one token per outer frame, only when `_vision_pad_streak >= LIVE_PROMPT_BOUNDARY_STREAK` and never within `POST_USER_TURN_INJECT_HOLDOFF_FRAMES` (~2 s) of a completed user turn, so a caption cannot displace reply formation. Outbound PCM zeroed during inject (with a ~10 ms boundary fade via `_gate_outbound_pcm`). Cap at `LIVE_PROMPT_MAX_STEPS`. An interrupted or capped drip window seals with one `_context_seal_token` (".") so no half-sentence dangles in the model's history. A persona-reinforcement drip interrupted mid-clause defers its seal to the first quiet frame, even inside the post-user-turn holdoff, so reply formation starts with a closed clause.
- Context drips replicate the t=0 conditioning ritual: forced text over forced-silent agent audio with the user channel swapped to `_encode_sine_frame()` (the 440 Hz sine every trained text prompt rode on), never on stop-latch or abort frames where the model must keep hearing the room. A fully delivered packet is followed by `CONTEXT_SEAL_HOLD_FRAMES` (~0.5 s) of forced PAD under the same ritual, mirroring the t=0 post-prompt silence slots, so the sentence files as a completed thought; user speech abandons the hold. Captions are phrased as second-person scenario context ("You are talking with ...") because that is the only text-conditioning distribution PersonaPlex trained on.
- Caption-CFG (`--caption-cfg` / `PERSONAPLEX_CAPTION_CFG=1`) runs two streaming rows for the process lifetime: row 0 conditions on injected context, row 1 receives PAD in those windows, text logits are guided as `uncond + cfg_gamma * (cond - uncond)`, one sample feeds both rows, and row 0's depformer codes overwrite row 1's audio slot so the KV timelines differ only by context text. The batch size is captured into the CUDA graphs at startup warmup and can never change live; Mimi stays single-row and only `tokens[0:1]` is ever decoded. `cfg_gamma` boosts to the session's `caption_cfg_gamma` when a packet completes and decays back to 1.0 (~2.5 s half-life). Roughly doubles per-frame transformer compute and adds ~1.6 GB of permanent streaming KV. Single-row voice sidecars restore fine (the tensor restore broadcasts rows via `Tensor.copy_`); legacy embedding replay expands to both rows. Rewind/restore resets `cfg_gamma` to 1.0 because the restored KV predates the boosting caption. `caption_cfg_gamma` is live-tunable; a live change applies to the next completed packet's boost, never the currently decaying value.
- Auto-rewind: `_pad_force_remaining` tripping `COLLAPSE_TRIGGER_THRESHOLD` times in `COLLAPSE_WINDOW_SEC` restores the latest snapshot via `set_streaming_state_inplace` and resets sampling/anti-collapse controls to the active model's safe defaults. Manual/bookmark rewind preserves tuning. Always pass `dict(state_dict)` so subsequent rewinds still find the keys. It only accepts snapshots younger than `AUTO_REWIND_SNAPSHOT_MAX_AGE_SEC` (90 s), so it depends on periodic snapshots (60 s cadence). Periodic refreshes default OFF (`PERSONAPLEX_PERIODIC_SNAPSHOTS=1` opts in), so by default sessions keep only the session-start baseline plus explicit bookmarks and auto-rewind goes inert 90 s in. Snapshot capture defers while a context drip is active.
- Audio top-k is a scalar CUDA tensor consumed by a fixed-cardinality masked sampler, and audio temperature is a per-level `[dep_q]` CUDA tensor (level 0, the semantic codebook, is capped at `semantic_temperature_cap`, default 0.7 via `PERSONAPLEX_SEMANTIC_TEMP_CAP`). Live tuning must never reset or recapture the depformer CUDA graph. Text sampling supports optional min-p truncation (`text_min_p`, default 0 = off).
- Attention sink (`--kv-sink-frames` / `PERSONAPLEX_KV_SINK_FRAMES`, default 0 = off): reserves the first N slots of the main temporal transformer's `RingKVCache` as never-evicted anchors and exempts positions `[0, N)` from the sliding-window mask, so the t=0 conditioning stays attendable past the 3000-frame context on long sessions. RoPE is applied at write time, so anchor keys keep their true t=0 rotation; only the main transformer's cache is touched (the depformer runs `context=None` and never receives a sink). N is a construction-time int fixed before CUDA-graph capture (`RingKVCache.sink`), never streaming state, so `asdict()` stays `{cache, end_offset}` and snapshot/restore is schema-identical to stock. N=0 collapses every `torch.where` to the original ring expression (byte-identical). `RingKVCache` raises when `sink >= capacity`. Behavioral quality (does it reduce drift) needs the `long_session_soak` A/B on a running server and is not yet measured.
- The default checkpoint is `kyutai/personaplex-rl-seamless` at its pinned revision; `PERSONAPLEX_MODEL=base` selects the pinned NVIDIA rollback. Repository and revision are one startup-time identity and voice markers include both.
- Native-duplex defaults: `audio_temperature=0.8`, `padding_bonus=0.0`, `repetition_penalty=1.0`, `max_turn_text_tokens=120`. User configuration cannot lower max-turn below 40, and only caps at or above `COLLAPSE_SIGNAL_MIN_TURN_TOKENS` (120) count toward auto-rewind; lower caps truncate without feeding collapse detection. Max-turn counts text across brief PAD gaps and resets only after `REPETITION_TURN_BREAK_FRAMES` natural PAD/EPAD frames. Native mode never turns client-detected overlap into an automatic interrupt; Assisted mode does. The repetition ring uses the same natural turn boundary. Turn-cap accounting reads the text token through an async pinned D2H copy gated by a CUDA event; the hot path never blocks on a per-frame `.item()`.
- Stop is latched: `_stop_response_latched` forces assistant PAD and zero PCM until the next valid user turn completes. A fixed one-second gate alone is insufficient because the model can resume the abandoned answer. `STOP_LATCH_MAX_HOLD_SEC` (12 s) force-releases a latch whose release detectors starve in sustained room noise. Rewind/new-session reset the latch; transport resume preserves it.
- The slow `stat` envelope exposes cumulative inbound queue and outbound-buffer pressure. Keep those fields numeric and explicitly allowlisted; never put PCM, paths, SDP/candidates, or free-form secrets in telemetry.
- A `context_note` control message injects typed text through the same manual-priority drip path as captions (fit, ritual, seal, hold). Notes displace waiting ambient captions, never an undelivered manual packet. The dashboard input lives under the transcript.
- The `stat` envelope carries `opus_encode_failures`, a process-wide count of dropped-frame encode failures from the mono Opus patch.
- Vision captions are safe/UI-only by default. After-speech and on-demand injection are retired because GPU traces proved they are neither reliable next-reply grounding nor a private context channel. Unsafe Ambient react remains explicit, injects at most once per 8 s, and may speak without a user prompt. Live Gemini captions use an 80-token budget and schema-valid JSON is accepted even if the provider marks an interaction incomplete.
- `torch.cuda.empty_cache()` in `_run_rtc_session.finally`. Model weights and KV cache buffer stay resident across sessions.
- Transport recovery is fresh-pc resume, not ICE restart (aiortc can't restart a live transport): unexpected transport death records a 25 s `_resume_grant`; a new offer with `resume_session_id` skips reset/warmup and continues from resident state. Server-initiated ends (`send_end`) and client `goodbye` must NOT record a grant; `goodbye` is flagged at receive time, before the serialized control queue, so a teardown that cancels the queued handler still suppresses the grant.
- WebRTC connectivity is direct: `handle_ice_servers` serves an empty `iceServers` list by default and aiortc advertises its own host candidate, so the server must run where that candidate carries the public IP (on the host, or a container with host networking). `WEBRTC_STUN_URLS` (comma-separated) is an optional escape hatch for NAT'd hosts; there is no TURN relay.
- Oversized vision frames travel as `vision_frame_chunk` sequences (48 KB chunks under the 64 KB SCTP message cap) reassembled server-side into the normal `vision_frame` path.
- Outbound audio stays float from Mimi through the 24->48 k resampler, passes a soft limiter (knee at 0.97), and quantizes to s16 exactly once at frame emit. An outbound underrun is a due playout frame the buffer cannot fill; an exact drain-to-zero on the 80 ms Mimi lump cadence is healthy and never fades or raises the prebuffer floor.

## Gotchas

- `LMGen.text_prompt_tokens` is startup-only. Mid-stream context goes in `_vision_pending`.
- Mid-stream text injection is off-distribution. Boundary-streak + audio-gate + drip-feed is empirical; burst inject causes degenerate single-token loops.
- `<system>` wrap is t=0 only. Strip from mid-stream injected text.
- `set_streaming_state_inplace` pops the dict it's given. Pass a shallow copy.
- Streaming restore validates the full schema (dtype plus exact shape, or 1-row to N-row dim-0 broadcast only) before mutating any tensor; a failed restore leaves live state untouched.
- Aiortc `DataChannel.send` is not thread-safe. Schedule cross-thread sends via `loop.call_soon_threadsafe`.

## Pointers

- Moshi paper §3.4.4 (inner monologue, EPAD forcing): https://arxiv.org/abs/2410.00037
- NVIDIA/personaplex PR #69 (closed; origin of `LIVE_PROMPT_BOUNDARY_STREAK`): https://github.com/NVIDIA/personaplex/pull/69
