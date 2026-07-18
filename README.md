# PersonaPlex WebRTC

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/kyutai/personaplex-rl-seamless)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
[![Interactivity](https://img.shields.io/badge/📄-RL%20Interactivity-violet)](https://arxiv.org/abs/2606.11167)

PersonaPlex is a real-time, full-duplex speech-to-speech model with persona control via text prompts and voice conditioning. This fork defaults to Kyutai's interactivity-aligned PersonaPlex checkpoint and serves it over WebRTC with a browser dashboard, on any CUDA GPU host you control.

## Credits

- **Model and research**: NVIDIA PersonaPlex team. All credit for the core AI belongs to the original authors. See [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex).
- **Interactivity post-training**: Kyutai and Gradium. The default checkpoint is [kyutai/personaplex-rl-seamless](https://huggingface.co/kyutai/personaplex-rl-seamless), trained for pause handling, turn-taking, backchanneling, and interruption behavior.
- **Windows-installer fork this repo branched from**: [Suresh Pydikondala (SurAiverse)](https://www.youtube.com/@suraiverse).

## Requirements

- An NVIDIA GPU with at least 24 GB VRAM for the resident model plus its baseline rewind snapshot (RTX 4090, RTX 6000 Ada, A6000, and L40S all work). Lower-memory cards require CPU offload and have substantially higher latency.
- A host with a **public IP and open UDP**, reachable from the browsers that will connect. WebRTC media is UDP, and the server connects peers directly (see [Networking](#networking)).
- Linux with a recent CUDA driver, plus [`uv`](https://docs.astral.sh/uv/) and [`bun`](https://bun.sh/).

## Setup

### 1. HuggingFace token

Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the gated license at [kyutai/personaplex-rl-seamless](https://huggingface.co/kyutai/personaplex-rl-seamless). The RL checkpoint combines CC BY-NC 4.0 with the NVIDIA Open Model License and is non-commercial. To use the base-model rollback, also accept [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1).

### 2. Install and build

```bash
git clone https://github.com/camjac251/personaplex-webrtc.git
cd personaplex-webrtc
export HF_TOKEN=hf_...            # required; see step 1
uv sync --frozen                 # Python environment
bun install --frozen-lockfile    # dashboard dependencies
bun run frontend:build           # builds the dashboard into moshi/moshi/web_client
```

### 3. Prefetch model and voice assets (optional)

The server downloads the checkpoint on first launch, but prefetching makes the first session instant. The prefetch script resolves the repository and pinned revision from `PERSONAPLEX_MODEL` (`rl-seamless` default, or `base`):

```bash
uv run python scripts/prefetch_assets.py --voice-dir voices
```

### 4. Run

```bash
uv run moshi-server \
  --host 0.0.0.0 \
  --port 8998 \
  --voice-prompt-dir voices
```

`moshi-server` reads `PERSONAPLEX_MODEL` (and the optional `PERSONAPLEX_HF_REPO` / `PERSONAPLEX_HF_REVISION` overrides) directly, so no wrapper script is needed to pick the checkpoint.

First boot downloads the 16.7 GB model plus tokenizer and voice assets; expect 30-60 minutes depending on the host's bandwidth. Subsequent boots reach "ready" in under a minute once the HuggingFace cache is warm. Keeping both RL and base checkpoints requires disk for both weight files.

When the log prints `serving static content from`, open `http://<public-ip>:8998/` in a browser. Click **Start**, allow microphone access, and speak.

The dashboard reports the active repository, revision, and license. Checkpoint selection stays at startup because hot-swapping an 8B model would discard CUDA graphs, snapshots, and the live conversation. Restart with `PERSONAPLEX_MODEL=base` for A/B comparisons. On a fresh browser, the base selection also restores Assisted overlap handling, audio temperature `0.7`, and repetition penalty `1.15`; saved user tuning remains untouched.

### Networking

WebRTC needs two paths reachable from the client:

- **TCP `8998`** for the dashboard and signaling.
- **UDP** for the media (ICE and RTP).

The server connects peers directly: it advertises its own host ICE candidate, so it must run where that candidate carries the public IP. On a VM whose network interface holds the public IP, run the server on the host. If you containerize it, use host networking (`docker run --network=host ...`) so aiortc sees the public address rather than a private bridge IP. No STUN or TURN relay is used by default.

If you deploy behind NAT and need STUN to discover a server-reflexive candidate, set `WEBRTC_STUN_URLS` to a comma-separated list (for example `stun:stun.l.google.com:19302`). It is empty by default, which keeps connectivity fully direct.

### Environment variables

| Name | Value |
|---|---|
| `HF_TOKEN` | HuggingFace Read token (required to download the model) |
| `GEMINI_API_KEY` | Google AI Studio key (optional; enables Vision) |
| `PERSONAPLEX_MODEL` | `rl-seamless` (default) or `base` |
| `PERSONAPLEX_HF_REPO` | Custom model repository override (optional) |
| `PERSONAPLEX_HF_REVISION` | Override the pinned model revision (optional; required with a custom repo) |
| `PERSONAPLEX_PERIODIC_SNAPSHOTS` | `0` by default; set `1` to enable 60 s snapshot refreshes for long-session auto-rewind |
| `WEBRTC_STUN_URLS` | Optional comma-separated STUN URLs for NAT'd hosts; empty means fully direct |

The launcher defaults to the pinned Seamless RL checkpoint. Set `PERSONAPLEX_MODEL=base` to roll back to the pinned NVIDIA base checkpoint. Both aliases select a matching immutable revision automatically, so restarting an unchanged environment cannot silently pick up different assets. A custom `PERSONAPLEX_HF_REPO` must be paired with `PERSONAPLEX_HF_REVISION`.

Periodic full-state snapshots are off by default: each fresh session keeps one baseline snapshot for manual Rewind, and explicit bookmarks capture on demand. Set `PERSONAPLEX_PERIODIC_SNAPSHOTS=1` to refresh a snapshot every 60 s, which keeps manual Rewind recent and powers long-session auto-rewind (each capture is a ~1.6 GB in-VRAM copy, usually inside one 80 ms frame on a modern GPU).

## Profiles and diagnostics

The dashboard includes checkpoint-aware **Balanced**, **Concise**, and **Expressive** profiles. Expressive uses the tested VARF4 voice and selects Native duplex only for the aligned checkpoint; Base and unknown checkpoints use Assisted overlap handling. Raw sampling controls remain available under Advanced.

The diagnostics rail reports model/build identity, RTF, WebRTC jitter/loss, input queue pressure, discarded buffered audio, reconnects, confirmed interrupts, and auto-recoveries. **Export bug report** downloads a bounded JSON trace with the applied seed, numeric configuration, prompt hashes, timing, and structured events. It deliberately excludes transcript text, prompt text, audio, images, SDP/ICE addresses, session/device IDs, URLs, and credentials.

To inspect connectivity while a session is live, open `chrome://webrtc-internals` in another tab. The active pair under `selectedCandidatePairId` should show a `host` candidate over UDP pointing at the server's public address.

For reproducible GPU checks, start the server and run a checked-in scenario with a mono PCM16 48 kHz speech fixture whose timing matches the manifest:

```bash
uv run python scripts/run_duplex_regression.py \
  --base-url http://127.0.0.1:8998 \
  --input-wav /tmp/turn-taking.wav \
  moshi/tests/fixtures/duplex/turn_taking.json
```

The runner uses the production WebRTC/DataChannel protocol and captures the actual remote audio. It scores pause takeover, turn latency, Stop acknowledgement and audible yield, cap events, clipping, repetition, text bursts, and RTF. Artifacts include both audio stems, raw control events, configuration, model revision, tool hashes, and metrics. Treat these developer artifacts as sensitive: unlike the dashboard's privacy-safe report, they can contain conversation audio and text, prompts and full config, the server URL and session ID, absolute local paths, and raw server/control metadata. Keep them on trusted storage and inspect/redact them before sharing.

Runtime metrics also retain input-queue depth/high-water/drop counters and output-buffer high-water/drop/flush counters from the server's periodic stat envelope. Any dropped inbound microphone audio is a hard run failure. Outbound backlog shedding is the intentional latency guard seen in normal GPU runs, so up to 200 ms is recorded without failing; more than 200 ms is a suppressible quality-threshold failure. Explicit output flushes are informational and never fail a run.

`--no-fail-on-thresholds` is only for collecting results that exceed numeric quality limits. Missing required turns/events, config mismatches, absent RTF telemetry, failed actions, server errors, early disconnects, and signaling failures still produce a non-zero exit status.

## Voices

Pre-packaged voice embeddings:

- **Natural (female)**: NATF0, NATF1, NATF2, NATF3
- **Natural (male)**: NATM0, NATM1, NATM2, NATM3
- **Variety (female)**: VARF0 through VARF4
- **Variety (male)**: VARM0 through VARM4

You can also upload 10-30 s of clean audio for any speaker via the **Clone a voice** panel. Mono or stereo, any common format. The model uses it as a voice prefix and continues in that timbre. Not zero-shot perfect, but recognisable.

## Vision (optional)

Adds situational awareness from a screen share or virtual camera. Frames are sent to **Gemini 3.5 Flash** via the Interactions API. Captions stay outside PersonaPlex by default; reaction modes can drip a compact scene fact into its text channel during natural silence windows.

Enable it by providing `GEMINI_API_KEY` and clicking **Add Vision** in the UI. Without the key the button stays disabled and a toast explains why.

Controls:

- **Add Vision / Stop Vision**: start or end the capture stream.
- **Pause Vision / Resume Vision**: keep the stream open but stop sending frames.
- **Capture Now**: force a high-detail frame send immediately (bypasses motion gate and pause).
- **Rewind**: restore the last KV-cache snapshot if the model gets stuck. Auto-rewind also fires when the safety net trips 3+ times in 30 s.

The reaction selector defaults to **Captions only**, which keeps descriptions outside the speech model. **Ambient react** is explicitly unsafe: it injects captions into PersonaPlex's own text stream and may speak about changing scenes without being asked. Ambient injections are rate-limited to one every eight seconds. Captions only is the cleanest mode for native duplex behavior.

Real GPU/WebRTC testing confirmed that PersonaPlex has no separate visual-input role: injected captions enter its own text stream. Automatic after-turn and on-demand "next reply" grounding were therefore removed rather than presented as reliable features. The caption panel itself remains independent and safe.

The **Vision Prompt** textarea in the config panel customizes the system prompt sent to Gemini at the start of each session. Frames are motion-gated client-side so static scenes don't waste calls. A live cost meter and a rolling caption history sit below the preview. The fallback frame interval is configurable; most frames are server-requested when the model just went silent, so the timer rarely fires in practice.

## Architecture notes

Audio path:

1. Browser captures mic via `getUserMedia` and sends Opus-encoded frames over `RTCPeerConnection`.
2. Server (aiortc) decodes to 48 kHz, resamples to Mimi's 24 kHz, and feeds the inference pipeline. All model work stays on one persistent inference thread so CUDA context and graph state remain warm while the asyncio loop stays responsive.
3. TTS PCM goes back the same way: 24 kHz -> 48 kHz -> Opus -> `<audio>` element in the browser.
4. A `RTCDataChannel` labelled `control` carries the session config (voice, sampling parameters, prompts) and streams text tokens for the transcript.

WebRTC connectivity is direct: the server advertises its own host candidate (the public IP when run on the host or with host networking) and the browser reaches it over UDP, with no STUN or TURN relay in the default path. Browser AEC, noise suppression, and AGC handle echo and ambient noise. Backgrounded tabs keep playing AI audio because WebRTC is treated as active media by the browser.

Single-session: `self.lock` in `ServerState` enforces one peer connection at a time. A second connect attempt while a session is live returns HTTP 409 `session_busy` instead of hanging.

Vision path (when `GEMINI_API_KEY` is set and the user enables it):

1. Browser motion-gates each captured frame and sends a base64 JPEG over the `control` DataChannel.
2. Server forwards the frame to Gemini 3.5 Flash via the Interactions API. Conversation state chains turn-to-turn through `previous_interaction_id`, so the model has long-term memory of prior frames.
3. The one-sentence description is mirrored to the browser as a subtitle and added to the rolling history log. In the default Captions-only mode it never enters PersonaPlex.
4. Unsafe Ambient react may tokenise and queue the caption in `ServerState._vision_pending`. `_process_audio_frame` drains one token per Mimi frame only at a confirmed silent boundary, with outbound PCM gated for the duration.

## Known issues

These come from the upstream model, not this packaging:

- **Response looping**: under certain prompts the model can repeat itself. Native RL defaults leave repetition and PAD bias off so they do not distort learned turn timing; max-turn and auto-rewind remain circuit breakers, and the Advanced panel exposes anti-loop overrides.
- **Research checkpoint**: the default RL model is non-commercial and its published evaluation is automated. Conversation-data style can affect safety behavior; review the model card before deploying it beyond research or personal evaluation.
- **Pipeline efficiency**: GPU utilisation is occasionally spiky; some kernels are not yet optimised.

Base-model issues belong upstream at [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex/issues); RL-checkpoint behavior belongs with [Kyutai's model](https://huggingface.co/kyutai/personaplex-rl-seamless). Bugs in this packaging or the WebRTC client belong here.

## Local dev

For local, same-machine testing you can bind to loopback and skip the public-IP requirement entirely (both peers are on the same host):

```bash
uv sync --frozen
bun install --frozen-lockfile
bun run frontend:build
uv run moshi-server --host 127.0.0.1 --port 8998 --voice-prompt-dir voices
```

Voice prompts download automatically on first launch, or prefetch them with `scripts/prefetch_assets.py` (see step 3).

Run the focused CPU regression checks:

```bash
uv run python moshi/tests/test_rtc_resampler.py
uv run python moshi/tests/test_duplex_scenarios.py
bun test frontend/src/utils/sessionTrace.test.js
```

## License

Code: MIT. The default `kyutai/personaplex-rl-seamless` checkpoint combines CC BY-NC 4.0 and the NVIDIA Open Model License and is non-commercial. The rollback `nvidia/personaplex-7b-v1` checkpoint uses the NVIDIA Open Model License. Always follow the license shown for the active model revision.

## Citation

```bibtex
@article{roy2026personaplex,
  title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models},
  author={Roy, Rajarshi and Raiman, Jonathan and Lee, Sang-gil and Ene, Teodor-Dumitru and Kirby, Robert and Kim, Sungwon and Kim, Jaehyeon and Catanzaro, Bryan},
  year={2026}
}
```
