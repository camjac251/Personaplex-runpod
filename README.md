# PersonaPlex on RunPod

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/kyutai/personaplex-rl-seamless)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
[![Interactivity](https://img.shields.io/badge/📄-RL%20Interactivity-violet)](https://arxiv.org/abs/2606.11167)

PersonaPlex is a real-time, full-duplex speech-to-speech model with persona control via text prompts and voice conditioning. This fork defaults to Kyutai's interactivity-aligned PersonaPlex checkpoint and deploys on RunPod's maintained CUDA base image with a WebRTC browser client and a one-shot bootstrap script.

## Credits

- **Model and research**: NVIDIA PersonaPlex team. All credit for the core AI belongs to the original authors. See [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex).
- **Interactivity post-training**: Kyutai and Gradium. The default checkpoint is [kyutai/personaplex-rl-seamless](https://huggingface.co/kyutai/personaplex-rl-seamless), trained for pause handling, turn-taking, backchanneling, and interruption behavior.
- **Windows-installer fork this repo branched from**: [Suresh Pydikondala (SurAiverse)](https://www.youtube.com/@suraiverse).

## Deploy on RunPod

### 1. HuggingFace token

Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the gated license at [kyutai/personaplex-rl-seamless](https://huggingface.co/kyutai/personaplex-rl-seamless). The RL checkpoint combines CC BY-NC 4.0 with the NVIDIA Open Model License and is non-commercial. To use the base-model rollback, also accept [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1).

### 2. Cloudflare TURN credentials

WebRTC media is UDP. RunPod's `*.proxy.runpod.net` is Cloudflare-fronted and only carries HTTP/WS, so the browser needs a TURN relay to reach the pod. Cloudflare's free tier covers a personal voice agent comfortably (1 TB egress/month).

1. Sign in to Cloudflare. Dashboard -> **Realtime** -> **TURN Server**.
2. Click **Create TURN App**, name it.
3. Copy the **Turn Token ID** and **API Token**. The API token is shown once. Lose it and you regenerate.

### 3. RunPod secrets

In the RunPod console, go to **Secrets** -> **Create secret** and add the following. `GEMINI_API_KEY` is optional and only required if you want the Vision feature.

| Secret name | Value |
|---|---|
| `HF_TOKEN` | HuggingFace Read token |
| `TURN_KEY_ID` | Cloudflare Turn Token ID |
| `TURN_KEY_API_TOKEN` | Cloudflare API Token |
| `GEMINI_API_KEY` | Google AI Studio key (optional; enables Vision) |

### 4. Pod template

Create a Pod template (Templates -> New Template). Settings:

- **Template name**: `PersonaPlex`
- **Type**: Pod
- **Compute**: NVIDIA GPU
- **Public template**: Off (recommended unless you intend to publish it)
- **Container image**: `runpod/base:1.0.7-cuda1281-ubuntu2404@sha256:abd8ebde05b7027fb95913aefbcb9a236381633a481962877f658c2ed721fc37`
- **Registry authentication**: None
- **Container disk**: 20 GB
- **Volume disk**: 60 GB minimum, 80 GB comfortable
- **Volume mount path**: `/workspace`

This pins RunPod's maintained CUDA 12.8.1 base rather than publishing a second
application image. The project, exact locked Python environment, HuggingFace
assets, and voice prompts live under `/workspace`; the first boot creates them
and subsequent starts reuse them. No registry authentication is required.

**Container start command**:

```bash
bash -lc set${IFS}-e;curl${IFS}-fsSL${IFS}https://raw.githubusercontent.com/camjac251/Personaplex-runpod/main/start.sh${IFS}-o${IFS}/workspace/start.sh;chmod${IFS}+x${IFS}/workspace/start.sh;/start.sh&exec${IFS}/workspace/start.sh
```

**Networking configuration**:

| Label | Protocol | Port | Requirement |
|---|---|---:|---|
| `PersonaPlex` | HTTP | `8998` | Required |
| `JupyterLab` | HTTP | `8888` | Optional; also set `JUPYTER_PASSWORD` |

Leave TCP ports empty for RunPod's basic proxied SSH. Add your full public key
under RunPod account settings -> **SSH Public Keys**; RunPod injects it as
`PUBLIC_KEY`, and the base image's `/start.sh` starts `sshd`. Connect with the
`ssh <pod>-<suffix>@ssh.runpod.io -i <private-key>` command shown by RunPod.

Only add an `SSH` TCP port on `22` if you need direct public-IP SSH, SCP/SFTP,
rsync, or VS Code/Cursor Remote-SSH. Direct TCP also requires a host with a
public IP.

**Environment variables**:

| Name | Value |
|---|---|
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
| `TURN_KEY_ID` | `{{ RUNPOD_SECRET_TURN_KEY_ID }}` |
| `TURN_KEY_API_TOKEN` | `{{ RUNPOD_SECRET_TURN_KEY_API_TOKEN }}` |
| `GEMINI_API_KEY` | `{{ RUNPOD_SECRET_GEMINI_API_KEY }}` (optional) |
| `JUPYTER_PASSWORD` | A strong RunPod secret (optional; enables JupyterLab) |
| `PERSONAPLEX_MODEL` | `rl-seamless` (default) or `base` |
| `PERSONAPLEX_HF_REPO` | Custom model repository override (optional) |
| `PERSONAPLEX_HF_REVISION` | Override the pinned tested model revision (optional) |
| `PERSONAPLEX_PERIODIC_SNAPSHOTS` | `0` by default; set `1` for 60 s snapshot refreshes |

The launcher defaults to the pinned Seamless RL checkpoint. Set
`PERSONAPLEX_MODEL=base` to roll back to the pinned NVIDIA base checkpoint.
Both aliases select a matching immutable revision automatically, so restarting
an unchanged image cannot silently pick up different assets. A custom
`PERSONAPLEX_HF_REPO` must be paired with `PERSONAPLEX_HF_REVISION`. Launchers
ignore the old NVIDIA revision pin when it is the only model variable left in
an existing pod; set `PERSONAPLEX_MODEL=base` when that rollback is intended.

The RunPod launcher disables periodic full-state snapshots so their GPU copy
cannot interrupt the realtime inference worker once per minute. Each fresh
session still keeps one baseline snapshot for manual Rewind, and explicit
bookmarks still capture on demand. Set `PERSONAPLEX_PERIODIC_SNAPSHOTS=1` to
restore fresh 60-second snapshots and long-session auto-rewind coverage.

Use **Stop** / **Start** on the same Pod to keep `/workspace`. Terminating a
regular Pod deletes its volume disk; use a network volume if you need the cache
to survive Pod deletion or move between Pods.

### 5. Launch and connect

Use a GPU with at least 24 GB VRAM for the default resident model plus rewind
snapshot (RTX 4090 / A6000 / L40S all work). Lower-memory cards require CPU
offload and have substantially higher latency.

First boot downloads the 16.7 GB model plus tokenizer and voice assets. Expect 30-60 minutes depending on the data centre. The volume disk caches them, so subsequent boots reach "ready" in under a minute. Keeping both RL and base checkpoints requires space for both model weight files.

The dashboard reports the active repository, revision, and license. Checkpoint
selection stays at pod startup because hot-swapping an 8B model would discard
CUDA graphs, snapshots, and the live conversation. Use separate templates or
restart with `PERSONAPLEX_MODEL=base` for A/B comparisons. On a fresh browser,
the base selection also restores Assisted overlap handling, audio temperature
`0.7`, and repetition penalty `1.15`; saved user tuning remains untouched.

When the server log prints `serving static content from`, open the proxy URL from the pod (looks like `https://<pod-id>-8998.proxy.runpod.net/`). Click **Start**, allow microphone access, and speak.

To confirm TURN is doing its job: open `chrome://webrtc-internals` in another tab while a session is live. The active candidate pair under `selectedCandidatePairId` should have `relayProtocol: tcp` or `udp` and a remote address pointing at `turn.cloudflare.com`. If it shows `host` or `srflx` and you see no audio, TURN didn't engage.

## Profiles and diagnostics

The dashboard includes checkpoint-aware **Balanced**, **Concise**, and
**Expressive** profiles. Expressive uses the tested VARF4 voice and selects
Native duplex only for the aligned checkpoint; Base and unknown checkpoints
use Assisted overlap handling. Raw sampling controls remain available under
Advanced.

The diagnostics rail reports model/build identity, RTF, WebRTC jitter/loss,
input queue pressure, discarded buffered audio, reconnects, confirmed
interrupts, and auto-recoveries. **Export bug report** downloads a bounded JSON
trace with the applied seed, numeric configuration, prompt hashes, timing, and
structured events. It deliberately excludes transcript text, prompt text,
audio, images, SDP/ICE addresses, session/device IDs, URLs, and credentials.

For reproducible GPU checks, start the server and run a checked-in scenario
with a mono PCM16 48 kHz speech fixture whose timing matches the manifest:

```bash
uv run python scripts/run_duplex_regression.py \
  --base-url http://127.0.0.1:8998 \
  --input-wav /tmp/turn-taking.wav \
  moshi/tests/fixtures/duplex/turn_taking.json
```

The runner uses the production WebRTC/DataChannel protocol and captures the
actual remote audio. It scores pause takeover, turn latency, Stop
acknowledgement and audible yield, cap events, clipping, repetition, text
bursts, and RTF. Artifacts include both audio stems, raw control events,
configuration, model revision, tool hashes, and metrics. Treat these developer
artifacts as sensitive: unlike the dashboard's privacy-safe report, they can
contain conversation audio and text, prompts and full config, the server URL
and session ID, absolute local paths, and raw server/control metadata. Keep
them on trusted storage and inspect/redact them before sharing.

Runtime metrics also retain input-queue depth/high-water/drop counters and
output-buffer high-water/drop/flush counters from the server's periodic stat
envelope. Any dropped inbound microphone audio is a hard run failure. Outbound
backlog shedding is the intentional latency guard seen in normal GPU runs, so
up to 200 ms is recorded without failing; more than 200 ms is a suppressible
quality-threshold failure. Explicit output flushes are informational and never
fail a run.

`--no-fail-on-thresholds` is only for collecting results that exceed numeric
quality limits. Missing required turns/events, config mismatches, absent RTF
telemetry, failed actions, server errors, early disconnects, and signaling
failures still produce a non-zero exit status.

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

The reaction selector defaults to **Captions only**, which keeps descriptions
outside the speech model. **Ambient react** is explicitly unsafe: it injects
captions into PersonaPlex's own text stream and may speak about changing scenes
without being asked. Ambient injections are rate-limited to one every eight
seconds. Captions only is the cleanest mode for native duplex behavior.

Real GPU/WebRTC testing confirmed that PersonaPlex has no separate visual-input
role: injected captions enter its own text stream. Automatic after-turn and
on-demand "next reply" grounding were therefore removed rather than presented
as reliable features. The caption panel itself remains independent and safe.

The **Vision Prompt** textarea in the config panel customizes the system prompt sent to Gemini at the start of each session. Frames are motion-gated client-side so static scenes don't waste calls. A live cost meter and a rolling caption history sit below the preview. The fallback frame interval is configurable; most frames are server-requested when the model just went silent, so the timer rarely fires in practice.

## Hardware

Use at least 24 GB VRAM for the resident model and baseline rewind snapshot.
The RunPod launcher disables periodic refreshes by default; the baseline still
uses snapshot memory. Smaller cards require CPU offload and have higher
latency.

## Architecture notes

Audio path:

1. Browser captures mic via `getUserMedia` and sends Opus-encoded frames over `RTCPeerConnection`.
2. Server (aiortc) decodes to 48 kHz, resamples to Mimi's 24 kHz, and feeds the inference pipeline. All model work stays on one persistent inference thread so CUDA context and graph state remain warm while the asyncio loop stays responsive.
3. TTS PCM goes back the same way: 24 kHz -> 48 kHz -> Opus -> `<audio>` element in the browser.
4. A `RTCDataChannel` labelled `control` carries the session config (voice, sampling parameters, prompts) and streams text tokens for the transcript.

Browser AEC, noise suppression, and AGC handle echo and ambient noise. Backgrounded tabs keep playing AI audio because WebRTC is treated as active media by the browser.

Single-session: `self.lock` in `ServerState` enforces one peer connection at a time. A second connect attempt while a session is live returns HTTP 409 `session_busy` instead of hanging.

Vision path (when `GEMINI_API_KEY` is set and the user enables it):

1. Browser motion-gates each captured frame and sends a base64 JPEG over the `control` DataChannel.
2. Server forwards the frame to Gemini 3.5 Flash via the Interactions API. Conversation state chains turn-to-turn through `previous_interaction_id`, so the model has long-term memory of prior frames.
3. The one-sentence description is mirrored to the browser as a subtitle and added to the rolling history log. In the default Captions-only mode it never enters PersonaPlex.
4. Unsafe Ambient react may tokenise and queue the caption in `ServerState._vision_pending`. `_process_audio_frame` drains one token per Mimi frame only at a confirmed silent boundary, with outbound PCM gated for the duration.

## Known issues

These come from the upstream model, not the RunPod packaging:

- **Response looping**: under certain prompts the model can repeat itself. Native RL defaults leave repetition and PAD bias off so they do not distort learned turn timing; max-turn and auto-rewind remain circuit breakers, and the Advanced panel exposes anti-loop overrides.
- **Research checkpoint**: the default RL model is non-commercial and its published evaluation is automated. Conversation-data style can affect safety behavior; review the model card before deploying it beyond research or personal evaluation.
- **Pipeline efficiency**: GPU utilisation is occasionally spiky; some kernels are not yet optimised.

Base-model issues belong upstream at [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex/issues); RL-checkpoint behavior belongs with [Kyutai's model](https://huggingface.co/kyutai/personaplex-rl-seamless). Bugs in the RunPod packaging or WebRTC client belong here.

## Local dev

If you want to run outside RunPod (LAN only, no TURN required since both peers can reach each other directly):

```bash
uv sync --frozen
bun install --frozen-lockfile
bun run frontend:build
uv run moshi-server --host 127.0.0.1 --port 8998 --voice-prompt-dir voices
```

Voice prompts need to be downloaded manually (see `start.sh` for the HuggingFace pull recipe) or symlinked from a previous RunPod volume.

Run the focused CPU regression checks:

```bash
uv run python moshi/tests/test_rtc_resampler.py
uv run python moshi/tests/test_duplex_scenarios.py
bun test frontend/src/utils/sessionTrace.test.js
```

## License

Code: MIT. The default `kyutai/personaplex-rl-seamless` checkpoint combines
CC BY-NC 4.0 and the NVIDIA Open Model License and is non-commercial. The
rollback `nvidia/personaplex-7b-v1` checkpoint uses the NVIDIA Open Model
License. Always follow the license shown for the active model revision.

## Citation

```bibtex
@article{roy2026personaplex,
  title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models},
  author={Roy, Rajarshi and Raiman, Jonathan and Lee, Sang-gil and Ene, Teodor-Dumitru and Kirby, Robert and Kim, Sungwon and Kim, Jaehyeon and Catanzaro, Bryan},
  year={2026}
}
```
