# PersonaPlex on RunPod

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)

PersonaPlex is a real-time, full-duplex speech-to-speech model with persona control via text prompts and voice conditioning. This fork packages it as a single-template RunPod deployment with a WebRTC browser client and a one-shot bootstrap script.

## Credits

- **Model and research**: NVIDIA PersonaPlex team. All credit for the core AI belongs to the original authors. See [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex).
- **Windows-installer fork this repo branched from**: [Suresh Pydikondala (SurAiverse)](https://www.youtube.com/@suraiverse).

## Deploy on RunPod

### 1. HuggingFace token

Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and accept the model license at [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1).

### 2. Cloudflare TURN credentials

WebRTC media is UDP. RunPod's `*.proxy.runpod.net` is Cloudflare-fronted and only carries HTTP/WS, so the browser needs a TURN relay to reach the pod. Cloudflare's free tier covers a personal voice agent comfortably (1 TB egress/month).

1. Sign in to Cloudflare. Dashboard -> **Realtime** -> **TURN Server**.
2. Click **Create TURN App**, name it.
3. Copy the **Turn Token ID** and **API Token**. The API token is shown once. Lose it and you regenerate.

### 3. RunPod secrets

In the RunPod console, go to **Secrets** -> **Create secret** and add three:

| Secret name | Value |
|---|---|
| `HF_TOKEN` | HuggingFace Read token |
| `TURN_KEY_ID` | Cloudflare Turn Token ID |
| `TURN_KEY_API_TOKEN` | Cloudflare API Token |

### 4. Pod template

Create a Pod template (Templates -> New Template). Settings:

- **Type**: Pod
- **Compute**: NVIDIA GPU
- **Container image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Container disk**: 20 GB
- **Volume disk**: 40 GB
- **Volume mount path**: `/workspace`

**Container start command**:

```bash
bash -c "curl -sL https://raw.githubusercontent.com/camjac251/Personaplex-runpod/main/start.sh -o /workspace/start.sh && chmod +x /workspace/start.sh && /workspace/start.sh & /start.sh"
```

**HTTP ports**: `8998` (PersonaPlex). `8888` (JupyterLab) is optional.

**TCP ports**: none.

**Environment variables**:

| Name | Value |
|---|---|
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
| `TURN_KEY_ID` | `{{ RUNPOD_SECRET_TURN_KEY_ID }}` |
| `TURN_KEY_API_TOKEN` | `{{ RUNPOD_SECRET_TURN_KEY_API_TOKEN }}` |

### 5. Launch and connect

Pick a GPU with at least 12 GB VRAM (RTX 4090 / A6000 / L40S all work). Start the pod with the template above.

First boot downloads ~14 GB of weights and voice prompts. Expect 30-60 minutes depending on the data centre. The volume disk caches them, so subsequent boots reach "ready" in under a minute.

When the server log prints `Serving embedded web client (no build required)`, open the proxy URL from the pod (looks like `https://<pod-id>-8998.proxy.runpod.net/`). Click **Connect**, allow microphone access, and speak.

To confirm TURN is doing its job: open `chrome://webrtc-internals` in another tab while a session is live. The active candidate pair under `selectedCandidatePairId` should have `relayProtocol: tcp` or `udp` and a remote address pointing at `turn.cloudflare.com`. If it shows `host` or `srflx` and you see no audio, TURN didn't engage.

## Voices

Pre-packaged voice embeddings:

- **Natural (female)**: NATF0, NATF1, NATF2, NATF3
- **Natural (male)**: NATM0, NATM1, NATM2, NATM3
- **Variety (female)**: VARF0 through VARF4
- **Variety (male)**: VARM0 through VARM4

You can also upload 10-30 s of clean audio for any speaker via the **Clone a voice** panel. Mono or stereo, any common format. The model uses it as a voice prefix and continues in that timbre. Not zero-shot perfect, but recognisable.

## Hardware

Tested on RTX 4090 (24 GB) with the default RunPod driver. Any modern NVIDIA card with 12 GB+ VRAM should work. Smaller cards can run with CPU offload at the cost of latency.

## Architecture notes

Audio path:

1. Browser captures mic via `getUserMedia` and sends Opus-encoded frames over `RTCPeerConnection`.
2. Server (aiortc) decodes to 48 kHz, resamples to Mimi's 24 kHz, feeds the inference pipeline. GPU work runs in a thread executor so the asyncio loop stays responsive.
3. TTS PCM goes back the same way: 24 kHz -> 48 kHz -> Opus -> `<audio>` element in the browser.
4. A `RTCDataChannel` labelled `control` carries the session config (voice, sampling parameters, prompts) and streams text tokens for the transcript.

Browser AEC, noise suppression, and AGC handle echo and ambient noise. Backgrounded tabs keep playing AI audio because WebRTC is treated as active media by the browser.

Single-session: `self.lock` in `ServerState` enforces one peer connection at a time. A second connect attempt while a session is live returns HTTP 409 `session_busy` instead of hanging.

## Known issues

These come from the upstream model, not the RunPod packaging:

- **Response looping**: under certain prompts the model can repeat itself. The repetition penalty / context window sliders in the Advanced panel usually break the loop.
- **Pipeline efficiency**: GPU utilisation is occasionally spiky; some kernels are not yet optimised.

Core model issues belong upstream at [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex/issues). Bugs in the RunPod packaging or WebRTC client belong here.

## Local dev

If you want to run outside RunPod (LAN only, no TURN required since both peers can reach each other directly):

```bash
uv sync --frozen
uv run moshi-server --host 127.0.0.1 --port 8998 --static none --voice-prompt-dir voices
```

Voice prompts need to be downloaded manually (see `start.sh` for the HuggingFace pull recipe) or symlinked from a previous RunPod volume.

Run the resampler smoke tests:

```bash
uv run python moshi/tests/test_rtc_resampler.py
```

## License

Code: MIT. Model weights: NVIDIA Open Model License.

## Citation

```bibtex
@article{roy2026personaplex,
  title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models},
  author={Roy, Rajarshi and Raiman, Jonathan and Lee, Sang-gil and Ene, Teodor-Dumitru and Kirby, Robert and Kim, Sungwon and Kim, Jaehyeon and Catanzaro, Bryan},
  year={2026}
}
```
