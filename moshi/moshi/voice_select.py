"""Best-of-N voice reference window selection by speaker similarity.

When a voice reference clip primes the model through a window shorter than
the clip, the slice position is a free choice. ``select_voice_window``
scores every candidate window against the whole clip and returns the start
of the most speaker-representative one; the exact tail window is always a
candidate, and ties or scoring failures fall back to it, so the selector
can never do worse than the plain tail slice.

The bundled embedder wraps the ``microsoft/wavlm-base-plus-sv`` speaker-
verification model from ``transformers``. transformers is an optional
dependency, deliberately absent from the project manifest: the embedder
activates only when the package is installed in the environment
(``uv pip install transformers``), and ``wavlm_embedder`` returns ``None``
otherwise so callers degrade to tail behavior. The model itself loads
lazily on the first embedding call, on the same device policy the ASR
engine uses (CUDA when the requested device is CUDA and available, else
CPU), and any load or inference failure logs one warning and degrades the
selection to the tail window.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from typing import Callable, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)

WAVLM_MODEL_ID = "microsoft/wavlm-base-plus-sv"
# WavLM is trained on 16 kHz input; clips arrive at Mimi's rate and are
# resampled before embedding.
WAVLM_SAMPLE_RATE = 16_000

EmbedFn = Callable[[np.ndarray, int], np.ndarray]


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample a mono float array between sample rates.

    Approximate by design: the speaker-verification embedder is robust to
    mild resampling artifacts, so a dependency-free linear interpolation is
    enough to feed it. Returns the input unchanged when the rates already
    match or the input is empty.
    """
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[-1] / float(src_rate)
    dst_len = int(round(duration * dst_rate))
    if dst_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0.0, audio.shape[-1] - 1, num=dst_len, dtype=np.float64)
    resampled = np.interp(
        src_idx, np.arange(audio.shape[-1], dtype=np.float64), audio
    )
    return resampled.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


def select_voice_window(
    wav: np.ndarray,
    sample_rate: int,
    window_samples: int,
    embed_fn: EmbedFn,
) -> int:
    """Pick the start of the window most similar to the whole clip.

    ``wav`` is a mono float array, ``window_samples`` the slice length the
    caller will keep, and ``embed_fn`` maps ``(mono_window, sample_rate)``
    to an embedding vector. Candidate windows slide at a hop of half the
    window, and the exact tail window is always a candidate so the selector
    can reproduce the plain tail slice. The winner is the candidate whose
    embedding has the highest cosine similarity to the full-clip embedding;
    a tie for the top score, a non-finite score, or any embedding failure
    returns the tail start instead. Clips no longer than the window return
    start 0 without embedding anything.
    """
    mono = np.asarray(wav).reshape(-1)
    total = int(mono.shape[-1])
    window = int(window_samples)
    if window <= 0 or total <= window:
        return 0

    tail_start = total - window
    hop = max(1, window // 2)
    starts = list(range(0, tail_start + 1, hop))
    if starts[-1] != tail_start:
        starts.append(tail_start)

    try:
        reference = np.asarray(embed_fn(mono, sample_rate), dtype=np.float64)
        reference = reference.reshape(-1)
        scores = np.asarray(
            [
                _cosine_similarity(
                    reference,
                    np.asarray(
                        embed_fn(mono[start : start + window], sample_rate),
                        dtype=np.float64,
                    ).reshape(-1),
                )
                for start in starts
            ]
        )
    except Exception as exc:
        logger.warning(
            "voice window scoring failed (%s: %s); keeping the tail slice",
            type(exc).__name__,
            exc,
        )
        return tail_start

    if not np.all(np.isfinite(scores)):
        return tail_start
    best = float(scores.max())
    winners = np.flatnonzero(scores == best)
    if winners.size != 1:
        return tail_start
    return int(starts[int(winners[0])])


class _WavLMEmbedder:
    """Lazy WavLM speaker-verification embedder.

    Construction is cheap and touches neither disk nor GPU; the model loads
    on the first call. A failed load is remembered so exactly one warning is
    logged and later calls fail fast, which callers of
    ``select_voice_window`` observe as the tail fallback.
    """

    def __init__(self, device: str | torch.device):
        self._requested_device = device
        self._model = None
        self._extractor = None
        self._device: Optional[torch.device] = None
        self._load_failed = False

    def _ensure_loaded(self) -> None:
        if self._load_failed:
            raise RuntimeError("WavLM embedder failed to load")
        if self._model is not None:
            return
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            requested = torch.device(self._requested_device)
            if requested.type == "cuda" and torch.cuda.is_available():
                target = torch.device("cuda")
            else:
                target = torch.device("cpu")
            t = time.monotonic()
            extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
            model = WavLMForXVector.from_pretrained(WAVLM_MODEL_ID)
            model.eval()
            model.to(target)
            self._extractor = extractor
            self._model = model
            self._device = target
            logger.info(
                "voice-picker embedder %r loaded on %s in %.1f s",
                WAVLM_MODEL_ID,
                target,
                time.monotonic() - t,
            )
        except Exception as exc:
            self._load_failed = True
            logger.warning(
                "voice-picker embedder %r failed to load (%s: %s); voice "
                "window selection keeps the tail slice",
                WAVLM_MODEL_ID,
                type(exc).__name__,
                exc,
            )
            raise

    def __call__(self, wav: np.ndarray, sample_rate: int) -> np.ndarray:
        self._ensure_loaded()
        mono = np.asarray(wav, dtype=np.float32).reshape(-1)
        resampled = _resample_linear(mono, int(sample_rate), WAVLM_SAMPLE_RATE)
        inputs = self._extractor(
            [resampled], sampling_rate=WAVLM_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self._model(**inputs).embeddings
        return embeddings[0].detach().cpu().numpy()


def wavlm_embedder(device: str | torch.device) -> Optional[_WavLMEmbedder]:
    """Return a lazy WavLM embedder, or None when transformers is absent.

    Mirrors the guarded ``faster_whisper`` pattern: the availability probe
    imports nothing heavy, and a missing package is logged and degrades
    gracefully to tail slicing rather than breaking the server.
    """
    if importlib.util.find_spec("transformers") is None:
        logger.warning(
            "voice window selection requested but transformers is not "
            "installed; keeping tail slices. Install with "
            "`uv pip install transformers` to enable it."
        )
        return None
    return _WavLMEmbedder(device)
