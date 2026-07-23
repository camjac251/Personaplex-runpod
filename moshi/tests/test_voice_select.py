"""CPU checks for best-of-N voice reference window selection.

Run directly: ``uv run python moshi/tests/test_voice_select.py``.

All tests drive ``select_voice_window`` with stub embedding functions; no
network access and no speaker model are involved.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "moshi")

from moshi.models.lm import LMGen  # noqa: E402
from moshi.voice_select import select_voice_window  # noqa: E402


SAMPLE_RATE = 1_000


def _mix_embed(wav: np.ndarray, sr: int) -> np.ndarray:
    """Amplitude-mix signature: positive vs negative sample counts.

    Cosine similarity against the full clip then favors the window whose
    positive/negative mix matches the whole clip's, independent of window
    length.
    """
    wav = np.asarray(wav)
    return np.array(
        [float(np.count_nonzero(wav > 0)), float(np.count_nonzero(wav < 0))],
        dtype=np.float64,
    )


def test_selection_picks_most_representative_window() -> None:
    # Four 1 s segments: all-positive, all-negative, a 65/35 tiled mix, and
    # all-positive again at the tail. The full clip mixes 2650 positive to
    # 1350 negative samples, so the tiled segment (650/350, the same ratio
    # shape) is the most representative window, while the tail window is
    # all-positive and scores far lower.
    seg_a = np.full(1_000, 0.5, dtype=np.float32)
    seg_b = np.full(1_000, -0.5, dtype=np.float32)
    tile = np.concatenate(
        [np.full(13, 0.5, dtype=np.float32), np.full(7, -0.5, dtype=np.float32)]
    )
    seg_c = np.tile(tile, 50)
    seg_d = np.full(1_000, 0.5, dtype=np.float32)
    clip = np.concatenate([seg_a, seg_b, seg_c, seg_d])

    start = select_voice_window(clip, SAMPLE_RATE, 1_000, _mix_embed)

    assert start == 2_000
    assert start != clip.size - 1_000


def test_tail_window_is_candidate_and_can_win() -> None:
    # 4321 samples with a 1000-sample window: the exact tail start (3321)
    # is not on the 500-sample hop grid, so it must be appended explicitly.
    # The stub scores only the exact tail window close to the full clip.
    total = 4_321
    window = 1_000
    tail_start = total - window
    clip = np.arange(total, dtype=np.float32)
    seen_starts: list[int] = []

    def embed(wav: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(wav)
        if wav.size == total:
            return np.array([1.0, 0.0])
        start = int(wav[0])
        seen_starts.append(start)
        if start == tail_start:
            return np.array([1.0, 0.1])
        return np.array([0.0, 1.0])

    start = select_voice_window(clip, SAMPLE_RATE, window, embed)

    assert tail_start in seen_starts
    assert start == tail_start


def test_scoring_failure_falls_back_to_tail() -> None:
    clip = np.arange(4_000, dtype=np.float32)

    def broken_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        raise RuntimeError("embedder unavailable")

    start = select_voice_window(clip, SAMPLE_RATE, 1_000, broken_embed)

    assert start == 3_000


def test_degenerate_scores_fall_back_to_tail() -> None:
    clip = np.arange(4_000, dtype=np.float32)

    def zero_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        return np.zeros(4, dtype=np.float64)

    start = select_voice_window(clip, SAMPLE_RATE, 1_000, zero_embed)

    assert start == 3_000


def test_tied_scores_fall_back_to_tail() -> None:
    clip = np.arange(4_000, dtype=np.float32)

    def constant_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        return np.array([1.0, 1.0])

    start = select_voice_window(clip, SAMPLE_RATE, 1_000, constant_embed)

    assert start == 3_000


def test_short_clip_passes_through_without_embedding() -> None:
    calls: list[int] = []

    def recording_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        calls.append(np.asarray(wav).size)
        return np.array([1.0, 0.0])

    exact = np.ones(1_000, dtype=np.float32)
    assert select_voice_window(exact, SAMPLE_RATE, 1_000, recording_embed) == 0
    shorter = np.ones(500, dtype=np.float32)
    assert select_voice_window(shorter, SAMPLE_RATE, 1_000, recording_embed) == 0
    assert calls == []


def test_priming_slice_uses_selected_window() -> None:
    lm_gen = LMGen.__new__(LMGen)
    audio = np.arange(1_000, dtype=np.float32)[None, :]
    lm_gen.voice_prompt_audio = audio
    lm_gen.voice_prompt_strength = 0.5
    lm_gen._frame_size = 100
    lm_gen._sample_rate = SAMPLE_RATE

    # No embedder configured: the strength slice is the plain clip tail.
    lm_gen.voice_window_embedder = None
    tail = lm_gen._strength_sliced_voice_prompt_audio()
    assert np.array_equal(tail, audio[:, -500:])

    # With an embedder, the slice moves to the best-scoring window.
    def favor_250(wav: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(wav)
        if wav.size == 1_000:
            return np.array([1.0, 0.0])
        if int(wav[0]) == 250:
            return np.array([1.0, 0.1])
        return np.array([0.0, 1.0])

    lm_gen.voice_window_embedder = favor_250
    picked = lm_gen._strength_sliced_voice_prompt_audio()
    assert np.array_equal(picked, audio[:, 250:750])


def test_full_strength_and_empty_keep_never_embed() -> None:
    lm_gen = LMGen.__new__(LMGen)
    audio = np.arange(1_000, dtype=np.float32)[None, :]
    lm_gen.voice_prompt_audio = audio
    lm_gen._frame_size = 100
    lm_gen._sample_rate = SAMPLE_RATE
    calls: list[int] = []

    def recording_embed(wav: np.ndarray, sr: int) -> np.ndarray:
        calls.append(np.asarray(wav).size)
        return np.array([1.0, 0.0])

    lm_gen.voice_window_embedder = recording_embed

    # Full strength keeps the whole clip object untouched; this is also the
    # _synthesize_voice_preview path, which pins strength to 1.0.
    lm_gen.voice_prompt_strength = 1.0
    assert lm_gen._strength_sliced_voice_prompt_audio() is audio

    # Zero strength keeps nothing; there is no window to score.
    lm_gen.voice_prompt_strength = 0.0
    assert lm_gen._strength_sliced_voice_prompt_audio().shape == (1, 0)

    assert calls == []


if __name__ == "__main__":
    tests = [
        test_selection_picks_most_representative_window,
        test_tail_window_is_candidate_and_can_win,
        test_scoring_failure_falls_back_to_tail,
        test_degenerate_scores_fall_back_to_tail,
        test_tied_scores_fall_back_to_tail,
        test_short_clip_passes_through_without_embedding,
        test_priming_slice_uses_selected_window,
        test_full_strength_and_empty_keep_never_embed,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all voice select tests passed")
