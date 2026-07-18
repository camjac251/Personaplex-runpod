"""CPU checks for caption-CFG guided text sampling and branch bookkeeping.

Caption-CFG runs two streaming rows: row 0 conditions on injected context,
row 1 is the clean counterfactual. These tests drive
``LMGen.process_transformer_output`` directly with crafted logits and a
fake depformer graph, so the guidance math, per-row forced windows, and
the shared-audio row sync are all verified without a GPU. Run directly:
``uv run python moshi/tests/test_caption_cfg.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import LMGen  # noqa: E402

PAD_ID = 3
VOCAB = 4
DEP_Q = 8
CACHE_T = 4
TARGET_POS = 3
MODEL_INPUT_POS = 2


class _FakeLmModel:
    text_padding_token_id = PAD_ID
    end_of_text_padding_id = 0
    dep_q = DEP_Q
    audio_offset = 1


def _fake_depth(rows: torch.Tensor):
    def graphed_depth(
        next_text_token,
        transformer_out,
        audio_tokens,
        audio_provided,
        audio_temperature,
        audio_top_k,
    ):
        assert next_text_token.shape == (2,)
        return rows

    return graphed_depth


def _cfg_lm_gen(depth_rows: torch.Tensor) -> LMGen:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.lm_model = _FakeLmModel()
    lm_gen.caption_cfg = True
    lm_gen.cfg_gamma = 1.0
    lm_gen.use_sampling = False
    lm_gen.temp_text = 0.7
    lm_gen.top_k_text = 25
    lm_gen.min_p_text = 0.0
    lm_gen.repetition_penalty = 1.0
    lm_gen.repetition_penalty_context = 8
    lm_gen.padding_bonus = 0.0
    lm_gen.max_turn_text_tokens = 0
    lm_gen._non_pad_streak = 0
    lm_gen._turn_pad_streak = 0
    lm_gen._pad_force_remaining = 0
    lm_gen.report_loss = False
    lm_gen.return_logits = False
    lm_gen._audio_temperature = torch.full((DEP_Q,), 0.8)
    lm_gen._audio_top_k = torch.tensor(250, dtype=torch.long)
    lm_gen.max_delay = 0
    lm_gen.delays_cuda = torch.zeros(DEP_Q + 9, dtype=torch.long)
    lm_gen._streaming_state = SimpleNamespace(
        cache=torch.zeros(2, DEP_Q + 9, CACHE_T, dtype=torch.long),
        provided=torch.zeros(2, DEP_Q + 9, CACHE_T, dtype=torch.bool),
        offset=TARGET_POS,
        graphed_depth=_fake_depth(depth_rows),
        recent_text_tokens=torch.full((2, 8), -1, dtype=torch.long),
        recent_text_offset=torch.zeros(2, dtype=torch.long),
        repetition_pad_streak=torch.zeros(2, dtype=torch.long),
    )
    return lm_gen


def _text_logits(cond: list[float], uncond: list[float]) -> torch.Tensor:
    logits = torch.zeros(2, 1, 1, VOCAB, dtype=torch.float32)
    logits[0, 0, 0] = torch.tensor(cond)
    logits[1, 0, 0] = torch.tensor(uncond)
    return logits


def _natural_slices() -> tuple[torch.Tensor, torch.Tensor]:
    provided = torch.zeros(2, DEP_Q + 9, 1, dtype=torch.bool)
    target = torch.zeros(2, DEP_Q + 9, 1, dtype=torch.long)
    return provided, target


def _run(lm_gen: LMGen, logits, provided, target, *, forced: bool):
    return lm_gen.process_transformer_output(
        torch.zeros(2, 1, 8),
        logits,
        provided,
        target,
        MODEL_INPUT_POS,
        TARGET_POS,
        text_was_forced=forced,
    )


def test_gamma_amplifies_the_context_delta() -> None:
    cond = [0.0, 1.1, 1.0, 0.0]  # conditional narrowly prefers token 1
    uncond = [0.0, 1.2, 0.0, 0.0]  # counterfactual has no interest in 2

    # gamma 1.0 is exactly the conditional branch.
    depth = torch.zeros(2, DEP_Q, dtype=torch.long)
    lm_gen = _cfg_lm_gen(depth)
    _run(lm_gen, _text_logits(cond, uncond), *_natural_slices(), forced=False)
    chosen = lm_gen._streaming_state.cache[:, 0, TARGET_POS]
    assert chosen.tolist() == [1, 1], chosen

    # Amplified guidance surfaces the token the context uniquely raised
    # (delta on token 2 is +1.0 while token 1's delta is negative).
    lm_gen = _cfg_lm_gen(depth)
    lm_gen.cfg_gamma = 3.0
    _run(lm_gen, _text_logits(cond, uncond), *_natural_slices(), forced=False)
    chosen = lm_gen._streaming_state.cache[:, 0, TARGET_POS]
    assert chosen.tolist() == [2, 2], chosen


def test_forced_window_routes_context_to_row_zero_only() -> None:
    depth = torch.zeros(2, DEP_Q, dtype=torch.long)
    lm_gen = _cfg_lm_gen(depth)
    provided, target = _natural_slices()
    provided[:, 0, 0] = True
    target[0, 0, 0] = 41  # context token, conditional row only
    target[1, 0, 0] = PAD_ID
    _run(
        lm_gen,
        _text_logits([9.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]),
        provided,
        target,
        forced=True,
    )
    state = lm_gen._streaming_state
    assert state.cache[:, 0, TARGET_POS].tolist() == [41, PAD_ID]
    # Forced frames never enter the repetition ring.
    assert state.recent_text_tokens.eq(-1).all()


def test_rows_share_row_zero_audio_history() -> None:
    depth = torch.stack(
        [
            torch.arange(100, 100 + DEP_Q, dtype=torch.long),
            torch.arange(200, 200 + DEP_Q, dtype=torch.long),
        ]
    )
    lm_gen = _cfg_lm_gen(depth)
    _run(
        lm_gen,
        _text_logits([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        *_natural_slices(),
        forced=False,
    )
    audio = lm_gen._streaming_state.cache[:, 1 : DEP_Q + 1, TARGET_POS]
    # Row 1's own sample (200..) is overwritten by row 0's codes so both
    # branches hear the same assistant audio.
    assert audio[0].tolist() == list(range(100, 100 + DEP_Q))
    assert audio[1].tolist() == list(range(100, 100 + DEP_Q))


def test_repetition_penalty_applies_to_the_single_guided_row() -> None:
    depth = torch.zeros(2, DEP_Q, dtype=torch.long)
    lm_gen = _cfg_lm_gen(depth)
    lm_gen.repetition_penalty = 1.5
    state = lm_gen._streaming_state
    state.recent_text_tokens[:, 0] = 1  # token 1 was recently emitted
    state.recent_text_offset.fill_(1)
    # The guided logits are a single row while the ring keeps one row per
    # branch; the penalty must run on the sliced ring (an unsliced ring
    # makes the gather raise on the row-count mismatch). Token 1 leads raw
    # (3.0 vs 2.5) but the penalty divides it to 2.0, so token 0 wins.
    _run(
        lm_gen,
        _text_logits([2.5, 3.0, 0.0, 0.0], [2.5, 3.0, 0.0, 0.0]),
        *_natural_slices(),
        forced=False,
    )
    chosen = lm_gen._streaming_state.cache[:, 0, TARGET_POS]
    assert chosen.tolist() == [0, 0], chosen


def test_gamma_one_does_not_mutate_the_graph_logits_buffer() -> None:
    depth = torch.zeros(2, DEP_Q, dtype=torch.long)
    lm_gen = _cfg_lm_gen(depth)
    lm_gen.padding_bonus = 2.0
    logits = _text_logits([1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0])
    before = logits.clone()
    _run(lm_gen, logits, *_natural_slices(), forced=False)
    # The padding bonus lands on a clone of the guided row, never on the
    # tensor that aliases the CUDA-graph output buffer.
    assert torch.equal(logits, before)


if __name__ == "__main__":
    tests = [
        test_gamma_amplifies_the_context_delta,
        test_forced_window_routes_context_to_row_zero_only,
        test_rows_share_row_zero_audio_history,
        test_repetition_penalty_applies_to_the_single_guided_row,
        test_gamma_one_does_not_mutate_the_graph_logits_buffer,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all caption-CFG tests passed")
