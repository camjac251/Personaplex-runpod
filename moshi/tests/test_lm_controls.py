"""Branch checks for live acoustic sampling controls.

Run directly: ``uv run python moshi/tests/test_lm_controls.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import (  # noqa: E402
    DEFAULT_SEMANTIC_TEMPERATURE_CAP,
    REPETITION_TURN_BREAK_FRAMES,
    LMGen,
)
from moshi.utils.sampling import (  # noqa: E402
    sample_token,
    sample_top_k_dynamic,
)


class _Graph:
    def __init__(self) -> None:
        self.resets: list[int] = []

    def reset(self, warmup_steps: int = 0) -> None:
        self.resets.append(warmup_steps)


def _bare_lm_gen() -> tuple[LMGen, _Graph]:
    graph = _Graph()
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.temp = 0.8
    lm_gen.top_k = 250
    lm_gen.semantic_temperature_cap = DEFAULT_SEMANTIC_TEMPERATURE_CAP
    lm_gen._audio_temperature = torch.full((8,), 0.8)
    lm_gen._audio_top_k = torch.tensor(250, dtype=torch.long)
    lm_gen._streaming_state = SimpleNamespace(graphed_depth=graph)
    return lm_gen, graph


def test_temperature_updates_graph_input_without_reset() -> None:
    lm_gen, graph = _bare_lm_gen()
    changed = lm_gen.set_audio_sampling(1.1, 250)
    assert changed is False
    assert lm_gen.temp == 1.1
    # Same tensor object, same shape: the graph replays the new values.
    assert lm_gen._audio_temperature.shape == (8,)
    assert torch.allclose(
        lm_gen._audio_temperature[1:], torch.full((7,), 1.1)
    )
    assert graph.resets == []


def test_semantic_codebook_temperature_is_capped() -> None:
    lm_gen, _ = _bare_lm_gen()

    # Hot acoustic settings leave the semantic level at the cap.
    lm_gen.set_audio_sampling(1.2, 250)
    assert torch.isclose(
        lm_gen._audio_temperature[0],
        torch.tensor(DEFAULT_SEMANTIC_TEMPERATURE_CAP),
    )
    assert torch.allclose(
        lm_gen._audio_temperature[1:], torch.full((7,), 1.2)
    )

    # At or below the cap every level matches the request exactly.
    lm_gen.set_audio_sampling(0.6, 250)
    assert torch.allclose(
        lm_gen._audio_temperature, torch.full((8,), 0.6)
    )


def test_min_p_masks_low_probability_text_tokens() -> None:
    # Token 2 dominates; token 0 sits far below half of its probability.
    logits = torch.tensor([[0.0, 2.2, 3.0]], dtype=torch.float32)
    torch.manual_seed(7)
    with_min_p = {
        sample_token(logits, True, 1.0, 0, min_p=0.5).item()
        for _ in range(200)
    }
    assert 0 not in with_min_p
    assert 2 in with_min_p

    # Disabled min-p keeps the full distribution reachable.
    torch.manual_seed(7)
    without = {
        sample_token(logits, True, 1.0, 0, min_p=0.0).item()
        for _ in range(400)
    }
    assert 0 in without


def test_top_k_updates_graph_input_without_reset() -> None:
    lm_gen, graph = _bare_lm_gen()
    changed = lm_gen.set_audio_sampling(0.7, 512)
    assert changed is True
    assert lm_gen.temp == 0.7
    assert lm_gen.top_k == 512
    assert lm_gen._audio_top_k.item() == 512
    assert graph.resets == []


def test_dynamic_top_k_masks_candidates_without_graph_shape_changes() -> None:
    probs = torch.tensor([[0.05, 0.15, 0.80]], dtype=torch.float32)
    for _ in range(20):
        assert sample_top_k_dynamic(probs, torch.tensor(1)).item() == 2

    torch.manual_seed(1234)
    top_two = {
        sample_top_k_dynamic(probs, torch.tensor(2)).item()
        for _ in range(100)
    }
    assert top_two <= {1, 2}
    assert 0 not in top_two

    # Zero retains the legacy "no top-k limit" meaning; oversized values
    # clamp to the fixed vocabulary size.
    for k in (0, 99):
        samples = {
            sample_top_k_dynamic(probs, torch.tensor(k)).item()
            for _ in range(200)
        }
        assert samples <= {0, 1, 2}


def _ring_lm_gen(ctx: int = 4) -> tuple[LMGen, SimpleNamespace]:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.repetition_penalty_context = ctx
    state = SimpleNamespace(
        recent_text_tokens=torch.full((1, 8), -1, dtype=torch.long),
        recent_text_offset=torch.zeros(1, dtype=torch.long),
        repetition_pad_streak=torch.zeros(1, dtype=torch.long),
    )
    lm_gen._streaming_state = state
    return lm_gen, state


def test_repetition_ring_is_turn_scoped() -> None:
    lm_gen, state = _ring_lm_gen()

    def step(token: int) -> None:
        lm_gen._update_repetition_ring(torch.tensor([token], dtype=torch.long))

    # Meaningful tokens fill the ring; PAD (3) and EPAD (0) never enter it.
    for token in (11, 0, 12, 3, 13):
        step(token)
    ring = state.recent_text_tokens[0].tolist()
    assert {11, 12, 13} <= set(ring), ring
    assert 0 not in ring and 3 not in ring, ring

    # An inter-word pad gap shorter than a turn break keeps the ring.
    for _ in range(REPETITION_TURN_BREAK_FRAMES - 1):
        step(3)
    step(14)
    ring = state.recent_text_tokens[0].tolist()
    assert {11, 12, 13, 14} <= set(ring), ring

    # A sustained natural pad run marks a turn boundary: the next turn's
    # first word starts against an empty ring.
    for _ in range(REPETITION_TURN_BREAK_FRAMES + 1):
        step(3)
    step(15)
    ring = state.recent_text_tokens[0].tolist()
    assert 15 in ring, ring
    assert not ({11, 12, 13, 14} & set(ring)), ring
    assert state.recent_text_offset.item() == 1


def test_forced_pads_do_not_break_the_turn() -> None:
    lm_gen, state = _ring_lm_gen()

    def step(token: int, *, forced: bool = False) -> None:
        lm_gen._update_repetition_ring(
            torch.tensor([token], dtype=torch.long), pad_was_forced=forced
        )

    for token in (11, 12, 13):
        step(token)

    # A max-turn cap trip forces a full turn-break's worth of PAD frames.
    # Forced silence is not the model yielding: the streak stays frozen and
    # the ring survives, so the repetition penalty still applies when the
    # model resumes.
    for _ in range(REPETITION_TURN_BREAK_FRAMES * 2):
        step(3, forced=True)
    step(14)
    ring = state.recent_text_tokens[0].tolist()
    assert {11, 12, 13, 14} <= set(ring), ring

    # Natural pads after the forced window still accumulate to a boundary.
    for _ in range(REPETITION_TURN_BREAK_FRAMES + 1):
        step(3)
    step(15)
    ring = state.recent_text_tokens[0].tolist()
    assert 15 in ring, ring
    assert not ({11, 12, 13, 14} & set(ring)), ring


def test_new_turn_clears_history_before_penalty() -> None:
    lm_gen, state = _ring_lm_gen()
    lm_gen.repetition_penalty = 2.0
    state.recent_text_tokens[0, 0] = 1
    state.recent_text_offset.fill_(1)
    state.repetition_pad_streak.fill_(REPETITION_TURN_BREAK_FRAMES)
    # Token 1 wins without a penalty. If stale history leaked into the new
    # turn it would be divided to 1.0 and token 2 would win at 1.5.
    logits = torch.tensor([[[[0.0, 2.0, 1.5, 0.0]]]])
    penalized = lm_gen._apply_text_repetition_penalty(logits)
    assert penalized.argmax(dim=-1).item() == 1
    assert state.recent_text_tokens.eq(-1).all()


def test_interrupt_force_window_works_with_turn_cap_disabled() -> None:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.max_turn_text_tokens = 0
    lm_gen._pad_force_remaining = 2
    natural = torch.tensor([17], dtype=torch.long)

    first, forced = lm_gen._consume_forced_pad(
        natural, 3, text_was_forced=False
    )
    assert forced is True and first.item() == 3
    second, forced = lm_gen._consume_forced_pad(
        natural, 3, text_was_forced=False
    )
    assert forced is True and second.item() == 3
    third, forced = lm_gen._consume_forced_pad(
        natural, 3, text_was_forced=False
    )
    assert forced is False and third.item() == 17


def test_turn_cap_counts_across_short_natural_pauses() -> None:
    lm_gen = LMGen.__new__(LMGen)
    lm_gen.max_turn_text_tokens = 3
    lm_gen._non_pad_streak = 0
    lm_gen._turn_pad_streak = 0
    lm_gen._pad_force_remaining = 0

    def account(token: int, *, forced_text: bool = False) -> None:
        lm_gen._update_turn_cap(
            torch.tensor([token], dtype=torch.long),
            3,
            text_was_forced=forced_text,
            turn_pad_forced=False,
        )

    account(11)
    account(3)
    account(12)
    account(3)
    assert lm_gen._pad_force_remaining == 0
    account(13)
    assert lm_gen._pad_force_remaining == REPETITION_TURN_BREAK_FRAMES
    assert lm_gen._non_pad_streak == 0

    # A sustained natural PAD run is a true boundary, so prior words no
    # longer count toward the next turn's cap.
    lm_gen._pad_force_remaining = 0
    account(21)
    account(22)
    for _ in range(REPETITION_TURN_BREAK_FRAMES):
        account(3)
    account(23)
    assert lm_gen._pad_force_remaining == 0
    assert lm_gen._non_pad_streak == 1

    # External text injection/Stop padding never advances either counter.
    account(24, forced_text=True)
    assert lm_gen._non_pad_streak == 1


if __name__ == "__main__":
    tests = [
        test_temperature_updates_graph_input_without_reset,
        test_semantic_codebook_temperature_is_capped,
        test_min_p_masks_low_probability_text_tokens,
        test_top_k_updates_graph_input_without_reset,
        test_dynamic_top_k_masks_candidates_without_graph_shape_changes,
        test_repetition_ring_is_turn_scoped,
        test_forced_pads_do_not_break_the_turn,
        test_new_turn_clears_history_before_penalty,
        test_interrupt_force_window_works_with_turn_cap_disabled,
        test_turn_cap_counts_across_short_natural_pauses,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all LM control tests passed")
