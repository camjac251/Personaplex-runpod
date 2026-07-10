"""Branch checks for live acoustic sampling controls.

Run directly: ``uv run python moshi/tests/test_lm_controls.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "moshi")

from moshi.models.lm import LMGen  # noqa: E402


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
    lm_gen._audio_temperature = torch.tensor(0.8)
    lm_gen._streaming_state = SimpleNamespace(graphed_depth=graph)
    return lm_gen, graph


def test_temperature_updates_graph_input_without_reset() -> None:
    lm_gen, graph = _bare_lm_gen()
    changed = lm_gen.set_audio_sampling(1.1, 250)
    assert changed is False
    assert lm_gen.temp == 1.1
    assert torch.isclose(lm_gen._audio_temperature, torch.tensor(1.1))
    assert graph.resets == []


def test_top_k_invalidates_only_depformer_graph() -> None:
    lm_gen, graph = _bare_lm_gen()
    changed = lm_gen.set_audio_sampling(0.7, 512)
    assert changed is True
    assert lm_gen.temp == 0.7
    assert lm_gen.top_k == 512
    assert graph.resets == [0]


if __name__ == "__main__":
    tests = [
        test_temperature_updates_graph_input_without_reset,
        test_top_k_invalidates_only_depformer_graph,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all LM control tests passed")
