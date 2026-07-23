"""Checks for the attention-sink ring KV cache.

The sink reserves the first ``sink`` slots of ``RingKVCache`` as never-evicted
anchors while the rest of the cache rolls. ``sink=0`` must be byte-identical to
the plain ring; ``sink>0`` must keep the anchor slots intact across wraparound,
report their true absolute positions, and keep them attendable past the window.

Run directly: ``uv run python moshi/tests/test_sink_kv.py``.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "moshi")

from moshi.modules.transformer import RingKVCache  # noqa: E402


def _feed(cache: RingKVCache, step: int) -> torch.Tensor:
    """Write one frame whose key/value encode its absolute step, return positions."""
    k = torch.full((1, 1, 1, 1), float(step))
    v = k.clone()
    _keys, _values, positions = cache.complete(k, v)
    return positions.reshape(-1)


def test_sink_zero_matches_plain_ring() -> None:
    capacity = 8
    plain = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32)
    sink0 = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=0)
    for step in range(3 * capacity + 5):
        pos_plain = _feed(plain, step)
        pos_sink = _feed(sink0, step)
        assert torch.equal(plain.cache, sink0.cache), step
        assert torch.equal(plain.end_offset, sink0.end_offset), step
        assert torch.equal(pos_plain, pos_sink), step


def test_sink_slots_never_evicted() -> None:
    capacity = 8
    sink = 3
    cache = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    for step in range(4 * capacity):
        _feed(cache, step)
        if step + 1 >= sink:
            stored = cache.cache[0].reshape(-1)
            for j in range(sink):
                # Slot j holds the j-th frame forever, and its value equals j.
                assert float(stored[j]) == float(j), (step, j, float(stored[j]))


def test_sink_positions_are_absolute_and_visible() -> None:
    capacity = 8
    sink = 2
    context = capacity
    cache = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    for step in range(4 * capacity):
        positions = _feed(cache, step)
        pos_q = step
        delta = pos_q - positions
        is_sink = (positions >= 0) & (positions < sink)
        within = (delta < context) | is_sink
        visible = (positions >= 0) & (delta >= 0) & within
        visible_positions = positions[visible].tolist()
        # Every filled sink position stays visible with its true absolute value.
        for j in range(min(sink, step + 1)):
            assert j in visible_positions, (step, j, sorted(visible_positions))
        # The most recent frame is always visible and never in the future.
        assert step in visible_positions, (step, sorted(visible_positions))
        assert all(0 <= vp <= step for vp in visible_positions), (
            step,
            sorted(visible_positions),
        )


def test_sink_multi_token_write_freezes_anchor() -> None:
    capacity = 8
    sink = 3
    cache = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    step = 0
    for chunk in (2, 2, 3, 1, 4, 2, 5):
        k = torch.arange(step, step + chunk, dtype=torch.float32).view(1, 1, chunk, 1)
        v = k.clone()
        cache.complete(k, v)
        step += chunk
        if int(cache.end_offset) >= sink:
            stored = cache.cache[0].reshape(-1)
            for j in range(sink):
                assert float(stored[j]) == float(j), (step, j, float(stored[j]))


def test_sink_survives_snapshot_restore() -> None:
    capacity = 8
    sink = 3
    src = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    for step in range(3 * capacity + 2):
        _feed(src, step)
    snapshot = {key: value.detach().clone() for key, value in src.asdict().items()}
    assert set(snapshot) == {"cache", "end_offset"}

    dst = RingKVCache(1, 1, 1, capacity, device="cpu", dtype=torch.float32, sink=sink)
    dst.cache.copy_(snapshot["cache"])
    dst.end_offset.copy_(snapshot["end_offset"])
    # Position/visibility bookkeeping after restore matches the source's next step.
    next_step = int(src.end_offset)
    pos_src = _feed(src, next_step)
    pos_dst = _feed(dst, next_step)
    assert torch.equal(pos_src, pos_dst)
    assert torch.equal(src.cache, dst.cache)


def test_sink_out_of_range_raises() -> None:
    for bad in (8, 9, -1):
        try:
            RingKVCache(1, 1, 1, 8, device="cpu", dtype=torch.float32, sink=bad)
        except ValueError:
            continue
        raise AssertionError(f"sink={bad} should have been rejected")


if __name__ == "__main__":
    tests = (
        test_sink_zero_matches_plain_ring,
        test_sink_slots_never_evicted,
        test_sink_positions_are_absolute_and_visible,
        test_sink_multi_token_write_freezes_anchor,
        test_sink_survives_snapshot_restore,
        test_sink_out_of_range_raises,
    )
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all sink KV tests passed")
