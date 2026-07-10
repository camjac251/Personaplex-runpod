"""Branch checks for the vision_frame_chunk reassembly state machine.

Run directly: ``uv run python moshi/tests/test_vision_chunk.py``.
No pytest dependency to keep the project deps lean; assertions raise.
"""

from __future__ import annotations

import sys

# Allow running this script from inside the repo without installing.
sys.path.insert(0, "moshi")

from moshi.rtc_session import (  # noqa: E402
    VISION_CHUNK_MAX_PARTIALS,
    VISION_CHUNK_MAX_PARTS,
    VISION_CHUNK_STALE_SEC,
    VISION_FRAME_MAX_CHARS,
    reassemble_vision_chunk,
)


def _log_sink(entries: list) -> object:
    def log(level: str, text: str) -> None:
        entries.append((level, text))

    return log


def chunk(
    frame_id: str,
    seq: int,
    total: int,
    data: str = "x",
    detail: bool = False,
    historical_detail: bool = False,
    source_generation: int = 0,
) -> dict:
    return {
        "type": "vision_frame_chunk",
        "frame_id": frame_id,
        "seq": seq,
        "total": total,
        "data": data,
        "detail": detail,
        "historical_detail": historical_detail,
        "source_generation": source_generation,
    }


def test_in_order_completion() -> None:
    partials, logs = {}, []
    log = _log_sink(logs)
    assert reassemble_vision_chunk(partials, chunk("f1", 0, 2, "AB"), 0.0, log) is None
    done = reassemble_vision_chunk(partials, chunk("f1", 1, 2, "CD"), 0.1, log)
    assert done == {
        "type": "vision_frame",
        "frame_id": "f1",
        "data": "ABCD",
        "detail": False,
        "historical_detail": False,
        "source_generation": 0,
    }
    assert partials == {}
    assert logs == []


def test_out_of_order_completion_and_detail_flag() -> None:
    partials = {}
    log = _log_sink([])
    assert reassemble_vision_chunk(partials, chunk("f1", 1, 2, "CD", detail=True), 0.0, log) is None
    done = reassemble_vision_chunk(
        partials,
        chunk("f1", 0, 2, "AB", detail=True),
        0.1,
        log,
    )
    assert done is not None
    assert done["data"] == "ABCD"
    assert done["detail"] is True


def test_generation_and_historical_detail_round_trip() -> None:
    partials = {}
    log = _log_sink([])
    first = chunk(
        "history-1",
        0,
        2,
        "AB",
        detail=True,
        historical_detail=True,
        source_generation=17,
    )
    second = chunk(
        "history-1",
        1,
        2,
        "CD",
        detail=True,
        historical_detail=True,
        source_generation=17,
    )
    assert reassemble_vision_chunk(partials, first, 0.0, log) is None
    done = reassemble_vision_chunk(partials, second, 0.1, log)
    assert done is not None
    assert done["historical_detail"] is True
    assert done["source_generation"] == 17


def test_inconsistent_source_metadata_drops_partial() -> None:
    for changed in (
        {"source_generation": 8},
        {"historical_detail": True},
        {"detail": True},
    ):
        partials = {}
        log = _log_sink([])
        assert reassemble_vision_chunk(
            partials,
            chunk("f1", 0, 2, source_generation=7),
            0.0,
            log,
        ) is None
        next_part = {
            **chunk("f1", 1, 2, source_generation=7),
            **changed,
        }
        assert reassemble_vision_chunk(partials, next_part, 0.1, log) is None
        assert "f1" not in partials


def test_duplicate_seq_drops_partial() -> None:
    partials = {}
    log = _log_sink([])
    assert reassemble_vision_chunk(partials, chunk("f1", 0, 3), 0.0, log) is None
    assert reassemble_vision_chunk(partials, chunk("f1", 0, 3), 0.1, log) is None
    assert "f1" not in partials


def test_inconsistent_total_drops_partial() -> None:
    partials = {}
    log = _log_sink([])
    assert reassemble_vision_chunk(partials, chunk("f1", 0, 3), 0.0, log) is None
    assert reassemble_vision_chunk(partials, chunk("f1", 1, 2), 0.1, log) is None
    assert "f1" not in partials


def test_bad_parts_rejected() -> None:
    log = _log_sink([])
    bad_messages = [
        chunk("f1", 0, 1),  # total below 2
        chunk("f1", 0, VISION_CHUNK_MAX_PARTS + 1),  # total above cap
        chunk("f1", 2, 2),  # seq == total
        chunk("f1", -1, 2),  # negative seq
        chunk("", 0, 2),  # empty frame_id
        chunk("f1", 0, 2, ""),  # empty data
        {**chunk("f1", 0, 2), "seq": None},  # unparseable seq
        {**chunk("f1", 0, 2), "total": "x"},  # unparseable total
        {**chunk("f1", 0, 2), "data": 123},  # non-string data
    ]
    for msg in bad_messages:
        partials = {}
        assert reassemble_vision_chunk(partials, msg, 0.0, log) is None, msg
        assert partials == {}, msg


def test_bad_part_clears_existing_partial_for_frame() -> None:
    partials = {}
    log = _log_sink([])
    assert reassemble_vision_chunk(partials, chunk("f1", 0, 2), 0.0, log) is None
    assert "f1" in partials
    assert reassemble_vision_chunk(partials, {**chunk("f1", 1, 2), "seq": None}, 0.1, log) is None
    assert "f1" not in partials


def test_oversize_combined_frame_dropped() -> None:
    partials = {}
    log = _log_sink([])
    half = "x" * (VISION_FRAME_MAX_CHARS // 2 + 1)
    assert reassemble_vision_chunk(partials, chunk("f1", 0, 2, half), 0.0, log) is None
    assert reassemble_vision_chunk(partials, chunk("f1", 1, 2, half), 0.1, log) is None
    assert "f1" not in partials


def test_stale_partial_swept_by_other_frame() -> None:
    partials = {}
    log = _log_sink([])
    assert reassemble_vision_chunk(partials, chunk("old", 0, 2), 0.0, log) is None
    late = VISION_CHUNK_STALE_SEC + 1.0
    assert reassemble_vision_chunk(partials, chunk("new", 0, 2), late, log) is None
    assert "old" not in partials
    assert "new" in partials


def test_oldest_partial_evicted_at_capacity() -> None:
    partials = {}
    log = _log_sink([])
    for i in range(VISION_CHUNK_MAX_PARTIALS):
        assert reassemble_vision_chunk(partials, chunk(f"f{i}", 0, 2), 0.0, log) is None
    assert len(partials) == VISION_CHUNK_MAX_PARTIALS
    assert reassemble_vision_chunk(partials, chunk("overflow", 0, 2), 0.1, log) is None
    assert len(partials) == VISION_CHUNK_MAX_PARTIALS
    assert "f0" not in partials
    assert "overflow" in partials
    # The evicted frame's straggler part re-registers as a fresh partial
    # rather than completing; its first half is gone.
    assert reassemble_vision_chunk(partials, chunk("f0", 1, 2), 0.2, log) is None


def test_completion_after_eviction_still_works_for_survivors() -> None:
    partials = {}
    log = _log_sink([])
    for i in range(VISION_CHUNK_MAX_PARTIALS):
        assert reassemble_vision_chunk(partials, chunk(f"f{i}", 0, 2, f"a{i}"), 0.0, log) is None
    done = reassemble_vision_chunk(partials, chunk("f3", 1, 2, "b3"), 0.1, log)
    assert done is not None
    assert done["data"] == "a3b3"


if __name__ == "__main__":
    tests = [
        test_in_order_completion,
        test_out_of_order_completion_and_detail_flag,
        test_generation_and_historical_detail_round_trip,
        test_inconsistent_source_metadata_drops_partial,
        test_duplicate_seq_drops_partial,
        test_inconsistent_total_drops_partial,
        test_bad_parts_rejected,
        test_bad_part_clears_existing_partial_for_frame,
        test_oversize_combined_frame_dropped,
        test_stale_partial_swept_by_other_frame,
        test_oldest_partial_evicted_at_capacity,
        test_completion_after_eviction_still_works_for_survivors,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all vision chunk tests passed")
