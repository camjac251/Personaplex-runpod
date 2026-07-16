"""Checks for the GPU-hang watchdog's stall detector and its safety invariant.

Run directly: ``uv run python moshi/tests/test_hang_watchdog.py``.
No pytest dependency to keep the project deps lean; assertions raise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, "moshi")

from moshi.server import (  # noqa: E402
    INFER_STALL_CONFIRM_POLLS,
    INFER_STALL_EXIT_MS,
    _StallDetector,
)

SERVER_PY = Path(__file__).resolve().parents[1] / "moshi" / "server.py"

# The watchdog hard-exits when a phase ages past the threshold, so phase
# writes are only safe from the bounded per-frame path. These are the only
# scopes allowed to call _set_inflight_phase: the _track_inflight_frame
# decorator body ("wrapped"), the tracked-lock contextmanager, and the
# frame body itself.
ALLOWED_PHASE_SETTER_SCOPES = {
    "wrapped",
    "_tracked_inference_lock",
    "_process_audio_frame",
}

# Direct assignments to the phase fields are only safe in the setter, the
# clear, and field initialization.
ALLOWED_FIELD_WRITE_SCOPES = {
    "__init__",
    "_set_inflight_phase",
    "_clear_inflight_frame",
}

PHASE_FIELDS = {"_inflight_phase", "_inflight_phase_started_at"}


def test_detector_never_fires_when_idle() -> None:
    det = _StallDetector(threshold_ms=100.0, confirm_polls=3)
    for i in range(10):
        assert det.observe("idle", 0.0, float(i)) is False


def test_detector_never_fires_under_threshold() -> None:
    det = _StallDetector(threshold_ms=100.0, confirm_polls=3)
    # Phase 50 ms old on every poll: healthy frames churning.
    for i in range(10):
        now = float(i)
        assert det.observe("gpu_sync_to_cpu", now - 0.05, now) is False


def test_detector_fires_on_confirmed_identical_stall() -> None:
    det = _StallDetector(threshold_ms=100.0, confirm_polls=3)
    started_at = 1.0
    # Same (phase, started_at) pair over threshold: strike 1, 2, then fire.
    assert det.observe("gpu_sync_to_cpu", started_at, 2.0) is False
    assert det.observe("gpu_sync_to_cpu", started_at, 3.0) is False
    assert det.observe("gpu_sync_to_cpu", started_at, 4.0) is True


def test_detector_ignores_torn_zero_timestamp() -> None:
    # The _clear_inflight_frame race: phase read pre-clear pairs with a
    # zeroed timestamp, which would compute an astronomical age.
    det = _StallDetector(threshold_ms=100.0, confirm_polls=3)
    for i in range(10):
        assert det.observe("gpu_sync_to_cpu", 0.0, 1000.0 + i) is False


def test_detector_resets_on_new_frame() -> None:
    det = _StallDetector(threshold_ms=100.0, confirm_polls=3)
    # Each poll sees an over-threshold age but a different started_at
    # (e.g. clock skew or SIGSTOP resume artifacts): identity is broken
    # every time, so it never accumulates strikes.
    for i in range(10):
        now = 10.0 + i
        assert det.observe("lm_step", now - 0.5 - i * 0.001, now) is False


def test_detector_resets_after_recovery() -> None:
    det = _StallDetector(threshold_ms=100.0, confirm_polls=3)
    started_at = 1.0
    assert det.observe("mimi_decode", started_at, 2.0) is False
    assert det.observe("mimi_decode", started_at, 3.0) is False
    # Frame resolves; the pipeline goes idle.
    assert det.observe("idle", 0.0, 4.0) is False
    # A new stall must re-earn all its strikes.
    assert det.observe("mimi_decode", 5.0, 6.0) is False
    assert det.observe("mimi_decode", 5.0, 7.0) is False
    assert det.observe("mimi_decode", 5.0, 8.0) is True


def test_detector_phase_change_breaks_identity() -> None:
    det = _StallDetector(threshold_ms=100.0, confirm_polls=3)
    # A livelock cycling phases (frames progressing through the pipeline)
    # never matches the same pair twice.
    assert det.observe("mimi_encode", 1.0, 2.0) is False
    assert det.observe("lm_step", 1.0, 3.0) is False
    assert det.observe("mimi_decode", 1.0, 4.0) is False


def test_shipped_constants_require_confirmation() -> None:
    assert INFER_STALL_EXIT_MS >= 10_000.0
    assert INFER_STALL_CONFIRM_POLLS >= 3


def _enclosing_function_names(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Yield (enclosing function name, node) for every node in the tree."""
    pairs: list[tuple[str, ast.AST]] = []

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_scope = scope
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                child_scope = child.name
            pairs.append((child_scope, child))
            walk(child, child_scope)

    walk(tree, "<module>")
    return pairs


def test_phase_writes_confined_to_frame_path() -> None:
    tree = ast.parse(SERVER_PY.read_text())
    setter_scopes = set()
    field_write_scopes = set()
    for scope, node in _enclosing_function_names(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_set_inflight_phase"
        ):
            setter_scopes.add(scope)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in PHASE_FIELDS
                ):
                    field_write_scopes.add(scope)

    assert setter_scopes, "expected _set_inflight_phase call sites"
    unexpected_setters = setter_scopes - ALLOWED_PHASE_SETTER_SCOPES
    assert not unexpected_setters, (
        "_set_inflight_phase called outside the bounded frame path: "
        f"{sorted(unexpected_setters)}. The hang watchdog hard-exits when a "
        f"phase ages past INFER_STALL_EXIT_MS ({INFER_STALL_EXIT_MS:.0f} ms), "
        "so long-running work must never set a phase."
    )
    unexpected_writes = field_write_scopes - ALLOWED_FIELD_WRITE_SCOPES
    assert not unexpected_writes, (
        "inflight phase fields written outside the setter/clear/init: "
        f"{sorted(unexpected_writes)}"
    )


if __name__ == "__main__":
    print("test_detector_never_fires_when_idle ...")
    test_detector_never_fires_when_idle()
    print("  ok")
    print("test_detector_never_fires_under_threshold ...")
    test_detector_never_fires_under_threshold()
    print("  ok")
    print("test_detector_fires_on_confirmed_identical_stall ...")
    test_detector_fires_on_confirmed_identical_stall()
    print("  ok")
    print("test_detector_ignores_torn_zero_timestamp ...")
    test_detector_ignores_torn_zero_timestamp()
    print("  ok")
    print("test_detector_resets_on_new_frame ...")
    test_detector_resets_on_new_frame()
    print("  ok")
    print("test_detector_resets_after_recovery ...")
    test_detector_resets_after_recovery()
    print("  ok")
    print("test_detector_phase_change_breaks_identity ...")
    test_detector_phase_change_breaks_identity()
    print("  ok")
    print("test_shipped_constants_require_confirmation ...")
    test_shipped_constants_require_confirmation()
    print("  ok")
    print("test_phase_writes_confined_to_frame_path ...")
    test_phase_writes_confined_to_frame_path()
    print("  ok")
    print("all hang watchdog tests passed")
