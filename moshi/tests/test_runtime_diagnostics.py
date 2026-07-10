"""Focused checks for realtime runtime policies and stall diagnostics.

Run directly: ``uv run python moshi/tests/test_runtime_diagnostics.py``.
"""

from __future__ import annotations

import inspect
import re
import sys
import threading
import time
from collections import deque

import torch
import numpy as np

sys.path.insert(0, "moshi")

from moshi.server import ServerState  # noqa: E402


def _bare_diagnostics_state() -> ServerState:
    state = ServerState.__new__(ServerState)
    state._frame_audio_sec = 0.08
    state._process_frame_count = 12
    state._rtf_last = 0.5
    state._rtf_ema = 0.55
    state._process_frame_ms_last = 40.0
    state._process_frame_ms_ema = 44.0
    state._lm_frame_ms_last = 39.0
    state._lm_frame_ms_ema = 43.0
    state._gpu_util_last = None
    state._vram_used_last = None
    state._inflight_phase = "lm_step"
    state._inflight_phase_started_at = time.perf_counter() - 1.25
    state._inflight_frame_started_at = time.perf_counter() - 1.5
    state._inflight_frame = 13
    return state


def test_periodic_snapshots_default_off() -> None:
    parameter = inspect.signature(ServerState.__init__).parameters.get(
        "periodic_snapshots"
    )
    assert parameter is not None
    assert parameter.default is False


def test_stale_baseline_is_not_an_auto_rewind_target() -> None:
    state = ServerState.__new__(ServerState)
    snapshot = {"version": 2}
    state._session_snapshots = {"session": [(100.0, snapshot)]}
    assert state._recent_auto_rewind_snapshot("session", now=189.0) is snapshot
    assert state._recent_auto_rewind_snapshot("session", now=191.0) is None


def test_backpressure_status_names_active_inference_phase() -> None:
    status = _bare_diagnostics_state()._backpressure_status()
    assert "inflight_phase=lm_step" in status
    assert "inflight_frame=13" in status
    phase_age = re.search(r"phase_age_ms=([0-9.]+)", status)
    frame_age = re.search(r"frame_age_ms=([0-9.]+)", status)
    assert phase_age is not None and float(phase_age.group(1)) >= 1_000
    assert frame_age is not None and float(frame_age.group(1)) >= 1_250


def test_tracked_inference_lock_clears_phase_after_error() -> None:
    state = _bare_diagnostics_state()
    state._infer_lock = threading.Lock()
    try:
        with state._tracked_inference_lock():
            assert state._inflight_phase == "mimi_encode"
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    assert state._inflight_phase == "idle"
    assert state._inflight_phase_started_at == 0.0
    assert state._inflight_frame_started_at == 0.0


def test_failed_input_transfer_clears_inflight_frame() -> None:
    state = _bare_diagnostics_state()
    original_from_numpy = torch.from_numpy
    try:
        def _fail_from_numpy(_samples):
            raise RuntimeError("input transfer failed")

        torch.from_numpy = _fail_from_numpy
        try:
            state._process_audio_frame(np.zeros(16, dtype=np.float32))
        except RuntimeError:
            pass
    finally:
        torch.from_numpy = original_from_numpy

    assert state._inflight_phase == "idle"
    assert state._inflight_phase_started_at == 0.0
    assert state._inflight_frame_started_at == 0.0
    assert state._inflight_frame == 0


def test_snapshot_waits_for_cuda_copy_completion() -> None:
    state = ServerState.__new__(ServerState)
    state.device = torch.device("cuda:0")
    state._infer_lock = threading.Lock()
    state.lm_gen = object()
    state.mimi = object()
    state._clone_streaming_state = lambda _module: {
        "state": torch.zeros(4, dtype=torch.float32)
    }

    original_is_available = torch.cuda.is_available
    original_get_rng_state = torch.cuda.get_rng_state
    original_synchronize = torch.cuda.synchronize
    sync_calls: list[int | None] = []
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.get_rng_state = lambda _device=None: torch.zeros(4, dtype=torch.uint8)
        torch.cuda.synchronize = lambda device=None: sync_calls.append(device)
        state._take_snapshot()
    finally:
        torch.cuda.is_available = original_is_available
        torch.cuda.get_rng_state = original_get_rng_state
        torch.cuda.synchronize = original_synchronize

    assert sync_calls == [0]


def test_restore_waits_for_cuda_copy_completion() -> None:
    class _RestoreModule:
        def set_streaming_state_inplace(self, _state: dict) -> None:
            return None

    state = ServerState.__new__(ServerState)
    state.device = torch.device("cuda:0")
    state.mimi = _RestoreModule()
    state.lm_gen = _RestoreModule()
    state.lm_gen._non_pad_streak = 0
    state.lm_gen._pad_force_remaining = 0
    state._clear_vision_pending = lambda: None
    state._clear_reinforce_pending = lambda: None
    state._collapse_triggers = deque()

    snapshot = {
        "version": 2,
        "lm": {},
        "mimi": {},
        "rng_cpu": torch.get_rng_state().clone(),
        "rng_cuda": None,
    }
    original_is_available = torch.cuda.is_available
    original_synchronize = torch.cuda.synchronize
    sync_calls: list[int | None] = []
    try:
        torch.cuda.is_available = lambda: True
        torch.cuda.synchronize = lambda device=None: sync_calls.append(device)
        state._restore_snapshot_locked(snapshot)
    finally:
        torch.cuda.is_available = original_is_available
        torch.cuda.synchronize = original_synchronize

    assert sync_calls == [0]


if __name__ == "__main__":
    tests = [
        test_periodic_snapshots_default_off,
        test_stale_baseline_is_not_an_auto_rewind_target,
        test_backpressure_status_names_active_inference_phase,
        test_tracked_inference_lock_clears_phase_after_error,
        test_failed_input_transfer_clears_inflight_frame,
        test_snapshot_waits_for_cuda_copy_completion,
        test_restore_waits_for_cuda_copy_completion,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all runtime diagnostics tests passed")
