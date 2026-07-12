"""Focused checks for realtime runtime policies and stall diagnostics.

Run directly: ``uv run python moshi/tests/test_runtime_diagnostics.py``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
import threading
import time
from collections import deque

import torch
import numpy as np

sys.path.insert(0, "moshi")

from moshi.server import (  # noqa: E402
    BASE_HF_REPO,
    GEMINI_VISION_MODEL,
    RL_HF_REPO,
    ServerState,
    SnapshotDeferred,
    _model_identity,
)


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


def test_periodic_snapshots_default_on() -> None:
    # Auto-rewind only accepts snapshots younger than 90 s, so the 60 s
    # periodic capture must be on by default or the collapse safety net is
    # inert past the first minutes of every session (capture measures ~3 ms).
    parameter = inspect.signature(ServerState.__init__).parameters.get(
        "periodic_snapshots"
    )
    assert parameter is not None
    assert parameter.default is True


def test_model_identity_distinguishes_rl_base_and_custom() -> None:
    rl = _model_identity(RL_HF_REPO, "a" * 40)
    assert rl["model_variant"] == "rl-seamless"
    assert rl["native_duplex_recommended"] is True
    assert "CC BY-NC" in rl["model_license"]

    base = _model_identity(BASE_HF_REPO, "b" * 40)
    assert base["model_variant"] == "base"
    assert base["native_duplex_recommended"] is False
    assert base["model_license"] == "NVIDIA OML"

    custom = _model_identity("example/custom", None)
    assert custom["model_label"] == "custom"
    assert custom["model_revision"] == "main"

    local = _model_identity("local:checkpoint.safetensors", None)
    assert local["model_label"] == "Local · checkpoint.safetensors"
    assert local["model_variant"] == "local"
    assert local["model_revision"] == "local file"


def test_server_info_reports_active_vision_model() -> None:
    state = ServerState.__new__(ServerState)
    state.model_identity = _model_identity(RL_HF_REPO, "a" * 40)
    state.gpu_name = "test-gpu"
    state.vram_total = 24 * 1024**3
    state.server_build = "test-build"
    state._gemini_api_key = "configured"

    response = asyncio.run(state.handle_server_info(None))
    payload = json.loads(response.text)

    assert payload["vision_available"] is True
    assert payload["vision_model"] == GEMINI_VISION_MODEL


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


def test_snapshot_defers_mid_context_injection() -> None:
    state = ServerState.__new__(ServerState)
    state._infer_lock = threading.Lock()
    state._inject_active = True
    state._vision_active = deque([1])
    state._reinforce_pending = deque()
    try:
        state._take_snapshot("periodic")
    except SnapshotDeferred:
        pass
    else:
        raise AssertionError("mid-inject snapshot was not deferred")


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


def test_short_noise_burst_does_not_complete_user_turn() -> None:
    def bare_state() -> ServerState:
        state = ServerState.__new__(ServerState)
        state._user_audio_active = False
        state._user_audio_attack_streak = 0
        state._user_audio_active_frames = 0
        state._user_audio_silence_streak = 0
        return state

    state = bare_state()
    for _ in range(3):
        started, ended = state._update_user_turn_activity(0.01)
    assert started is True and ended is False
    for _ in range(7):
        _, ended = state._update_user_turn_activity(0.0)
    assert ended is False

    state = bare_state()
    for _ in range(4):
        state._update_user_turn_activity(0.01)
    for _ in range(7):
        _, ended = state._update_user_turn_activity(0.0)
    assert ended is True


def test_live_turn_cap_change_resets_tracking_but_preserves_interrupt() -> None:
    class _Lm:
        _non_pad_streak = 77
        _pad_force_remaining = 5

    state = ServerState.__new__(ServerState)
    state.lm_gen = _Lm()
    state._collapse_triggers = deque([1.0, 2.0])
    state._prev_pad_force_remaining = 0

    state._reset_turn_cap_tracking_for_config_change()

    assert state.lm_gen._non_pad_streak == 0
    assert state.lm_gen._pad_force_remaining == 5
    assert list(state._collapse_triggers) == []
    assert state._prev_pad_force_remaining == 5


def test_auto_recovery_replaces_extreme_tuning() -> None:
    class _Model:
        text_card = 32_000
        card = 2_048

    class _Lm:
        lm_model = _Model()
        temp_text = 1.5
        top_k_text = 500
        repetition_penalty = 2.0
        repetition_penalty_context = 256
        padding_bonus = 6.0
        max_turn_text_tokens = 40
        _non_pad_streak = 39
        _pad_force_remaining = 0

        def set_audio_sampling(self, temperature, top_k) -> None:
            self.temp = temperature
            self.top_k = top_k

        def reset_repetition_state(self) -> None:
            self.repetition_reset = True

    state = ServerState.__new__(ServerState)
    state.model_identity = {"model_variant": "rl-seamless"}
    state.lm_gen = _Lm()
    state._collapse_triggers = deque([1.0, 2.0])
    state._prev_pad_force_remaining = 0

    state._apply_auto_recovery_tuning_locked()

    assert state.lm_gen.temp_text == 0.7
    assert state.lm_gen.top_k_text == 25
    assert state.lm_gen.temp == 0.8
    assert state.lm_gen.top_k == 250
    assert state.lm_gen.repetition_penalty == 1.0
    assert state.lm_gen.repetition_penalty_context == 64
    assert state.lm_gen.padding_bonus == 0.0
    assert state.lm_gen.max_turn_text_tokens == 120
    assert state.lm_gen.repetition_reset is True


def test_clearing_resume_grant_cancels_snapshot_retaining_timer() -> None:
    class _Handle:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    state = ServerState.__new__(ServerState)
    handle = _Handle()
    state._resume_grant = {"snapshots": [object()]}
    state._resume_grant_expiry_handle = handle
    state._clear_resume_grant()
    assert state._resume_grant is None
    assert state._resume_grant_expiry_handle is None
    assert handle.cancelled is True


if __name__ == "__main__":
    tests = [
        test_periodic_snapshots_default_on,
        test_model_identity_distinguishes_rl_base_and_custom,
        test_server_info_reports_active_vision_model,
        test_stale_baseline_is_not_an_auto_rewind_target,
        test_backpressure_status_names_active_inference_phase,
        test_tracked_inference_lock_clears_phase_after_error,
        test_failed_input_transfer_clears_inflight_frame,
        test_snapshot_waits_for_cuda_copy_completion,
        test_snapshot_defers_mid_context_injection,
        test_restore_waits_for_cuda_copy_completion,
        test_short_noise_burst_does_not_complete_user_turn,
        test_live_turn_cap_change_resets_tracking_but_preserves_interrupt,
        test_auto_recovery_replaces_extreme_tuning,
        test_clearing_resume_grant_cancels_snapshot_retaining_timer,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all runtime diagnostics tests passed")
