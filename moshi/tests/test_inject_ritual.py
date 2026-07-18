"""Checks for the context-inject conditioning ritual and completion seal.

A context drip must replicate the t=0 conditioning ritual (sine user
channel, silent agent audio, forced text) and a fully delivered packet
must be followed by a PAD hold so the injected sentence lands as a
completed thought. Run directly:
``uv run python moshi/tests/test_inject_ritual.py``.
"""

from __future__ import annotations

import sys
import threading
from collections import deque

import numpy as np
import torch

sys.path.insert(0, "moshi")

from moshi.server import (  # noqa: E402
    CONTEXT_SEAL_HOLD_FRAMES,
    INJECT_SILENCE_RMS_DEFAULT,
    INJECT_SILENCE_STREAK_DEFAULT,
    ServerState,
)

PAD_ID = 3
SINE_MARK = 111
ZERO_MARK = 222
MIC_MARK = 55
FRAME_SAMPLES = 1920


class _FakeLmModel:
    text_padding_token_id = PAD_ID
    dep_q = 8


class _FakeLmGen:
    """Records every step() call so tests can assert the exact ritual."""

    def __init__(self) -> None:
        self.lm_model = _FakeLmModel()
        self._pad_force_remaining = 0
        self._sine = torch.full((1, 8, 1), SINE_MARK, dtype=torch.long)
        self._zero = torch.full((1, 8, 1), ZERO_MARK, dtype=torch.long)
        self.steps: list[dict] = []
        self.natural_text_token = PAD_ID

    def _encode_sine_frame(self) -> torch.Tensor:
        return self._sine

    def _encode_zero_frame(self) -> torch.Tensor:
        return self._zero

    def step(self, input_tokens, moshi_tokens=None, text_token=None):
        forced = None if text_token is None else int(text_token.reshape(-1)[0])
        self.steps.append(
            {
                "input_mark": int(input_tokens.reshape(-1)[0]),
                "agent_silenced": moshi_tokens is not None,
                "forced_text": forced,
            }
        )
        tokens = torch.zeros(1, 9, 1, dtype=torch.long)
        tokens[0, 0, 0] = self.natural_text_token if forced is None else forced
        return tokens


class _FakeMimi:
    def encode(self, chunk: torch.Tensor) -> torch.Tensor:
        return torch.full((1, 8, 1), MIC_MARK, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, 1, FRAME_SAMPLES, dtype=torch.float32)


def _pipeline_state() -> tuple[ServerState, _FakeLmGen]:
    state = ServerState.__new__(ServerState)
    lm_gen = _FakeLmGen()
    state.lm_gen = lm_gen
    state.mimi = _FakeMimi()
    state.device = "cpu"
    state.asr = None
    state.caption_cfg = False
    state._caption_cfg_gamma = 2.0
    state._infer_lock = threading.Lock()
    state._session_recorder = None
    state._active_session = None
    state._active_session_id = None
    state._main_loop = None

    # Inject machinery.
    state._vision_pending = deque()
    state._vision_pending_source = ""
    state._vision_pending_meta = {}
    state._vision_active = deque()
    state._vision_active_source = ""
    state._vision_active_meta = {}
    state._vision_inject_steps = 0
    state._inject_seal_remaining = 0
    state._vision_pad_streak = 8
    state._audio_silence_streak = 8
    state._inject_silence_rms = INJECT_SILENCE_RMS_DEFAULT
    state._inject_silence_streak = INJECT_SILENCE_STREAK_DEFAULT
    state._post_turn_inject_holdoff = 0
    state._interrupt_gate_remaining = 0
    state._stop_response_latched = False
    state._context_seal_token = 7
    state._last_injected_vision_key = ""
    state._last_ambient_context_queued_at = 0.0
    state._active_context_meta = {}
    state._inject_active = False
    state._inject_end_status = "complete"
    state._observed_idle_rms_ema = 0.0
    state._outbound_muted_prev = False

    # Reinforce machinery (disabled).
    state._reinforce_enabled = False
    state._reinforce_prompt_tokens = []
    state._reinforce_prompt_text = ""
    state._reinforce_pending = deque()
    state._reinforce_pending_meta = {}
    state._reinforce_inject_steps = 0
    state._last_reinforce_at = 0.0

    # User-turn activity tracking.
    state._user_audio_active = False
    state._user_audio_attack_streak = 0
    state._user_audio_active_frames = 0
    state._user_audio_silence_streak = 0
    state._stop_user_audio_active = False
    state._stop_user_audio_attack_streak = 0
    state._stop_user_audio_silence_streak = 0
    state._stop_latched_at = 0.0
    state._asr_assistant_silent = False
    state._asr_user_active = False

    # Collapse / diagnostics bookkeeping touched by the frame path.
    state._collapse_triggers = deque(maxlen=16)
    state._prev_pad_force_remaining = 0
    state._vision_ground_user_turns = False
    state._vision_request_pending = False
    state._vision_request_force = False
    state._vision_request_reason = "cadence"
    state._inflight_phase = "idle"
    state._inflight_phase_started_at = 0.0
    state._inflight_frame_started_at = 0.0
    state._inflight_frame = 0
    state._rtf_ema = 0.0
    state._rtf_last = 0.0
    state._process_frame_ms_last = 0.0
    state._process_frame_ms_ema = 0.0
    state._lm_frame_ms_last = 0.0
    state._lm_frame_ms_ema = 0.0
    state._process_frame_count = 0
    state.frame_size = FRAME_SAMPLES
    state._frame_audio_sec = FRAME_SAMPLES / 24000.0
    return state, lm_gen


def _silent_chunk() -> np.ndarray:
    return np.zeros(FRAME_SAMPLES, dtype=np.float32)


def _loud_chunk() -> np.ndarray:
    return np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)


def test_drip_frames_ride_the_t0_ritual() -> None:
    state, lm_gen = _pipeline_state()
    state._vision_active.extend([41, 42])
    state._vision_active_source = "ambient"
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    state._process_audio_frame(_silent_chunk())
    state._process_audio_frame(_silent_chunk())

    first, second = lm_gen.steps[0], lm_gen.steps[1]
    assert first == {
        "input_mark": SINE_MARK,
        "agent_silenced": True,
        "forced_text": 41,
    }
    assert second == {
        "input_mark": SINE_MARK,
        "agent_silenced": True,
        "forced_text": 42,
    }
    # Delivering the last token arms the completion hold.
    assert state._inject_seal_remaining == CONTEXT_SEAL_HOLD_FRAMES


def test_completion_hold_pads_then_releases() -> None:
    state, lm_gen = _pipeline_state()
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    state._process_audio_frame(_silent_chunk())
    for _ in range(CONTEXT_SEAL_HOLD_FRAMES):
        state._process_audio_frame(_silent_chunk())
    state._process_audio_frame(_silent_chunk())

    hold_frames = lm_gen.steps[1 : 1 + CONTEXT_SEAL_HOLD_FRAMES]
    assert all(
        frame
        == {
            "input_mark": SINE_MARK,
            "agent_silenced": True,
            "forced_text": PAD_ID,
        }
        for frame in hold_frames
    ), hold_frames
    assert state._inject_seal_remaining == 0
    # After the hold the model runs free again on real mic audio.
    released = lm_gen.steps[1 + CONTEXT_SEAL_HOLD_FRAMES]
    assert released["input_mark"] == MIC_MARK
    assert released["forced_text"] is None
    assert released["agent_silenced"] is False


def test_user_speech_cancels_hold_and_keeps_mic_audio() -> None:
    state, lm_gen = _pipeline_state()
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    state._process_audio_frame(_silent_chunk())
    assert state._inject_seal_remaining == CONTEXT_SEAL_HOLD_FRAMES
    state._process_audio_frame(_loud_chunk())

    assert state._inject_seal_remaining == 0
    spoken = lm_gen.steps[1]
    assert spoken["input_mark"] == MIC_MARK
    assert spoken["forced_text"] is None


def test_stop_latch_frames_keep_real_mic_audio() -> None:
    state, lm_gen = _pipeline_state()
    state._stop_response_latched = True
    state._stop_latched_at = 1e18  # far future; the latch must not expire
    state._vision_active.extend([41])
    state._vision_active_meta = {"source": "ambient", "text": "scene"}

    state._process_audio_frame(_silent_chunk())

    latched = lm_gen.steps[0]
    # The latch forces PAD but the model must keep hearing the room.
    assert latched["forced_text"] == PAD_ID
    assert latched["input_mark"] == MIC_MARK
    assert latched["agent_silenced"] is True


if __name__ == "__main__":
    tests = [
        test_drip_frames_ride_the_t0_ritual,
        test_completion_hold_pads_then_releases,
        test_user_speech_cancels_hold_and_keeps_mic_audio,
        test_stop_latch_frames_keep_real_mic_audio,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all inject ritual tests passed")
