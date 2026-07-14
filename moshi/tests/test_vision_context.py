"""Focused checks for factual vision-context normalization and queueing.

Run directly: ``uv run python moshi/tests/test_vision_context.py``.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections import deque

sys.path.insert(0, "moshi")

from moshi.server import (  # noqa: E402
    DEFAULT_VISION_SYSTEM_PROMPT,
    ServerState,
    VISION_AMBIENT_MIN_INTERVAL_SEC,
    VISION_QUEUE_MAX,
    _clip_vision_context,
    _gemini_caption_from_interaction,
    _sanitize_vision_text,
)


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


class _GateLock:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        self.entered.set()
        if not self.release.wait(timeout=1.0):
            raise RuntimeError("gate lock release timed out")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def _bare_state() -> ServerState:
    state = ServerState.__new__(ServerState)
    state._infer_executor = None
    state.text_tokenizer = _Tokenizer()
    state._vision_pending = deque()
    state._vision_pending_source = ""
    state._vision_pending_meta = {}
    state._vision_active = deque()
    state._vision_active_source = ""
    state._vision_active_meta = {}
    state._vision_inject_steps = 0
    state._last_injected_vision_key = ""
    state._active_context_meta = {}
    state._inject_end_status = "complete"
    state._vision_pad_streak = 8
    state._audio_silence_streak = 8
    return state


def test_sanitize_removes_labels_and_extra_sentences() -> None:
    assert (
        _sanitize_vision_text(
            "Scene; A person stands on a country road. They are facing uphill."
        )
        == "A person stands on a country road."
    )
    assert _sanitize_vision_text("Observation: Trees line the road") == (
        "Trees line the road"
    )


def test_default_prompt_uses_second_person_viewpoint_without_role_tags() -> None:
    assert '"In your current view,"' in DEFAULT_VISION_SYSTEM_PROMPT
    assert "no more than 20 words" in DEFAULT_VISION_SYSTEM_PROMPT
    assert "opener counts toward the 20-word limit" in (
        DEFAULT_VISION_SYSTEM_PROMPT
    )
    assert 'Use "your" only to establish the viewpoint' in (
        DEFAULT_VISION_SYSTEM_PROMPT
    )
    assert "Do not address anyone" not in DEFAULT_VISION_SYSTEM_PROMPT
    assert "<system>" not in DEFAULT_VISION_SYSTEM_PROMPT
    assert _sanitize_vision_text(
        "In your current view, a red mug rests beside the keyboard."
    ) == "In your current view, a red mug rests beside the keyboard."


def test_incomplete_interaction_keeps_valid_caption_json() -> None:
    data = {
        "status": "incomplete",
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": '{"caption":"An orange square is visible."}',
                    }
                ],
            }
        ],
    }
    caption, step_count = _gemini_caption_from_interaction(data)
    assert caption == "An orange square is visible."
    assert step_count == 1


def test_clipping_preserves_complete_untruncated_text() -> None:
    text = "A person stands beside the road"
    assert _clip_vision_context(text, len(text)) == text
    assert _clip_vision_context(text, 20) == "A person stands"


def test_fit_context_keeps_complete_words_within_token_window() -> None:
    state = _bare_state()
    words = [f"word{i}" for i in range(VISION_QUEUE_MAX + 4)]
    context, tokens = state._fit_vision_context(" ".join(words))
    # Whichever budget binds first (chars or tokens), the fit must stay
    # inside the token window and emit only complete leading words.
    assert 0 < len(tokens) <= VISION_QUEUE_MAX
    assert context.endswith(".")
    emitted = context.rstrip(".").split()
    assert emitted == words[: len(emitted)]
    assert len(emitted) < len(words)


def test_waiting_packet_never_splices_active_packet() -> None:
    state = _bare_state()
    state._vision_active.extend([1, 2])
    state._vision_active_source = "manual"
    state._vision_active_meta = {"caption": "person by the road"}
    queued, blocked_by, duplicate = state._queue_waiting_vision_context(
        [3, 4],
        "ambient",
        {"caption": "a car approaches"},
    )
    assert (queued, blocked_by, duplicate) == (True, "", False)
    assert list(state._vision_active) == [1, 2]
    assert state._vision_active_source == "manual"
    assert list(state._vision_pending) == [3, 4]


def test_ambient_context_is_rate_limited_without_blocking_captions() -> None:
    state = _bare_state()
    state._last_ambient_context_queued_at = 0.0
    first = state._queue_waiting_vision_context(
        [1, 2],
        "ambient",
        {"caption": "a red square"},
    )
    second = state._queue_waiting_vision_context(
        [3, 4],
        "ambient",
        {"caption": "a green square"},
    )
    state._last_ambient_context_queued_at -= VISION_AMBIENT_MIN_INTERVAL_SEC
    third = state._queue_waiting_vision_context(
        [5, 6],
        "ambient",
        {"caption": "a purple square"},
    )

    assert first == (True, "", False)
    assert second == (False, "rate_limit", False)
    assert third == (True, "", False)
    assert list(state._vision_pending) == [5, 6]


def test_source_clear_preserves_other_packet() -> None:
    state = _bare_state()
    state._vision_active.extend([1, 2])
    state._vision_active_source = "manual"
    state._vision_active_meta = {"caption": "manual context"}
    state._vision_pending.extend([3, 4])
    state._vision_pending_source = "ambient"
    state._vision_pending_meta = {"caption": "ambient context"}
    assert state._clear_vision_source("ambient") == (True, False)
    assert list(state._vision_active) == [1, 2]
    assert list(state._vision_pending) == []

    state._vision_pending.extend([5, 6])
    state._vision_pending_source = "manual"
    state._vision_pending_meta = {"caption": "new manual context"}
    state._vision_active.clear()
    state._vision_active.extend([7, 8])
    state._vision_active_source = "ambient"
    state._vision_active_meta = {"caption": "active ambient context"}
    assert state._clear_vision_source("ambient") == (False, True)
    assert list(state._vision_active) == []
    assert list(state._vision_pending) == [5, 6]
    assert state._inject_end_status == "dropped"
    assert state._vision_pad_streak == 0
    assert state._audio_silence_streak == 0


def test_stale_source_generation_cannot_queue_context() -> None:
    async def run_case() -> None:
        state = _bare_state()
        gate = _GateLock()
        state._infer_lock = gate
        state._active_session_id = "session"
        state._active_session = None
        state._vision_source_active = True
        state._vision_source_generation = 4
        state._latest_vision_caption = "A person stands beside a road."
        state._latest_vision_at = time.monotonic()
        state._latest_vision_frame_id = "frame-1"

        task = asyncio.create_task(
            state._queue_latest_vision_context(
                "session",
                "manual",
                "user_requested",
            )
        )
        assert await asyncio.to_thread(gate.entered.wait, 1.0)
        state._vision_source_generation = 5
        gate.release.set()
        assert await task is False
        assert list(state._vision_pending) == []

    asyncio.run(run_case())


if __name__ == "__main__":
    tests = [
        test_sanitize_removes_labels_and_extra_sentences,
        test_default_prompt_uses_second_person_viewpoint_without_role_tags,
        test_incomplete_interaction_keeps_valid_caption_json,
        test_clipping_preserves_complete_untruncated_text,
        test_fit_context_keeps_complete_words_within_token_window,
        test_waiting_packet_never_splices_active_packet,
        test_ambient_context_is_rate_limited_without_blocking_captions,
        test_source_clear_preserves_other_packet,
        test_stale_source_generation_cannot_queue_context,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        test()
        print("  ok")
    print("all vision context tests passed")
