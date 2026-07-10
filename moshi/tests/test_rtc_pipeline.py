"""Focused checks for rewind-safe RTC pipeline generations.

Run directly: ``uv run python moshi/tests/test_rtc_pipeline.py``.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import numpy as np

sys.path.insert(0, "moshi")

from moshi.rtc_session import RTCSession  # noqa: E402


class _OutputTrack:
    def __init__(self) -> None:
        self.pushed: list[np.ndarray] = []
        self.clear_count = 0

    async def push_24k_f32(self, samples: np.ndarray) -> None:
        self.pushed.append(samples.copy())

    async def clear_buffer(self) -> None:
        self.clear_count += 1
        self.pushed.clear()


def _bare_session(process_fn) -> RTCSession:
    session = RTCSession.__new__(RTCSession)
    session._frame_size = 4
    session._process_fn = process_fn
    session._log = lambda _level, _text: None
    session._pcm_queue = asyncio.Queue(maxsize=10)
    session._processing_started = True
    session._processing_paused = False
    session._pipeline_generation = 2
    session._pending_pcm = None
    session._process_idle = asyncio.Event()
    session._process_idle.set()
    session._output_track = _OutputTrack()
    session._on_pcm = None
    session._control = None
    session._closed = asyncio.Event()
    session.close_reason = None
    return session


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        await asyncio.sleep(0.005)


async def _stop_loop(session: RTCSession, task: asyncio.Task) -> None:
    session._closed.set()
    await asyncio.wait_for(task, timeout=1.0)


async def test_stale_queued_generation_is_discarded() -> None:
    session = _bare_session(lambda chunk: [(chunk, None)])
    task = asyncio.create_task(session._process_loop())
    session._pcm_queue.put_nowait((1, np.ones(4, dtype=np.float32)))
    session._pcm_queue.put_nowait((2, np.full(4, 2.0, dtype=np.float32)))
    await _wait_until(lambda: len(session._output_track.pushed) == 1)
    assert np.all(session._output_track.pushed[0] == 2.0)
    await _stop_loop(session, task)


async def test_in_flight_result_is_discarded_across_pause() -> None:
    started = threading.Event()
    release = threading.Event()

    def process(chunk: np.ndarray):
        started.set()
        if not release.wait(timeout=1.0):
            raise RuntimeError("test process release timed out")
        return [(chunk, None)]

    session = _bare_session(process)
    session._pipeline_generation = 0
    task = asyncio.create_task(session._process_loop())
    session._pcm_queue.put_nowait((0, np.ones(4, dtype=np.float32)))
    assert await asyncio.to_thread(started.wait, 1.0)
    pause_task = asyncio.create_task(session.pause_and_flush_audio())
    await _wait_until(lambda: session._pipeline_generation == 1)
    release.set()
    generation = await asyncio.wait_for(pause_task, timeout=1.0)
    assert generation == 1
    assert session._output_track.pushed == []
    assert session._output_track.clear_count == 1
    session.resume_audio(generation)
    assert session._processing_paused is False
    await _stop_loop(session, task)


if __name__ == "__main__":
    tests = [
        test_stale_queued_generation_is_discarded,
        test_in_flight_result_is_discarded_across_pause,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        asyncio.run(test())
        print("  ok")
    print("all RTC pipeline tests passed")
