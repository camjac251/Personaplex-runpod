"""Focused checks for rewind-safe RTC pipeline generations.

Run directly: ``uv run python moshi/tests/test_rtc_pipeline.py``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading

import numpy as np

sys.path.insert(0, "moshi")

from moshi.rtc_session import (  # noqa: E402
    OUTBOUND_FRAME_SAMPLES,
    OUTBOUND_PREBUFFER_DECAY_RECVS,
    OUTBOUND_PREBUFFER_MAX_SAMPLES,
    OUTBOUND_PREBUFFER_MIN_SAMPLES,
    OUTBOUND_PREBUFFER_START_SAMPLES,
    OUTBOUND_SHED_CROSSFADE_SAMPLES,
    OUTBOUND_SHED_WINDOW_SAMPLES,
    MimiOutputTrack,
    RTCSession,
)
from moshi.server import ServerState  # noqa: E402


class _OutputTrack:
    def __init__(self) -> None:
        self.pushed: list[np.ndarray] = []
        self.clear_count = 0

    async def push_24k_f32(self, samples: np.ndarray) -> None:
        self.pushed.append(samples.copy())

    async def clear_buffer(self) -> None:
        self.clear_count += 1
        self.pushed.clear()

    async def diagnostics_snapshot(self) -> dict[str, int | float]:
        return {
            "outbound_buffer_ms": 20.0,
            "outbound_drop_events": 2,
        }


def _bare_session(process_fn) -> RTCSession:
    session = RTCSession.__new__(RTCSession)
    session._frame_size = 4
    session._process_fn = process_fn
    session._process_executor = None
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
    session._inbound_task = None
    session._process_task = None
    session._control_tasks = set()
    session._control_message_lock = asyncio.Lock()
    session._accept_control = True
    session._active_control_task = None
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


async def test_stop_processing_freezes_and_drains_in_flight_model_work() -> None:
    started = threading.Event()
    release = threading.Event()

    def process(chunk: np.ndarray):
        started.set()
        if not release.wait(timeout=1.0):
            raise RuntimeError("test process release timed out")
        return [(chunk, None)]

    session = _bare_session(process)
    session._process_task = asyncio.create_task(session._process_loop())
    session._pcm_queue.put_nowait((2, np.ones(4, dtype=np.float32)))
    assert await asyncio.to_thread(started.wait, 1.0)

    stop_task = asyncio.create_task(session.stop_processing())
    await asyncio.sleep(0)
    assert not stop_task.done()
    release.set()
    await asyncio.wait_for(stop_task, timeout=1.0)

    assert session._closed.is_set()
    assert session._processing_paused is True
    assert session._output_track.pushed == []
    assert session._process_task.done()


def test_standing_inbound_backlog_is_trimmed_to_one_frame() -> None:
    session = RTCSession.__new__(RTCSession)
    session._frame_size = 4
    session._pipeline_generation = 7
    session._pcm_queue = asyncio.Queue(maxsize=10)
    session._pcm_drop_events = 0
    session._pcm_dropped_ms = 0.0
    for value in range(8):
        session._pcm_queue.put_nowait(
            (7, np.full(1, value, dtype=np.float32))
        )

    dropped_ms = session._trim_standing_inbound_backlog()
    kept = []
    while not session._pcm_queue.empty():
        _, samples = session._pcm_queue.get_nowait()
        kept.append(int(samples[0]))
    assert kept == [4, 5, 6, 7], kept
    assert dropped_ms > 0
    assert session._pcm_drop_events == 1


async def test_transport_diagnostics_expose_counts_without_audio() -> None:
    session = _bare_session(lambda chunk: [(chunk, None)])
    session._pcm_queue_high_water = 7
    session._pcm_drop_events = 3
    session._pcm_dropped_ms = 240.0
    session._pcm_queue.put_nowait((2, np.ones(4, dtype=np.float32)))

    snapshot = await session.diagnostics_snapshot()

    assert snapshot == {
        "pcm_queue_depth": 1,
        "pcm_queue_capacity": 10,
        "pcm_queue_high_water": 7,
        "pcm_drop_events": 3,
        "pcm_dropped_ms": 240.0,
        "outbound_buffer_ms": 20.0,
        "outbound_drop_events": 2,
    }


def test_outbound_underrun_fades_and_reprimes() -> None:
    async def scenario() -> None:
        track = MimiOutputTrack()
        # Below the prebuffer floor nothing is emitted yet.
        track._buffer = np.full(OUTBOUND_FRAME_SAMPLES, 0.5, dtype=np.float32)
        assert np.all(await track._pop_chunk() == 0.0)
        assert track._underrun_events == 0

        # Floor filled: emission resumes with a fade-in from silence.
        track._buffer = np.full(
            OUTBOUND_FRAME_SAMPLES * 3, 0.5, dtype=np.float32
        )
        resumed = await track._pop_chunk()
        assert resumed[0] == 0.0
        assert resumed[-1] == np.float32(0.5)
        assert track._primed is True

        # Consuming the last queued frame lands its tail on silence and
        # reprimes, so a possible underrun next frame cannot click.
        await track._pop_chunk()
        emptied = await track._pop_chunk()
        assert emptied[0] == np.float32(0.5)
        assert emptied[-1] == 0.0
        assert track._primed is False
        assert track._underrun_events == 1
        assert track._prebuffer_target == (
            OUTBOUND_PREBUFFER_START_SAMPLES + OUTBOUND_FRAME_SAMPLES
        )

        # While repriming, queued-but-below-floor audio stays held.
        track._buffer = np.full(OUTBOUND_FRAME_SAMPLES, 0.5, dtype=np.float32)
        assert np.all(await track._pop_chunk() == 0.0)

        # A stranded partial frame is emitted faded, never hard-truncated.
        track._primed = True
        track._buffer = np.full(400, 0.5, dtype=np.float32)
        partial = await track._pop_chunk()
        assert partial[0] == np.float32(0.5)
        assert partial[399] == 0.0
        assert np.all(partial[400:] == 0.0)
        assert track._underrun_events == 2
        assert track._primed is False
        assert track._prebuffer_target == (
            OUTBOUND_PREBUFFER_START_SAMPLES + OUTBOUND_FRAME_SAMPLES * 2
        )

    asyncio.run(scenario())


def test_outbound_prebuffer_decays_after_clean_stretch() -> None:
    track = MimiOutputTrack()
    track._prebuffer_target = OUTBOUND_PREBUFFER_MAX_SAMPLES
    for _ in range(OUTBOUND_PREBUFFER_DECAY_RECVS):
        track._note_clean_recv_locked()
    assert track._prebuffer_target == (
        OUTBOUND_PREBUFFER_MAX_SAMPLES - OUTBOUND_FRAME_SAMPLES
    )

    # An underrun resets the clean streak before it can earn a decay.
    track._clean_recvs = OUTBOUND_PREBUFFER_DECAY_RECVS - 1
    track._note_underrun_locked()
    assert track._clean_recvs == 0
    assert track._prebuffer_target == OUTBOUND_PREBUFFER_MAX_SAMPLES

    # Decay never sinks below the floor minimum.
    track._prebuffer_target = OUTBOUND_PREBUFFER_MIN_SAMPLES
    for _ in range(OUTBOUND_PREBUFFER_DECAY_RECVS * 2):
        track._note_clean_recv_locked()
    assert track._prebuffer_target == OUTBOUND_PREBUFFER_MIN_SAMPLES


def test_outbound_shed_prefers_silence_and_crossfades() -> None:
    window = OUTBOUND_SHED_WINDOW_SAMPLES
    track = MimiOutputTrack()
    speech_a = (
        np.sin(np.linspace(0.0, 40.0 * np.pi, window * 4)) * 0.3
    ).astype(np.float32)
    silence = np.zeros(window * 4, dtype=np.float32)
    speech_b = (
        np.cos(np.linspace(0.0, 40.0 * np.pi, window * 4)) * 0.3
    ).astype(np.float32)
    track._buffer = np.concatenate([speech_a, silence, speech_b])

    track._shed_buffer_locked(silence.size)

    # The silent span absorbed the whole cut; speech survived untouched.
    assert track._buffer.size == speech_a.size + speech_b.size
    assert np.allclose(track._buffer[: speech_a.size], speech_a)
    assert np.allclose(track._buffer[speech_a.size:], speech_b)
    assert track._dropped_samples == silence.size

    # An all-speech buffer is cut with a crossfade, never a hard splice.
    loud = (
        np.sin(np.linspace(0.0, 200.0 * np.pi, window * 8)) * 0.5
    ).astype(np.float32)
    track._buffer = loud.copy()
    track._dropped_samples = 0
    track._shed_buffer_locked(window * 2)
    assert track._buffer.size == loud.size - window * 2
    assert track._dropped_samples == window * 2
    natural_step = float(np.max(np.abs(np.diff(loud))))
    splice_region = track._buffer[: OUTBOUND_SHED_CROSSFADE_SAMPLES * 2]
    splice_step = float(np.max(np.abs(np.diff(splice_region))))
    assert splice_step <= natural_step * 2.0, (splice_step, natural_step)


async def test_outbound_diagnostics_separate_flush_from_backlog_drop() -> None:
    track = MimiOutputTrack()
    track._buffer = np.zeros(4800, dtype=np.float32)
    track._buffer_high_water = 9600
    track._drop_events = 2
    track._dropped_samples = 2400

    await track.clear_buffer()
    snapshot = await track.diagnostics_snapshot()

    assert snapshot["outbound_buffer_ms"] == 0.0
    assert snapshot["outbound_high_water_ms"] == 200.0
    assert snapshot["outbound_drop_events"] == 2
    assert snapshot["outbound_dropped_ms"] == 50.0
    assert snapshot["outbound_flush_events"] == 1
    assert snapshot["outbound_flushed_ms"] == 100.0


def test_stat_envelope_only_forwards_numeric_diagnostics() -> None:
    class _Control:
        readyState = "open"

        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, payload: str) -> None:
            self.sent.append(payload)

    session = RTCSession.__new__(RTCSession)
    session._control = _Control()
    session.send_stat(
        rtf=0.42,
        diagnostics={
            "pcm_queue_depth": 4,
            "pcm_dropped_ms": 80.04,
            "private_path": "/private/secret",
            "outbound_drop_events": "not numeric",
        },
    )

    payload = json.loads(session._control.sent[0])
    assert payload == {
        "type": "stat",
        "rtf": 0.42,
        "pcm_queue_depth": 4,
        "pcm_dropped_ms": 80.0,
    }


class _Peer:
    async def close(self) -> None:
        return None


def _bare_control_session(handler) -> tuple[RTCSession, list[str]]:
    session = RTCSession.__new__(RTCSession)
    logs: list[str] = []
    session._log = lambda level, message: logs.append(f"{level}:{message}")
    session._control_tasks = set()
    session._control_message_lock = asyncio.Lock()
    session._accept_control = True
    session._active_control_task = None
    session._closed = asyncio.Event()
    session._on_config = None
    session._on_message = handler
    session._control = None
    session._inbound_task = None
    session._process_task = None
    session._pc = _Peer()
    session.close_reason = None
    return session, logs


def _start_control_task(session: RTCSession, payload: dict) -> asyncio.Task:
    task = asyncio.create_task(
        session._handle_control_message(json.dumps(payload))
    )
    session._control_tasks.add(task)
    task.add_done_callback(session._control_task_done)
    return task


async def test_control_commands_preserve_wire_order() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def handler(payload: dict) -> None:
        name = payload["name"]
        order.append(f"start-{name}")
        if name == "first":
            first_started.set()
            await release_first.wait()
        order.append(f"end-{name}")

    session, _ = _bare_control_session(handler)
    first = _start_control_task(session, {"type": "command", "name": "first"})
    await first_started.wait()
    second = _start_control_task(session, {"type": "command", "name": "second"})
    await asyncio.sleep(0)
    assert order == ["start-first"]
    release_first.set()
    await asyncio.gather(first, second)
    assert order == [
        "start-first",
        "end-first",
        "start-second",
        "end-second",
    ]


async def test_close_drains_active_control_and_cancels_waiters() -> None:
    active_started = asyncio.Event()
    release_active = asyncio.Event()
    order: list[str] = []

    async def handler(payload: dict) -> None:
        name = payload["name"]
        order.append(f"start-{name}")
        if name == "active":
            active_started.set()
            await release_active.wait()
        order.append(f"end-{name}")

    session, _ = _bare_control_session(handler)
    active = _start_control_task(session, {"type": "command", "name": "active"})
    await active_started.wait()
    queued = _start_control_task(session, {"type": "command", "name": "queued"})
    close_task = asyncio.create_task(session.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    release_active.set()
    await close_task
    await asyncio.gather(active, queued, return_exceptions=True)
    assert order == ["start-active", "end-active"], order
    assert not session._control_tasks


async def test_control_failure_is_retrieved_and_closes_session() -> None:
    async def handler(_payload: dict) -> None:
        raise RuntimeError("boom")

    session, logs = _bare_control_session(handler)
    task = _start_control_task(session, {"type": "command"})
    await session._closed.wait()
    await asyncio.gather(task, return_exceptions=True)
    assert session.close_reason == "error"
    assert any("control handler: RuntimeError: boom" in line for line in logs)


async def test_cancelled_session_lock_waiter_cannot_orphan_lock() -> None:
    state = ServerState.__new__(ServerState)
    state.lock = asyncio.Lock()
    await state.lock.acquire()
    waiter = asyncio.create_task(state._try_acquire_session_lock(timeout=10.0))
    await asyncio.sleep(0)
    waiter.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass
    state.lock.release()
    await asyncio.sleep(0)
    assert not state.lock.locked()


if __name__ == "__main__":
    tests = [
        test_stale_queued_generation_is_discarded,
        test_in_flight_result_is_discarded_across_pause,
        test_stop_processing_freezes_and_drains_in_flight_model_work,
        test_standing_inbound_backlog_is_trimmed_to_one_frame,
        test_transport_diagnostics_expose_counts_without_audio,
        test_outbound_underrun_fades_and_reprimes,
        test_outbound_prebuffer_decays_after_clean_stretch,
        test_outbound_shed_prefers_silence_and_crossfades,
        test_outbound_diagnostics_separate_flush_from_backlog_drop,
        test_stat_envelope_only_forwards_numeric_diagnostics,
        test_control_commands_preserve_wire_order,
        test_close_drains_active_control_and_cancels_waiters,
        test_control_failure_is_retrieved_and_closes_session,
        test_cancelled_session_lock_waiter_cannot_orphan_lock,
    ]
    for test in tests:
        print(f"{test.__name__} ...")
        if asyncio.iscoroutinefunction(test):
            asyncio.run(test())
        else:
            test()
        print("  ok")
    print("all RTC pipeline tests passed")
