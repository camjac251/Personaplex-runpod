"""Smoke tests for the WebRTC resampler chain.

Run directly: ``uv run python moshi/tests/test_rtc_resampler.py``.
No pytest dependency to keep the project deps lean; assertions raise.
"""

from __future__ import annotations

import asyncio
import fractions
import sys
import time
from unittest.mock import patch

import numpy as np
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
import aiortc.codecs as aiortc_codecs
from aiortc.codecs import opus as aiortc_opus

# Allow running this script from inside the repo without installing.
sys.path.insert(0, "moshi")

from moshi.rtc_session import (  # noqa: E402
    MIMI_SAMPLE_RATE,
    OUTBOUND_DRAIN_BACKLOG_SAMPLES,
    OUTBOUND_FRAME_SAMPLES,
    OUTBOUND_PREBUFFER_START_SAMPLES,
    WEBRTC_SAMPLE_RATE,
    MimiOutputTrack,
    _f32_to_s16,
    _frame_to_mono_24k_f32,
    _s16_to_f32,
    _soft_limit_f32,
)
import moshi.rtc_opus as rtc_opus  # noqa: E402
from moshi.rtc_opus import (  # noqa: E402
    MonoOpusEncoder,
    install_mono_opus_encoder,
)


def _sine_wave_f32(freq_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    n = min(len(reference), len(candidate))
    ref = reference[:n]
    cand = candidate[:n]
    noise = ref - cand
    ref_power = float(np.mean(ref**2))
    noise_power = float(np.mean(noise**2)) + 1e-12
    return 10.0 * np.log10(ref_power / noise_power)


def test_int_float_round_trip() -> None:
    rng = np.random.default_rng(42)
    samples = rng.uniform(-0.9, 0.9, size=4096).astype(np.float32)
    s16 = _f32_to_s16(samples)
    back = _s16_to_f32(s16)
    # int16 quantisation puts an upper bound on the round-trip error
    # (1/32768). Anything looser is a sign something is off.
    assert s16.dtype == np.int16
    assert back.dtype == np.float32
    max_err = float(np.max(np.abs(samples - back)))
    assert max_err <= 1.0 / 32768, f"round-trip max error too high: {max_err}"
    print(f"  s16 round-trip max error: {max_err:.2e}")


def test_mono_opus_encoder_installation_updates_both_factories() -> None:
    previous_opus_encoder = aiortc_opus.OpusEncoder
    previous_codecs_encoder = aiortc_codecs.OpusEncoder
    try:
        aiortc_opus.OpusEncoder = object
        aiortc_codecs.OpusEncoder = object

        assert install_mono_opus_encoder()
        assert aiortc_opus.OpusEncoder is MonoOpusEncoder
        assert aiortc_codecs.OpusEncoder is MonoOpusEncoder
    finally:
        aiortc_opus.OpusEncoder = previous_opus_encoder
        aiortc_codecs.OpusEncoder = previous_codecs_encoder


def test_mono_opus_encoder_emits_payload_for_pcm_frame() -> None:
    encoder = MonoOpusEncoder()
    frame = AudioFrame.from_ndarray(
        np.zeros((1, OUTBOUND_FRAME_SAMPLES), dtype=np.int16),
        format="s16",
        layout="mono",
    )
    frame.sample_rate = WEBRTC_SAMPLE_RATE
    frame.pts = 0
    frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)

    payloads, timestamp = encoder.encode(frame)

    assert timestamp == 0
    assert payloads
    assert all(isinstance(payload, bytes) and payload for payload in payloads)


def test_mono_opus_encoder_timestamps_advance_one_frame() -> None:
    encoder = MonoOpusEncoder()
    timestamps: list[int] = []
    for index in range(20):
        frame = AudioFrame.from_ndarray(
            np.zeros((1, OUTBOUND_FRAME_SAMPLES), dtype=np.int16),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = index * OUTBOUND_FRAME_SAMPLES
        frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
        payloads, timestamp = encoder.encode(frame)
        assert len(payloads) == 1
        assert isinstance(payloads[0], bytes) and payloads[0]
        assert timestamp is not None
        timestamps.append(timestamp)

    assert all(
        later - earlier == OUTBOUND_FRAME_SAMPLES
        for earlier, later in zip(timestamps, timestamps[1:])
    )


def test_mono_opus_encoder_drops_codec_failures_without_raising() -> None:
    class _Resampler:
        @staticmethod
        def resample(frame):
            return [frame]

    class _FailingCodec:
        @staticmethod
        def encode(_frame):
            raise RuntimeError("codec failure")

    encoder = MonoOpusEncoder.__new__(MonoOpusEncoder)
    encoder.resampler = _Resampler()
    encoder.codec = _FailingCodec()
    encoder._first_packet_pts = None
    encoder._encode_failure_count = 0
    encoder._last_encode_log_at = 0.0
    frame = AudioFrame.from_ndarray(
        np.zeros((1, OUTBOUND_FRAME_SAMPLES), dtype=np.int16),
        format="s16",
        layout="mono",
    )

    with (
        patch.object(rtc_opus.logger, "warning") as warning,
        patch.object(rtc_opus.logger, "error") as error,
    ):
        for _ in range(5):
            assert encoder.encode(frame) == ([], None)

    assert warning.call_count == 1
    assert error.call_count == 1


def test_soft_limit_preserves_samples_through_knee() -> None:
    samples = np.array([-0.97, -0.5, 0.0, 0.5, 0.97], dtype=np.float32)

    limited = _soft_limit_f32(samples)

    assert limited.dtype == np.float32
    np.testing.assert_array_equal(limited, samples)


def test_soft_limit_bounds_outliers_symmetrically() -> None:
    samples = np.array([-1.0, -0.98, 0.98, 1.0], dtype=np.float32)

    limited = _soft_limit_f32(samples)

    assert np.all(np.abs(limited) < 1.0)
    assert np.all(np.abs(limited) < np.abs(samples))
    np.testing.assert_allclose(limited[:2], -limited[:1:-1], rtol=0, atol=1e-7)


def test_inbound_resample_preserves_sine() -> None:
    """48 kHz sine -> mono 24 kHz float32 should round-trip with high SNR.

    Validates the inbound path's resampler. We synthesise a 48 kHz
    sine wave (1 kHz, well within both sample-rate Nyquist limits),
    feed it through `_frame_to_mono_24k_f32`, and check the output is
    a recognisable sine.
    """
    duration = 0.5  # seconds
    freq = 1000.0
    src_rate = 48_000
    expected = _sine_wave_f32(freq, duration, MIMI_SAMPLE_RATE)
    src = _sine_wave_f32(freq, duration, src_rate)
    s16 = _f32_to_s16(src).reshape(1, -1)
    frame = AudioFrame.from_ndarray(s16, format="s16", layout="mono")
    frame.sample_rate = src_rate
    resampler = AudioResampler(format="s16", layout="mono", rate=MIMI_SAMPLE_RATE)
    out = _frame_to_mono_24k_f32(frame, resampler)
    # Resampler may emit slightly fewer samples than the ratio implies on
    # the first call; allow up to 5% tolerance.
    expected_n = int(duration * MIMI_SAMPLE_RATE)
    assert out.size >= int(expected_n * 0.9), (
        f"too few samples: {out.size} vs expected ~{expected_n}"
    )
    snr = _snr_db(expected, out)
    print(f"  48k -> 24k SNR for 1 kHz sine: {snr:.1f} dB")
    # 30 dB is a generous floor; the FFmpeg resampler typically lands
    # around 60+ dB for clean sines well below Nyquist.
    assert snr > 30.0, f"resampler SNR too low: {snr:.1f} dB"


def test_output_track_pacing_and_resample() -> None:
    """`MimiOutputTrack.recv()` pulls 20 ms 48 kHz s16 mono frames.

    Push some 24 kHz Mimi-style audio, pull a few frames, and check
    they decode back to something close to the original sine.
    """
    track = MimiOutputTrack()
    # Push 200 ms of 1 kHz sine at 24 kHz.
    sine_24k = _sine_wave_f32(1000.0, 0.2, MIMI_SAMPLE_RATE)

    async def run() -> list[np.ndarray]:
        await track.push_24k_f32(sine_24k)
        # 200 ms of audio -> 10 outbound 20 ms frames at 48 kHz.
        out_frames: list[np.ndarray] = []
        for _ in range(10):
            frame = await track.recv()
            assert frame.sample_rate == WEBRTC_SAMPLE_RATE
            assert frame.format.name == "s16"
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr[0]
            assert arr.size == OUTBOUND_FRAME_SAMPLES, (
                f"expected {OUTBOUND_FRAME_SAMPLES} samples, got {arr.size}"
            )
            out_frames.append(arr)
        return out_frames

    out_frames = asyncio.run(run())
    pcm_48k = np.concatenate(out_frames).astype(np.float32) / 32768.0
    # Phase-aligning a buffered/paced output track against a freshly
    # generated reference sine is fragile (resampler pre-roll + pacing
    # delay shift everything). Instead, verify the output is a clean
    # 1 kHz tone via an FFT: dominant bin at 1 kHz, with most of the
    # total spectral energy concentrated there.
    rms = float(np.sqrt(np.mean(pcm_48k**2)))
    assert rms > 0.05, f"output track too quiet: rms={rms:.3f}"
    spectrum = np.abs(np.fft.rfft(pcm_48k))
    freqs = np.fft.rfftfreq(len(pcm_48k), 1 / WEBRTC_SAMPLE_RATE)
    peak_bin = int(np.argmax(spectrum))
    peak_freq = float(freqs[peak_bin])
    energy_in_peak = float(spectrum[peak_bin] ** 2)
    total_energy = float(np.sum(spectrum**2))
    fraction = energy_in_peak / total_energy
    print(
        f"  24k -> 48k via track.recv(): "
        f"peak={peak_freq:.0f} Hz, peak energy fraction={fraction:.2f}, rms={rms:.3f}"
    )
    assert abs(peak_freq - 1000.0) < 25.0, (
        f"expected 1 kHz peak, got {peak_freq:.1f} Hz"
    )
    assert fraction > 0.5, (
        f"expected most energy at 1 kHz, got {fraction:.2f}"
    )


def test_output_track_rebases_after_scheduler_stall() -> None:
    """A delayed sender must not burst overdue audio frames."""
    track = MimiOutputTrack()

    async def run() -> tuple[float, list[int]]:
        await track.recv()
        await asyncio.sleep(0.25)
        started_at = time.perf_counter()
        pts: list[int] = []
        for _ in range(10):
            frame = await track.recv()
            assert frame.pts is not None
            pts.append(frame.pts)
        return time.perf_counter() - started_at, pts

    elapsed, pts = asyncio.run(run())
    print(f"  ten frames after scheduler stall: {elapsed * 1000:.1f} ms")
    assert elapsed >= 0.16, (
        "output track burst overdue frames instead of resuming real-time pacing: "
        f"{elapsed * 1000:.1f} ms"
    )
    assert elapsed <= 0.35, f"output track resumed too slowly: {elapsed * 1000:.1f} ms"
    assert all(
        later - earlier == OUTBOUND_FRAME_SAMPLES
        for earlier, later in zip(pts, pts[1:])
    ), f"RTP timestamp spacing changed after pacing rebase: {pts}"


def test_output_track_drops_stale_backlog_after_stall() -> None:
    """Audio stranded by a stall is dropped at a stable RTP cadence.

    Sending packets faster does not speed up receiver playout because RTP
    timestamps still advance at 20 ms. The sender must discard stale PCM and
    then resume ordinary pacing.
    """
    track = MimiOutputTrack()
    # 480 ms of audio resamples to ~24 outbound frames: three times the
    # drain threshold, as if a long stall queued a burst of decodes.
    sine_24k = _sine_wave_f32(1000.0, 0.48, MIMI_SAMPLE_RATE)

    async def run() -> tuple[float, list[int], int]:
        # Establish the sender clock, then simulate a scheduler/network stall
        # while decoded speech accumulates.
        await track.recv()
        await asyncio.sleep(0.25)
        await track.push_24k_f32(sine_24k)
        started_at = time.perf_counter()
        pts: list[int] = []
        for _ in range(20):
            frame = await track.recv()
            assert frame.pts is not None
            pts.append(frame.pts)
        elapsed = time.perf_counter() - started_at
        async with track._buffer_lock:
            remaining = int(track._buffer.size)
        return elapsed, pts, remaining

    elapsed, pts, remaining = asyncio.run(run())
    print(
        f"  twenty frames with 480 ms backlog: {elapsed * 1000:.1f} ms, "
        f"{remaining} samples still queued"
    )
    # First frame after the rebase is immediate, then 19 frames keep 20 ms
    # pacing. Most stale content was discarded, not burst into a jitter buffer.
    assert 0.33 <= elapsed <= 0.48, (
        f"sender did not resume stable pacing: {elapsed * 1000:.1f} ms"
    )
    assert remaining < OUTBOUND_FRAME_SAMPLES, remaining
    assert all(
        later - earlier == OUTBOUND_FRAME_SAMPLES
        for earlier, later in zip(pts, pts[1:])
    ), f"RTP timestamp spacing changed during backlog drain: {pts}"


def test_output_track_sheds_standing_subthreshold_backlog() -> None:
    """Residue below the level gate must still drain via the backlog floor.

    A stall can leave 1-7 frames of standing latency: production and
    consumption both advance at 1x afterwards, so the residue never grows
    past the 8-frame drain gate and never shrinks on its own. The rolling
    backlog floor identifies it and recv() sheds it.
    """
    track = MimiOutputTrack()
    # 70 ms residue: more than three outbound frames. Small enough that even the
    # sawtooth peak (residue + one fresh lump) stays below the 8-frame
    # level gate, so only the backlog-floor path can shed it.
    residue_24k = _sine_wave_f32(1000.0, 0.07, MIMI_SAMPLE_RATE)
    # One healthy 80 ms producer lump.
    lump_24k = _sine_wave_f32(1000.0, 0.08, MIMI_SAMPLE_RATE)

    async def run() -> list[int]:
        loop = asyncio.get_event_loop()
        await track.push_24k_f32(residue_24k)

        async def producer() -> None:
            # Absolute schedule so sleep drift cannot starve the consumer
            # and fake a zero floor. The first lump lands immediately so the
            # residue is standing latency from the start, not warm-up food.
            started_at = loop.time()
            for i in range(200):
                await asyncio.sleep(max(0.0, started_at + i * 0.08 - loop.time()))
                await track.push_24k_f32(lump_24k)

        feed = asyncio.ensure_future(producer())
        backlogs: list[int] = []
        try:
            for _ in range(80):
                await track.recv()
                async with track._buffer_lock:
                    backlogs.append(int(track._buffer.size))
        finally:
            feed.cancel()
            try:
                await feed
            except asyncio.CancelledError:
                pass
        return backlogs

    backlogs = asyncio.run(run())
    early_floor = min(backlogs[5:25])
    early_peak = max(backlogs[:25])
    final_floor = min(backlogs[-25:])
    print(
        f"  standing 70 ms residue: early floor {early_floor}, "
        f"early peak {early_peak}, final floor {final_floor} samples"
    )
    standing_threshold = (
        OUTBOUND_PREBUFFER_START_SAMPLES + OUTBOUND_FRAME_SAMPLES
    )
    assert early_floor >= standing_threshold, (
        "test setup failed to create a standing backlog: "
        f"early floor {early_floor}"
    )
    assert early_peak < OUTBOUND_DRAIN_BACKLOG_SAMPLES, (
        "residue reached the level gate; this test must exercise the "
        f"backlog-floor path: early peak {early_peak}"
    )
    assert final_floor <= OUTBOUND_PREBUFFER_START_SAMPLES, (
        "standing backlog did not return to the adaptive prebuffer floor: "
        f"final floor {final_floor} samples"
    )


if __name__ == "__main__":
    print("test_int_float_round_trip ...")
    test_int_float_round_trip()
    print("  ok")
    print("test_mono_opus_encoder_installation_updates_both_factories ...")
    test_mono_opus_encoder_installation_updates_both_factories()
    print("  ok")
    print("test_mono_opus_encoder_emits_payload_for_pcm_frame ...")
    test_mono_opus_encoder_emits_payload_for_pcm_frame()
    print("  ok")
    print("test_mono_opus_encoder_timestamps_advance_one_frame ...")
    test_mono_opus_encoder_timestamps_advance_one_frame()
    print("  ok")
    print("test_mono_opus_encoder_drops_codec_failures_without_raising ...")
    test_mono_opus_encoder_drops_codec_failures_without_raising()
    print("  ok")
    print("test_soft_limit_preserves_samples_through_knee ...")
    test_soft_limit_preserves_samples_through_knee()
    print("  ok")
    print("test_soft_limit_bounds_outliers_symmetrically ...")
    test_soft_limit_bounds_outliers_symmetrically()
    print("  ok")
    print("test_inbound_resample_preserves_sine ...")
    test_inbound_resample_preserves_sine()
    print("  ok")
    print("test_output_track_pacing_and_resample ...")
    test_output_track_pacing_and_resample()
    print("  ok")
    print("test_output_track_rebases_after_scheduler_stall ...")
    test_output_track_rebases_after_scheduler_stall()
    print("  ok")
    print("test_output_track_drops_stale_backlog_after_stall ...")
    test_output_track_drops_stale_backlog_after_stall()
    print("  ok")
    print("test_output_track_sheds_standing_subthreshold_backlog ...")
    test_output_track_sheds_standing_subthreshold_backlog()
    print("  ok")
    print("all resampler tests passed")
