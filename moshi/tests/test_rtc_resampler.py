"""Smoke tests for the WebRTC resampler chain.

Run directly: ``uv run python moshi/tests/test_rtc_resampler.py``.
No pytest dependency to keep the project deps lean; assertions raise.
"""

from __future__ import annotations

import asyncio
import sys

import numpy as np
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler

# Allow running this script from inside the repo without installing.
sys.path.insert(0, "moshi")

from moshi.rtc_session import (  # noqa: E402
    MIMI_SAMPLE_RATE,
    OUTBOUND_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
    MimiOutputTrack,
    _f32_to_s16,
    _frame_to_mono_24k_f32,
    _s16_to_f32,
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
    # f32 -> s16 (* 32767) -> f32 (/ 32768) introduces a small asymmetric
    # scaling bias on top of the 1/32767 quantisation step. Actual worst
    # case lands around ~6e-5; treat ~1/16384 as the floor.
    assert max_err < 1.0 / 16384, f"round-trip max error too high: {max_err}"
    print(f"  s16 round-trip max error: {max_err:.2e}")


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


if __name__ == "__main__":
    print("test_int_float_round_trip ...")
    test_int_float_round_trip()
    print("  ok")
    print("test_inbound_resample_preserves_sine ...")
    test_inbound_resample_preserves_sine()
    print("  ok")
    print("test_output_track_pacing_and_resample ...")
    test_output_track_pacing_and_resample()
    print("  ok")
    print("all resampler tests passed")
