"""Mono Opus encoder integration for aiortc."""

from __future__ import annotations

import fractions
import logging
import time
from typing import Optional

import av
from aiortc import codecs
from aiortc.codecs import opus
from aiortc.codecs.base import Encoder
from aiortc.mediastreams import convert_timebase
from av import AudioFrame
from av.frame import Frame
from av.packet import Packet


logger = logging.getLogger(__name__)

OPUS_SAMPLE_RATE = 48_000
OPUS_FRAME_SAMPLES = 960
OPUS_TIME_BASE = fractions.Fraction(1, OPUS_SAMPLE_RATE)
ENCODE_FAILURE_LOG_INTERVAL_SEC = 5.0
ENCODE_FAILURE_ESCALATE_COUNT = 5


class MonoOpusEncoder(Encoder):
    """Encode 48 kHz mono PCM with libopus at 64 kbps."""

    def __init__(self) -> None:
        self.codec = self._create_codec()
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=OPUS_SAMPLE_RATE,
            frame_size=OPUS_FRAME_SAMPLES,
        )
        self._first_packet_pts: Optional[int] = None
        self._encode_failure_count = 0
        self._last_encode_log_at = 0.0

    @staticmethod
    def _create_codec() -> av.CodecContext:
        codec = av.CodecContext.create("libopus", "w")
        codec.sample_rate = OPUS_SAMPLE_RATE
        codec.format = "s16"
        codec.layout = "mono"
        codec.bit_rate = 64_000
        codec.time_base = OPUS_TIME_BASE
        codec.options = {
            "application": "audio",
            "vbr": "on",
            "frame_duration": "20",
        }
        codec.open()
        return codec

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> tuple[list[bytes], Optional[int]]:
        # An exception escaping here permanently stops aiortc's RTP sender
        # coroutine while the DataChannel and inference stay alive, so every
        # operational failure (resampler, codec, packet format) is contained
        # to a dropped frame with rate-limited diagnostics.
        try:
            assert isinstance(frame, AudioFrame)
            assert frame.format.name == "s16"
            assert frame.layout.name in ("mono", "stereo")

            payloads: list[bytes] = []
            timestamp: Optional[int] = None
            for resampled in self.resampler.resample(frame):
                for packet in self.codec.encode(resampled):
                    packet_pts = convert_timebase(
                        packet.pts, packet.time_base, OPUS_TIME_BASE
                    )
                    if self._first_packet_pts is None:
                        self._first_packet_pts = packet_pts
                    if timestamp is None:
                        timestamp = packet_pts - self._first_packet_pts
                    payloads.append(bytes(packet))
        except Exception as exc:
            self._encode_failure_count += 1
            now = time.monotonic()
            if self._encode_failure_count == ENCODE_FAILURE_ESCALATE_COUNT:
                logger.error(
                    "mono Opus encode failing repeatedly "
                    "(%d consecutive failures); outbound audio degraded: "
                    "%s: %s",
                    self._encode_failure_count,
                    type(exc).__name__,
                    exc,
                )
                self._last_encode_log_at = now
            elif (
                now - self._last_encode_log_at
                >= ENCODE_FAILURE_LOG_INTERVAL_SEC
            ):
                logger.warning(
                    "mono Opus encode failed; dropping frame: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self._last_encode_log_at = now
            return [], None
        self._encode_failure_count = 0
        return payloads, timestamp

    def pack(self, packet: Packet) -> tuple[list[bytes], int]:
        timestamp = convert_timebase(packet.pts, packet.time_base, OPUS_TIME_BASE)
        return [bytes(packet)], timestamp


def install_mono_opus_encoder() -> bool:
    """Install the mono encoder in every aiortc Opus factory binding."""
    if (
        opus.OpusEncoder is MonoOpusEncoder
        and codecs.OpusEncoder is MonoOpusEncoder
    ):
        return True
    try:
        probe = MonoOpusEncoder()
        probe.codec.close()
    except Exception as exc:
        logger.warning(
            "mono Opus encoder unavailable; using aiortc stock encoder: %s: %s",
            type(exc).__name__,
            exc,
        )
        return False
    opus.OpusEncoder = MonoOpusEncoder
    codecs.OpusEncoder = MonoOpusEncoder
    return True
