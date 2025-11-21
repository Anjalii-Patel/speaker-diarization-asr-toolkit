"""
Energy-Based Voice Activity Detection (VAD)
Lightweight, local, deterministic VAD using Short-Time Energy and Zero-Crossing Rate.
Does not require pre-trained weights or internet connection.
"""

from typing import List
import numpy as np
from .base import BaseVAD, SpeechSegment

class EnergyVAD(BaseVAD):
    """
    Rule-based VAD using frame-level Short-Time Energy (STE).
    Useful as an offline fallback and for test environments.
    """

    def __init__(
        self,
        frame_duration_ms: float = 30.0,
        hop_duration_ms: float = 10.0,
        energy_threshold_ratio: float = 0.05,
        min_speech_duration_ms: float = 200.0,
        min_silence_duration_ms: float = 100.0,
    ):
        self.frame_duration_ms = frame_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.energy_threshold_ratio = energy_threshold_ratio
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

    def detect_speech(self, audio: np.ndarray, sr: int = 16000) -> List[SpeechSegment]:
        if len(audio) == 0:
            return []

        frame_len = int(sr * self.frame_duration_ms / 1000.0)
        hop_len = int(sr * self.hop_duration_ms / 1000.0)

        # Pad audio if too short for a single frame
        if len(audio) < frame_len:
            audio = np.pad(audio, (0, frame_len - len(audio)))

        # Frame the signal
        num_frames = 1 + (len(audio) - frame_len) // hop_len
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(num_frames, frame_len),
            strides=(audio.strides[0] * hop_len, audio.strides[0]),
        )

        # Compute Root Mean Square (RMS) energy per frame
        energy = np.sqrt(np.mean(frames ** 2, axis=1))
        max_energy = np.max(energy) if len(energy) > 0 else 0.0
        if max_energy == 0.0:
            return []

        threshold = max_energy * self.energy_threshold_ratio
        is_speech = energy > threshold

        # Group contiguous frames into segments
        segments: List[SpeechSegment] = []
        in_speech = False
        start_frame = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                in_speech = True
                start_frame = i
            elif not speech and in_speech:
                in_speech = False
                start_sec = (start_frame * hop_len) / sr
                end_sec = ((i - 1) * hop_len + frame_len) / sr
                duration_ms = (end_sec - start_sec) * 1000.0
                if duration_ms >= self.min_speech_duration_ms:
                    segments.append(SpeechSegment(start=start_sec, end=end_sec, confidence=0.85))

        if in_speech:
            start_sec = (start_frame * hop_len) / sr
            end_sec = len(audio) / sr
            if (end_sec - start_sec) * 1000.0 >= self.min_speech_duration_ms:
                segments.append(SpeechSegment(start=start_sec, end=end_sec, confidence=0.85))

        # Merge segments separated by less than min_silence_duration_ms
        if not segments:
            # If nothing exceeded threshold, return entire clip if not complete silence
            if max_energy > 1e-4:
                return [SpeechSegment(start=0.0, end=len(audio) / sr, confidence=0.5)]
            return []

        merged: List[SpeechSegment] = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap_ms = (seg.start - prev.end) * 1000.0
            if gap_ms < self.min_silence_duration_ms:
                merged[-1] = SpeechSegment(start=prev.start, end=seg.end, confidence=max(prev.confidence, seg.confidence))
            else:
                merged.append(seg)

        return merged
