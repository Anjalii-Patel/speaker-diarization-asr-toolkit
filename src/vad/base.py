"""
Base Voice Activity Detection (VAD) Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class SpeechSegment:
    """Represents a continuous segment of detected speech."""
    start: float       # Start time in seconds
    end: float         # End time in seconds
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "confidence": round(self.confidence, 3),
        }

class BaseVAD(ABC):
    """Abstract base class for all VAD models."""

    @abstractmethod
    def detect_speech(self, audio: np.ndarray, sr: int = 16000) -> List[SpeechSegment]:
        """
        Detect speech segments in a 1D audio waveform.

        Args:
            audio: 1D numpy array of audio samples.
            sr: Sampling rate (expected: 16000 Hz).

        Returns:
            List of SpeechSegment objects.
        """
        pass
