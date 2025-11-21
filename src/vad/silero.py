"""
Silero VAD Wrapper
Integrates the Silero Voice Activity Detector with clean fallback mechanisms.
"""

import logging
from typing import List, Optional
import numpy as np
import torch
from .base import BaseVAD, SpeechSegment
from .energy import EnergyVAD

logger = logging.getLogger(__name__)

class SileroVAD(BaseVAD):
    """
    Silero VAD implementation supporting pre-trained neural speech detection.
    Falls back to EnergyVAD if silero_vad package or weights cannot be loaded.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: float = 250.0,
        min_silence_duration_ms: float = 100.0,
        speech_pad_ms: float = 30.0,
        device: str = "cpu",
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda"else "cpu")

        self.model = None
        self.get_speech_timestamps = None
        self._fallback_vad = EnergyVAD(
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        self._init_model()

    def _init_model(self):
        # Method 1: silero_vad python package
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            self.model = load_silero_vad()
            self.get_speech_timestamps = get_speech_timestamps
            logger.info("Silero VAD initialized via 'silero_vad' package.")
            return
        except ImportError:
            logger.debug("'silero_vad' package not installed.")

        # Method 2: torch.hub
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.model = model.to(self.device)
            self.get_speech_timestamps = utils[0]  # get_speech_timestamps is first element
            logger.info("Silero VAD initialized via torch.hub.")
            return
        except Exception as e_hub:
            logger.warning(
                "Could not initialize Silero VAD (%s). Falling back to rule-based EnergyVAD.",
                e_hub,
            )

    def detect_speech(self, audio: np.ndarray, sr: int = 16000) -> List[SpeechSegment]:
        if len(audio) == 0:
            return []

        if self.model is None or self.get_speech_timestamps is None:
            return self._fallback_vad.detect_speech(audio, sr)

        try:
            tensor_audio = torch.from_numpy(audio).float().to(self.device)
            timestamps = self.get_speech_timestamps(
                tensor_audio,
                self.model,
                threshold=self.threshold,
                sampling_rate=sr,
                min_speech_duration_ms=int(self.min_speech_duration_ms),
                min_silence_duration_ms=int(self.min_silence_duration_ms),
                speech_pad_ms=int(self.speech_pad_ms),
                return_seconds=True,
            )

            segments: List[SpeechSegment] = []
            for item in timestamps:
                segments.append(
                    SpeechSegment(
                        start=float(item["start"]),
                        end=float(item["end"]),
                        confidence=self.threshold,
                    )
                )
            return segments
        except Exception as e:
            logger.warning("Silero VAD inference failed (%s). Using EnergyVAD fallback.", e)
            return self._fallback_vad.detect_speech(audio, sr)
