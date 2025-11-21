"""
Whisper Automatic Speech Recognition (ASR) Module
Wraps OpenAI Whisper models loaded via Hugging Face Transformers.
Supports standalone inference, chunked pipelines, and segment-by-segment decoding.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch

logger = logging.getLogger(__name__)

class WhisperASR:
    """
    Automatic Speech Recognition using Whisper architectures.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        language: str = "en",
        task: str = "transcribe",
    ):
        self.model_name = model_name
        self.language = language
        self.task = task

        # Device selection
        if device is None:
            self.device = "cuda:0"if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if "cuda"in self.device else torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.processor = None
        self.model = None
        self._pipe = None

    def load_model(self) -> None:
        """Load model and processor into memory."""
        if self.model is not None and self.processor is not None:
            return

        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            logger.info("Loading Whisper model '%s' on %s...", self.model_name, self.device)
            try:
                # Try loading from local cache first for instant offline inference
                self.processor = WhisperProcessor.from_pretrained(self.model_name, local_files_only=True)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            except Exception:
                # Download if not present in cache
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model '{self.model_name}' on {self.device}: {e}"
            )

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
        task: Optional[str] = None,
    ) -> str:
        """
        Transcribe a single 1D audio waveform.

        Args:
            audio: 1D numpy array of audio samples.
            sr: Sampling rate (must be 16000 Hz).
            language: Language code (e.g., 'en', 'es', None for auto).
            task: 'transcribe' or 'translate'.

        Returns:
            Transcribed text.
        """
        if len(audio) == 0:
            return ""

        self.load_model()
        lang = language or self.language
        tsk = task or self.task

        # Pad very short segments to at least 0.5s for Whisper stability
        min_samples = int(sr * 0.5)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        input_features = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        ).input_features.to(self.device, dtype=self.torch_dtype)

        # Decoder prompt IDs for language and task
        forced_decoder_ids = None
        try:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=lang,
                task=tsk,
            )
        except Exception:
            pass

        with torch.no_grad():
            if forced_decoder_ids is not None:
                predicted_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                )
            else:
                predicted_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0].strip() if transcription else ""

    def transcribe_segments(
        self,
        audio: np.ndarray,
        sr: int,
        intervals: List[Tuple[float, float]],
        language: Optional[str] = None,
    ) -> List[str]:
        """
        Transcribe multiple audio slices extracted by start and end timestamps.

        Args:
            audio: Full continuous 1D audio waveform.
            sr: Sampling rate in Hz.
            intervals: List of (start_seconds, end_seconds) tuples.
            language: Target language code.

        Returns:
            List of transcribed strings matching the input interval count.
        """
        transcriptions: List[str] = []
        for start_sec, end_sec in intervals:
            start_idx = int(max(0.0, start_sec) * sr)
            end_idx = int(min(len(audio) / sr, end_sec) * sr)
            if end_idx <= start_idx:
                transcriptions.append("")
                continue

            chunk = audio[start_idx:end_idx]
            text = self.transcribe(chunk, sr=sr, language=language)
            transcriptions.append(text)

        return transcriptions
