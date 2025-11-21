"""
MFCC Feature Extraction Module
Extracts Mel-Frequency Cepstral Coefficients (MFCCs) with statistical pooling across time frames.
Serves as the classical acoustic baseline for experimental clustering diarization.
"""

import logging
from typing import Optional
import numpy as np
import torch
from .base import BaseEmbeddingExtractor

logger = logging.getLogger(__name__)

class MFCCFeatureExtractor(BaseEmbeddingExtractor):
    """
    Extracts MFCC features with mean and standard deviation pooling.
    Uses torchaudio if available, with a pure NumPy/SciPy fallback.
    """

    def __init__(
        self,
        n_mfcc: int = 20,
        n_fft: int = 512,
        hop_length: int = 160,
        include_std: bool = True,
        normalize: bool = True,
    ):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.include_std = include_std
        self.normalize = normalize

        # Try initializing torchaudio MFCC transform
        self._torchaudio_mfcc = None
        try:
            import torchaudio

            self._torchaudio_mfcc = torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "n_mels": 40,
                    "center": False,
                },
            )
        except Exception as e:
            logger.debug("torchaudio MFCC initialization skipped: %s", e)

    def extract_segment(self, audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
        if len(audio_chunk) < self.n_fft:
            # Pad with zeros if chunk is smaller than window
            audio_chunk = np.pad(audio_chunk, (0, self.n_fft - len(audio_chunk)))

        mfcc_matrix = None

        # Try torchaudio backend
        if self._torchaudio_mfcc is not None:
            try:
                tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
                mfcc_t = self._torchaudio_mfcc(tensor)  # shape: [1, n_mfcc, frames]
                mfcc_matrix = mfcc_t.squeeze(0).cpu().numpy()  # shape: [n_mfcc, frames]
            except Exception as e:
                logger.debug("torchaudio MFCC computation failed: %s", e)

        # Fallback to librosa or numpy FFT
        if mfcc_matrix is None:
            try:
                import librosa

                mfcc_matrix = librosa.feature.mfcc(
                    y=audio_chunk,
                    sr=sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                )
            except ImportError:
                # Basic power spectrum mel-filter approximation
                from scipy.signal import spectrogram

                f, t, Sxx = spectrogram(audio_chunk, fs=sr, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
                log_spec = np.log1p(Sxx[: self.n_mfcc, :])
                mfcc_matrix = log_spec

        # Temporal pooling: mean and optional standard deviation across frames
        mean_feat = np.mean(mfcc_matrix, axis=1)
        if self.include_std:
            std_feat = np.std(mfcc_matrix, axis=1)
            feature_vector = np.concatenate([mean_feat, std_feat])
        else:
            feature_vector = mean_feat

        if self.normalize:
            norm = np.linalg.norm(feature_vector)
            if norm > 1e-8:
                feature_vector = feature_vector / norm

        return feature_vector.astype(np.float32)
