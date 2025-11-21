"""
Base Speaker Embedding Extractor Interface
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbeddingExtractor(ABC):
    """Abstract base class for extracting speaker embeddings / acoustic features."""

    @abstractmethod
    def extract_segment(self, audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract a 1D embedding vector for a single audio chunk.

        Args:
            audio_chunk: 1D numpy array of audio samples.
            sr: Sampling rate in Hz.

        Returns:
            1D numpy array representing the feature/embedding vector.
        """
        pass

    def extract_batch(self, chunks: List[np.ndarray], sr: int = 16000) -> np.ndarray:
        """
        Extract embeddings for a list of audio chunks.

        Args:
            chunks: List of 1D numpy arrays.
            sr: Sampling rate in Hz.

        Returns:
            2D numpy array of shape (num_chunks, embedding_dim).
        """
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = [self.extract_segment(chunk, sr=sr) for chunk in chunks]
        return np.vstack(embeddings)
