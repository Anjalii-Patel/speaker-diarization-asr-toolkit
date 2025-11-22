"""
Unit Tests for Embedding Extraction and Clustering Compatibility
Validates:
- Dimensionality of MFCC (40-d: mean + std pooling) vs ECAPA-TDNN (192-d d-vector).
- L2-normalization of embedding vectors.
- Distinct acoustic representations between MFCC and neural embeddings.
- End-to-end integration into the clustering stage.
"""

import pytest
import numpy as np
from src.embeddings.mfcc import MFCCFeatureExtractor
from src.embeddings.ecapa_tdnn import ECAPATDNNEmbeddingExtractor
from src.diarization.clustering_diarizer import ClusteringDiarizer

@pytest.fixture
def synthetic_speech_chunks():
    """Generate synthetic audio chunks with distinct spectral profiles."""
    sr = 16000
    # Speaker 1: Low-frequency harmonic tone (150 Hz + harmonics)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    spk1_chunk = (
        0.5 * np.sin(2 * np.pi * 150 * t)
        + 0.3 * np.sin(2 * np.pi * 300 * t)
        + 0.1 * np.random.normal(0, 0.05, sr)
    ).astype(np.float32)

    # Speaker 2: High-frequency harmonic tone (400 Hz + harmonics)
    spk2_chunk = (
        0.5 * np.sin(2 * np.pi * 400 * t)
        + 0.3 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.random.normal(0, 0.05, sr)
    ).astype(np.float32)

    return [spk1_chunk, spk1_chunk, spk2_chunk, spk2_chunk]

def test_mfcc_embedding_dimensions_and_norm(synthetic_speech_chunks):
    extractor = MFCCFeatureExtractor(n_mfcc=20, include_std=True)
    single_emb = extractor.extract_segment(synthetic_speech_chunks[0], sr=16000)

    # 20 mean + 20 std = 40 dimensions
    assert single_emb.ndim == 1
    assert single_emb.shape[0] == 40
    assert abs(np.linalg.norm(single_emb) - 1.0) < 1e-4

    batch_emb = extractor.extract_batch(synthetic_speech_chunks, sr=16000)
    assert batch_emb.shape == (4, 40)

def test_ecapa_embedding_dimensions_and_norm(synthetic_speech_chunks):
    try:
        extractor = ECAPATDNNEmbeddingExtractor(device="cpu")
    except Exception as e:
        pytest.skip(f"ECAPA-TDNN unavailable: {e}")

    single_emb = extractor.extract_segment(synthetic_speech_chunks[0], sr=16000)
    assert single_emb.ndim == 1
    assert single_emb.shape[0] == 192
    assert abs(np.linalg.norm(single_emb) - 1.0) < 1e-4

    batch_emb = extractor.extract_batch(synthetic_speech_chunks, sr=16000)
    assert batch_emb.shape == (4, 192)

def test_mfcc_and_ecapa_produce_distinct_representations(synthetic_speech_chunks):
    mfcc_ext = MFCCFeatureExtractor(n_mfcc=20)
    try:
        ecapa_ext = ECAPATDNNEmbeddingExtractor(device="cpu")
    except Exception as e:
        pytest.skip(f"ECAPA-TDNN unavailable: {e}")

    mfcc_matrix = mfcc_ext.extract_batch(synthetic_speech_chunks, sr=16000)
    ecapa_matrix = ecapa_ext.extract_batch(synthetic_speech_chunks, sr=16000)

    # Confirm different dimensions
    assert mfcc_matrix.shape[1] != ecapa_matrix.shape[1]
    assert mfcc_matrix.shape == (4, 40)
    assert ecapa_matrix.shape == (4, 192)

    # Both reach clustering successfully
    diarizer = ClusteringDiarizer(num_speakers=2, linkage="ward")
    mfcc_labels = diarizer.fit_predict(mfcc_matrix)
    ecapa_labels = diarizer.fit_predict(ecapa_matrix)

    assert len(mfcc_labels) == 4
    assert len(ecapa_labels) == 4
    assert len(set(mfcc_labels)) == 2
    assert len(set(ecapa_labels)) == 2

