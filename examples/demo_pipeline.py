#!/usr/bin/env python3
"""
Quickstart Demo:
Demonstrates speaker diarization, embedding inspection, and RTTM generation
on the CMU AN4 public benchmark sample with both MFCC and ECAPA-TDNN feature extractors.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Set local matplotlib cache directory
os.environ["MPLCONFIGDIR"] = os.path.join(str(Path(__file__).resolve().parent.parent), "tmp", "matplotlib")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.io import load_audio, get_audio_duration
from src.vad.energy import EnergyVAD
from src.embeddings.mfcc import MFCCFeatureExtractor
from src.embeddings.ecapa_tdnn import ECAPATDNNEmbeddingExtractor
from src.diarization.clustering_diarizer import ClusteringDiarizer
from src.diarization.rttm import write_rttm, parse_rttm
from src.evaluation.metrics import compute_der
from sklearn.metrics.pairwise import cosine_similarity

def run_demo(embedding_type: str = "mfcc", linkage: str = "ward", device: str = "cpu"):
    print("="* 65)
    print("Speaker Diarization Pipeline Demo")
    print(f"Configuration: Embedding={embedding_type.upper()}, Linkage={linkage}, Device={device}")
    print("="* 65)

    audio_path = os.path.join("data", "an4_diarize_test.wav")
    ref_rttm_path = os.path.join("data", "an4_diarize_test.rttm")

    if not os.path.exists(audio_path):
        print(f"Sample audio not found at {audio_path}")
        return

    # 1. Load Audio
    signal, sr = load_audio(audio_path, target_sr=16000, mono=True)
    duration = get_audio_duration(signal, sr)
    print(f"[1/5] Loaded {audio_path} ({duration:.2f}s, {sr} Hz)")

    # 2. VAD
    vad = EnergyVAD(min_speech_duration_ms=150, min_silence_duration_ms=50)
    speech_segments = vad.detect_speech(signal, sr=sr)
    print(f"[2/5] Detected {len(speech_segments)} speech segments via VAD:")
    for i, s in enumerate(speech_segments):
        print(f"Segment {i}: {s.start:.2f}s - {s.end:.2f}s (dur: {s.end - s.start:.2f}s)")

    # 3. Feature Extraction
    if embedding_type.lower() == "ecapa":
        extractor = ECAPATDNNEmbeddingExtractor(device=device)
    else:
        extractor = MFCCFeatureExtractor(n_mfcc=20)

    chunks = [signal[int(s.start * sr): int(s.end * sr)] for s in speech_segments]
    intervals = [(s.start, s.end) for s in speech_segments]
    embeddings = extractor.extract_batch(chunks, sr=sr)
    print(f"[3/5] Extracted {embedding_type.upper()} embeddings: matrix shape {embeddings.shape}, dtype {embeddings.dtype}")

    # 4. Pairwise Cosine Similarity Matrix
    sim_mat = cosine_similarity(embeddings)
    print("\n[4/5] Pairwise Cosine Similarity Matrix:")
    print(""+ "".join([f"Seg{i} "for i in range(len(speech_segments))]))
    for i, row in enumerate(sim_mat):
        row_str = "".join([f"{val:.3f}"for val in row])
        print(f"Seg{i}: {row_str}")

    # 5. Clustering Diarization
    diarizer = ClusteringDiarizer(method="agglomerative", num_speakers=2, linkage=linkage)
    labels = diarizer.fit_predict(embeddings)
    pred_rttm_segs = diarizer.assign_speaker_labels(intervals, labels, uri="an4")

    unique, counts = np.unique(labels, return_counts=True)
    dist_str = ", ".join([f"SPEAKER_{u:02d}: {c} turns"for u, c in zip(unique, counts)])
    print(f"\n[5/5] Clustering completed ({dist_str}):")
    for seg in pred_rttm_segs:
        print(f"{seg.to_line()}")

    # Compare against ground-truth if available
    if os.path.exists(ref_rttm_path):
        ref_segs = parse_rttm(ref_rttm_path)
        metrics = compute_der(ref_segs, pred_rttm_segs, collar=0.25)
        print("\nGround-Truth Comparison (NIST pyannote.metrics):")
        print(f"Reference turns : {len(ref_segs)}")
        print(f"Diarization Error Rate (DER): {metrics['der'] * 100:.2f}% (Collar: 0.25s)")
        print(f"Missed Speech   : {metrics['missed_speech'] * 100:.2f}%")
        print(f"False Alarm     : {metrics['false_alarm'] * 100:.2f}%")
        print(f"Speaker Error   : {metrics['speaker_error'] * 100:.2f}%")

    print("="* 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline demo.")
    parser.add_argument("--embedding", type=str, default="mfcc", choices=["mfcc", "ecapa"], help="Embedding type")
    parser.add_argument("--linkage", type=str, default="ward", choices=["ward", "average", "complete"], help="Clustering linkage")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    args = parser.parse_args()
    run_demo(embedding_type=args.embedding, linkage=args.linkage, device=args.device)
