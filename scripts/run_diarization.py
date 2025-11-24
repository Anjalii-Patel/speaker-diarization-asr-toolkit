#!/usr/bin/env python3
"""
Speaker Diarization CLI:
Detects speech segments and assigns speaker cluster IDs to produce standard RTTM files.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.io import load_audio, get_audio_duration
from src.audio.validation import validate_audio_file
from src.vad.silero import SileroVAD
from src.vad.energy import EnergyVAD
from src.embeddings.mfcc import MFCCFeatureExtractor
from src.embeddings.ecapa_tdnn import ECAPATDNNEmbeddingExtractor
from src.diarization.clustering_diarizer import ClusteringDiarizer
from src.diarization.rttm import write_rttm

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    )

def main():
    parser = argparse.ArgumentParser(description="Run clustering-based speaker diarization.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--vad", type=str, default="silero", choices=["silero", "energy"], help="VAD engine.")
    parser.add_argument("--embedding", type=str, default="mfcc", choices=["mfcc", "ecapa_tdnn"], help="Feature extraction method.")
    parser.add_argument("--clusterer", type=str, default="agglomerative", choices=["agglomerative", "spectral", "dbscan"], help="Clustering method.")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers (optional).")
    parser.add_argument("--linkage", type=str, default="ward", choices=["ward", "average", "complete", "single"], help="Agglomerative linkage.")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda').")
    parser.add_argument("--output-rttm", type=str, default=None, help="Output RTTM destination path.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("run_diarization")

    validate_audio_file(args.audio)
    logger.info("Loading audio: %s", args.audio)
    signal, sr = load_audio(args.audio, target_sr=16000, mono=True)
    duration = get_audio_duration(signal, sr)
    logger.info("Audio duration: %.2f seconds.", duration)

    # 1. VAD
    vad = SileroVAD(device=args.device) if args.vad == "silero"else EnergyVAD()
    logger.info("Running %s VAD...", args.vad.upper())
    speech_segments = vad.detect_speech(signal, sr=sr)
    logger.info("Detected %d speech segments.", len(speech_segments))

    if not speech_segments:
        logger.warning("No speech detected. Exiting.")
        sys.exit(0)

    # 2. Extract features
    logger.info("Extracting embeddings using %s...", args.embedding)
    if args.embedding == "mfcc":
        extractor = MFCCFeatureExtractor()
    else:
        extractor = ECAPATDNNEmbeddingExtractor(device=args.device)

    chunks = []
    intervals = []
    for s in speech_segments:
        s_idx = int(max(0.0, s.start) * sr)
        e_idx = int(min(duration, s.end) * sr)
        if e_idx > s_idx:
            chunks.append(signal[s_idx:e_idx])
            intervals.append((s.start, s.end))

    embeddings = extractor.extract_batch(chunks, sr=sr)

    # 3. Clustering
    diarizer = ClusteringDiarizer(
        method=args.clusterer,
        num_speakers=args.num_speakers,
        linkage=args.linkage,
        metric=args.metric,
    )
    labels = diarizer.fit_predict(embeddings)
    uri = os.path.splitext(os.path.basename(args.audio))[0]
    rttm_segments = diarizer.assign_speaker_labels(intervals, labels, uri=uri)

    print("\n"+ "="* 60)
    print("DIARIZATION RESULTS (RTTM Format):")
    print("="* 60)
    for seg in rttm_segments:
        print(seg.to_line())
    print("="* 60 + "\n")

    out_rttm = args.output_rttm or f"{uri}_diarization.rttm"
    write_rttm(out_rttm, rttm_segments)
    logger.info("RTTM successfully saved to %s", out_rttm)

if __name__ == "__main__":
    main()
