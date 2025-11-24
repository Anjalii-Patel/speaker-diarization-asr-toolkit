#!/usr/bin/env python3
"""
Speaker-Attributed Transcription CLI:
Executes end-to-end VAD, clustering diarization, and Whisper ASR to generate
speaker-attributed conversational transcripts.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.validation import validate_audio_file
from src.vad.silero import SileroVAD
from src.vad.energy import EnergyVAD
from src.embeddings.mfcc import MFCCFeatureExtractor
from src.embeddings.ecapa_tdnn import ECAPATDNNEmbeddingExtractor
from src.diarization.clustering_diarizer import ClusteringDiarizer
from src.asr.whisper_asr import WhisperASR
from src.pipeline.speaker_attributed_asr import SpeakerAttributedASRPipeline

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    )

def main():
    parser = argparse.ArgumentParser(description="End-to-End Speaker-Attributed Transcription.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers.")
    parser.add_argument("--vad", type=str, default="silero", choices=["silero", "energy"], help="VAD model.")
    parser.add_argument("--embedding", type=str, default="mfcc", choices=["mfcc", "ecapa_tdnn"], help="Feature extractor.")
    parser.add_argument("--asr-model", type=str, default="openai/whisper-base", help="Whisper model name.")
    parser.add_argument("--language", type=str, default="en", help="Target spoken language.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda').")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output files.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("run_attributed_transcription")

    validate_audio_file(args.audio)

    # Initialize submodules
    vad = SileroVAD(device=args.device) if args.vad == "silero"else EnergyVAD()
    embedding = MFCCFeatureExtractor() if args.embedding == "mfcc"else ECAPATDNNEmbeddingExtractor(device=args.device)
    diarizer = ClusteringDiarizer(num_speakers=args.num_speakers)
    asr = WhisperASR(model_name=args.asr_model, device=args.device, language=args.language)

    pipeline = SpeakerAttributedASRPipeline(
        vad=vad,
        embedding_extractor=embedding,
        diarizer=diarizer,
        asr=asr,
        device=args.device,
    )

    logger.info("Executing pipeline on %s...", args.audio)
    result = pipeline.run(
        audio_path=args.audio,
        num_speakers=args.num_speakers,
        language=args.language,
        output_dir=args.output_dir,
    )

    print("\n"+ "="* 65)
    print("SPEAKER-ATTRIBUTED TRANSCRIPT:")
    print("="* 65)
    print(result.to_formatted_transcript())
    print("="* 65)
    print(f"Summary: {result.num_speakers_detected} speakers, {result.num_segments} turns, "
          f"RTF: {result.rtf} ({result.total_processing_time:.2f}s for {result.audio_duration:.2f}s audio)")
    print("="* 65 + "\n")

if __name__ == "__main__":
    main()
