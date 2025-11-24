#!/usr/bin/env python3
"""
NVIDIA NeMo Diarization CLI:
Executes NeMo ClusteringDiarizer with TitaNet speaker embeddings and MarbleNet VAD.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.validation import validate_audio_file
from src.diarization.nemo_diarizer import NeMoDiarizer
from src.diarization.rttm import write_rttm

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    )

def main():
    parser = argparse.ArgumentParser(description="Run NVIDIA NeMo Speaker Diarization.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--config", type=str, default="configs/diar_infer_telephonic.yaml", help="NeMo inference YAML config.")
    parser.add_argument("--out-dir", type=str, default="output_nemo", help="Directory for intermediate/output files.")
    parser.add_argument("--reference-rttm", type=str, default=None, help="Reference RTTM (activates Oracle VAD).")
    parser.add_argument("--num-speakers", type=int, default=None, help="Known speaker count (optional).")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda:0').")
    parser.add_argument("--speaker-model", type=str, default="titanet_large", help="NeMo speaker embedding model name.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("run_nemo_diarize")

    validate_audio_file(args.audio)

    diarizer = NeMoDiarizer(
        config_path=args.config,
        out_dir=args.out_dir,
        device=args.device,
        oracle_vad=bool(args.reference_rttm),
        speaker_model=args.speaker_model,
    )

    logger.info("Executing NeMo Diarization...")
    segments = diarizer.diarize(
        audio_path=args.audio,
        reference_rttm=args.reference_rttm,
        num_speakers=args.num_speakers,
    )

    print("\n"+ "="* 60)
    print(f"NEMO DIARIZATION PREDICTED RTTM ({len(segments)} segments):")
    print("="* 60)
    for seg in segments:
        print(seg.to_line())
    print("="* 60 + "\n")

if __name__ == "__main__":
    main()
