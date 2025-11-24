#!/usr/bin/env python3
"""
ASR CLI: Transcribe audio files using OpenAI Whisper models.
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
from src.asr.whisper_asr import WhisperASR

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    )

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper ASR.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--model", type=str, default="openai/whisper-base", help="Whisper model name.")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda').")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g., 'en', 'es').")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task.")
    parser.add_argument("--output", type=str, default=None, help="Optional output text file path.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("run_asr")

    validate_audio_file(args.audio)
    logger.info("Loading audio: %s", args.audio)
    signal, sr = load_audio(args.audio, target_sr=16000, mono=True)
    duration = get_audio_duration(signal, sr)
    logger.info("Loaded audio duration: %.2f seconds.", duration)

    asr = WhisperASR(
        model_name=args.model,
        device=args.device,
        language=args.language,
        task=args.task,
    )
    logger.info("Transcribing audio with model '%s'...", args.model)
    transcription = asr.transcribe(signal, sr=sr)

    print("\n"+ "="* 50)
    print("TRANSCRIPTION:")
    print("="* 50)
    print(transcription)
    print("="* 50 + "\n")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(transcription + "\n")
        logger.info("Transcription written to %s", args.output)

if __name__ == "__main__":
    main()
