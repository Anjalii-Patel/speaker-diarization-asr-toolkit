#!/usr/bin/env python3
"""
Evaluation CLI:
Evaluates Word Error Rate (WER) and Diarization Error Rate (DER) against ground-truth references.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diarization.rttm import parse_rttm
from src.evaluation.metrics import compute_wer, compute_der

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    )

def read_text_file_or_string(text_or_path: str) -> str:
    if os.path.isfile(text_or_path):
        with open(text_or_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return text_or_path.strip()

def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR (WER) and Diarization (DER).")
    parser.add_argument("--ref-rttm", type=str, default=None, help="Reference (ground-truth) RTTM file path.")
    parser.add_argument("--hyp-rttm", type=str, default=None, help="Hypothesis (predicted) RTTM file path.")
    parser.add_argument("--collar", type=float, default=0.25, help="DER boundary forgiveness collar in seconds (default: 0.25s).")
    parser.add_argument("--ref-transcript", type=str, default=None, help="Reference transcript text or file path.")
    parser.add_argument("--hyp-transcript", type=str, default=None, help="Hypothesis transcript text or file path.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON report path.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("evaluate")

    has_rttm = args.ref_rttm is not None and args.hyp_rttm is not None
    has_text = args.ref_transcript is not None and args.hyp_transcript is not None

    if not has_rttm and not has_text:
        logger.error(
            "No evaluation pairs provided. Provide either (--ref-rttm and --hyp-rttm) for DER, "
            "or (--ref-transcript and --hyp-transcript) for WER."
        )
        sys.exit(1)

    print("\n"+ "="* 55)
    print("EVALUATION REPORT")
    print("="* 55)

    if has_rttm:
        if not os.path.exists(args.ref_rttm):
            raise FileNotFoundError(f"Reference RTTM not found: {args.ref_rttm}")
        if not os.path.exists(args.hyp_rttm):
            raise FileNotFoundError(f"Hypothesis RTTM not found: {args.hyp_rttm}")

        ref_segs = parse_rttm(args.ref_rttm)
        hyp_segs = parse_rttm(args.hyp_rttm)

        der_metrics = compute_der(ref_segs, hyp_segs, collar=args.collar)
        print(f"DIARIZATION PERFORMANCE (Collar: {args.collar}s):")
        print(f"* Diarization Error Rate (DER) : {der_metrics['der'] * 100:.2f}%")
        print(f"* Jaccard Error Rate (JER)     : {der_metrics['jer'] * 100:.2f}%")
        print(f"* Missed Speech Rate (MS)     : {der_metrics['missed_speech'] * 100:.2f}%")
        print(f"* False Alarm Rate (FA)       : {der_metrics['false_alarm'] * 100:.2f}%")
        print(f"* Speaker Error Rate (SE)     : {der_metrics['speaker_error'] * 100:.2f}%")
        print(f"* Evaluated Speech Time       : {der_metrics['total_ref_time']:.2f} seconds")
        print("-"* 55)

    if has_text:
        ref_txt = read_text_file_or_string(args.ref_transcript)
        hyp_txt = read_text_file_or_string(args.hyp_transcript)
        wer_score = compute_wer(ref_txt, hyp_txt)
        print("ASR TRANSCRIPTION PERFORMANCE:")
        print(f"* Word Error Rate (WER)       : {wer_score * 100:.2f}%")
        print("-"* 55)

    if args.output_json:
        import json
        report = {}
        if has_rttm:
            report["diarization"] = der_metrics
            report["collar_sec"] = args.collar
        if has_text:
            report["asr"] = {"wer": round(wer_score, 4)}
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved evaluation JSON report to %s", args.output_json)

    print("="* 55 + "\n")

if __name__ == "__main__":
    main()
