"""Tests for evaluation metrics (WER, DER, JER, and RTF)."""

import pytest
from src.diarization.rttm import RTTMSegment
from src.evaluation.metrics import compute_wer, compute_der, compute_rtf

def test_wer_identical():
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumps over the lazy dog"
    assert compute_wer(ref, hyp) == 0.0

def test_wer_known_errors():
    ref = "speech recognition and diarization"
    # 1 substitution ("in"for "and") out of 4 words -> 0.25 WER
    hyp = "speech recognition in diarization"
    assert compute_wer(ref, hyp) == 0.25

    # 1 deletion ("diarization") -> 0.25 WER
    hyp2 = "speech recognition and"
    assert compute_wer(ref, hyp2) == 0.25

    # 1 insertion ("very") -> 0.25 WER
    hyp3 = "speech recognition and diarization very"
    assert compute_wer(ref, hyp3) == 0.25

def test_der_identical_segments():
    ref_segs = [
        RTTMSegment(uri="rec", speaker="A", start=0.0, duration=2.0),
        RTTMSegment(uri="rec", speaker="B", start=3.0, duration=2.0),
    ]
    hyp_segs = [
        RTTMSegment(uri="rec", speaker="SPK_1", start=0.0, duration=2.0),
        RTTMSegment(uri="rec", speaker="SPK_2", start=3.0, duration=2.0),
    ]
    # Despite different speaker label strings ("A"vs "SPK_1"), optimal mapping aligns them perfectly
    metrics = compute_der(ref_segs, hyp_segs, collar=0.0)
    assert metrics["der"] == 0.0
    assert metrics["speaker_error"] == 0.0
    assert metrics["missed_speech"] == 0.0
    assert metrics["false_alarm"] == 0.0

def test_der_missed_speech():
    ref_segs = [
        RTTMSegment(uri="rec", speaker="A", start=0.0, duration=10.0),
    ]
    # Hypothesis is half duration
    hyp_segs = [
        RTTMSegment(uri="rec", speaker="A", start=0.0, duration=5.0),
    ]
    metrics = compute_der(ref_segs, hyp_segs, collar=0.0)
    assert metrics["speaker_error"] == 0.0
    assert abs(metrics["missed_speech"] - 0.5) < 0.05
    assert abs(metrics["der"] - 0.5) < 0.05

def test_der_false_alarm():
    ref_segs = [
        RTTMSegment(uri="rec", speaker="A", start=0.0, duration=5.0),
    ]
    # Hypothesis predicts 10 seconds (5s reference + 5s non-speech)
    hyp_segs = [
        RTTMSegment(uri="rec", speaker="A", start=0.0, duration=10.0),
    ]
    metrics = compute_der(ref_segs, hyp_segs, collar=0.0)
    assert metrics["speaker_error"] == 0.0
    assert metrics["missed_speech"] == 0.0
    assert abs(metrics["false_alarm"] - 1.0) < 0.05  # 5s FA / 5s ref = 100% FA
    assert abs(metrics["der"] - 1.0) < 0.05

def test_der_overlapping_speakers():
    """Ensure multi-speaker overlap is handled accurately."""
    ref_segs = [
        RTTMSegment(uri="rec", speaker="A", start=0.0, duration=4.0),
        RTTMSegment(uri="rec", speaker="B", start=2.0, duration=4.0),  # Overlap between 2.0s and 4.0s
    ]
    # Hypothesis correctly detects both speakers
    hyp_segs = [
        RTTMSegment(uri="rec", speaker="SPK_1", start=0.0, duration=4.0),
        RTTMSegment(uri="rec", speaker="SPK_2", start=2.0, duration=4.0),
    ]
    metrics = compute_der(ref_segs, hyp_segs, collar=0.0)
    assert metrics["der"] == 0.0
    assert metrics["total_ref_time"] == 8.0  # 4s + 4s = 8s total speech time

def test_compute_rtf():
    # 10s audio processed in 2s -> RTF 0.2
    assert compute_rtf(10.0, 2.0) == 0.2
    assert compute_rtf(0.0, 5.0) == 0.0
