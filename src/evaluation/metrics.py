"""
Evaluation Metrics Module
Implements standard speech AI performance benchmarks:
- Word Error Rate (WER) with Levenshtein dynamic programming alignment.
- Diarization Error Rate (DER) adhering to NIST RT guidelines (Collar, Speaker Error, Missed, False Alarm).
- Jaccard Error Rate (JER).
- Real-Time Factor (RTF) and inference throughput.

Uses pyannote.metrics as the primary gold standard when available,
with a robust 2D multi-label binary matrix fallback supporting speaker overlaps.
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..diarization.rttm import RTTMSegment

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text for consistent WER scoring."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "", text).strip()
    return text

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis transcripts.
    Uses jiwer if available; otherwise uses a native Levenshtein distance implementation.

    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    """
    try:
        import jiwer
        return float(jiwer.wer(reference, hypothesis))
    except ImportError:
        pass

    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 0.0 if m == 0 else 1.0

    # Distance matrix of size (n+1) x (m+1)
    d = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,        # Deletion
                d[i, j - 1] + 1,        # Insertion
                d[i - 1, j - 1] + cost  # Substitution
            )

    return float(d[n, m]) / float(n)

def _compute_der_pyannote(
    reference_segments: List[RTTMSegment],
    hypothesis_segments: List[RTTMSegment],
    collar: float = 0.25,
) -> Optional[Dict[str, float]]:
    """
    Compute DER and JER using pyannote.metrics (industry gold standard).
    """
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
        from pyannote.core import Annotation, Segment

        uri = reference_segments[0].uri if reference_segments else "audio"

        ref_ann = Annotation(uri=uri)
        for s in reference_segments:
            if s.end > s.start:
                ref_ann[Segment(s.start, s.end)] = s.speaker

        hyp_ann = Annotation(uri=uri)
        for s in hypothesis_segments:
            if s.end > s.start:
                hyp_ann[Segment(s.start, s.end)] = s.speaker

        der_metric = DiarizationErrorRate(collar=collar)
        der_val = float(der_metric(ref_ann, hyp_ann))

        details = der_metric.accumulated_
        total_time = float(details.get("total", 0.0))
        miss = float(details.get("missed detection", 0.0))
        fa = float(details.get("false alarm", 0.0))
        conf = float(details.get("confusion", 0.0))

        miss_rate = miss / total_time if total_time > 0 else 0.0
        fa_rate = fa / total_time if total_time > 0 else 0.0
        spk_err_rate = conf / total_time if total_time > 0 else 0.0

        try:
            jer_metric = JaccardErrorRate(collar=collar)
            jer_val = float(jer_metric(ref_ann, hyp_ann))
        except Exception:
            jer_val = 0.0

        return {
            "der": round(der_val, 4),
            "jer": round(jer_val, 4),
            "missed_speech": round(miss_rate, 4),
            "false_alarm": round(fa_rate, 4),
            "speaker_error": round(spk_err_rate, 4),
            "total_ref_time": round(total_time, 2),
        }
    except Exception as e:
        logger.debug("pyannote.metrics DER computation unavailable or failed: %s", e)
        return None

def compute_der(
    reference_segments: List[RTTMSegment],
    hypothesis_segments: List[RTTMSegment],
    collar: float = 0.25,
    resolution: float = 0.01,
) -> Dict[str, float]:
    """
    Calculate Diarization Error Rate (DER) according to NIST standards.
    Supports overlapping speakers and boundary collar forgiveness.

    Components:
    - Missed Speech (MS): Reference speech marked as non-speech in hypothesis.
    - False Alarm (FA): Non-speech in reference marked as speech in hypothesis.
    - Speaker Error (SE): Speech attributed to wrong speaker identity.
    - DER = (MS + FA + SE) / Total Reference Duration

    Args:
        reference_segments: Ground-truth RTTMSegments.
        hypothesis_segments: Predicted RTTMSegments.
        collar: Forgiveness collar around segment boundaries in seconds (default: 0.25s).
        resolution: Time frame quantization in seconds (default: 10ms).

    Returns:
        Dictionary with 'der', 'jer', 'missed_speech', 'false_alarm', 'speaker_error', and 'total_ref_time'.
    """
    if not reference_segments:
        raise ValueError("Reference segments must not be empty to compute DER.")

    # Try pyannote.metrics gold standard first
    res = _compute_der_pyannote(reference_segments, hypothesis_segments, collar=collar)
    if res is not None:
        return res

    # Robust 2D multi-label binary matrix fallback (handles overlap & multi-speaker)
    ref_speakers = sorted(list(set(s.speaker for s in reference_segments)))
    hyp_speakers = sorted(list(set(s.speaker for s in hypothesis_segments))) if hypothesis_segments else []

    max_ref = max([s.end for s in reference_segments], default=0.0)
    max_hyp = max([s.end for s in hypothesis_segments], default=0.0) if hypothesis_segments else 0.0
    total_time = max(max_ref, max_hyp) + collar + 1.0
    num_frames = int(np.ceil(total_time / resolution))
    collar_frames = int(np.round(collar / resolution))

    # Multi-label reference matrix: (num_ref_speakers, num_frames)
    ref_mat = np.zeros((len(ref_speakers), num_frames), dtype=bool)
    collar_mask = np.zeros(num_frames, dtype=bool)

    for seg in reference_segments:
        if seg.end <= seg.start:
            continue
        spk_idx = ref_speakers.index(seg.speaker)
        s_f = int(np.round(seg.start / resolution))
        e_f = int(np.round(seg.end / resolution))
        ref_mat[spk_idx, s_f:e_f] = True

        # Apply collar to transition boundaries
        if collar_frames > 0:
            collar_mask[max(0, s_f - collar_frames):min(num_frames, s_f + collar_frames)] = True
            collar_mask[max(0, e_f - collar_frames):min(num_frames, e_f + collar_frames)] = True

    # Multi-label hypothesis matrix: (num_hyp_speakers, num_frames)
    hyp_mat = np.zeros((len(hyp_speakers), num_frames), dtype=bool)
    for seg in hypothesis_segments:
        if seg.end <= seg.start:
            continue
        spk_idx = hyp_speakers.index(seg.speaker)
        s_f = int(np.round(seg.start / resolution))
        e_f = int(np.round(seg.end / resolution))
        hyp_mat[spk_idx, s_f:e_f] = True

    valid_frames = ~collar_mask
    ref_valid = ref_mat[:, valid_frames]
    hyp_valid = hyp_mat[:, valid_frames]

    total_ref_frames = int(np.sum(ref_valid))
    if total_ref_frames == 0:
        return {
            "der": 0.0,
            "jer": 0.0,
            "missed_speech": 0.0,
            "false_alarm": 0.0,
            "speaker_error": 0.0,
            "total_ref_time": 0.0,
        }

    # Hungarian alignment on intersection matrix
    num_r = len(ref_speakers)
    num_h = len(hyp_speakers)
    cost = np.zeros((num_r, max(num_h, 1)), dtype=int)
    for r in range(num_r):
        for h in range(num_h):
            cost[r, h] = -int(np.sum(ref_valid[r] & hyp_valid[h]))

    r_ind, h_ind = linear_sum_assignment(cost)
    mapping = {h: r for r, h in zip(r_ind, h_ind) if h < num_h}

    # Frame-by-frame speaker counts
    N_ref = np.sum(ref_valid, axis=0)  # active ref speakers per frame
    N_hyp = np.sum(hyp_valid, axis=0)  # active hyp speakers per frame

    # Correct speaker-match frames
    N_corr = np.zeros(ref_valid.shape[1], dtype=int)
    for h, r in mapping.items():
        N_corr += (ref_valid[r] & hyp_valid[h]).astype(int)

    # NIST error decomposition
    miss_frames = np.sum(np.maximum(0, N_ref - N_hyp))
    fa_frames = np.sum(np.maximum(0, N_hyp - N_ref))
    conf_frames = np.sum(np.minimum(N_ref, N_hyp) - N_corr)

    miss_rate = float(miss_frames / total_ref_frames)
    fa_rate = float(fa_frames / total_ref_frames)
    spk_err_rate = float(conf_frames / total_ref_frames)
    der = miss_rate + fa_rate + spk_err_rate
    total_ref_sec = float(total_ref_frames * resolution)

    # Jaccard Error Rate (JER)
    speaker_jers = []
    for r_id in range(num_r):
        h_matched = [h for h, r in mapping.items() if r == r_id]
        if h_matched and h_matched[0] < num_h:
            h_id = h_matched[0]
            intersection = np.sum(ref_valid[r_id] & hyp_valid[h_id])
            union = np.sum(ref_valid[r_id] | hyp_valid[h_id])
            speaker_jers.append(1.0 - (intersection / union) if union > 0 else 0.0)
        else:
            speaker_jers.append(1.0)

    jer = float(np.mean(speaker_jers)) if speaker_jers else 0.0

    return {
        "der": round(der, 4),
        "jer": round(jer, 4),
        "missed_speech": round(miss_rate, 4),
        "false_alarm": round(fa_rate, 4),
        "speaker_error": round(spk_err_rate, 4),
        "total_ref_time": round(total_ref_sec, 2),
    }

def compute_rtf(audio_duration: float, processing_time: float) -> float:
    """
    Compute Real-Time Factor (RTF).
    RTF = processing_time / audio_duration
    Values < 1.0 indicate faster-than-realtime execution.
    """
    if audio_duration <= 0.0:
        return 0.0
    return round(processing_time / audio_duration, 4)
