"""
RTTM (Rich Transcription Time Marked) File Utilities
Standard format used in NIST RT evaluations and speaker diarization benchmarks.
Format specification:
SPEAKER <file_id> <channel_id> <tbeg> <tdur> <ortho> <stype> <speaker_name> <conf> <slat>
"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RTTMSegment:
    """Represents a single speaker turn in an RTTM file."""
    uri: str               # File identifier (e.g., 'an4' or audio basename)
    speaker: str           # Speaker label (e.g., 'SPEAKER_00')
    start: float           # Start time in seconds
    duration: float        # Duration in seconds
    channel: int = 1       # Channel ID (default: 1)
    confidence: float = 1.0

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_line(self) -> str:
        """Format as standard NIST RTTM line."""
        return (
            f"SPEAKER {self.uri} {self.channel} {self.start:.3f} {self.duration:.3f} "
            f"<NA> <NA> {self.speaker} <NA> <NA>"
        )

def parse_rttm(rttm_source: str) -> List[RTTMSegment]:
    """
    Parse an RTTM file path or RTTM content string into a list of RTTMSegments.

    Args:
        rttm_source: File path to .rttm file, or raw multi-line RTTM string.

    Returns:
        List of RTTMSegment instances, sorted by start time.
    """
    lines: List[str] = []
    if os.path.isfile(rttm_source):
        with open(rttm_source, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = rttm_source.strip().splitlines()

    segments: List[RTTMSegment] = []
    for line in lines:
        parts = line.strip().split()
        if not parts or parts[0] != "SPEAKER":
            continue
        if len(parts) >= 8:
            uri = parts[1]
            try:
                channel = int(parts[2])
            except ValueError:
                channel = 1
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append(
                RTTMSegment(
                    uri=uri,
                    channel=channel,
                    start=start,
                    duration=duration,
                    speaker=speaker,
                )
            )

    segments.sort(key=lambda s: s.start)
    return segments

def write_rttm(file_path: str, segments: List[RTTMSegment]) -> str:
    """
    Write a list of RTTMSegments to an RTTM file.

    Args:
        file_path: Target destination path.
        segments: List of RTTMSegment instances.

    Returns:
        Absolute path to the created RTTM file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    sorted_segments = sorted(segments, key=lambda s: s.start)
    with open(file_path, "w", encoding="utf-8") as f:
        for seg in sorted_segments:
            f.write(seg.to_line() + "\n")
    return os.path.abspath(file_path)
