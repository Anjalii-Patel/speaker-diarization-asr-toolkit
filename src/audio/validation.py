"""
Audio Validation Utilities
Ensures input files conform to expected specifications prior to processing.
"""

import os
from typing import Optional, List
import numpy as np

SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]

def validate_audio_file(
    file_path: str,
    min_duration: float = 0.1,
    max_duration: Optional[float] = None,
    supported_extensions: Optional[List[str]] = None,
) -> bool:
    """
    Validate that an audio file exists, has a valid extension, and is non-empty.

    Args:
        file_path: Path to the audio file.
        min_duration: Minimum acceptable audio duration in seconds.
        max_duration: Maximum acceptable audio duration in seconds (optional).
        supported_extensions: Allowed file extensions.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file format, size, or duration is invalid.

    Returns:
        True if valid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file does not exist: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a regular file: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Audio file is empty (0 bytes): {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    valid_exts = supported_extensions or SUPPORTED_AUDIO_EXTENSIONS
    if ext not in valid_exts:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Expected one of: {', '.join(valid_exts)}"
        )

    return True

def validate_sample_rate(actual_sr: int, expected_sr: int = 16000) -> None:
    """
    Validate that the sample rate matches the expected rate.

    Args:
        actual_sr: The actual sample rate of the signal.
        expected_sr: Expected sample rate (typically 16000 Hz for ASR/Diarization models).

    Raises:
        ValueError: If the rates do not match.
    """
    if actual_sr != expected_sr:
        raise ValueError(
            f"Sample rate mismatch: received {actual_sr} Hz, expected {expected_sr} Hz. "
            f"Please resample before passing to downstream models."
        )
