"""Tests for audio IO, duration, and validation."""

import os
import tempfile
import numpy as np
import pytest

from src.audio.io import load_audio, save_audio, get_audio_duration
from src.audio.validation import validate_audio_file, validate_sample_rate

def test_save_and_load_audio_roundtrip():
    sr = 16000
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Generate 440 Hz test tone
    test_signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_wav = os.path.join(tmp_dir, "test_tone.wav")
        saved_path = save_audio(tmp_wav, test_signal, sr=sr)
        assert os.path.exists(saved_path)

        loaded_signal, loaded_sr = load_audio(saved_path, target_sr=16000, mono=True)
        assert loaded_sr == 16000
        assert len(loaded_signal) == len(test_signal)
        # Check signal fidelity within float-int16 quantization limits
        np.testing.assert_allclose(loaded_signal, test_signal, atol=1e-3)

def test_get_audio_duration():
    sr = 16000
    signal = np.zeros(32000, dtype=np.float32)
    duration = get_audio_duration(signal, sr)
    assert duration == 2.0

    empty_signal = np.zeros(0, dtype=np.float32)
    assert get_audio_duration(empty_signal, sr) == 0.0

def test_validate_audio_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        non_existent = os.path.join(tmp_dir, "does_not_exist.wav")
        with pytest.raises(FileNotFoundError):
            validate_audio_file(non_existent)

        empty_file = os.path.join(tmp_dir, "empty.wav")
        with open(empty_file, "wb") as f:
            pass
        with pytest.raises(ValueError):
            validate_audio_file(empty_file)

        unsupported = os.path.join(tmp_dir, "file.xyz")
        with open(unsupported, "w") as f:
            f.write("content")
        with pytest.raises(ValueError):
            validate_audio_file(unsupported)

def test_validate_sample_rate():
    validate_sample_rate(16000, 16000)
    with pytest.raises(ValueError):
        validate_sample_rate(8000, 16000)
