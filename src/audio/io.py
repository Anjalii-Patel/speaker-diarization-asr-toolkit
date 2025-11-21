"""
Audio Input/Output Module
Provides robust, multi-backend audio reading, writing, resampling, and format conversion.
"""

import os
import logging
from typing import Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

def load_audio(
    file_path: str,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono if requested, and resample to target_sr.

    Backends supported with automatic fallback:
    1. torchaudio
    2. soundfile
    3. scipy.io.wavfile (standard PCM WAV)

    Args:
        file_path: Path to the audio file.
        target_sr: Desired output sampling rate (default: 16000 Hz).
        mono: Whether to downmix multi-channel audio to mono (default: True).

    Returns:
        Tuple of (audio_signal: np.ndarray [float32, 1D if mono], sample_rate: int).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    signal = None
    orig_sr = None

    # Backend 1: torchaudio
    try:
        import torch
        import torchaudio

        waveform, orig_sr = torchaudio.load(file_path)
        # waveform is [channels, time]
        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif not mono and waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            waveform = resampler(waveform)
            orig_sr = target_sr

        signal = waveform.squeeze().cpu().numpy().astype(np.float32)
        return signal, target_sr
    except Exception as e_torch:
        logger.debug("torchaudio failed (%s), trying soundfile...", e_torch)

    # Backend 2: soundfile
    try:
        import soundfile as sf

        audio_data, orig_sr = sf.read(file_path, dtype="float32")
        if mono and audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        signal = audio_data
    except Exception as e_sf:
        logger.debug("soundfile failed (%s), trying scipy.io.wavfile...", e_sf)

    # Backend 3: scipy.io.wavfile
    if signal is None:
        try:
            import scipy.io.wavfile as wav

            orig_sr, audio_data = wav.read(file_path)
            # Convert integer types to float32 [-1.0, 1.0]
            if audio_data.dtype == np.int16:
                signal = (audio_data / 32768.0).astype(np.float32)
            elif audio_data.dtype == np.int32:
                signal = (audio_data / 2147483648.0).astype(np.float32)
            elif audio_data.dtype == np.uint8:
                signal = ((audio_data - 128) / 128.0).astype(np.float32)
            else:
                signal = audio_data.astype(np.float32)

            if mono and signal.ndim > 1:
                signal = np.mean(signal, axis=1)
        except Exception as e_wav:
            raise RuntimeError(
                f"Failed to load audio file {file_path} using torchaudio, soundfile, or scipy: {e_wav}"
            )

    # Resample if not target_sr
    if orig_sr != target_sr:
        from scipy import signal as scipy_signal

        num_target_samples = int(len(signal) * target_sr / orig_sr)
        signal = scipy_signal.resample(signal, num_target_samples).astype(np.float32)

    return signal, target_sr

def save_audio(
    file_path: str,
    audio_data: np.ndarray,
    sr: int = 16000,
) -> str:
    """
    Save floating point or integer audio data to a WAV file.

    Args:
        file_path: Target destination path.
        audio_data: 1D or 2D numpy array of audio samples.
        sr: Sampling rate in Hz.

    Returns:
        Absolute path to the saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    # Normalize float to int16 for universal compatibility
    if audio_data.dtype in (np.float32, np.float64):
        # Clip to prevent overflow distortion
        clipped = np.clip(audio_data, -1.0, 1.0)
        pcm_data = (clipped * 32767.0).astype(np.int16)
    else:
        pcm_data = audio_data.astype(np.int16)

    try:
        import scipy.io.wavfile as wav

        wav.write(file_path, sr, pcm_data)
        return os.path.abspath(file_path)
    except Exception:
        pass

    try:
        import soundfile as sf

        sf.write(file_path, audio_data, sr)
        return os.path.abspath(file_path)
    except Exception as e:
        raise IOError(f"Could not save audio to {file_path}: {e}")

def get_audio_duration(audio_data: np.ndarray, sr: int = 16000) -> float:
    """Return the duration of the audio in seconds."""
    if len(audio_data) == 0:
        return 0.0
    return float(len(audio_data)) / float(sr)
