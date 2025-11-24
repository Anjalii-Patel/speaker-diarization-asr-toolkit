"""
Synthetic & Empirical Multi-Speaker Evaluation Dataset Generator
Generates an expanded benchmark test suite across 6 conditions with 3 independent trials each:
- 2-Speaker Clean (3 trials)
- 2-Speaker Noisy (3 trials, 12dB SNR colored noise)
- 3-Speaker Clean (3 trials)
- 3-Speaker Noisy (3 trials, 12dB SNR colored noise)
- 4-Speaker Clean (3 trials)
- 4-Speaker Noisy (3 trials, 12dB SNR colored noise)

Total: 18 independent audio recordings with reference RTTMs and transcripts.
Ensures generalization testing with mean +/- std reporting.
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.io import load_audio
from src.diarization.rttm import RTTMSegment, write_rttm

def add_colored_noise(signal: np.ndarray, target_snr_db: float = 12.0, seed: int = 42) -> np.ndarray:
    """Add pink/colored noise at target SNR (dB)."""
    np.random.seed(seed)
    white = np.random.normal(0, 1, len(signal))
    # 1/f filter approximation for pink noise
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter
    pink = lfilter(b, a, white)

    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(pink ** 2)
    if noise_power < 1e-10:
        return signal

    factor = np.sqrt(sig_power / (10 ** (target_snr_db / 10.0) * noise_power))
    noisy = signal + factor * pink
    # Prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 0.95:
        noisy = (noisy / max_val) * 0.95
    return noisy.astype(np.float32)

def generate_suite(out_dir: str = "data/eval_suite"):
    os.makedirs(out_dir, exist_ok=True)
    sr = 16000

    # Load base acoustic sources
    an4_sig, _ = load_audio("data/an4_diarize_test.wav", target_sr=sr)
    # Split an4 into distinct speaker utterances
    spk_a_raw = an4_sig[int(0.3 * sr):int(2.8 * sr)]  # 2.5s
    spk_b_raw = an4_sig[int(3.2 * sr):int(5.1 * sr)]  # 1.9s

    sample_sig, _ = load_audio("data/conversation_2spk_clean_30s.wav", target_sr=sr)
    # Distinct sections of sample.wav (Diane and Sheila)
    # Diane utterance: 10.57s to 14.49s (~3.9s)
    spk_diane = sample_sig[int(10.6 * sr):int(14.4 * sr)]
    # Sheila utterance: 14.49s to 17.92s (~3.4s)
    spk_sheila = sample_sig[int(14.5 * sr):int(17.9 * sr)]

    # Acoustic transformations for additional speakers
    spk_c = librosa.effects.pitch_shift(spk_a_raw, sr=sr, n_steps=-3.5)
    spk_d = librosa.effects.pitch_shift(spk_b_raw, sr=sr, n_steps=4.0)

    speakers_bank = {
        "SPK_A": {"audio": spk_a_raw, "text": "11 27 57"},
        "SPK_B": {"audio": spk_b_raw, "text": "October 24 1970"},
        "SPK_C": {"audio": spk_c, "text": "11 27 57"},
        "SPK_D": {"audio": spk_d, "text": "October 24 1970"},
        "SPK_DIANE": {"audio": spk_diane, "text": "Okay I thought you know I heard a beep this is Diane in New Jersey"},
        "SPK_SHEILA": {"audio": spk_sheila, "text": "and I'm Sheila in Texas originally from Chicago"},
    }

    # Configuration for trials
    scenarios = [
        # (condition, trial_id, speakers_used, turn_sequence, total_dur)
        ("2spk", 1, ["SPK_DIANE", "SPK_SHEILA"], [
            ("SPK_DIANE", 1.0, 3.8),
            ("SPK_SHEILA", 5.2, 3.4),
            ("SPK_DIANE", 9.0, 3.8),
            ("SPK_SHEILA", 13.2, 3.4),
            ("SPK_DIANE", 17.0, 3.8),
            ("SPK_SHEILA", 21.2, 3.4),
        ], 26.0),
        ("2spk", 2, ["SPK_A", "SPK_B"], [
            ("SPK_A", 0.5, 2.5),
            ("SPK_B", 3.5, 1.9),
            ("SPK_A", 6.0, 2.5),
            ("SPK_B", 9.0, 1.9),
            ("SPK_A", 11.5, 2.5),
            ("SPK_B", 14.5, 1.9),
            ("SPK_A", 17.0, 2.5),
            ("SPK_B", 20.0, 1.9),
        ], 23.0),
        ("2spk", 3, ["SPK_C", "SPK_D"], [
            ("SPK_C", 1.0, 2.5),
            ("SPK_D", 4.0, 1.9),
            ("SPK_C", 6.5, 2.5),
            ("SPK_D", 9.5, 1.9),
            ("SPK_C", 12.0, 2.5),
            ("SPK_D", 15.0, 1.9),
            ("SPK_C", 17.5, 2.5),
        ], 21.0),

        # 3 Speakers
        ("3spk", 1, ["SPK_A", "SPK_B", "SPK_C"], [
            ("SPK_A", 0.5, 2.5),
            ("SPK_B", 3.5, 1.9),
            ("SPK_C", 6.0, 2.5),
            ("SPK_A", 9.0, 2.5),
            ("SPK_B", 12.0, 1.9),
            ("SPK_C", 14.5, 2.5),
            ("SPK_A", 17.5, 2.5),
            ("SPK_B", 20.5, 1.9),
        ], 24.0),
        ("3spk", 2, ["SPK_DIANE", "SPK_SHEILA", "SPK_D"], [
            ("SPK_DIANE", 0.8, 3.8),
            ("SPK_SHEILA", 5.0, 3.4),
            ("SPK_D", 8.8, 1.9),
            ("SPK_DIANE", 11.2, 3.8),
            ("SPK_SHEILA", 15.5, 3.4),
            ("SPK_D", 19.3, 1.9),
            ("SPK_DIANE", 21.8, 3.8),
        ], 26.5),
        ("3spk", 3, ["SPK_B", "SPK_C", "SPK_D"], [
            ("SPK_B", 0.5, 1.9),
            ("SPK_C", 2.8, 2.5),
            ("SPK_D", 5.8, 1.9),
            ("SPK_B", 8.2, 1.9),
            ("SPK_C", 10.5, 2.5),
            ("SPK_D", 13.5, 1.9),
            ("SPK_B", 16.0, 1.9),
            ("SPK_C", 18.5, 2.5),
        ], 22.0),

        # 4 Speakers
        ("4spk", 1, ["SPK_A", "SPK_B", "SPK_C", "SPK_D"], [
            ("SPK_A", 0.5, 2.5),
            ("SPK_B", 3.5, 1.9),
            ("SPK_C", 6.0, 2.5),
            ("SPK_D", 9.0, 1.9),
            ("SPK_A", 11.5, 2.5),
            ("SPK_B", 14.5, 1.9),
            ("SPK_C", 17.0, 2.5),
            ("SPK_D", 20.0, 1.9),
            ("SPK_A", 22.5, 2.5),
            ("SPK_C", 25.5, 2.5),
        ], 29.0),
        ("4spk", 2, ["SPK_DIANE", "SPK_SHEILA", "SPK_A", "SPK_B"], [
            ("SPK_DIANE", 0.8, 3.8),
            ("SPK_SHEILA", 5.0, 3.4),
            ("SPK_A", 8.8, 2.5),
            ("SPK_B", 11.8, 1.9),
            ("SPK_DIANE", 14.2, 3.8),
            ("SPK_SHEILA", 18.5, 3.4),
            ("SPK_A", 22.3, 2.5),
            ("SPK_B", 25.3, 1.9),
        ], 28.5),
        ("4spk", 3, ["SPK_B", "SPK_C", "SPK_D", "SPK_SHEILA"], [
            ("SPK_B", 0.5, 1.9),
            ("SPK_C", 2.8, 2.5),
            ("SPK_SHEILA", 5.8, 3.4),
            ("SPK_D", 9.6, 1.9),
            ("SPK_B", 12.0, 1.9),
            ("SPK_C", 14.3, 2.5),
            ("SPK_SHEILA", 17.2, 3.4),
            ("SPK_D", 21.0, 1.9),
            ("SPK_C", 23.3, 2.5),
        ], 27.0),
    ]

    manifest = []

    for cond, trial, spk_list, turns, total_dur in scenarios:
        num_spk = len(spk_list)
        base_id = f"{cond}_trial{trial}_{int(total_dur)}s"

        # Construct clean timeline
        total_samples = int(total_dur * sr)
        timeline = np.zeros(total_samples, dtype=np.float32)
        rttm_segs = []
        transcript_lines = []

        for spk, start_t, dur_t in turns:
            spk_info = speakers_bank[spk]
            chunk = spk_info["audio"]
            chunk_len = len(chunk)
            s_idx = int(start_t * sr)
            e_idx = s_idx + chunk_len
            if e_idx > total_samples:
                chunk = chunk[:total_samples - s_idx]
                e_idx = total_samples

            timeline[s_idx:e_idx] += chunk
            act_dur = len(chunk) / sr

            rttm_segs.append(
                RTTMSegment(
                    uri=base_id,
                    channel=1,
                    start=round(start_t, 3),
                    duration=round(act_dur, 3),
                    speaker=spk,
                )
            )
            transcript_lines.append(f"[{start_t:.2f}s - {start_t + act_dur:.2f}s] {spk}: {spk_info['text']}")

        # Normalize clean audio
        max_v = np.max(np.abs(timeline))
        if max_v > 0.95:
            timeline = (timeline / max_v) * 0.95

        # 1. Clean recording
        clean_id = f"{base_id}_clean"
        clean_wav = os.path.join(out_dir, f"{clean_id}.wav")
        clean_rttm = os.path.join(out_dir, f"{clean_id}.rttm")
        clean_txt = os.path.join(out_dir, f"{clean_id}.txt")

        sf.write(clean_wav, timeline, sr)
        # Update URI in RTTM
        for s in rttm_segs:
            s.uri = clean_id
        write_rttm(clean_rttm, rttm_segs)
        with open(clean_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines) + "\n")

        manifest.append({
            "id": clean_id,
            "condition": "Clean",
            "speakers_category": cond,
            "trial": trial,
            "num_speakers": num_spk,
            "audio": clean_wav,
            "ref_rttm": clean_rttm,
            "ref_text": clean_txt,
            "duration": total_dur,
        })

        # 2. Noisy recording (12dB SNR colored noise)
        noisy_id = f"{base_id}_noisy"
        noisy_wav = os.path.join(out_dir, f"{noisy_id}.wav")
        noisy_rttm = os.path.join(out_dir, f"{noisy_id}.rttm")
        noisy_txt = os.path.join(out_dir, f"{noisy_id}.txt")

        noisy_timeline = add_colored_noise(timeline, target_snr_db=12.0, seed=trial * 100)
        sf.write(noisy_wav, noisy_timeline, sr)
        for s in rttm_segs:
            s.uri = noisy_id
        write_rttm(noisy_rttm, rttm_segs)
        with open(noisy_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines) + "\n")

        manifest.append({
            "id": noisy_id,
            "condition": "Noisy",
            "speakers_category": cond,
            "trial": trial,
            "num_speakers": num_spk,
            "audio": noisy_wav,
            "ref_rttm": noisy_rttm,
            "ref_text": noisy_txt,
            "duration": total_dur,
        })

    import json
    manifest_path = os.path.join(out_dir, "eval_suite_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated {len(manifest)} test audio recordings in {out_dir}")
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    generate_suite()
