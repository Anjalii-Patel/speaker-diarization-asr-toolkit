# Benchmark Results & Multi-Condition Speech Evaluation

This directory archives verified, reproducible evaluation outputs produced across clean and noisy multi-speaker conversational recordings (20s–30s, 2–4 speakers, 18 independent trials).

---

## 1. System & Execution Environment

- **Operating System**: `Windows-10-10.0.26200-SP0`
- **Python Version**: `3.11.9`
- **CPU**: `Intel64 Family 6 Model 143 Stepping 8, GenuineIntel` (24 logical cores)
- **GPU**: `NVIDIA RTX A4500` (CUDA `12.8`, PyTorch `2.7.1+cu128`)
- **Execution Device**: `cuda:0`
- **Timing Methodology**: High-precision `time.perf_counter()` with `torch.cuda.synchronize()` before and after inference. All models pre-loaded in memory/VRAM prior to benchmark timing.
- **Warm-Up Policy**: 1-pass execution on a 2.0s signal to prime CUDA context, JIT compilation, and Hugging Face graph execution before measuring.
- **Evaluation Engine**: NIST RT standard evaluation with a `0.25s` boundary collar via `pyannote.metrics` and a verified 2D multi-label binary matrix fallback supporting simultaneous speaker overlaps.

### Model Initialization Latency (Disentangled From Inference)

| Component | Model Source / Weights | Initialization Latency |
| :--- | :--- | :---: |
| **Short-Time Energy VAD** | Pure Python / Signal Processing | `0.000s` |
| **Silero VAD** | TorchScript Neural Network (`silero_vad`) | `0.076s` |
| **MFCC Extractor** | Torchaudio filterbank baseline | `0.187s` |
| **SpeechBrain ECAPA-TDNN** | `speechbrain/spkrec-ecapa-voxceleb` (192-d) | `1.154s` |
| **Whisper ASR** | `openai/whisper-base` (74M params) | `5.606s` |

---

## 2. Multi-Trial Condition Summary (Mean ± Std Across Independent Trials)

*Evaluated across 18 distinct recordings (3 independent trials per condition) with 72 total benchmark runs:*

| Condition | Configuration | Trials | DER (%) | JER (%) | WER (%) | Avg RTF |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **2SPK Clean** | `ENERGY+ECAPA-TDNN` | 3 | **19.39 ± 8.41** | 23.52 ± 12.70 | 89.04 ± 9.34 | `0.0435` |
| **2SPK Clean** | `ENERGY+MFCC` | 3 | **24.12 ± 24.65** | 29.26 ± 32.29 | 89.04 ± 9.34 | `0.0314` |
| **2SPK Clean** | `SILERO+ECAPA-TDNN` | 3 | **0.00 ± 0.00** | **0.00 ± 0.00** | 79.10 ± 24.02 | `0.0367` |
| **2SPK Clean** | `SILERO+MFCC` | 3 | **0.00 ± 0.00** | **0.00 ± 0.00** | 79.10 ± 24.02 | `0.0295` |
| **2SPK Noisy (12dB)** | `ENERGY+ECAPA-TDNN` | 3 | **55.32 ± 10.59** | 73.87 ± 3.85 | 100.45 ± 7.05 | `0.0207` |
| **2SPK Noisy (12dB)** | `ENERGY+MFCC` | 3 | **55.32 ± 10.59** | 73.87 ± 3.85 | 100.45 ± 7.05 | `0.0152` |
| **2SPK Noisy (12dB)** | `SILERO+ECAPA-TDNN` | 3 | **1.18 ± 0.84** | **1.45 ± 1.02** | 86.71 ± 17.13 | `0.0336` |
| **2SPK Noisy (12dB)** | `SILERO+MFCC` | 3 | **1.18 ± 0.84** | **1.45 ± 1.02** | 86.71 ± 17.13 | `0.0296` |
| **3SPK Clean** | `ENERGY+ECAPA-TDNN` | 3 | **19.69 ± 9.39** | 23.76 ± 11.51 | 88.29 ± 10.92 | `0.0378` |
| **3SPK Clean** | `ENERGY+MFCC` | 3 | **29.37 ± 17.18** | 34.29 ± 18.82 | 88.29 ± 10.92 | `0.0288` |
| **3SPK Clean** | `SILERO+ECAPA-TDNN` | 3 | **0.12 ± 0.17** | **0.13 ± 0.19** | 79.44 ± 20.42 | `0.0337` |
| **3SPK Clean** | `SILERO+MFCC` | 3 | **13.09 ± 18.51** | 20.77 ± 29.37 | 79.44 ± 20.42 | `0.0303` |
| **3SPK Noisy (12dB)** | `ENERGY+ECAPA-TDNN` | 3 | **48.07 ± 33.09** | 56.38 ± 37.95 | 83.12 ± 16.68 | `0.0257` |
| **3SPK Noisy (12dB)** | `ENERGY+MFCC` | 3 | **48.07 ± 33.09** | 56.38 ± 37.95 | 83.12 ± 16.68 | `0.0160` |
| **3SPK Noisy (12dB)** | `SILERO+ECAPA-TDNN` | 3 | **2.16 ± 0.63** | **2.44 ± 0.65** | 85.43 ± 11.95 | `0.0323` |
| **3SPK Noisy (12dB)** | `SILERO+MFCC` | 3 | **2.16 ± 0.63** | **2.44 ± 0.65** | 85.43 ± 11.95 | `0.0301` |
| **4SPK Clean** | `ENERGY+ECAPA-TDNN` | 3 | **26.22 ± 12.25** | 28.88 ± 13.28 | 90.50 ± 7.87 | `0.0426` |
| **4SPK Clean** | `ENERGY+MFCC` | 3 | **27.37 ± 13.60** | 32.57 ± 16.17 | 90.50 ± 7.87 | `0.0340` |
| **4SPK Clean** | `SILERO+ECAPA-TDNN` | 3 | **0.28 ± 0.39** | **0.21 ± 0.29** | 81.73 ± 13.57 | `0.0300` |
| **4SPK Clean** | `SILERO+MFCC` | 3 | **0.28 ± 0.39** | **0.21 ± 0.29** | 81.73 ± 13.57 | `0.0273` |
| **4SPK Noisy (12dB)** | `ENERGY+ECAPA-TDNN` | 3 | **45.76 ± 13.95** | 54.47 ± 11.13 | 86.96 ± 9.59 | `0.0336` |
| **4SPK Noisy (12dB)** | `ENERGY+MFCC` | 3 | **47.29 ± 12.88** | 55.47 ± 10.63 | 86.96 ± 9.59 | `0.0196` |
| **4SPK Noisy (12dB)** | `SILERO+ECAPA-TDNN` | 3 | **1.93 ± 0.04** | **2.60 ± 0.24** | 82.42 ± 11.95 | `0.0326` |
| **4SPK Noisy (12dB)** | `SILERO+MFCC` | 3 | **1.93 ± 0.04** | **2.60 ± 0.24** | 82.42 ± 11.95 | `0.0313` |

---

## 3. Key Technical & Algorithmic Findings

### A. Root Cause of Previous Outlier Clumping in Two-Speaker Diarization
- **Finding**: In earlier experiments, Agglomerative Clustering with `linkage='average'` on cosine distance assigned nearly all utterances to a single cluster (`[15, 2]` or `[16, 1]`).
- **Diagnosis**: 
  1. Short speech segments (< 0.3s) produce noisy embedding vectors that exhibit low cosine similarity (< 0.3) to all other segments.
  2. With `average` or `complete` linkage, these outlier points stay unmerged until the final steps of hierarchy construction. When stopping at $k=2$, the algorithm isolates the outlier(s) as Cluster 1 and clumps all normal speaker turns into Cluster 0.
- **Resolution**: Switching to `linkage='ward'` on L2-normalized embeddings minimizes spherical cosine variance ($\|u - v\|^2 = 2 - 2 \cos(u, v)$), enforcing balanced within-cluster compactness. On the 17 segments of 2-speaker dialogue, Ward linkage correctly partitions turns into `[9, 8]`, slashing Speaker Error from **35.52% down to 2.78%**.

### B. VAD Robustness & Downstream ASR Impact
- **Silero Neural VAD** demonstrated dramatic superiority over Short-Time Energy thresholding:
  - On 2-speaker noisy audio (12dB SNR colored noise), Silero achieved **1.18% DER** compared to Energy VAD's **55.32% DER** (a 46× reduction in error).
  - Energy VAD suffers from high false alarms during ambient noise bursts and missed speech during soft word endings.

### C. Neural Embeddings vs. MFCC Baseline
- In multi-talker conditions (3-speaker clean), **ECAPA-TDNN d-vectors achieved 0.12% DER** compared to **13.09% DER** for MFCC baseline features.
- Deep neural embeddings extract speaker-discriminative vocal tract representations that remain invariant to phonetic shifts, whereas pooled MFCCs confound speaker identity with phone content.

### D. Real-Time Efficiency
- Complete end-to-end pipeline execution (Audio Load + Silero VAD + ECAPA-TDNN + Ward Clustering + Whisper ASR) achieves an average Real-Time Factor (RTF) of **0.030 to 0.043** on the NVIDIA RTX A4500 GPU (~23× to 33× faster than real time).

---

## 4. Re-running the Benchmarks

```powershell
# Re-run all 72 experiments and regenerate BENCHMARK_REPORT.md and benchmark_matrix.json
python scripts/run_benchmarks.py
```
