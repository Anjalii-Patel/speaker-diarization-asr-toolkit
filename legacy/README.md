# Legacy Exploratory Scripts

This directory preserves the original experimental and research scripts created during initial exploration of speech diarization, VAD, and ASR workflows.

## Historical Scripts Overview

1. `pyannote.py`: Early experiment testing Pyannote audio embedding wrapper with SpeechBrain ECAPA-TDNN and chunk-level MFCC feature extraction followed by Agglomerative Clustering and Whisper translation.
2. `agglo-clustering.py` / `agglo-test.py`: Experiments evaluating Silero VAD segmentation combined with MFCC feature extraction and Agglomerative Clustering (Ward linkage).
3. `clustering_audio.py`: Exploration of DBSCAN density-based clustering on MFCC features.
4. `multi-speaker.py`: Multi-speaker chunk splitting (15-second cap) with Silero VAD and average-linkage Agglomerative Clustering.
5. `test-1.py`: Neural embedding extraction using SpeechBrain ECAPA-TDNN on fixed 15-second chunks followed by Agglomerative Clustering.
6. `return_timestamps.py`: Exploration of token-level timestamps using Whisper Large.
7. `whisper-pipline.py`: Hugging Face `transformers.pipeline` automatic speech recognition with chunking and generation parameters.
8. `nemo/`:
   - `nemo-diarizer.py`: NeMo Clustering Diarizer with Oracle VAD on the CMU AN4 benchmark dataset.
   - `nemo-diarizer-1.py`: NeMo Clustering Diarizer with System VAD (no ground-truth RTTM).
   - `nemo-diarizer-ASR.py`: RTTM parser, audio segment slicing, and Whisper ASR transcription attribution.
   - `nemo-diarize.py`: Exploratory manual VAD thresholding and frame-level MFCC cosine similarity clustering.

> [!NOTE]
> For production-ready, modular, and reproducible pipelines, please use the modules in `src/` and the executable entry points in `scripts/`.
