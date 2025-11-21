"""
Speaker-Attributed ASR Pipeline
End-to-End Orchestrator:
Audio Ingestion -> VAD -> Feature/Embedding Extraction -> Clustering -> Segment Transcription -> Attribution
"""

import os
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np

from ..audio.io import load_audio, get_audio_duration
from ..audio.validation import validate_audio_file
from ..vad.base import BaseVAD, SpeechSegment
from ..vad.silero import SileroVAD
from ..vad.energy import EnergyVAD
from ..embeddings.base import BaseEmbeddingExtractor
from ..embeddings.mfcc import MFCCFeatureExtractor
from ..diarization.clustering_diarizer import ClusteringDiarizer
from ..diarization.rttm import RTTMSegment, write_rttm
from ..asr.whisper_asr import WhisperASR
from ..evaluation.metrics import compute_rtf

logger = logging.getLogger(__name__)

@dataclass
class AttributedUtterance:
    """A transcription unit attributed to a specific speaker."""
    speaker: str
    start: float
    end: float
    duration: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "text": self.text,
        }

@dataclass
class PipelineResult:
    """Complete result payload produced by the pipeline."""
    audio_file: str
    audio_duration: float
    total_processing_time: float
    rtf: float
    num_speakers_detected: int
    num_segments: int
    rttm_segments: List[RTTMSegment]
    utterances: List[AttributedUtterance]

    def to_formatted_transcript(self) -> str:
        """Format as human-readable conversational transcript."""
        lines = []
        for u in self.utterances:
            lines.append(f"[{u.start:.2f}s - {u.end:.2f}s] {u.speaker}: {u.text}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_file": self.audio_file,
            "audio_duration_sec": round(self.audio_duration, 2),
            "processing_time_sec": round(self.total_processing_time, 2),
            "real_time_factor": self.rtf,
            "num_speakers_detected": self.num_speakers_detected,
            "num_segments": self.num_segments,
            "utterances": [u.to_dict() for u in self.utterances],
        }

class SpeakerAttributedASRPipeline:
    """
    Modular, reproducible end-to-end pipeline connecting VAD, speaker clustering,
    and ASR decoding.
    """

    def __init__(
        self,
        vad: Optional[BaseVAD] = None,
        embedding_extractor: Optional[BaseEmbeddingExtractor] = None,
        diarizer: Optional[ClusteringDiarizer] = None,
        asr: Optional[WhisperASR] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.vad = vad or SileroVAD(device=device)
        self.embedding_extractor = embedding_extractor or MFCCFeatureExtractor()
        self.diarizer = diarizer or ClusteringDiarizer(num_speakers=2)
        self.asr = asr  # Lazy-loaded on request to avoid unnecessary memory allocations

    def run(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        language: str = "en",
        output_dir: Optional[str] = None,
        save_rttm: bool = True,
    ) -> PipelineResult:
        """
        Execute full speaker-attributed transcription.

        Args:
            audio_path: Path to target audio file.
            num_speakers: Expected number of speakers (optional, overrides default).
            language: Whisper ASR target language.
            output_dir: Optional directory to save output RTTM and transcript files.
            save_rttm: Whether to write RTTM file to disk.

        Returns:
            PipelineResult dataclass instance.
        """
        validate_audio_file(audio_path)
        start_time = time.time()

        # Step 1: Load and resample audio
        logger.info("Loading audio: %s", audio_path)
        audio, sr = load_audio(audio_path, target_sr=16000, mono=True)
        duration = get_audio_duration(audio, sr)

        # Step 2: Voice Activity Detection (VAD)
        logger.info("Running Voice Activity Detection...")
        speech_segments = self.vad.detect_speech(audio, sr=sr)
        if not speech_segments:
            logger.warning("No speech detected by VAD in %s", audio_path)
            speech_segments = [SpeechSegment(start=0.0, end=duration, confidence=0.5)]

        # Step 3: Extract Speech Chunks & Acoustic/Neural Embeddings
        logger.info("Extracting features for %d speech segments...", len(speech_segments))
        audio_chunks: List[np.ndarray] = []
        intervals: List[tuple] = []
        for seg in speech_segments:
            s_idx = int(max(0.0, seg.start) * sr)
            e_idx = int(min(duration, seg.end) * sr)
            if e_idx > s_idx:
                audio_chunks.append(audio[s_idx:e_idx])
                intervals.append((seg.start, seg.end))

        embeddings = self.embedding_extractor.extract_batch(audio_chunks, sr=sr)

        # Step 4: Clustering & Speaker Assignment
        logger.info("Clustering speaker embeddings...")
        if num_speakers is not None:
            self.diarizer.num_speakers = num_speakers

        labels = self.diarizer.fit_predict(embeddings)
        uri = os.path.splitext(os.path.basename(audio_path))[0]
        rttm_segments = self.diarizer.assign_speaker_labels(intervals, labels, uri=uri)

        # Step 5: Automatic Speech Recognition (ASR)
        utterances: List[AttributedUtterance] = []
        if self.asr is not None:
            logger.info("Transcribing speech segments with Whisper ASR...")
            self.asr.load_model()
            transcriptions = self.asr.transcribe_segments(audio, sr, intervals, language=language)
            for rttm_seg, text in zip(rttm_segments, transcriptions):
                utterances.append(
                    AttributedUtterance(
                        speaker=rttm_seg.speaker,
                        start=rttm_seg.start,
                        end=rttm_seg.end,
                        duration=rttm_seg.duration,
                        text=text,
                    )
                )
        else:
            for rttm_seg in rttm_segments:
                utterances.append(
                    AttributedUtterance(
                        speaker=rttm_seg.speaker,
                        start=rttm_seg.start,
                        end=rttm_seg.end,
                        duration=rttm_seg.duration,
                        text="[ASR disabled / model omitted]",
                    )
                )

        total_time = time.time() - start_time
        rtf = compute_rtf(duration, total_time)
        unique_speakers = len(set(s.speaker for s in rttm_segments))

        result = PipelineResult(
            audio_file=audio_path,
            audio_duration=duration,
            total_processing_time=total_time,
            rtf=rtf,
            num_speakers_detected=unique_speakers,
            num_segments=len(rttm_segments),
            rttm_segments=rttm_segments,
            utterances=utterances,
        )

        # Step 6: Save Outputs if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if save_rttm:
                rttm_file = os.path.join(output_dir, f"{uri}.rttm")
                write_rttm(rttm_file, rttm_segments)
                logger.info("Saved RTTM to %s", rttm_file)

            transcript_file = os.path.join(output_dir, f"{uri}_transcript.txt")
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(result.to_formatted_transcript() + "\n")
            logger.info("Saved transcript to %s", transcript_file)

            json_file = os.path.join(output_dir, f"{uri}_result.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info("Saved metadata to %s", json_file)

        return result
