import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from sklearn.cluster import DBSCAN

# Constants
num_speakers = 3
language = 'English'
import os
audio_path = os.getenv("AUDIO_PATH", os.path.join("data", "an4_diarize_test.wav"))
device = "cpu"

# Load Whisper and VAD
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)
vad_model = load_silero_vad()

# ASR Transcription
def ASR_Translate(audio_data, language, operation="translate"):
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language, task=operation)
    input_features = whisper_processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0].lower()

# Load Audio
y, sr = librosa.load(audio_path, sr=16000)

# VAD Detection
speech_timestamps = get_speech_timestamps(y, vad_model,threshold=0.5,min_speech_duration_ms=150,min_silence_duration_ms=50,visualize_probs= True,speech_pad_ms=20, return_seconds=True)

# Split Audio into 15-second Chunks
audio_chunks = []
for segment in speech_timestamps:
    start_time = int(segment['start'] * sr)
    end_time = int(segment['end'] * sr)
    segment_length = end_time - start_time

    # Force split if segment is longer than 15 seconds
    if segment_length > 15 * sr:
        for i in range(0, segment_length, 15 * sr):
            audio_chunks.append(y[start_time + i: min(start_time + i + 15 * sr, end_time)])
    else:
        audio_chunks.append(y[start_time:end_time])

# MFCC Feature Extraction
mfcc_features = np.array([librosa.feature.mfcc(y=chunk, sr=sr).mean(axis=1) for chunk in audio_chunks])

# Agglomerative Clustering
if len(mfcc_features) >= 2:
    clusterer = AgglomerativeClustering(n_clusters=num_speakers,linkage="average")
    labels = clusterer.fit_predict(mfcc_features)
else:
    labels = np.zeros(len(audio_chunks))  # Default to single speaker

# Transcription with Speaker Labels
# speaker_labels = ["Speaker 1", "Speaker 2", "Speaker 3"]  # List for three speakers

for i, label in enumerate(labels):
    transcription = ASR_Translate(audio_chunks[i], language)
    # speaker_label = speaker_labels[label]  # Assign based on clustering label
    print(f"Segment {i + 1}: Speaker {label if label != -1 else 'Noise'}: {transcription}")
    # print(f"Segment {i + 1}: {speaker_label}: {transcription}")
