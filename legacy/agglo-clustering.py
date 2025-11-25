import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer  # Install jiwer to compute WER

# Constants
num_speakers = 2  # .rttm
language = 'English'
# model_size = 'large'
import os
audio_path = os.getenv("AUDIO_PATH", os.path.join("data", "an4_diarize_test.wav"))
device = "cpu"

# Load the Whisper model and processor
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)

# Load Silero VAD model
model = load_silero_vad()

# Function to transcribe and calculate WER
def ASR_Translate(audio_data, language, operation="translate", label_text=""):
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language, task=operation)
    input_features = whisper_processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids, return_timestamps=True)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # Calculate WER if ground truth is provided
    error = wer(label_text.lower(), transcription[0].lower()) if label_text else None
    return transcription[0].lower(), error

# Load and process the audio
y, sr = librosa.load(audio_path, sr=16000)
speech_timestamps = get_speech_timestamps(y, model,threshold=0.5,min_speech_duration_ms=250,min_silence_duration_ms=50,visualize_probs= True,speech_pad_ms=20, return_seconds=True)

# Chunk audio based on VAD timestamps
audio_chunks = []
for segment in speech_timestamps:
    start_time = int(segment['start'] * sr)
    end_time = int(segment['end'] * sr)
    audio_segment_slice = y[start_time:end_time]
    audio_chunks.append(audio_segment_slice)

print(len(audio_chunks))

# Calculate MFCC features
mfcc_features = np.array([librosa.feature.mfcc(y=chunk, sr=sr).mean(axis=1) for chunk in audio_chunks])

print(len(mfcc_features))

# Apply Agglomerative Clustering
# agglomerative_clusterer = AgglomerativeClustering(n_clusters=num_speakers, linkage="single")
agglomerative_clusterer = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')

# print("checkpoint-1")

labels = agglomerative_clusterer.fit_predict(mfcc_features)

# print("checkpoint-2")

# print("checkpoint-3")

# Transcribe and print clusters
sample = audio_chunks
for i, label in enumerate(labels):
    transcription, error = ASR_Translate(audio_chunks[i], language, "translate")
    speaker_label = "Doctor"if label == 1 else "Patient"
    print(f"Segment {i + 1}: {speaker_label}: {transcription}")
    if error is not None:
        print(f"WER for segment {i + 1}: {error}")

# print("Dimensions:", mfcc_features.shape, labels.shape)
