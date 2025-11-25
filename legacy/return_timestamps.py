from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Load the Whisper processor and model
model_name = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load the audio file and preprocess
def load_audio(file_path, sampling_rate=16000):
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    return audio, sr

import os
audio_path = os.getenv("AUDIO_PATH", os.path.join("data", "an4_diarize_test.wav"))  # Path to your audio file
audio, sr = load_audio(audio_path)

# Convert audio to input features
input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

# Generate transcription with token timestamps
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        return_token_timestamps=True,  # Assuming this is supported
    )

# Decode the transcription
transcription = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]

# Extract timestamps if available
if hasattr(predicted_ids, "timestamps"):
    token_timestamps = predicted_ids.timestamps
    print("Token Timestamps:", token_timestamps)

print("Transcription:", transcription)
