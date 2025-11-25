import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from speechbrain.pretrained import SpeakerRecognition
import os

# Constants
num_speakers = 3
language = 'English'
audio_path = os.getenv("AUDIO_PATH", os.path.join("data", "an4_diarize_test.wav"))
device = "cpu"

# Load Whisper and Speaker Embedding Model
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)

# Load Speaker Embedding Model from Local Directory
spk_embedder = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",  # Still use the Hugging Face identifier
    savedir="tmp",
    run_opts={"device": device},
    model_dir=os.getenv("SPEECHBRAIN_CACHE_DIR", None)
).to(device)

# ASR Transcription
def ASR_Translate(audio_data, language, operation="translate"):
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language, task=operation)
    input_features = whisper_processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0].lower()

# Load Audio
y, sr = librosa.load(audio_path, sr=16000)

# Split Audio into 15-second Chunks
audio_chunks = [y[i:i + 15 * sr] for i in range(0, len(y), 15 * sr)]

# Extract Speaker Embeddings
embeddings = np.array([spk_embedder.encode_batch(chunk[np.newaxis, :]) for chunk in audio_chunks])
embeddings = embeddings.reshape(len(audio_chunks), -1)

# Agglomerative Clustering
if len(embeddings) >= 2:
    clusterer = AgglomerativeClustering(n_clusters=num_speakers)
    labels = clusterer.fit_predict(embeddings)
else:
    labels = np.zeros(len(audio_chunks))  # Default to single speaker

speaker_labels = [f"Speaker {i + 1}"for i in range(num_speakers)]

# Transcription with Speaker Labels
for i, label in enumerate(labels):
    transcription = ASR_Translate(audio_chunks[i], language)
    speaker_label = speaker_labels[label] if label != -1 else "Unknown"
    print(f"Segment {i + 1}: {speaker_label}: {transcription}")
