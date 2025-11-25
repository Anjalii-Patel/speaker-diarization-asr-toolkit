import warnings
warnings.filterwarnings("ignore")
import torch
import librosa
import numpy as np
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Inference
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from pyannote.core import Segment
import os
from sklearn.cluster import AgglomerativeClustering

# Hugging Face Authentication
from huggingface_hub import login
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    warnings.warn(
        "HF_TOKEN environment variable not set. Using public/cached Hugging Face models."
    )

# Constants
num_speakers = 2
language = 'English'
audio_path = os.getenv("AUDIO_PATH", os.path.join("data", "an4_diarize_test.wav"))
device = "cuda"if torch.cuda.is_available() else "cpu"

# Validate audio path
if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Audio file not found at {audio_path}")

# Load Whisper ASR model
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)

# Load Pyannote VAD (pre-trained model)
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

vad_pipeline = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device(device))

inference = Inference(vad_pipeline)

# ASR Translation Function
def ASR_Translate(audio_data, language, operation="translate", label_text=""):
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language, task=operation)
    input_features = whisper_processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    error = wer(label_text.lower(), transcription[0].lower()) if label_text else None
    return transcription[0].lower(), error

# Load and process the audio
y, sr = librosa.load(audio_path, sr=16000)

# Perform VAD using Pyannote
vad_results = inference(audio_path)

# Chunk audio based on VAD results
audio_chunks = []
timestamps = []
for segment in vad_results.get_timeline():
    start_time = int(segment.start * sr)
    end_time = int(segment.end * sr)
    audio_segment_slice = y[start_time:end_time]
    audio_chunks.append(audio_segment_slice)
    timestamps.append(segment)

print(f"Detected {len(audio_chunks)} speech segments.")

# Calculate MFCC features for clustering
mfcc_features = np.array([librosa.feature.mfcc(y=chunk, sr=sr).mean(axis=1) for chunk in audio_chunks])

# Clustering using Agglomerative Clustering
agglomerative_clusterer = AgglomerativeClustering(n_clusters=num_speakers)
labels = agglomerative_clusterer.fit_predict(mfcc_features)

# Transcribe and print segments
for i, (label, segment) in enumerate(zip(labels, timestamps)):
    transcription, error = ASR_Translate(audio_chunks[i], language, "translate")
    speaker_label = "Doctor"if label == 1 else "Patient"
    print(f"Segment {i + 1} [{segment.start:.2f}s - {segment.end:.2f}s]: {speaker_label}: {transcription}")
    if error is not None:
        print(f"WER for segment {i + 1}: {error}")
