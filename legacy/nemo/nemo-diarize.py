
# import torch
# from nemo.collections.asr.models import EncDecClassificationModel
# import soundfile as sf
# import numpy as np

# # Load NeMo's pre-trained VAD model
# vad_model = EncDecClassificationModel.from_pretrained(model_name="vad_marblenet")

# # Audio path
# audio_path = os.getenv("AUDIO_PATH", "data/an4_diarize_test.wav")

# # Load audio (NeMo expects 16kHz mono audio)
# audio, sr = sf.read(audio_path)

# # Convert stereo to mono by averaging channels
# if len(audio.shape) == 2:  # Stereo audio (2 channels)
#     audio = np.mean(audio, axis=1)

# # Ensure audio is 16kHz
# if sr != 16000:
#     raise ValueError("Audio sampling rate must be 16kHz for NeMo VAD")

# # Save the mono audio as a temporary file
# sf.write("temp_mono_audio.wav", audio, 16000)

# # Run VAD and get predictions (raw logits)
# vad_predictions = vad_model.transcribe(["temp_mono_audio.wav"])[0]  # Get first output

# # Check the output structure
# print(vad_predictions)
# print(f"Shape of vad_predictions: {vad_predictions.shape if isinstance(vad_predictions, torch.Tensor) else 'N/A'}")
# print(f"Type of vad_predictions: {type(vad_predictions)}")

# # Check if the output is scalar (single prediction)
# if isinstance(vad_predictions, torch.Tensor) and vad_predictions.ndimension() == 1:
#     # In this case, the tensor has only one element, so no iteration is needed
#     prediction = vad_predictions.item()  # Get the scalar value
#     if prediction > 0.5:
#         print("Speech detected in the entire audio.")
#     else:
#         print("No speech detected in the entire audio.")
# else:
#     # If it's not a scalar, proceed with normal frame-level processing (as earlier)
#     threshold = 0.5  # Adjust as needed
#     speech_segments = []

#     # Ensure predictions are in tensor/numpy format
#     if isinstance(vad_predictions, torch.Tensor):
#         vad_predictions = vad_predictions.squeeze().cpu().numpy()  # Convert to numpy if needed

#     start, end = None, None

#     # Iterate through predictions and apply thresholding
#     for idx, logit in enumerate(vad_predictions):
#         if logit > threshold:
#             if start is None:
#                 start = idx * 0.01  # Assuming 10ms frame shift
#             end = (idx + 1) * 0.01
#         elif start is not None:
#             speech_segments.append({'start': start, 'end': end, 'confidence': logit})
#             start, end = None, None

#     # Handle trailing segments
#     if start is not None:
#         speech_segments.append({'start': start, 'end': end, 'confidence': logit})

#     # Print speech segments
#     if speech_segments:
#         for segment in speech_segments:
#             print(f"Speech from {segment['start']:.2f}s to {segment['end']:.2f}s (Confidence: {segment['confidence']:.2f})")
#     else:
#         print("No speech segments detected.")

print("-----------------------------------------------------------------------------------------------------")

# import torch
# from nemo.collections.asr.models import EncDecClassificationModel
# import soundfile as sf
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics.pairwise import cosine_similarity
# import librosa

# # Load NeMo's pre-trained VAD model
# vad_model = EncDecClassificationModel.from_pretrained(model_name="")

# # Audio path
# audio_path = os.getenv("AUDIO_PATH", "data/an4_diarize_test.wav")

# # Load audio (NeMo expects 16kHz mono audio)
# audio, sr = sf.read(audio_path)

# # Convert stereo to mono by averaging channels
# if len(audio.shape) == 2:  # Stereo audio (2 channels)
#     audio = np.mean(audio, axis=1)

# if sr != 16000:
#     raise ValueError("Audio sampling rate must be 16kHz for NeMo VAD")

# # Save the mono audio as a temporary file
# sf.write("temp_mono_audio.wav", audio, 16000)

# # Run VAD and get predictions
# vad_predictions = vad_model.transcribe(["temp_mono_audio.wav"])

# # Print vad_predictions to understand its structure
# print("vad_predictions:", vad_predictions)

# # Set a threshold for VAD prediction
# threshold = 0.1  # You can adjust this threshold to control speech detection sensitivity

# # Extract speech segments from VAD predictions
# speech_segments = []
# start_time = 0

# for idx, logit in enumerate(vad_predictions[0]):
#     if logit > threshold:  # This is speech

#         start_time = idx / 10 # Adjust based on frame rate (here it's assumed to be 100 Hz)
#     else:
#         if start_time is not None:
#             end_time = idx / 10
#             speech_segments.append((start_time, end_time))
#             start_time = None

# # If there's a last segment that wasn't added
# if start_time is not None:
#     end_time = len(vad_predictions[0]) / 100
#     speech_segments.append((start_time, end_time))

# # Print out speech segments
# if len(speech_segments) == 0:
#     print("No speech detected.")
# else:
#     print("Detected speech segments:")
#     for start, end in speech_segments:
#         print(f"Speech from {start:.2f}s to {end:.2f}s")

# # Now perform speaker clustering using Agglomerative Clustering
# # Extract features from the audio for clustering
# y, sr = librosa.load("temp_mono_audio.wav", sr=16000)

# # Extract MFCC features
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# # Transpose to have the correct shape (time steps, features)
# mfcc = mfcc.T

# # Calculate cosine similarity between frames (this will be used as distance measure)
# cosine_sim = cosine_similarity(mfcc)

# # Perform Agglomerative Clustering on the similarity matrix
# agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='average')
# labels = agg_clustering.fit_predict(1 - cosine_sim)  # 1 - cosine similarity is used as the distance metric

# # Print diarized speech segments with speaker labels
# print("\nDiarized Speech Segments with Speaker Labels:")
# for idx, (start, end) in enumerate(speech_segments):
#     speaker = labels[idx]  # Assign speaker label based on clustering result
#     print(f"Speech from {start:.2f}s to {end:.2f}s (Speaker {speaker + 1})")

print("-----------------------------------------------------------------------------------------------------")

import torch
from nemo.collections.asr.models import EncDecClassificationModel
import soundfile as sf
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import librosa

 # Directory to store intermediate files and prediction outputs
pretrained_speaker_model = 'titanet_large'

from nemo.collections.asr.models import ClusteringDiarizer
oracle_vad_clusdiar_model = ClusteringDiarizer(cfg=pretrained_speaker_model)

oracle_vad_clusdiar_model.diarize()

# Load NeMo's pre-trained VAD model (Switching to vad_mlsd model)
vad_model = EncDecClassificationModel.from_pretrained(model_name='titanet_large')

# Audio path
audio_path = os.getenv("AUDIO_PATH", "data/an4_diarize_test.wav")

# Load audio (NeMo expects 16kHz mono audio)
audio, sr = sf.read(audio_path)

# Convert stereo to mono by averaging channels
if len(audio.shape) == 2:  # Stereo audio (2 channels)
    audio = np.mean(audio, axis=1)

if sr != 16000:
    raise ValueError("Audio sampling rate must be 16kHz for NeMo VAD")

# Save the mono audio as a temporary file
sf.write("temp_mono_audio.wav", audio, 16000)

# Run VAD and get predictions
vad_predictions = vad_model.transcribe(["temp_mono_audio.wav"])

# Print vad_predictions to understand its structure
print("vad_predictions:", vad_predictions)

# Extract speech segments from VAD predictions
speech_segments = []
for idx, logit in enumerate(vad_predictions[0]):
    if logit > 0.5:  # This is speech
        start_time = idx / 100  # Adjust based on frame rate (here it's assumed to be 100 Hz)
        end_time = (idx + 1) / 100
        speech_segments.append((start_time, end_time))

# Print out speech segments
if len(speech_segments) == 0:
    print("No speech detected.")
else:
    print("Detected speech segments:")
    for start, end in speech_segments:
        print(f"Speech from {start:.2f}s to {end:.2f}s")

# Now perform speaker clustering using Agglomerative Clustering
# Extract features from the audio for clustering
y, sr = librosa.load("temp_mono_audio.wav", sr=16000)

# Extract MFCC features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Transpose to have the correct shape (time steps, features)
mfcc = mfcc.T

# Calculate cosine similarity between frames (this will be used as distance measure)
cosine_sim = cosine_similarity(mfcc)

# Perform Agglomerative Clustering on the similarity matrix
agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='average')
labels = agg_clustering.fit_predict(1 - cosine_sim)  # 1 - cosine similarity is used as the distance metric

# Print diarized speech segments with speaker labels
print("\nDiarized Speech Segments with Speaker Labels:")
for idx, (start, end) in enumerate(speech_segments):
    speaker = labels[idx]  # Assign speaker label based on clustering result
    print(f"Speech from {start:.2f}s to {end:.2f}s (Speaker {speaker + 1})")
