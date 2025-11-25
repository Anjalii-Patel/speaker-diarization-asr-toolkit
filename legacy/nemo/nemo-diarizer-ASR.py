# import os
# import librosa
# import soundfile as sf
# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration

# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_PATH = os.getenv("AUDIO_PATH", os.path.join(BASE_DIR, "data", "an4_diarize_test.wav"))
# RTTM_PATH = os.getenv("RTTM_PATH", os.path.join(BASE_DIR, "data", "an4_diarize_test.rttm"))
# SEGMENTS_DIR = os.getenv("SEGMENTS_DIR", os.path.join(BASE_DIR, "output", "segments"))
# os.makedirs(SEGMENTS_DIR, exist_ok=True)

# # Load audio
# sr = 16000
# signal, sr = librosa.load(AUDIO_PATH, sr=sr)

# # Parse RTTM file to extract segments
# def parse_rttm(rttm_path):
#     segments = []
#     with open(rttm_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 5:
#                 start = float(parts[3])
#                 duration = float(parts[4])
#                 segments.append((start, start + duration))
#     return segments

# segments = parse_rttm(RTTM_PATH)

# # Split and save audio segments
# for idx, (start, end) in enumerate(segments):
#     start_sample = int(start * sr)
#     end_sample = int(end * sr)
#     segment_audio = signal[start_sample:end_sample]
#     segment_path = os.path.join(SEGMENTS_DIR, f"segment_{idx + 1}.wav")
#     sf.write(segment_path, segment_audio, sr)
#     print(f"Saved segment {idx + 1} to {segment_path}")

# # Load Whisper model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-large")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
# model.eval()

# # Perform ASR on each segment
# transcriptions = {}
# for segment_file in os.listdir(SEGMENTS_DIR):
#     if segment_file.endswith('.wav'):
#         segment_path = os.path.join(SEGMENTS_DIR, segment_file)
#         audio_input, _ = librosa.load(segment_path, sr=16000)
#         input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
#         with torch.no_grad():
#             predicted_ids = model.generate(input_features, language='en')
#             transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#         transcriptions[segment_file] = transcription
#         print(f"Transcription for {segment_file}: {transcription}")

# # Save transcriptions to a file
# with open(os.path.join(SEGMENTS_DIR, 'transcriptions.txt'), 'w') as f:
#     for segment, text in transcriptions.items():
#         f.write(f"{segment}: {text}\n")

# print("ASR completed. Transcriptions saved to transcriptions.txt")

import os
import librosa
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_PATH = os.getenv("AUDIO_PATH", os.path.join(BASE_DIR, "data", "an4_diarize_test.wav"))
RTTM_PATH = os.getenv("RTTM_PATH", os.path.join(BASE_DIR, "data", "an4_diarize_test.rttm"))
SEGMENTS_DIR = os.getenv("SEGMENTS_DIR", os.path.join(BASE_DIR, "output", "segments"))
os.makedirs(SEGMENTS_DIR, exist_ok=True)

# Load audio
sr = 16000
signal, sr = librosa.load(AUDIO_PATH, sr=sr)

# Parse RTTM file to extract segments and speakers
def parse_rttm(rttm_path):
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]  # Speaker label from RTTM
                segments.append((start, start + duration, speaker))
    return segments

segments = parse_rttm(RTTM_PATH)

# Split and save audio segments
for idx, (start, end, speaker) in enumerate(segments):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment_audio = signal[start_sample:end_sample]
    segment_path = os.path.join(SEGMENTS_DIR, f"segment_{idx + 1}.wav")
    sf.write(segment_path, segment_audio, sr)
    print(f"Saved segment {idx + 1} to {segment_path}")

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.eval()

# Perform ASR on each segment in order
transcriptions = {}
for idx in range(1, len(segments) + 1):
    segment_file = f"segment_{idx}.wav"
    segment_path = os.path.join(SEGMENTS_DIR, segment_file)
    if os.path.exists(segment_path):
        audio_input, _ = librosa.load(segment_path, sr=16000)
        input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = model.generate(input_features, language='en')
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions[segment_file] = (transcription, segments[idx - 1][2])  # Retrieve speaker
        print(f"Transcription for {segment_file} (Speaker {segments[idx - 1][2]}): {transcription}")

# Save transcriptions to a file
with open(os.path.join(SEGMENTS_DIR, 'transcriptions.txt'), 'w') as f:
    for segment, (text, speaker) in transcriptions.items():
        f.write(f"{segment} (Speaker {speaker}): {text}\n")

print("ASR completed. Transcriptions saved to transcriptions.txt")
