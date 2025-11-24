import pyttsx3
from pydub import AudioSegment

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Get available voices
voices = engine.getProperty('voices')

# Check the number of available voices
print(f"Number of voices available: {len(voices)}")

# List of text for each speaker
text_speaker_1 = "Hello, this is speaker one."
text_speaker_2 = "Hi, I am Speaker 2."
text_speaker_3 = "Good day, I am Speaker 1 again, due to limited voices."

# Set properties for Speaker 1 (first available voice)
engine.setProperty('voice', voices[0].id)  # Select first available voice
engine.save_to_file(text_speaker_1, 'speaker_1.wav')

# Set properties for Speaker 2 (second available voice)
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)  # Select second available voice
else:
    engine.setProperty('voice', voices[0].id)  # Fallback to first voice if only one is available
engine.save_to_file(text_speaker_2, 'speaker_2.wav')

# Set properties for Speaker 3 (fallback to Speaker 1 due to only two voices)
engine.setProperty('voice', voices[0].id)  # Select first available voice
engine.save_to_file(text_speaker_3, 'speaker_3.wav')

# Wait for the audio to finish
engine.runAndWait()

# Convert the WAV files to MP3
audio_speaker_1 = AudioSegment.from_wav('speaker_1.wav')
audio_speaker_2 = AudioSegment.from_wav('speaker_2.wav')
audio_speaker_3 = AudioSegment.from_wav('speaker_3.wav')

# Combine the audio segments
combined_audio = audio_speaker_1 + audio_speaker_2 + audio_speaker_3

# Export the combined audio to MP3
combined_audio.export('two_speakers.mp3', format='mp3')

print("The audio file 'two_speakers.mp3' has been generated.")
