import os
import wget
import librosa
import json
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer

# Set audio path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
an4_audio = os.getenv("AUDIO_PATH", os.path.join(BASE_DIR, "data", "an4_diarize_test.wav"))

# Load audio
sr = 16000
signal, sr = librosa.load(an4_audio, sr=sr)

# Create manifest without RTTM
meta = {
    'audio_filepath': an4_audio,
    'offset': 0,
    'duration': None,
    'label': 'infer',
    'text': '-',
    'num_speakers': 3,
    'rttm_filepath': None,  # No RTTM for reference
    'uem_filepath': None
}

# Write manifest to JSON file
os.makedirs('data', exist_ok=True)
with open(os.path.join('data', 'manifest.json'), 'w') as fp:
    json.dump(meta, fp)
    fp.close()
    # fp.write('\n')

# Print manifest content
# with open(os.path.join('data', 'manifest.json'), 'r') as fp:
#     print(fp.read())

# Create output directory
ROOT = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, 'output_nemo'))
output_dir = os.path.join(ROOT, 'vad_infer')
os.makedirs(output_dir, exist_ok=True)

# Load diarization model config
MODEL_CONFIG = os.path.join(BASE_DIR, "data", "diar_infer_telephonic.yaml")
config = OmegaConf.load(MODEL_CONFIG)

# Update configuration to infer without RTTM
config.diarizer.manifest_filepath = os.path.join(BASE_DIR, "data", "manifest.json")
config.diarizer.out_dir = output_dir
config.diarizer.speaker_embeddings.model_path = 'titanet_large'
config.diarizer.oracle_vad = False  # Disable oracle VAD for independent inference
config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.25, 1.0, 0.75, 0.5]
config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75, 0.625, 0.5, 0.375, 0.25]
config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1, 1, 1, 1, 1]
config.diarizer.clustering.parameters.oracle_num_speakers = False  # Infer speaker count

# Initialize diarizer model
vad_infer_diarizer = ClusteringDiarizer(cfg=config)

# Run diarization
if __name__ == '__main__':
    vad_infer_diarizer.diarize()

print("Diarization complete. Results saved to:", output_dir)
