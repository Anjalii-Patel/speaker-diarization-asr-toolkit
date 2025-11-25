import os
import wget
import librosa
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
# from nemo.collections.common.parts.preprocessing.collections import SpeechLabelEntity
import json
from omegaconf import OmegaConf
import torch
from nemo.collections.asr.models import ClusteringDiarizer

# torch.multiprocessing.set_start_method('spawn', force=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
an4_audio = os.getenv("AUDIO_PATH", os.path.join(BASE_DIR, "data", "an4_diarize_test.wav"))
an4_rttm = os.getenv("RTTM_PATH", os.path.join(BASE_DIR, "data", "an4_diarize_test.rttm"))
# if not os.path.exists(an4_audio):
#     an4_audio_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
#     an4_audio = wget.download(an4_audio_url, data_dir)
# if not os.path.exists(an4_rttm):
#     an4_rttm_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
#     an4_rttm = wget.download(an4_rttm_url, data_dir)

sr = 16000
signal, sr = librosa.load(an4_audio,sr=sr)

labels = rttm_to_labels(an4_rttm)
reference = labels_to_pyannote_object(labels)
print(labels)
reference

meta = {
    'audio_filepath': an4_audio,
    'offset': 0,
    'duration': None,
    'label': 'infer',
    'text': '-',
    'num_speakers': 2,
    'rttm_filepath': an4_rttm,
    'uem_filepath': None
}

# Write manifest to JSON file
with open(os.path.join('data', 'manifest.json'), 'w') as fp:
    json.dump(meta, fp)
    fp.write('\n')

# Read and print the manifest content
with open(os.path.join('data', 'manifest.json'), 'r') as fp:
    print(fp.read())

# Create output directory
ROOT = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, 'output_nemo'))
output_dir = os.path.join(ROOT, 'oracle_vad')
os.makedirs(output_dir, exist_ok=True)

# MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')
# if not os.path.exists(MODEL_CONFIG):
#     config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
#     MODEL_CONFIG = wget.download(config_url,data_dir)

MODEL_CONFIG = os.path.join(BASE_DIR, "data", "diar_infer_telephonic.yaml")

config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(config))

config.diarizer.manifest_filepath = os.path.join(BASE_DIR, "data", "manifest.json")
config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
pretrained_speaker_model = 'titanet_large'
config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5]
config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1]
config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1]
config.diarizer.oracle_vad = True # ----> ORACLE VAD
config.diarizer.clustering.parameters.oracle_num_speakers = False

oracle_vad_clusdiar_model = ClusteringDiarizer(cfg=config)

# And lets diarize
if __name__ == '__main__':
    oracle_vad_clusdiar_model.diarize()

print("Clustering Diarizer Result (RTTM format)")
pred_labels_neural = rttm_to_labels(an4_rttm)
hypothesis_neural = labels_to_pyannote_object(pred_labels_neural)

# hypothesis_neural
print("Ground-truth Speaker Labels")
