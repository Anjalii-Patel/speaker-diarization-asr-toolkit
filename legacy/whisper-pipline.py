import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np

# Setup device and data type based on available hardware
device = "cuda:0"if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Initialize the pipeline for automatic speech recognition (ASR)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=15,
    batch_size=16,  # batch size for inference
    torch_dtype=torch_dtype,
    device=device,
)

generate_kwargs = {
    "max_new_tokens": 445,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

import os
audio_path = os.getenv("AUDIO_PATH", os.path.join("data", "an4_diarize_test.wav"))
# Load the audio
sample, sr = librosa.load(audio_path, sr=16000)

result = pipe(sample, generate_kwargs=generate_kwargs)
print(result)

# import torch
# from transformers import pipeline
# from datasets import load_dataset

# model = "openai/whisper-tiny"
# device = 0 if torch.cuda.is_available() else "cpu"

# pipe = pipeline(
#     task="automatic-speech-recognition",
#     model=model,
#     chunk_length_s=30,
#     device=device,
# )

# # replace this with the loading/inference for your audio sample
# ls_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# out = pipe(ls_dummy[0]["audio"], return_timestamps=True)

# # join all the text together
# text = [chunk["text"] for chunk in out["chunks"]]
# text = "".join(text)

# # get first timestamp of first chunk
# start = out["chunks"][0]["timestamp"][0]
# # get last timestamp of last chunk
# end = out["chunks"][-1]["timestamp"][-1]

# print(f"{start} -> {end}: {text}")