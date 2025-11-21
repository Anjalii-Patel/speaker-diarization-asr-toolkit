"""
Neural Speaker Embedding Extractor (ECAPA-TDNN)
Extracts 192-dimensional speaker embeddings using SpeechBrain ECAPA-TDNN architecture.
"""

import os
import logging
from typing import Optional
import numpy as np
import torch
from .base import BaseEmbeddingExtractor

logger = logging.getLogger(__name__)

class ECAPATDNNEmbeddingExtractor(BaseEmbeddingExtractor):
    """
    SpeechBrain ECAPA-TDNN speaker embedding extractor.
    Model: speechbrain/spkrec-ecapa-voxceleb
    """

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cpu",
        savedir: Optional[str] = None,
    ):
        self.model_source = model_source
        if torch.cuda.is_available() and (device == "cuda"or device.startswith("cuda")):
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.savedir = savedir or os.getenv("SPEECHBRAIN_CACHE_DIR", "tmp/speechbrain")
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            try:
                from speechbrain.inference.speaker import SpeakerRecognition
                from speechbrain.utils.fetching import LocalStrategy
                strategy = LocalStrategy.COPY
            except ImportError:
                from speechbrain.pretrained import SpeakerRecognition
                strategy = None

            kwargs = {
                "source": self.model_source,
                "savedir": self.savedir,
                "run_opts": {"device": self.device},
            }
            if strategy is not None:
                kwargs["local_strategy"] = strategy

            self.model = SpeakerRecognition.from_hparams(**kwargs)
            logger.info("SpeechBrain ECAPA-TDNN loaded on device: %s", self.device)
        except Exception as e:
            logger.warning(
                "Could not load SpeechBrain ECAPA-TDNN model (%s). "
                "Ensure speechbrain is installed and model weights are accessible.",
                e,
            )

    def extract_segment(self, audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "ECAPA-TDNN model is not loaded. Please install speechbrain and verify network access."
            )

        # SpeechBrain expects tensor of shape [batch_size, time]
        tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_batch(tensor)
            # Embedding shape is [batch_size, 1, 192]
            vec = embeddings.squeeze().cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        return vec.astype(np.float32)
