"""
NVIDIA NeMo Diarization Interface
Encapsulates NeMo ClusteringDiarizer and NeuralDiarizer with automated manifest management.
"""

import os
import json
import logging
from typing import List, Optional
import numpy as np
from .rttm import RTTMSegment, parse_rttm

logger = logging.getLogger(__name__)

class NeMoDiarizer:
    """
    Wrapper around NVIDIA NeMo speaker diarization framework.
    """

    def __init__(
        self,
        config_path: str = "configs/diar_infer_telephonic.yaml",
        out_dir: str = "output_nemo",
        device: Optional[str] = None,
        oracle_vad: bool = False,
        speaker_model: str = "titanet_large",
    ):
        self.config_path = config_path
        self.out_dir = out_dir
        self.device = device
        self.oracle_vad = oracle_vad
        self.speaker_model = speaker_model

    def diarize(
        self,
        audio_path: str,
        reference_rttm: Optional[str] = None,
        num_speakers: Optional[int] = None,
    ) -> List[RTTMSegment]:
        """
        Execute NeMo diarization pipeline on an audio file.

        Args:
            audio_path: Path to the target audio file (16kHz mono recommended).
            reference_rttm: Optional path to reference RTTM for Oracle VAD mode.
            num_speakers: Known speaker count (optional).

        Returns:
            List of predicted RTTMSegment objects.
        """
        try:
            from omegaconf import OmegaConf
            from nemo.collections.asr.models import ClusteringDiarizer
        except ImportError as e:
            raise ImportError(
                f"NVIDIA NeMo is required for NeMoDiarizer but could not be imported: {e}. "
                "Install nemo_toolkit[asr] to enable this component."
            )

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"NeMo config YAML not found: {self.config_path}")

        abs_audio_path = os.path.abspath(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        os.makedirs(self.out_dir, exist_ok=True)

        # 1. Prepare temporary manifest
        manifest_path = os.path.join(self.out_dir, f"manifest_{base_name}.json")
        meta = {
            "audio_filepath": abs_audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": os.path.abspath(reference_rttm) if reference_rttm else None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")

        # 2. Configure Diarizer
        config = OmegaConf.load(self.config_path)
        config.diarizer.manifest_filepath = manifest_path
        config.diarizer.out_dir = os.path.abspath(self.out_dir)
        config.diarizer.oracle_vad = bool(self.oracle_vad and reference_rttm)

        if self.speaker_model:
            config.diarizer.speaker_embeddings.model_path = self.speaker_model

        if num_speakers is not None:
            config.diarizer.clustering.parameters.oracle_num_speakers = True
        else:
            config.diarizer.clustering.parameters.oracle_num_speakers = False

        if self.device:
            config.device = self.device

        # 3. Execute Diarization
        logger.info("Initializing NeMo ClusteringDiarizer with config: %s", self.config_path)
        diarizer_model = ClusteringDiarizer(cfg=config)
        diarizer_model.diarize()

        # 4. Locate predicted RTTM
        pred_rttm_path = os.path.join(self.out_dir, "pred_rttms", f"{base_name}.rttm")
        if not os.path.exists(pred_rttm_path):
            logger.warning("Expected output RTTM not found at %s", pred_rttm_path)
            return []

        return parse_rttm(pred_rttm_path)
