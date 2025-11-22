"""Tests for configuration loading and validation."""

import os
import yaml
import pytest

def test_configs_exist_and_valid_yaml():
    config_dir = "configs"
    assert os.path.exists(config_dir)

    expected_files = [
        "diar_infer_general.yaml",
        "diar_infer_meeting.yaml",
        "diar_infer_telephonic.yaml",
    ]

    for filename in expected_files:
        path = os.path.join(config_dir, filename)
        assert os.path.exists(path), f"Missing configuration file: {path}"

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "diarizer"in cfg
        assert "vad"in cfg["diarizer"]
        assert "speaker_embeddings"in cfg["diarizer"]
        assert "clustering"in cfg["diarizer"]

        # Ensure no hardcoded C:\ paths remain in configs
        manifest_path = str(cfg["diarizer"].get("manifest_filepath", ""))
        assert not manifest_path.startswith("C:"), f"Hardcoded C: path found in {filename}"
        out_dir = str(cfg["diarizer"].get("out_dir", ""))
        assert not out_dir.startswith("C:"), f"Hardcoded C: path found in {filename}"
