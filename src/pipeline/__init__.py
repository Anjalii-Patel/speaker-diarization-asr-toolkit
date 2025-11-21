"""End-to-End Speech Pipeline subpackage."""

from .speaker_attributed_asr import (
    SpeakerAttributedASRPipeline,
    AttributedUtterance,
    PipelineResult,
)

__all__ = [
    "SpeakerAttributedASRPipeline",
    "AttributedUtterance",
    "PipelineResult",
]
