"""Open-vocabulary grounding backends."""

from .grounding_dino import Detection, GroundingDinoDetector

__all__ = ["Detection", "GroundingDinoDetector"]
