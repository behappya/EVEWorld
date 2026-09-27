"""Multi-Instance Localisation Rate (MLR).

MLR counts the fraction of sampled timestamps at which at least one object the
instruction asks for is missing from the generated clip. A deviation only
becomes an event once it persists for two consecutive sampled timestamps, and a
missing instance that is plausibly hidden behind the robot arm is exempted.
"""

from .detector import InstanceCounter, count_instances, expected_count
from .merge import align_timestamps, build_metadata, merge_detections, merge_track_masks
from .metric import aggregate, clip_mlr, coverage, missing_rate
from .occlusion import adjust_counts, is_occluded, occlusion_ratio, occlusion_table
from .persistence import PersistenceResult, PersistenceTracker, detect_events
from .vlm_audit import VLMAuditor, build_audit_prompt, load_prompt_template, parse_audit_response

__all__ = [
    "InstanceCounter",
    "PersistenceResult",
    "PersistenceTracker",
    "VLMAuditor",
    "adjust_counts",
    "aggregate",
    "align_timestamps",
    "build_audit_prompt",
    "build_metadata",
    "clip_mlr",
    "count_instances",
    "coverage",
    "detect_events",
    "expected_count",
    "is_occluded",
    "load_prompt_template",
    "merge_detections",
    "merge_track_masks",
    "missing_rate",
    "occlusion_ratio",
    "occlusion_table",
    "parse_audit_response",
]
