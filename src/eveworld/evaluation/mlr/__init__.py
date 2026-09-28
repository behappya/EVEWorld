"""Model Laziness Rate (MLR).

MLR measures persistent violations of target-instance consistency over a
rollout: the adjusted count of target instances at a sampled timestamp
deviates from the count in the conditioning state, where under-counts that
robot occlusion explains are exempt and all over-counts are retained. A
deviation only becomes an event once it persists for two consecutive sampled
timestamps.
"""

from .detector import InstanceCounter, count_instances, expected_count
from .merge import align_timestamps, build_metadata, merge_detections, merge_track_masks
from .metric import aggregate, clip_mlr, coverage, mean_mlr
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
    "mean_mlr",
    "merge_detections",
    "merge_track_masks",
    "occlusion_ratio",
    "occlusion_table",
    "parse_audit_response",
]
