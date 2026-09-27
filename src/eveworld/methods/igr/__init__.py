"""Instance-Guided Restoration (IGR).

IGR keeps the interaction that the instruction asks for while re-grounding it on
a location the world model can actually render, so the repaired clip follows the
prompt without being dragged around by the layout of the source video.
"""

from .corruption import (
    IGREvent,
    IGRSample,
    as_clean_sample,
    build_sample,
    insert_duplicate,
    interaction_region,
    relocate_instance,
)
from .loss import edm_weight, igr_loss, resize_weight_map
from .paste_region import (
    admissible_regions,
    blend_patch,
    box_iou,
    candidate_regions,
    in_frame,
    inscribed_boxes,
    occupancy_ratio,
    overlaps_any,
    paste_box_region,
    select_paste_region,
)
from .trajectory import (
    Track,
    build_tracks,
    displacement,
    interaction_box,
    most_reliable_frame,
    reliability,
    trajectory_center,
)
from .weight_map import (
    BACKGROUND,
    DISTURBED,
    boxes_to_mask,
    build_weight_map,
    clip_boxes,
    normalize_unit_mean,
    stamp_regions,
    support_mask,
)

__all__ = [
    "BACKGROUND",
    "DISTURBED",
    "IGREvent",
    "IGRSample",
    "Track",
    "admissible_regions",
    "as_clean_sample",
    "blend_patch",
    "box_iou",
    "boxes_to_mask",
    "build_sample",
    "build_tracks",
    "build_weight_map",
    "candidate_regions",
    "clip_boxes",
    "displacement",
    "edm_weight",
    "igr_loss",
    "in_frame",
    "insert_duplicate",
    "inscribed_boxes",
    "interaction_box",
    "interaction_region",
    "most_reliable_frame",
    "normalize_unit_mean",
    "occupancy_ratio",
    "overlaps_any",
    "paste_box_region",
    "relocate_instance",
    "reliability",
    "resize_weight_map",
    "select_paste_region",
    "stamp_regions",
    "support_mask",
    "trajectory_center",
]
