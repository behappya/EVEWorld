"""A missing instance the robot plausibly hides is not counted as a failure.

The occlusion ratio is measured against the target mask only, the threshold is
applied strictly, and a gap is forgiven only when every missing instance has an
occluded instance to account for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from eveworld.evaluation.mlr.occlusion import (
    adjust_counts,
    is_occluded,
    occlusion_ratio,
    occlusion_table,
)

TAU_OCC = 0.15


def box_mask(shape, box):
    mask = np.zeros(shape, dtype=bool)
    y0, x0, y1, x1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def test_occlusion_ratio_is_measured_against_the_target_area():
    target = box_mask((10, 10), (0, 0, 10, 10))
    robot = box_mask((10, 10), (0, 0, 3, 5))
    assert occlusion_ratio(target, robot) == pytest.approx(0.15)


def test_an_empty_target_has_no_occlusion():
    target = np.zeros((4, 4), dtype=bool)
    assert occlusion_ratio(target, np.ones((4, 4), dtype=bool)) == 0.0


def test_the_threshold_is_exclusive():
    target = box_mask((10, 10), (0, 0, 10, 10))
    exactly_at_threshold = box_mask((10, 10), (0, 0, 3, 5))
    just_above = box_mask((10, 10), (0, 0, 4, 4))
    assert not is_occluded(target, exactly_at_threshold, tau_occ=TAU_OCC)
    assert is_occluded(target, just_above, tau_occ=TAU_OCC)


def test_the_table_covers_every_timestamp_and_instance():
    target = np.stack([box_mask((8, 8), (0, 0, 8, 8))] * 3)
    robot = np.stack(
        [
            np.zeros((8, 8), dtype=bool),
            box_mask((8, 8), (0, 0, 8, 2)),
            box_mask((8, 8), (0, 0, 8, 1)),
        ]
    )
    table = occlusion_table(target, robot, tau_occ=TAU_OCC)
    assert table.shape == (3, 1)
    assert table.dtype == bool
    assert table[:, 0].tolist() == [False, True, False]


def test_the_table_keeps_instances_apart():
    frame = box_mask((6, 6), (0, 0, 6, 6))
    robot = box_mask((6, 6), (0, 0, 6, 2))
    target = np.stack([frame, frame])[None]
    table = occlusion_table(target, robot[None], tau_occ=TAU_OCC)
    assert table.shape == (1, 2)
    assert table.tolist() == [[True, True]]


def test_adjust_counts_leaves_a_complete_observation_alone():
    counts = np.array([2, 2, 2])
    occlusion = np.zeros((3, 2), dtype=bool)
    assert adjust_counts(counts, occlusion, expected=2).tolist() == [2, 2, 2]


def test_adjust_counts_forgives_a_gap_that_is_fully_occluded():
    counts = np.array([2, 1])
    occlusion = np.zeros((2, 2), dtype=bool)
    occlusion[1, 0] = True
    assert adjust_counts(counts, occlusion, expected=2).tolist() == [2, 2]


def test_adjust_counts_keeps_a_gap_that_is_only_partly_occluded():
    counts = np.array([1, 0])
    occlusion = np.zeros((2, 2), dtype=bool)
    occlusion[0, 0] = True
    occlusion[1, 1] = True
    assert adjust_counts(counts, occlusion, expected=2).tolist() == [2, 0]
