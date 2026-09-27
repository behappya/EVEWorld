"""The IGR weight map is where the two interaction regions enter the loss.

It is binary by construction -- disturbed cells carry ``DISTURBED`` and the rest
``BACKGROUND`` -- and it is renormalised to unit mean, so concentrating weight on
a small region does not simply rescale the whole objective.
"""

from __future__ import annotations

import numpy as np
import pytest

from eveworld.methods.igr.weight_map import (
    BACKGROUND,
    DISTURBED,
    boxes_to_mask,
    build_weight_map,
    clip_boxes,
    normalize_unit_mean,
    stamp_regions,
    support_mask,
)

SHAPE = (8, 12)
BOXES = [(2, 1, 5, 4), (7, 6, 10, 9)]


def test_only_the_two_canonical_levels_are_used():
    weight = build_weight_map(SHAPE, BOXES, normalize=False)
    assert weight.shape == SHAPE
    assert set(np.unique(weight).tolist()) <= {BACKGROUND, DISTURBED}
    assert DISTURBED > BACKGROUND


def test_the_disturbed_level_lands_exactly_on_the_given_boxes():
    weight = build_weight_map(SHAPE, BOXES, normalize=False)
    assert (weight[1:4, 2:5] == DISTURBED).all()
    assert (weight[6:9, 7:10] == DISTURBED).all()
    assert weight[0, 0] == BACKGROUND
    assert weight[5, 5] == BACKGROUND


def test_no_region_leaves_a_flat_unit_map():
    weight = build_weight_map(SHAPE, [])
    assert np.allclose(weight, BACKGROUND)
    assert not support_mask(SHAPE, []).any()


def test_normalisation_gives_unit_mean_without_touching_the_ratio():
    raw = build_weight_map(SHAPE, BOXES, normalize=False)
    norm = normalize_unit_mean(raw.astype(np.float64))
    assert norm.mean() == pytest.approx(1.0, abs=1e-6)
    disturbed = norm[raw == DISTURBED].mean()
    background = norm[raw == BACKGROUND].mean()
    assert disturbed / background == pytest.approx(DISTURBED / BACKGROUND)


def test_the_support_is_the_union_of_the_regions():
    support = support_mask(SHAPE, BOXES)
    assert support.dtype == bool
    assert support.shape == SHAPE

    expected = np.zeros(SHAPE, dtype=bool)
    for x0, y0, x1, y1 in BOXES:
        expected[y0 : min(y1, SHAPE[0]), x0 : min(x1, SHAPE[1])] = True

    assert np.array_equal(support, expected)
    assert support[1:4, 2:5].all()
    assert support[6:9, 7:10].all()
    assert not support[0, 0]


def test_regions_can_be_given_as_masks():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[2:4, 3:6] = True

    stamped = stamp_regions(SHAPE, [mask], value=DISTURBED)
    assert stamped.shape == SHAPE
    assert (stamped[mask] == DISTURBED).all()
    assert (stamped[~mask] == 0.0).all()

    assert support_mask(SHAPE, [mask]).tolist() == mask.tolist()


def test_a_box_that_leaves_the_canvas_is_clipped_not_dropped():
    mask = boxes_to_mask(SHAPE, [(-4, -4, 3, 3), (9, 5, 40, 40)])
    assert mask.shape == SHAPE
    assert mask[0, 0] and mask[1, 1]
    assert not mask[3, 3]
    assert mask[5, 9] and mask[7, 11]


def test_clip_boxes_stays_inside_the_canvas():
    boxes = np.array([[-4, -4, 3, 3], [9, 5, 40, 40]], dtype=np.float64)
    clipped = clip_boxes(boxes, SHAPE)
    assert clipped.shape == boxes.shape
    assert (clipped[:, 0] >= 0).all() and (clipped[:, 1] >= 0).all()
    assert (clipped[:, 2] <= SHAPE[1]).all() and (clipped[:, 3] <= SHAPE[0]).all()
    assert clipped[0, 0] == 0 and clipped[0, 1] == 0
