"""The IGR corruption builds the disturbed clip and the matching weight map.

A sample either carries a duplicated instance with a binary weight map focused
on the interaction region, or falls back to the untouched clip -- there is no
partial in-between. The association step that turns per-frame detections into
the track consumed here is exercised at the end of the file.
"""

from __future__ import annotations

import numpy as np
import pytest

from eveworld.data.grounding.grounding_dino import Detection
from eveworld.data.parsers.instruction_parser import ParsedInstruction
from eveworld.data.tracking.sam2_tracker import associate_boxes
from eveworld.methods.igr.corruption import (
    as_clean_sample,
    build_sample,
    insert_duplicate,
    interaction_region,
)
from eveworld.methods.igr.trajectory import Track, build_tracks
from eveworld.methods.igr.weight_map import BACKGROUND, DISTURBED

SHAPE = (32, 32)
REGION = np.array([20, 20, 28, 28], dtype=np.int64)
PARSED = ParsedInstruction(
    raw="put the red cube on the plate",
    verb="put",
    target="red cube",
    source="table",
    destination="plate",
    objects=["red cube", "plate"],
)


def make_frames(num_frames: int = 4) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (num_frames, *SHAPE, 3), dtype=np.uint8)


def make_track(num_frames: int = 4) -> Track:
    return Track(
        track_id=0,
        label="red cube",
        boxes=np.tile(np.array([4.0, 4.0, 12.0, 12.0]), (num_frames, 1)),
        scores=np.full((num_frames,), 0.9),
    )


def test_the_fallback_sample_is_the_clean_clip():
    frames = make_frames()
    sample = as_clean_sample(frames)

    assert np.array_equal(sample.frames, frames)
    assert sample.events == []
    assert sample.fallback is True
    assert sample.weight_map.shape == frames.shape[1:3]
    assert np.allclose(sample.weight_map, BACKGROUND)
    assert not np.asarray(sample.support).any()


def test_the_duplicate_only_touches_its_paste_region():
    frames = make_frames()
    original = frames.copy()

    corrupted, event = insert_duplicate(frames, make_track(), REGION, rng=np.random.default_rng(0))

    assert corrupted.shape == frames.shape
    assert corrupted.dtype == frames.dtype
    assert np.array_equal(frames, original)

    changed = np.any(corrupted != original, axis=-1)
    assert changed.any()
    touched_frames, ys, xs = np.nonzero(changed)
    assert ys.min() >= REGION[1] and ys.max() <= REGION[3]
    assert xs.min() >= REGION[0] and xs.max() <= REGION[2]
    assert set(touched_frames.tolist()) == set(range(frames.shape[0]))
    assert event.kind


def test_the_duplicate_is_reproducible_for_a_given_seed():
    frames = make_frames()
    track = make_track()

    first, first_event = insert_duplicate(frames, track, REGION, rng=np.random.default_rng(0))
    second, second_event = insert_duplicate(frames, track, REGION, rng=np.random.default_rng(0))

    assert np.array_equal(first, second)
    assert first_event.paste_box == pytest.approx(second_event.paste_box)


def test_a_fresh_region_keeps_the_patch_inside_the_frame():
    frames = make_frames()
    corrupted, event = insert_duplicate(frames, make_track(), REGION, rng=np.random.default_rng(1))

    assert tuple(np.asarray(event.paste_box).shape) == (4,)
    assert event.track_id == 0
    x0, y0, x1, y1 = np.asarray(event.paste_box).tolist()
    assert 0 <= y0 < y1 <= SHAPE[0]
    assert 0 <= x0 < x1 <= SHAPE[1]

    changed = np.any(corrupted != frames, axis=-1)
    touched_frames, ys, xs = np.nonzero(changed)
    assert ys.size > 0
    assert set(touched_frames.tolist()) == set(range(frames.shape[0]))
    assert ys.min() >= y0 and ys.max() < y1
    assert xs.min() >= x0 and xs.max() < x1


def test_build_sample_is_reproducible_for_a_given_seed():
    frames = make_frames()
    track = make_track()

    first = build_sample(frames, track, PARSED, p_dup=1.0, shape=SHAPE, rng=np.random.default_rng(3))
    second = build_sample(frames, track, PARSED, p_dup=1.0, shape=SHAPE, rng=np.random.default_rng(3))

    assert np.array_equal(first.frames, second.frames)
    assert np.allclose(first.weight_map, second.weight_map)
    assert first.fallback == second.fallback


def test_a_duplicate_sample_reports_an_event_and_a_two_level_map():
    frames = make_frames()
    sample = build_sample(
        frames, make_track(), PARSED, p_dup=1.0, shape=SHAPE, rng=np.random.default_rng(3)
    )

    assert sample.fallback is False
    assert len(sample.events) >= 1
    assert isinstance(sample.metadata, dict)

    levels = np.unique(sample.weight_map)
    assert levels.size == 2
    background, disturbed = np.sort(levels).tolist()
    assert disturbed / background == pytest.approx(DISTURBED / BACKGROUND)
    disturbed_mean = sample.weight_map[sample.weight_map == disturbed].mean()
    background_mean = sample.weight_map[sample.weight_map == background].mean()
    assert disturbed_mean / background_mean == pytest.approx(DISTURBED / BACKGROUND)
    assert sample.weight_map.mean() == pytest.approx(1.0, abs=1e-6)


def test_a_sample_built_without_the_duplicate_branch_stays_consistent():
    frames = make_frames()
    sample = build_sample(
        frames, make_track(), PARSED, p_dup=0.0, shape=SHAPE, rng=np.random.default_rng(5)
    )

    assert sample.frames.shape == frames.shape
    assert sample.weight_map.shape == SHAPE
    assert np.asarray(sample.support).shape == SHAPE
    assert sample.weight_map.mean() == pytest.approx(1.0, abs=1e-6)
    if sample.fallback:
        assert np.array_equal(sample.frames, frames)


def test_the_interaction_region_is_a_box_inside_the_frame():
    frames = make_frames()
    track = make_track()
    _, event = insert_duplicate(frames, track, REGION, rng=np.random.default_rng(2))

    region = np.asarray(interaction_region(track, event, frames.shape[1:3]))

    assert region.shape == (4,)
    assert region[0] >= 0 and region[1] >= 0
    assert region[2] <= SHAPE[1] and region[3] <= SHAPE[0]
    assert region[2] > region[0] and region[3] > region[1]


def test_the_association_maps_reordered_boxes_to_their_track():
    previous = np.array([[0, 0, 4, 4], [10, 10, 14, 14]], dtype=np.float64)
    current = np.array([[10, 10, 14, 14], [0, 0, 4, 4]], dtype=np.float64)

    assert associate_boxes(previous, current).tolist() == [1, 0]
    assert associate_boxes(previous, current[:1]).tolist() == [-1, 0]


def test_detections_are_chained_into_a_single_track():
    square = np.array([2.0, 2.0, 10.0, 10.0])
    shifted = square + np.array([2.0, 0.0, 2.0, 0.0])
    detections = [[Detection(square, 0.9, "red cube")], [Detection(shifted, 0.8, "red cube")]]

    tracks = build_tracks(detections, iou_threshold=0.5)

    assert len(tracks) == 1
    assert tracks[0].label == "red cube"
    assert tracks[0].boxes.shape == (2, 4)
    assert tracks[0].scores.tolist() == pytest.approx([0.9, 0.8])
