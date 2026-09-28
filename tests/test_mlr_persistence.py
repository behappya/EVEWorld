"""A single deviating timestamp is noise; MLR only counts persistent deviation.

The tracker and the vectorised helper have to agree element by element, and an
exempted observation has to break the streak rather than extend it.
"""

from __future__ import annotations

import numpy as np

from eveworld.evaluation.mlr.persistence import PersistenceResult, PersistenceTracker, detect_events


def test_two_consecutive_deviations_are_needed():
    tracker = PersistenceTracker(k=2)
    assert not tracker.update(True)
    assert tracker.update(True)


def test_a_single_deviation_does_not_fire():
    tracker = PersistenceTracker(k=2)
    assert not tracker.update(True)
    assert not tracker.update(False)
    assert not tracker.update(False)


def test_an_exemption_resets_the_streak():
    tracker = PersistenceTracker(k=2)
    assert not tracker.update(True)
    assert not tracker.update(True, exempt=True)
    assert not tracker.update(True)
    assert tracker.update(True)


def test_reset_clears_the_streak():
    tracker = PersistenceTracker(k=2)
    tracker.update(True)
    tracker.reset()
    assert not tracker.update(True)
    assert tracker.update(True)


def test_events_are_labelled_from_the_first_persistent_timestamp():
    deviations = np.array([False, True, True, False, True, True, True])
    events = detect_events(deviations, k=2)
    assert events.dtype == bool
    assert events.tolist() == [False, False, True, False, False, True, True]


def test_exemptions_break_the_streak_in_the_vectorised_helper():
    deviations = np.array([True, True, True])
    exemptions = np.array([False, True, False])
    assert detect_events(deviations, exemptions, k=2).tolist() == [False] * 3


def test_tracker_and_vectorised_helper_agree():
    rng = np.random.default_rng(20240117)
    deviations = rng.random(256) < 0.5
    exemptions = rng.random(256) < 0.15

    tracker = PersistenceTracker(k=2)
    stepwise = np.array([tracker.update(bool(d), exempt=bool(e)) for d, e in zip(deviations, exemptions)])
    assert np.array_equal(stepwise, detect_events(deviations, exemptions, k=2))


def test_higher_k_needs_a_longer_streak():
    deviations = np.array([True, True, True, False])
    assert detect_events(deviations, k=3).tolist() == [False, False, True, False]
    assert detect_events(deviations, k=1).tolist() == [True, True, True, False]


def test_the_result_dataclass_carries_the_three_arrays():
    result = PersistenceResult(
        events=np.array([False, True]),
        deviations=np.array([False, True]),
        streak=np.array([0, 1]),
    )
    assert result.events.tolist() == [False, True]
    assert result.deviations.tolist() == [False, True]
    assert result.streak.tolist() == [0, 1]
