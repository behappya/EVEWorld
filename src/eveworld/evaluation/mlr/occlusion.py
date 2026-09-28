"""Occlusion exemption for the Multi-Instance Localisation Rate protocol.

Algorithm 1 skips a timestamp whose instance count falls below the frame-0 inventory only
when every instance that went missing is plausibly hidden behind the robot arm. For target
instance ``i`` at timestamp ``t`` the occlusion score is

    r_occ[t, i] = |target[t, i] & robot[t]| / |target[t, i]|,

and the instance is exempt when ``r_occ > tau_occ`` (:data:`DEFAULT_TAU_OCC`; the paper's
boundary is strict, an exact tie is not an exemption). A target whose mask area drops below
:data:`RELIABLE_AREA_RATIO` of its own temporal median carries no trustworthy visibility
evidence, so it does not vote in the exemption on its own.

Array shapes: target masks are ``(T, N, H, W)``, ``(N, H, W)`` for instances at a single
timestamp or ``(T, H, W)`` for a single tracked instance per timestamp; a robot mask is
``(H, W)`` (broadcast over time) or ``(T, H, W)``; the occlusion table is ``(T, N)`` (``(T, 1)``
for a ``(T, H, W)`` stack) and :func:`adjust_counts` returns the per-timestamp effective counts
``(T,)``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "DEFAULT_TAU_OCC",
    "RELIABLE_AREA_RATIO",
    "adjust_counts",
    "exempt_targets",
    "is_occluded",
    "occlusion_ratio",
    "occlusion_table",
    "reliable_targets",
]

DEFAULT_TAU_OCC = 0.15
RELIABLE_AREA_RATIO = 0.50


def _masks(values: Any, name: str) -> np.ndarray:
    """Boolean view of ``values`` after checking that it is an array of 3 or 4 dimensions."""
    array = np.asarray(values, dtype=bool)
    if array.ndim not in (3, 4):
        raise ValueError(f"{name} must have shape (N, H, W), (T, H, W) or (T, N, H, W), got {array.shape}")
    return array


def _robot_frames(robot_mask: Any, num_times: int) -> np.ndarray:
    """Robot mask as ``(T, H, W)``, broadcasting a single ``(H, W)`` mask over time."""
    robot = np.asarray(robot_mask, dtype=bool)
    if robot.ndim == 2:
        return np.broadcast_to(robot, (num_times,) + robot.shape)
    if robot.ndim == 3:
        if robot.shape[0] == num_times:
            return robot
        if robot.shape[0] == 1:
            return np.broadcast_to(robot[0], (num_times,) + robot.shape[1:])
        raise ValueError(f"robot_mask has {robot.shape[0]} frames but the target masks have {num_times}")
    raise ValueError(f"robot_mask must have shape (H, W) or (T, H, W), got {robot.shape}")


def _occlusion(areas: np.ndarray, hits: np.ndarray, tau_occ: float) -> np.ndarray:
    """``hits / areas > tau_occ`` with an empty mask scoring ``0.0`` instead of NaN."""
    ratio = np.divide(hits, areas, out=np.zeros_like(areas, dtype=float), where=areas > 0)
    return ratio > tau_occ


def occlusion_ratio(target_mask: np.ndarray, robot_mask: np.ndarray) -> float:
    """Share of the target mask covered by the robot mask, ``|target & robot| / |target|``.

    Args:
        target_mask: Boolean mask of shape ``(H, W)``.
        robot_mask: Boolean mask of shape ``(H, W)``.

    Returns:
        The overlap ratio in ``[0, 1]``; ``0.0`` for an empty target mask (never NaN).
    """
    target = np.asarray(target_mask, dtype=bool)
    robot = np.asarray(robot_mask, dtype=bool)
    if target.shape != robot.shape:
        raise ValueError(f"target_mask {target.shape} and robot_mask {robot.shape} differ in shape")
    area = int(np.count_nonzero(target))
    if area == 0:
        return 0.0
    return float(np.count_nonzero(target & robot)) / area


def is_occluded(target_mask: np.ndarray, robot_mask: np.ndarray, tau_occ: float = DEFAULT_TAU_OCC) -> bool:
    """Whether a single instance is exempt because the robot arm covers it.

    Args:
        target_mask: Boolean mask of shape ``(H, W)``.
        robot_mask: Boolean mask of shape ``(H, W)``.
        tau_occ: Occlusion threshold; the paper's test is strict, so a ratio of exactly
            ``tau_occ`` is not an exemption.

    Returns:
        ``True`` when ``occlusion_ratio(target_mask, robot_mask) > tau_occ``.
    """
    return occlusion_ratio(target_mask, robot_mask) > tau_occ


def occlusion_table(target_masks: np.ndarray, robot_mask: np.ndarray, tau_occ: float = DEFAULT_TAU_OCC) -> np.ndarray:
    """Per-instance occlusion table over the sampled timeline.

    Args:
        target_masks: Boolean masks in one of three layouts: ``(T, N, H, W)`` for a timeline of
            ``N`` tracked instances, ``(N, H, W)`` for instances at a single timestamp, or
            ``(T, H, W)`` for a single tracked instance per timestamp. A 3-D stack is read as a
            timeline when the robot mask is also a stack and as a bag of instances when the
            robot mask is a single frame.
        robot_mask: Boolean mask of shape ``(H, W)``, shared by every timestamp, or
            ``(T, H, W)`` when the arm moves.
        tau_occ: Occlusion threshold, applied as a strict ``>`` test.

    Returns:
        Boolean table of shape ``(T, N)`` for 4-D input, ``(T, 1)`` for a timeline of single
        instances and ``(N,)`` for instances at a single timestamp; entries are ``True`` where
        the instance is occluded by the robot.
    """
    targets = _masks(target_masks, "target_masks")
    robot = np.asarray(robot_mask, dtype=bool)
    if targets.ndim == 3 and robot.ndim == 2:
        # (N, H, W) at a single timestamp, with one robot mask shared by every instance.
        if robot.shape != targets.shape[1:]:
            raise ValueError(f"robot_mask {robot.shape} does not match {targets.shape[1:]}")
        areas = np.count_nonzero(targets, axis=(1, 2))
        hits = np.count_nonzero(targets & robot, axis=(1, 2))
        return _occlusion(areas, hits, tau_occ)
    if targets.ndim == 3:
        # (T, H, W): one tracked instance per timestamp, with a robot frame for each.
        frames = _robot_frames(robot, targets.shape[0])
        if frames.shape[1:] != targets.shape[1:]:
            raise ValueError(f"robot_mask {frames.shape[1:]} does not match {targets.shape[1:]}")
        areas = np.count_nonzero(targets, axis=(1, 2))
        hits = np.count_nonzero(targets & frames, axis=(1, 2))
        return _occlusion(areas, hits, tau_occ)[:, None]
    times, count = targets.shape[:2]
    robot_frames = _robot_frames(robot_mask, times)
    if robot_frames.shape[1:] != targets.shape[2:]:
        raise ValueError(f"robot_mask {robot_frames.shape[1:]} does not match {targets.shape[2:]}")
    areas = np.count_nonzero(targets, axis=(2, 3))
    hits = np.count_nonzero(targets & robot_frames[:, None], axis=(2, 3))
    return _occlusion(areas, hits, tau_occ)


def reliable_targets(target_masks: np.ndarray, *, area_ratio: float = RELIABLE_AREA_RATIO) -> np.ndarray:
    """Masks large enough to vote in the exemption.

    A tracked instance votes only while its mask area is at least ``area_ratio`` times its own
    temporal median, which drops spurious detections that survive for a single frame.

    Args:
        target_masks: Boolean masks of shape ``(T, N, H, W)`` or, for a stack, ``(K, H, W)``.
        area_ratio: Minimum area relative to the instance's median area; ``0.50`` follows
            the released protocol.

    Returns:
        Boolean array of shape ``(T, N)`` for 4-D input and ``(K,)`` for 3-D input, one entry
        per mask of the stack; a single-layer mask votes whenever it is non-empty.
    """
    targets = _masks(target_masks, "target_masks")
    if targets.ndim == 3:
        return np.count_nonzero(targets, axis=(1, 2)) > 0
    areas = np.count_nonzero(targets, axis=(2, 3))
    if areas.shape[0] == 0:
        return np.zeros(areas.shape, dtype=bool)
    median = np.median(areas, axis=0)
    return areas >= area_ratio * median


def exempt_targets(
    target_masks: np.ndarray,
    robot_mask: np.ndarray,
    *,
    tau_occ: float = DEFAULT_TAU_OCC,
    area_ratio: float = RELIABLE_AREA_RATIO,
) -> np.ndarray:
    """Occlusion table intersected with the reliability gate.

    Args:
        target_masks: Boolean masks in one of the layouts :func:`occlusion_table` accepts.
        robot_mask: Boolean mask of shape ``(H, W)`` or ``(T, H, W)``.
        tau_occ: Occlusion threshold, applied as a strict ``>`` test.
        area_ratio: Reliability threshold passed to :func:`reliable_targets`.

    Returns:
        Boolean array of shape ``(T, N)`` for 4-D input and ``(N,)`` or ``(T, 1)`` for 3-D
        input, matching :func:`occlusion_table`; ``True`` where the instance is both occluded
        and reliable enough to exempt the timestamp.
    """
    table = occlusion_table(target_masks, robot_mask, tau_occ)
    return table & reliable_targets(target_masks, area_ratio=area_ratio)


def adjust_counts(
    counts: np.ndarray | Sequence[Any],
    occlusion: np.ndarray | Sequence[Any],
    expected: int | np.ndarray | Sequence[Any] | None,
) -> np.ndarray:
    """Effective per-timestamp instance counts after the occlusion exemption.

    Paper rule: a timestamp is skipped, and its count falls back to the expected count, only
    when **every** instance that is missing at that timestamp is occluded; a single visible
    missing instance keeps the raw count, so the timestamp stays a deviation.

    Args:
        counts: Raw counts of shape ``(T,)``, or occupancy flags of shape ``(T, N)`` whose row
            sums are the raw counts.
        occlusion: Exemption witnesses of shape ``(T,)`` (per-timestamp flag, typically from
            :func:`exempt_targets` reduced over instances) or ``(T, N)`` (per-instance table).
        expected: Expected number of instances: an ``int``, shape ``(T,)`` per timestamp, or
            shape ``(T, N)`` per instance slot. ``None`` disables the exemption entirely.

    Returns:
        Integer array of shape ``(T,)`` with the effective count of every timestamp: the
        expected count where the exemption applies, the observed count everywhere else. An
        over-count (more instances than expected) is never exempted.
    """
    observed = np.asarray(counts, dtype=int)
    table = np.asarray(occlusion, dtype=bool)
    if observed.ndim == 2:
        observed_row = observed.sum(axis=1)
    elif observed.ndim == 1:
        observed_row = observed
    else:
        raise ValueError(f"counts must have shape (T,) or (T, N), got {observed.shape}")
    if expected is None:
        return observed_row.astype(int)
    target = np.asarray(expected, dtype=int)
    if observed.ndim == 2 and table.ndim == 2 and target.ndim == 2 and observed.shape == target.shape:
        # "Every missing instance of that timestamp must be occluded for it to be skipped."
        missing_slots = (target > 0) & (observed <= 0)
        every_missing_occluded = np.all(~missing_slots | table, axis=1)
        expected_row = target.sum(axis=1)
        missing_count = np.maximum(0, expected_row - observed_row)
        exempt = every_missing_occluded & (missing_count > 0)
        return np.where(exempt, expected_row, observed_row).astype(int)
    expected_row = target.sum(axis=1) if target.ndim == 2 else target
    witnesses = table.sum(axis=1) if table.ndim == 2 else table
    missing_count = np.maximum(0, expected_row - observed_row)
    exempt = (missing_count > 0) & (witnesses >= missing_count)
    return np.where(exempt, expected_row, observed_row).astype(int)
