#!/usr/bin/env python3
"""Build nohuman-v2 uniform-3x weightmaps with arm/source/target coverage fixes.

The previous `weightmap_cache_uniform3_nohuman_v2` is a value remap of the old
Track4Gen contract map: every non-background cell becomes 3x.  That preserves
old detector misses.  In practice several all-frame views miss the robot arm,
the manipulated object's true source position, or the destination region.

This script keeps the training contract simple:

  - background = 1.0
  - every selected supervision cell = 3.0

but expands the selected-cell set to:

  old uniform3 mask
  UNION robot arm corridors from base anchors to gripper/object/goal cells
  UNION dark robot-arm foreground only near those corridors
  UNION object trajectory candidates, target_cell_0, and distractor/object cells
  UNION a compact destination region derived from B cells

It writes a new cache/viz pair and never overwrites older results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

import t4g_probe as P


T_LAT, H_LAT, W_LAT = 24, 30, 48
NF, HPIX, WPIX, CELL = 93, 480, 768, 16
W_BG, W_MARK = 1.0, 3.0

ROOT = Path("/data/datasets/gagi")
RAW_ROOT = ROOT / "gr1_finetune_data/raw_data_t4g_nohuman_v2"
ANNO_DIR = ROOT / "eve_v2_outputs/track4gen_probe/t4g_anno_nohuman_v2"
GRIPPER_DIR = ROOT / "eve_v2_outputs/track4gen_probe/gripper_anno_nohuman_v2"
AUG_ASSETS_DIR = ROOT / "eve_v2_outputs/track4gen_probe/aug_assets_nohuman_v2"
BASE_CACHE_DIR = ROOT / "eve_v2_outputs/track4gen_probe/weightmap_cache_uniform3_nohuman_v2"

ARMFIX_VERSION = "v4"
OUT_CACHE_DIR = ROOT / f"eve_v2_outputs/track4gen_probe/weightmap_cache_uniform3_armfix_{ARMFIX_VERSION}_nohuman_v2"
OUT_VIZ_DIR = ROOT / f"eve_v2_outputs/track4gen_probe/weightmap_viz_uniform3_armfix_{ARMFIX_VERSION}_nohuman_v2"

LEFT_ANCHOR = (H_LAT - 1, 4)
RIGHT_ANCHOR = (H_LAT - 1, W_LAT - 5)
# GR-1 tabletop videos often include annotators/observers in the upper-left
# background.  They are not task state and must not receive weighted loss.  Keep
# this narrow so top-shelf objects on the right/center are still valid.
BACKGROUND_EXCLUDE_BOXES = [
    (0, 9, 0, 18),  # y0,y1,x0,x1 in latent grid
]

# Manual high-confidence corrections after human review.  Each entry is
# (t0, t1_inclusive, start_cell, end_cell, thickness).  These lines are only
# used when detector/corridor logic is too narrow for a visible robot arm.
MANUAL_ARM_LINES = {
    16: [
        # Late frames: both arms are clearly in the workspace but v2 missed
        # large parts of the black/gray links.  These lines follow the visible
        # arm bodies and stay away from the upper-left background person.
        (18, 23, LEFT_ANCHOR, (15, 17), 7),
        (18, 23, LEFT_ANCHOR, (20, 19), 7),
        (18, 23, RIGHT_ANCHOR, (18, 18), 7),
        (18, 23, RIGHT_ANCHOR, (21, 31), 6),
        # v4: the late right arm has a horizontal gray/black forearm and wrist
        # section that can fall between the base-to-target diagonals.  Add
        # short local segments that trace the visible mechanism instead of
        # enabling global dark foreground, which over-labels the paper/table.
        (18, 23, (22, 38), (18, 28), 6),
        (18, 23, (18, 28), (15, 19), 6),
        (20, 23, (26, 42), (22, 35), 6),
        (20, 23, (22, 35), (18, 24), 6),
        (20, 23, (18, 24), (16, 17), 5),
    ],
}

MANUAL_ARM_BOXES = {
    16: [
        # Local high-confidence boxes for vid16 late frames.  They cover the
        # visible joints/gripper rings that are wider than a single line.
        # Format: (t0, t1_inclusive, center_cell, half_size).
        (18, 23, (22, 38), (3, 4)),
        (18, 23, (19, 31), (3, 5)),
        (20, 23, (23, 35), (3, 5)),
        (20, 23, (18, 23), (3, 4)),
        (20, 23, (16, 17), (2, 4)),
    ],
}


def parse_vids(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return sorted(int(p.stem) for p in ANNO_DIR.glob("*.json") if p.stem.isdigit())
    return [int(x) for x in value.split(",") if x.strip()]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def instruction_for_vid(vid: int) -> str:
    path = RAW_ROOT / f"{vid}.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def latent_rep_indices() -> np.ndarray:
    return np.linspace(0, NF - 1, T_LAT).astype(int)


def grid_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def stamp_box(mask: np.ndarray, center: Iterable[int] | None, half: tuple[int, int], margin: int = 0) -> None:
    if center is None:
        return
    gy, gx = [int(v) for v in center]
    hh = max(0, int(half[0]) + margin)
    hw = max(0, int(half[1]) + margin)
    y0, y1 = max(0, gy - hh), min(H_LAT, gy + hh + 1)
    x0, x1 = max(0, gx - hw), min(W_LAT, gx + hw + 1)
    mask[y0:y1, x0:x1] = True


def stamp_cells(mask: np.ndarray, cells: Iterable[Iterable[int]], radius: int = 0) -> None:
    for c in cells or []:
        if c is None:
            continue
        gy, gx = [int(v) for v in c]
        stamp_box(mask, (gy, gx), (radius, radius), 0)


def stamp_line(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int], thickness: int = 3) -> None:
    canvas = mask.astype(np.uint8)
    # cv2 uses x,y; grid cells use gy,gx.
    cv2.line(canvas, (int(a[1]), int(a[0])), (int(b[1]), int(b[0])), 1, thickness=thickness)
    mask[:] = canvas.astype(bool)


def clear_background_exclusions(mask: np.ndarray) -> int:
    before = int(mask.sum())
    for y0, y1, x0, x1 in BACKGROUND_EXCLUDE_BOXES:
        mask[:, y0:y1, x0:x1] = False
    return before - int(mask.sum())


def apply_manual_arm_lines(mask: np.ndarray, vid: int) -> int:
    n = 0
    for t0, t1, start, end, thickness in MANUAL_ARM_LINES.get(int(vid), []):
        for t in range(max(0, t0), min(T_LAT - 1, t1) + 1):
            before = mask[t].copy()
            stamp_line(mask[t], tuple(start), tuple(end), int(thickness))
            n += int((mask[t] & ~before).sum())
    return n


def apply_manual_arm_boxes(mask: np.ndarray, vid: int) -> int:
    n = 0
    for t0, t1, center, half in MANUAL_ARM_BOXES.get(int(vid), []):
        for t in range(max(0, t0), min(T_LAT - 1, t1) + 1):
            before = mask[t].copy()
            stamp_box(mask[t], center, half, margin=0)
            n += int((mask[t] & ~before).sum())
    return n


def hand_anchor_from_instruction(text: str) -> tuple[int, int] | None:
    lower = text.lower()
    if "left hand" in lower:
        return LEFT_ANCHOR
    if "right hand" in lower:
        return RIGHT_ANCHOR
    return None


def object_halfsize(vid: int) -> tuple[int, int]:
    """Object half-size in latent cells, with a conservative floor.

    The old code reads the frame-0 aug asset bbox as y0,x0,y1,x1.  Some of the
    noisy detections are too tight for loss weighting, so this function floors
    the half-size at 2 cells.
    """
    fp = AUG_ASSETS_DIR / f"{vid}.npz"
    if fp.exists():
        try:
            d = np.load(fp)
            if "box" in d:
                y0, x0, y1, x1 = [float(v) for v in d["box"]]
                hh = max(2, int(round((y1 - y0) / 2 / CELL)))
                hw = max(2, int(round((x1 - x0) / 2 / CELL)))
                return min(hh, 6), min(hw, 6)
        except Exception:
            pass
    return (2, 2)


def compact_region_from_cells(cells: list[list[int]], cap_half: tuple[int, int] = (5, 6)) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return a compact center/half box from a potentially over-large detector mask."""
    if not cells:
        return None
    arr = np.asarray(cells, dtype=np.int32)
    cy = int(round(float(np.median(arr[:, 0]))))
    cx = int(round(float(np.median(arr[:, 1]))))
    hh = int(np.ceil((int(arr[:, 0].max()) - int(arr[:, 0].min())) / 2))
    hw = int(np.ceil((int(arr[:, 1].max()) - int(arr[:, 1].min())) / 2))
    hh = min(max(hh, 2), cap_half[0])
    hw = min(max(hw, 2), cap_half[1])
    return (cy, cx), (hh, hw)


def add_object_source_target(mask: np.ndarray, vid: int, anno: dict) -> dict:
    half = object_halfsize(vid)
    per = anno.get("per_lat_frame") or []

    # 1) Manipulated object trajectory candidates.  Even when the tracker jumps,
    # unioning candidates is safer than missing the object in supervision.
    for t in range(min(T_LAT, len(per))):
        c = per[t].get("target_cell")
        if c is not None:
            stamp_box(mask[t], c, half, margin=2)

    # 2) Explicit frame-0 target/source anchor.  This fixes cases where
    # per-frame target_cell drifts to a distractor while target_cell_0 is right.
    source_candidates: list[Iterable[int]] = []
    if anno.get("target_cell_0") is not None:
        source_candidates.append(anno["target_cell_0"])
    for p in per[:6]:
        if p.get("target_cell") is not None:
            source_candidates.append(p["target_cell"])
    for c in source_candidates:
        for t in range(T_LAT):
            stamp_box(mask[t], c, half, margin=2)

    # 3) Static object inventory/distractors.  These are small anchors; marking
    # them helps object permanence and also covers source/nearby items when the
    # prompt parser leaves a_cells empty.
    inventory = []
    inventory.extend(anno.get("inventory_cells") or [])
    inventory.extend(anno.get("distractor_cells") or [])
    for c in inventory:
        for t in range(T_LAT):
            stamp_box(mask[t], c, half, margin=1)

    # 4) Source A region when available.  Many current files have this empty, so
    # this is opportunistic.
    a_cells = anno.get("a_cells") or []
    if a_cells:
        region = compact_region_from_cells(a_cells, cap_half=(5, 6))
        if region:
            center, reg_half = region
            for t in range(T_LAT):
                stamp_box(mask[t], center, reg_half, margin=1)

    # 5) Destination B region.  The raw B detector can cover a large table/bag
    # area; use a compact robust box and mark it across the whole video so the
    # goal location is always supervised.
    b_pool: list[list[int]] = []
    t_arr = anno.get("t_arrival")
    cut = int(t_arr) if isinstance(t_arr, int) and t_arr > 0 else T_LAT
    for p in per[: min(cut, len(per))]:
        b_pool.extend(p.get("b_cells") or [])
    if not b_pool:
        for p in per:
            b_pool.extend(p.get("b_cells") or [])
    b_region = compact_region_from_cells(b_pool, cap_half=(5, 6))
    if b_region:
        center, reg_half = b_region
        for t in range(T_LAT):
            stamp_box(mask[t], center, reg_half, margin=1)

    return {
        "object_half": list(half),
        "n_source_candidates": len(source_candidates),
        "n_inventory_cells": len(inventory),
        "b_region": {"center": list(b_region[0]), "half": list(b_region[1])} if b_region else None,
    }


def build_robot_corridor_mask(vid: int, anno: dict, b_region: dict | None) -> tuple[np.ndarray, int]:
    """Build arm/gripper supervision corridors without using global foreground.

    This is the main v2 fix over v1.  People and background equipment can be
    dark, so color alone is unsafe.  Instead we mark only plausible robot-arm
    geometry:

    - compact boxes around gripper detection centers;
    - thick lines from left/right robot bases to those centers;
    - active-hand base to current target_cell;
    - active-hand base to compact B region near/after arrival.
    """
    corridor = np.zeros((T_LAT, H_LAT, W_LAT), dtype=bool)
    per = anno.get("per_lat_frame") or []
    active_anchor = hand_anchor_from_instruction(instruction_for_vid(vid) or anno.get("prompt", ""))
    t_arr = anno.get("t_arrival")
    b_center = tuple(b_region["center"]) if b_region else None
    n = 0

    # Base/root areas are always visible robot structure.
    for t in range(T_LAT):
        stamp_box(corridor[t], LEFT_ANCHOR, (3, 5), margin=0)
        stamp_box(corridor[t], RIGHT_ANCHOR, (3, 5), margin=0)

    # Active hand to manipulated object and destination.  This covers cases
    # where the gripper detector misses the carrying hand.
    if active_anchor is not None:
        for t in range(min(T_LAT, len(per))):
            target = per[t].get("target_cell")
            if target is not None:
                target_t = (int(target[0]), int(target[1]))
                stamp_line(corridor[t], active_anchor, target_t, thickness=5)
                stamp_box(corridor[t], target_t, (3, 4), margin=0)
                n += 1
            if b_center is not None and isinstance(t_arr, int) and t >= max(0, t_arr - 3):
                stamp_line(corridor[t], active_anchor, b_center, thickness=5)
                stamp_box(corridor[t], b_center, (4, 5), margin=0)
                n += 1

    # Gripper detections: use centers only.  The cached cell boxes can span a
    # large part of the image and were a source of over-labeling.
    fp = GRIPPER_DIR / f"{vid}.json"
    if fp.exists():
        data = read_json(fp)
        gper = data.get("per_lat_gripper") or []
        last_by_side: dict[str, tuple[int, int]] = {}
        for t in range(T_LAT):
            dets = gper[t] if t < len(gper) else []
            seen_side = set()
            for g in dets:
                c = g.get("center")
                if c is None:
                    continue
                cy, cx = int(c[0]), int(c[1])
                side = "left" if cx < W_LAT // 2 else "right"
                anchor = LEFT_ANCHOR if side == "left" else RIGHT_ANCHOR
                center = (cy, cx)
                stamp_box(corridor[t], center, (4, 5), margin=0)
                stamp_line(corridor[t], anchor, center, thickness=4)
                last_by_side[side] = center
                seen_side.add(side)
                n += 1
            # Short detector holes: reuse last center per side, but only draw a
            # corridor; this avoids global dark-mask hallucinations.
            for side, center in list(last_by_side.items()):
                if side in seen_side:
                    continue
                anchor = LEFT_ANCHOR if side == "left" else RIGHT_ANCHOR
                stamp_box(corridor[t], center, (3, 4), margin=0)
                stamp_line(corridor[t], anchor, center, thickness=3)
                n += 1
    return corridor, n


def dark_arm_grid(frame_rgb: np.ndarray) -> np.ndarray:
    """Detect dark robot arm/gripper material and convert it to latent cells.

    This is intentionally high-recall.  It favors covering robot structure over
    avoiding a few extra dark static cells.  The mask is restricted away from
    the very top background, then component-filtered to keep substantial
    connected components.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    y = np.arange(frame_rgb.shape[0])[:, None]

    dark_black = gray < 70
    dark_neutral = (gray < 84) & (hsv[..., 1] < 65)
    # Use dark pixels only in the lower workspace.  High shelf/contact cases are
    # handled by gripper corridors and object/source/target boxes; allowing dark
    # pixels in the upper half pulls in people, tripods, and background cables.
    lower_or_side = (y > 185)
    pix = (dark_black | dark_neutral) & lower_or_side

    m = (pix.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, n):
        x, y0, w, h, area = stats[i]
        x2, y2 = x + w, y0 + h
        touches_bottom_base = y2 > int(HPIX * 0.60)
        touches_low_side = (x < 80 or x2 > WPIX - 80) and y2 > int(HPIX * 0.45)
        if area >= 650 and (touches_bottom_base or touches_low_side):
            keep[labels == i] = 255

    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    grid = np.zeros((H_LAT, W_LAT), dtype=bool)
    for gy in range(H_LAT):
        for gx in range(W_LAT):
            cell = keep[gy * CELL : (gy + 1) * CELL, gx * CELL : (gx + 1) * CELL]
            if float((cell > 0).mean()) >= 0.055:
                grid[gy, gx] = True
    return grid_dilate(grid, radius=1)


def grid_to_pixel_mask(grid: np.ndarray, radius: int = 0) -> np.ndarray:
    if radius > 0:
        grid = grid_dilate(grid, radius=radius)
    pix = np.zeros((HPIX, WPIX), dtype=np.uint8)
    for gy in range(H_LAT):
        for gx in range(W_LAT):
            if grid[gy, gx]:
                pix[gy * CELL : (gy + 1) * CELL, gx * CELL : (gx + 1) * CELL] = 255
    return pix


def mechanical_foreground_grid(frame_rgb: np.ndarray, allow_grid: np.ndarray) -> np.ndarray:
    """Experimental high-recall robot foreground, gated by geometry.

    v2 was safe around people but too narrow in late frames because the gripper
    detector often disappears.  This v3 foreground recovers the whole arm by
    detecting dark/neutral robot material, but only keeps connected components
    that touch the current robot corridor or fixed robot bases.  Background
    humans/equipment are dark too, but they do not touch the robot bases/corridor.

    This is intentionally not called in v4: visual review showed it can pull in
    the paper bag/table as neutral foreground.  Prefer narrow manual arm
    lines/boxes for confirmed misses.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    y = np.arange(frame_rgb.shape[0])[:, None]

    dark = gray < 82
    neutral_metal = (gray < 142) & (hsv[..., 1] < 80)
    workspace = y > 115
    pix = (dark | neutral_metal) & workspace

    m = (pix.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    allow_pix = grid_to_pixel_mask(allow_grid, radius=4)
    base_grid = np.zeros((H_LAT, W_LAT), dtype=bool)
    stamp_box(base_grid, LEFT_ANCHOR, (4, 7), margin=0)
    stamp_box(base_grid, RIGHT_ANCHOR, (4, 7), margin=0)
    base_pix = grid_to_pixel_mask(base_grid, radius=1)
    gate_pix = cv2.dilate(
        np.maximum(allow_pix, base_pix),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 220:
            continue
        comp = labels == i
        overlap = int(np.logical_and(comp, gate_pix > 0).sum())
        if overlap >= 20 or overlap / max(area, 1) >= 0.02:
            keep[comp] = 255

    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    grid = np.zeros((H_LAT, W_LAT), dtype=bool)
    for gy in range(H_LAT):
        for gx in range(W_LAT):
            cell = keep[gy * CELL : (gy + 1) * CELL, gx * CELL : (gx + 1) * CELL]
            if float((cell > 0).mean()) >= 0.04:
                grid[gy, gx] = True
    return grid_dilate(grid, radius=1)


def add_gripper_corridors(mask: np.ndarray, vid: int) -> int:
    fp = GRIPPER_DIR / f"{vid}.json"
    if not fp.exists():
        return 0
    data = read_json(fp)
    per = data.get("per_lat_gripper") or []
    n = 0
    left_anchor = (H_LAT - 1, 4)
    right_anchor = (H_LAT - 1, W_LAT - 5)

    # Current detections already tend to be high recall but can miss late frames.
    # Draw thick base-to-detection corridors to cover the arm body, not only the
    # detected gripper rectangle.
    last_centers: list[tuple[int, int]] = []
    for t in range(T_LAT):
        dets = per[t] if t < len(per) else []
        centers = []
        for g in dets:
            c = g.get("center")
            if c is None:
                continue
            cy, cx = int(c[0]), int(c[1])
            centers.append((cy, cx))
            # The cached GDINO "robot gripper" cell list is often a huge bbox
            # around the whole arm.  Do not stamp it verbatim; use the center as
            # a compact hand/gripper anchor and cover the arm body with a
            # bottom-anchor corridor.
            stamp_box(mask[t], (cy, cx), (3, 4), margin=0)
            anchor = left_anchor if cx < W_LAT // 2 else right_anchor
            stamp_line(mask[t], anchor, (cy, cx), thickness=3)
            n += 1
        if centers:
            last_centers = centers
        elif last_centers:
            for cy, cx in last_centers:
                anchor = left_anchor if cx < W_LAT // 2 else right_anchor
                stamp_box(mask[t], (cy, cx), (3, 4), margin=0)
                stamp_line(mask[t], anchor, (cy, cx), thickness=3)
                n += 1

        # Always cover robot bases / visible lower-arm root zones for both arms.
        stamp_box(mask[t], (H_LAT - 3, 4), (2, 4), margin=0)
        stamp_box(mask[t], (H_LAT - 3, W_LAT - 5), (2, 4), margin=0)
    return n


def build_armfix_weightmap(vid: int) -> tuple[np.ndarray, dict]:
    base = np.load(BASE_CACHE_DIR / f"{vid}.npy").astype(np.float32)
    if base.shape != (T_LAT, H_LAT, W_LAT):
        raise ValueError(f"bad base shape for vid {vid}: {base.shape}")
    anno = read_json(ANNO_DIR / f"{vid}.json")
    old_mask = base > (W_BG + 1e-6)
    mask = old_mask.copy()

    obj_meta = add_object_source_target(mask, vid, anno)
    corridor, n_corr = build_robot_corridor_mask(vid, anno, obj_meta.get("b_region"))
    mask |= corridor

    frames = P.sample_frames_like_training(str(RAW_ROOT / f"{vid}.mp4"), NF, HPIX, WPIX)
    reps = latent_rep_indices()
    corridor_near = np.stack([grid_dilate(corridor[t], radius=4) for t in range(T_LAT)], axis=0)
    dark_counts = []
    dark_added_counts = []
    for t, pidx in enumerate(reps):
        g = dark_arm_grid(frames[int(pidx)]) & corridor_near[t]
        dark_counts.append(int(g.sum()))
        before = mask[t].copy()
        mask[t] |= g
        dark_added_counts.append(int((mask[t] & ~before).sum()))
    manual_added_cells = apply_manual_arm_lines(mask, vid)
    manual_added_cells += apply_manual_arm_boxes(mask, vid)
    excluded_cells = clear_background_exclusions(mask)
    mask = grid_dilate(mask.reshape(T_LAT * H_LAT, W_LAT), radius=0).reshape(T_LAT, H_LAT, W_LAT)

    out = np.full((T_LAT, H_LAT, W_LAT), W_BG, dtype=np.float32)
    out[mask] = W_MARK
    meta = {
        "video_id": vid,
        "instruction": instruction_for_vid(vid),
        "old_marked_pct": float(old_mask.mean()),
        "new_marked_pct": float(mask.mean()),
        "added_marked_pct": float((mask & ~old_mask).mean()),
        "corridor_marked_pct": float(corridor.mean()),
        "dark_arm_cells_near_corridor_per_frame": dark_counts,
        "dark_added_cells_per_frame": dark_added_counts,
        "manual_added_cells": manual_added_cells,
        "gripper_corridor_stamps": n_corr,
        "excluded_background_cells": excluded_cells,
        **obj_meta,
    }
    return out, meta


def render_viz(vid: int, w: np.ndarray, out_dir: Path) -> None:
    base = np.load(BASE_CACHE_DIR / f"{vid}.npy").astype(np.float32)
    old = base > (W_BG + 1e-6)
    new = w > (W_BG + 1e-6)
    added = new & ~old
    old_kept = old & new
    frames = P.sample_frames_like_training(str(RAW_ROOT / f"{vid}.mp4"), NF, HPIX, WPIX)
    reps = latent_rep_indices()
    rows = []
    for t, pidx in enumerate(reps):
        left = np.ascontiguousarray(frames[int(pidx)].copy())
        right = frames[int(pidx)].copy().astype(np.float32)
        overlay = np.zeros_like(right, dtype=np.uint8)
        for gy in range(H_LAT):
            for gx in range(W_LAT):
                y0, y1 = gy * CELL, (gy + 1) * CELL
                x0, x1 = gx * CELL, (gx + 1) * CELL
                if added[t, gy, gx]:
                    overlay[y0:y1, x0:x1] = (240, 65, 220)  # magenta: armfix additions
                elif old_kept[t, gy, gx]:
                    overlay[y0:y1, x0:x1] = (60, 210, 210)  # cyan: previous 3x
        m = (overlay.sum(2) > 0)[..., None]
        right = np.where(m, 0.48 * right + 0.52 * overlay, right).astype(np.uint8)
        for gy in range(0, H_LAT + 1, 2):
            cv2.line(right, (0, gy * CELL), (WPIX, gy * CELL), (90, 90, 90), 1)
        for gx in range(0, W_LAT + 1, 3):
            cv2.line(right, (gx * CELL, 0), (gx * CELL, HPIX), (90, 90, 90), 1)
        cv2.putText(left, f"{vid} t={t}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        cv2.putText(right, f"armfix_{ARMFIX_VERSION}: cyan=old3x magenta=added3x; train map all marked cells=3x",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        rows.append(np.concatenate([left, right], axis=1))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"wmap_{vid}_allframes.png"
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def write_index_and_summary(records: list[dict]) -> None:
    OUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    idx_path = OUT_VIZ_DIR / "wmap_instruction_index.jsonl"
    with idx_path.open("w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda x: x["video_id"]):
            vid = r["video_id"]
            rec = {
                "video_id": vid,
                "instruction": r["instruction"],
                "weightmap_viz_path": str(OUT_VIZ_DIR / f"wmap_{vid}_allframes.png"),
                "weightmap_cache_path": str(OUT_CACHE_DIR / f"{vid}.npy"),
                "raw_video_path": str(RAW_ROOT / f"{vid}.mp4"),
                "prompt_txt_path": str(RAW_ROOT / f"{vid}.txt"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_path = OUT_VIZ_DIR / f"armfix_{ARMFIX_VERSION}_summary.json"
    summary = {
        "cache_dir": str(OUT_CACHE_DIR),
        "viz_dir": str(OUT_VIZ_DIR),
        "count": len(records),
        "old_marked_pct_mean": float(np.mean([r["old_marked_pct"] for r in records])) if records else 0.0,
        "new_marked_pct_mean": float(np.mean([r["new_marked_pct"] for r in records])) if records else 0.0,
        "added_marked_pct_mean": float(np.mean([r["added_marked_pct"] for r in records])) if records else 0.0,
        "records": records,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vids", default="12,16,56,96", help="'all' or comma-separated video ids")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    OUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for vid in parse_vids(args.vids):
        if not (BASE_CACHE_DIR / f"{vid}.npy").exists():
            print(f"[skip] vid {vid}: missing base cache", flush=True)
            continue
        w, meta = build_armfix_weightmap(vid)
        np.save(OUT_CACHE_DIR / f"{vid}.npy", w.astype(np.float32))
        if not args.no_viz:
            render_viz(vid, w, OUT_VIZ_DIR)
        records.append(meta)
        print(
            f"[armfix] {vid}: old={meta['old_marked_pct']:.3f} "
            f"new={meta['new_marked_pct']:.3f} added={meta['added_marked_pct']:.3f} "
            f"b={meta['b_region']}",
            flush=True,
        )
    write_index_and_summary(records)
    print(f"[done] cache={OUT_CACHE_DIR}", flush=True)
    print(f"[done] viz={OUT_VIZ_DIR}", flush=True)


if __name__ == "__main__":
    main()
