"""Offline layer probe: sweep the backbone blocks and pick the one TIA is attached to.

The block is chosen by an offline experiment rather than by training. For every candidate block,
tokens of frame ``t`` are matched against the tokens of frame ``t + 1`` by bare nearest neighbour
on the block's normalised features, and the end-point error against the benchmark's ground-truth
correspondences is averaged over all tracked points. The paper's sweep over blocks 8-26 selected
block 23 for GigaWorld-0/AgiBot and block 12 for FlowWAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from eveworld.methods.tia.matcher import ProbeResult, retrieve, select_layer, token_grid_shape

__all__ = ["ProbeResult", "layer_epe", "probe_layers", "run_probe", "write_probe_report"]


def layer_epe(
    features: torch.Tensor | np.ndarray,
    query_coords: torch.Tensor | np.ndarray,
    target_coords: torch.Tensor | np.ndarray,
    grid: Sequence[int] | None = None,
) -> float:
    """Mean end-point error of nearest-neighbour tracking at one block.

    Frame ``t`` is queried at ``query_coords[t]`` and matched against **every** token of frame
    ``t + 1``; the retrieved cell is compared against ``target_coords[t + 1]``, i.e. the two arrays
    are the per-frame trajectory of the same point, the convention of the benchmark annotations.
    The error is the Euclidean distance in grid cells, averaged over all ``(frame, token)`` pairs
    whose coordinates are finite.

    Args:
        features: Float tensor ``(T, H, W, C)``, ``(T, N, C)`` or ``(N, C)`` of a single block. A
            single frame has no pair to score and yields ``nan``.
        query_coords: Query cells ``(T, 2)`` (or ``(T, K, 2)``) taken from frame ``t``, in
            ``(row, col)`` grid cells.
        target_coords: Ground-truth cells of the same points, same shape as ``query_coords``; entry
            ``t + 1`` is the ground truth of the query at entry ``t``. Non-finite entries mark a
            point without ground truth and are skipped.
        grid: Optional ``(height, width)`` of the token grid; validated against the token count and
            inferred by :func:`eveworld.methods.tia.matcher.token_grid_shape` otherwise.

    Returns:
        float: Mean end-point error in grid cells, or ``nan`` when no pair has ground truth.

    Raises:
        ValueError: If the features are not one of the accepted shapes, if the coordinates are
        mis-shaped or do not match the frame count, or if ``grid`` does not match the token count.
    """
    features = torch.as_tensor(features).float()
    if features.ndim == 4:
        frames, height, width = features.shape[0], int(features.shape[1]), int(features.shape[2])
        grid = token_grid_shape(height * width, grid)
        features = features.reshape(frames, height * width, features.shape[-1])
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3:
        raise ValueError(f"features must be (T, H, W, C), (T, N, C) or (N, C), got {tuple(features.shape)}")
    num_frames = int(features.shape[0])
    height, width = token_grid_shape(int(features.shape[1]), grid)

    queries = np.asarray(query_coords, dtype=np.float64)
    truth = np.asarray(target_coords, dtype=np.float64)
    if queries.ndim not in (2, 3) or queries.shape[0] != num_frames or queries.shape[-1] != 2:
        raise ValueError(f"query_coords must be (T, 2) or (T, K, 2) with T={num_frames}, got {queries.shape}")
    if truth.shape != queries.shape:
        raise ValueError(f"target_coords {truth.shape} must match query_coords {queries.shape}")
    if num_frames < 2:
        return float("nan")

    rows = torch.arange(height).repeat_interleave(width)
    cols = torch.arange(width).repeat(height)
    coords = torch.stack((rows, cols), dim=-1)
    with torch.no_grad():
        predicted = retrieve(features, coords, query_coords, grid=(height, width))
    predicted = predicted.detach().to(torch.float64).cpu().numpy()

    valid = np.isfinite(queries[:-1]).all(axis=-1) & np.isfinite(truth[1:]).all(axis=-1)
    distance = np.linalg.norm(predicted - truth[1:], axis=-1)
    distance = np.where(valid, distance, np.nan)
    if not np.isfinite(distance).any():
        return float("nan")
    return float(np.nanmean(distance))


def probe_layers(
    features_by_layer: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
    query_coords: torch.Tensor | np.ndarray,
    target_coords: torch.Tensor | np.ndarray,
    candidates: Sequence[int] | None = None,
    grid: Sequence[int] | None = None,
) -> list[ProbeResult]:
    """Score several blocks of a single clip and return them sorted by block index.

    Args:
        features_by_layer: Mapping from block index to that block's features, or a sequence of
            features whose entry ``i`` belongs to block ``candidates[i]``.
        query_coords: Query cells, as in :func:`layer_epe`.
        target_coords: Ground-truth cells, as in :func:`layer_epe`.
        candidates: Blocks to score. Defaults to the keys of the mapping, or to the positions of the
            sequence when it is a sequence.
        grid: Optional token grid ``(height, width)``, forwarded to :func:`layer_epe`.

    Returns:
        list[ProbeResult]: One entry per candidate, sorted by ascending block index; the EPE is in
        grid cells and is ``nan`` for blocks that could not be scored.

    Raises:
        ValueError: If a requested block has no features, if a sequence does not align with
        ``candidates``, or if the coordinates are invalid.
    """
    if isinstance(features_by_layer, Mapping):
        available = {int(layer): feats for layer, feats in features_by_layer.items()}
        if candidates is None:
            entries = sorted(available.items())
        else:
            layers = [int(layer) for layer in candidates]
            missing = [layer for layer in layers if layer not in available]
            if missing:
                raise ValueError(f"no features for layer(s) {missing}")
            entries = [(layer, available[layer]) for layer in layers]
    else:
        blocks = list(features_by_layer)
        if candidates is None:
            entries = list(enumerate(blocks))
        else:
            layers = [int(layer) for layer in candidates]
            if len(layers) != len(blocks):
                raise ValueError(f"got {len(blocks)} feature entries for {len(layers)} candidates")
            entries = list(zip(layers, blocks))
    results = [ProbeResult(layer=int(layer), epe=float(layer_epe(feats, query_coords, target_coords, grid=grid))) for layer, feats in entries]
    results.sort(key=lambda result: result.layer)
    return results


def run_probe(
    model: Any,
    videos: Iterable[Mapping[str, Any]],
    out_path: str | Path | None = None,
    candidates: Sequence[int] = range(8, 27),
    *,
    capture: Callable[[Any, Any, Sequence[int], str], Mapping[int, torch.Tensor]] | None = None,
    device: str = "cpu",
) -> list[ProbeResult]:
    """Sweep the candidate blocks over a list of clips and report the mean EPE per block.

    The probe is deliberately agnostic about the backbone: ``capture`` is the only part that knows
    how to run the model and pull per-block features out of it, so the same sweep works for any
    architecture that exposes its blocks.

    Args:
        model: Backbone, forwarded to ``capture``.
        videos: Clip records, each a mapping with the keys ``video`` (whatever ``capture`` expects),
            ``query_coords`` and ``target_coords`` as in :func:`layer_epe`, plus an optional
            ``grid``.
        out_path: When given, the report is written there with :func:`write_probe_report`.
        candidates: Blocks to sweep; the paper swept 8-26.
        capture: ``capture(model, video, candidates, device)`` returning a mapping from block index
            to that block's features for one clip. Required - a backbone's hook API cannot be
            guessed, so the caller plugs its own capture in.
        device: Device handed to ``capture``.

    Returns:
        list[ProbeResult]: Mean EPE over the clips per block, sorted by ascending block index.

    Raises:
        ValueError: If ``capture`` is missing, if a clip record lacks a required key, if there are
        no clips or no candidates.
        TypeError: If ``capture`` is not callable or a record is not a mapping.
    """
    if capture is None:
        raise ValueError(
            "run_probe needs a capture callable: pass capture(model, video, candidates, device) that runs "
            "the backbone on one clip and returns a mapping {block_index: features} for the candidate "
            "blocks, for example by registering forward hooks on the selected blocks"
        )
    if not callable(capture):
        raise TypeError(f"capture must be callable, got {type(capture).__name__}")
    records = list(videos)
    if not records:
        raise ValueError("run_probe needs at least one video")
    layers = [int(layer) for layer in candidates]
    if not layers:
        raise ValueError("candidates must not be empty")

    required = ("video", "query_coords", "target_coords")
    total = {layer: 0.0 for layer in layers}
    count = {layer: 0 for layer in layers}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"video records must be mappings, got {type(record).__name__}")
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"video record is missing key(s) {missing}")
        with torch.no_grad():
            features_by_layer = capture(model, record["video"], layers, device)
        if not isinstance(features_by_layer, Mapping):
            raise TypeError("capture must return a mapping from block index to features")
        available = {int(layer): feats for layer, feats in features_by_layer.items()}
        for layer in layers:
            feats = available.get(layer)
            if feats is None:
                continue
            epe = layer_epe(feats, record["query_coords"], record["target_coords"], grid=record.get("grid"))
            if np.isfinite(epe):
                total[layer] += float(epe)
                count[layer] += 1

    results = [ProbeResult(layer=layer, epe=total[layer] / count[layer] if count[layer] else float("nan")) for layer in layers]
    results.sort(key=lambda result: result.layer)
    if out_path is not None:
        write_probe_report(results, out_path)
    return results


def write_probe_report(results: Sequence[ProbeResult], path: str | Path) -> Path:
    """Write the EPE table as JSON, with the selected block alongside it.

    Args:
        results: Probe results, as returned by :func:`probe_layers` or :func:`run_probe`.
        path: Destination file; missing parent directories are created.

    Returns:
        Path: The written file. ``selected`` is ``null`` when no block has a finite EPE.
    """
    results = list(results)
    try:
        selected = select_layer([result.layer for result in results], [result.epe for result in results]).to_dict()
    except ValueError:
        selected = None
    payload = {"selected": selected, "blocks": [result.to_dict() for result in results]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
