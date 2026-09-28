"""DreamGen/GR1 and AgiBot clip datasets of the GigaWorld-0 integration.

Both corpora are 93-frame, 16 FPS manipulation clips whose metadata records carry the prompt and
the IGR annotation of :mod:`eveworld.methods.igr` (the target's cell on the 16 px grid at every
latent timestamp). A dataset item is a clean clip plus the annotation the IGR transform and the
TIA term need, and every item has the following keys:

===================  ============================================  =================================
key                  shape / type                                  meaning
===================  ============================================  =================================
``video``            ``(C, T, H, W)`` float32 in ``[-1, 1]``       clean clip
``caption``          ``str``                                       the clip prompt
``instruction``      ``str``                                       instruction the IGR parser reads
``weight_map``       ``(T_lat, H_lat, W_lat)`` float32, unit mean  IGR map of the cached corruption
``tia_cells``        ``(T_lat, 2)`` int64                          target cell ``(row, col)``; ``-1``
                                                                   when the target was not detected
``sample_id``        ``str``                                       clip id, e.g. ``"00001"``
``fallback``         ``bool``                                      no usable IGR annotation
===================  ============================================  =================================

The weight map is flat (all ones) when no cache is available: the disturbance of ``alg:igr_construction``
is applied on the fly by :class:`eveworld.integrations.gigaworld.hooks.IGRCollator`, which rewrites
``video``, ``weight_map``, ``fallback`` and ``events`` of each item between the dataset and
:func:`collate_fn`.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from eveworld.data.transforms.latent import latent_grid_size, resize_weight_map
from eveworld.data.transforms.video import frames_to_tensor, load_video
from eveworld.integrations.gigaworld.model import CELL_SIZE, latent_frames
from eveworld.methods.igr.weight_map import BACKGROUND, DISTURBED, normalize_unit_mean
from eveworld.utils.io import read_json, repo_root
from eveworld.utils.logging import get_logger

__all__ = ["AgiBotDataset", "DreamGenDataset", "collate_fn"]

logger = get_logger(__name__)

DEFAULT_SEED = 42
EVEWORLD_ROOT_VARIABLE = "EVEWORLD_ROOT"
VIDEO_KEYS = ("video", "video_path", "path", "clip")
CAPTION_KEYS = ("prompt", "caption", "text", "instruction")
INSTRUCTION_KEYS = ("instruction", "prompt", "text", "caption")
VALUE_TOLERANCE = 1e-3


def _resolve(value: str | os.PathLike[str]) -> Path:
    """Resolve a configured path: expand ``~`` and anchor relative paths at the repository root."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else repo_root() / path


def _fix_length(frames: np.ndarray, num_frames: int) -> np.ndarray:
    """Force a clip to ``num_frames`` frames by truncation or repetition of the last frame."""
    count = int(frames.shape[0])
    if count == num_frames:
        return frames
    if count > num_frames:
        return frames[:num_frames]
    if count == 0:
        raise ValueError("cannot pad an empty clip")
    pad = num_frames - count
    return np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)


def _takes_rng(transform: Any) -> bool:
    """Whether ``transform`` accepts a second positional argument, i.e. a random generator."""
    if transform is None:
        return False
    try:
        parameters = inspect.signature(transform).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2


def _annotation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """The IGR annotation of a metadata record, tolerating a nested ``igr`` mapping."""
    nested = record.get("igr")
    return nested if isinstance(nested, Mapping) else record


def _text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    """First non-empty text value of ``keys`` in a record."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _discrete_levels(array: np.ndarray, levels: Sequence[float], tolerance: float = VALUE_TOLERANCE) -> bool:
    """Whether every value of ``array`` is one of ``levels`` within ``tolerance``."""
    values = np.unique(np.asarray(array, dtype=np.float64))
    return all(any(abs(float(value) - float(level)) <= tolerance for level in levels) for value in values)


class _ClipDataset(Dataset):
    """Shared machinery of the DreamGen/GR1 and AgiBot clip datasets.

    Args:
        root: Data root holding the videos, resolved against the repository root.
        split: Split file listing the clip ids, one per line; ``None`` uses the class default.
        metadata_dir: Directory of ``<vid>.json`` records; ``None`` uses the class default.
        num_frames: Frames of a clip; longer clips are truncated, shorter ones repeat their last
            frame.
        resolution: ``(width, height)`` the clips are resized to.
        fps: Frame rate the clips are resampled to.
        with_weight_map: Whether the cached IGR weight maps are read at all.
        weight_map_dir: Directory of ``<vid>.npy`` weight maps; ``None`` means flat unit maps.
        video_dir: Explicit directory of the clips, tried before the environment roots.
        seed: Base seed of the per-item generator handed to ``transform``.
        transform: Optional callable applied to the item; it receives the item, plus a
            ``numpy.random.Generator`` when it takes a second argument, and may either mutate the
            item or return a mapping merged into it.

    Raises:
        FileNotFoundError: If the split file or the metadata directory does not exist.
        ValueError: If the split file holds no clip id.
    """

    corpus = "clip"
    default_split = ""
    default_metadata = ""
    video_roots: tuple[str, ...] = ()
    video_subdirs: tuple[str, ...] = ("", "video", "raw_data")

    def __init__(
        self,
        root: str | os.PathLike[str] = ".",
        split: str | os.PathLike[str] | None = None,
        metadata_dir: str | os.PathLike[str] | None = None,
        *,
        num_frames: int = 93,
        resolution: tuple[int, int] = (768, 480),
        fps: float = 16.0,
        with_weight_map: bool = True,
        weight_map_dir: str | os.PathLike[str] | None = None,
        video_dir: str | os.PathLike[str] | None = None,
        seed: int = DEFAULT_SEED,
        transform: Any = None,
    ) -> None:
        self.root = _resolve(root)
        self.split_path = _resolve(split if split is not None else self.default_split)
        if not self.split_path.is_file():
            raise FileNotFoundError(f"{self.corpus} split file not found: {self.split_path}")
        metadata = _resolve(metadata_dir if metadata_dir is not None else self.default_metadata)
        if not metadata.is_dir():
            raise FileNotFoundError(f"{self.corpus} metadata directory not found: {metadata}")
        self.metadata_dir = metadata
        self.ids = self._read_split(self.split_path)
        self.num_frames = int(num_frames)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.fps = float(fps)
        self.with_weight_map = bool(with_weight_map)
        self.weight_map_dir = _resolve(weight_map_dir) if weight_map_dir is not None else None
        self.video_dir = _resolve(video_dir) if video_dir is not None else None
        self.seed = int(seed)
        self.transform = transform
        self._transform_rng = _takes_rng(transform)
        self._warned: set[str] = set()
        self._warned_map_miss = False

    @staticmethod
    def _read_split(path: Path) -> list[str]:
        """Clip ids of a split file, skipping blank lines and ``#`` headers."""
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not ids:
            raise ValueError(f"split file {path} holds no clip id")
        return ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one clip with its annotation.

        Args:
            index: Item index into the split file.

        Returns:
            dict: ``video``, ``caption``, ``instruction``, ``weight_map``, ``tia_cells``,
            ``sample_id`` and ``fallback``; see the module docstring for the shapes.
        """
        vid = self.ids[index]
        record = self.annotation(vid)
        annotation = _annotation(record)
        frames = load_video(self.video_path(vid, record), size=self.resolution, out_fps=self.fps)
        sample: dict[str, Any] = {
            "video": frames_to_tensor(_fix_length(frames, self.num_frames)),
            "caption": _text(record, CAPTION_KEYS),
            "instruction": _text(record, INSTRUCTION_KEYS),
            "weight_map": self.weight_map(vid, annotation),
            "tia_cells": self.tia_cells(annotation),
            "sample_id": vid,
            "fallback": self.fallback(annotation),
        }
        return self._apply_transform(sample, index)

    def annotation(self, vid: str) -> Mapping[str, Any]:
        """Metadata record of one clip; an empty mapping when it is missing or unreadable."""
        path = self.metadata_dir / f"{vid}.json"
        if not path.is_file():
            self._warn(f"no metadata record for {vid}: expected {path}")
            return {}
        try:
            record = read_json(path)
        except (OSError, ValueError) as error:
            self._warn(f"unreadable metadata record {path}: {error}")
            return {}
        if not isinstance(record, Mapping):
            self._warn(f"metadata record {path} is not a mapping")
            return {}
        return record

    def grid_of(self, annotation: Mapping[str, Any]) -> tuple[int, int, int]:
        """``(n_lat, H_lat, W_lat)`` cell grid the weight map and the TIA cells live on."""
        declared = annotation.get("latent_grid")
        if isinstance(declared, Sequence) and not isinstance(declared, str) and len(declared) == 3:
            return (int(declared[0]), int(declared[1]), int(declared[2]))
        height, width = self.resolution[1], self.resolution[0]
        rows, columns = latent_grid_size((height, width), CELL_SIZE)
        return latent_frames(self.num_frames), rows, columns

    def weight_map(self, vid: str, annotation: Mapping[str, Any]) -> np.ndarray:
        """Unit-mean IGR weight map ``(T_lat, H_lat, W_lat)`` of one clip, flat without a cache."""
        grid = self.grid_of(annotation)
        if self.with_weight_map and self.weight_map_dir is not None:
            cached = self._read_cached_map(self.weight_map_dir / f"{vid}.npy", grid)
            if cached is not None:
                return cached
        return np.ones(grid, dtype=np.float32)

    def _read_cached_map(self, path: Path, grid: tuple[int, int, int]) -> np.ndarray | None:
        """Load and validate one cached weight map, or ``None`` when it cannot be used."""
        if not path.is_file():
            if not self._warned_map_miss:
                self._warned_map_miss = True
                logger.warning("no cached weight map at %s; using flat unit maps", path)
            return None
        try:
            array = np.load(path)
        except (OSError, ValueError) as error:
            self._warn(f"unreadable weight map {path}: {error}")
            return None
        array = np.asarray(array)
        if array.ndim != 3:
            self._warn(f"weight map {path} has rank {array.ndim}, expected 3")
            return None
        if array.shape[0] != grid[0]:
            self._warn(f"weight map {path} covers {array.shape[0]} frames, expected {grid[0]}")
            return None
        if not np.isfinite(array).all():
            self._warn(f"weight map {path} holds non-finite values")
            return None
        if not _discrete_levels(array, (BACKGROUND, DISTURBED)):
            self._warn(f"weight map {path} is not a {BACKGROUND}/{DISTURBED} map")
            return None
        if tuple(array.shape[1:]) != tuple(grid[1:]):
            array = resize_weight_map(array, size=(grid[1], grid[2]))
        return normalize_unit_mean(np.asarray(array, dtype=np.float32))

    def tia_cells(self, annotation: Mapping[str, Any]) -> np.ndarray:
        """Target cells ``(T_lat, 2)`` int64 of the tracked instance, ``-1`` where undetected."""
        grid = self.grid_of(annotation)
        cells = np.full((grid[0], 2), -1, dtype=np.int64)
        if not annotation.get("gate_enabled", False):
            return cells
        entries = annotation.get("per_lat_frame")
        if not isinstance(entries, Sequence) or isinstance(entries, str):
            return cells
        for index, entry in enumerate(entries[: grid[0]]):
            if not isinstance(entry, Mapping):
                continue
            if "detected" in entry and not entry.get("detected"):
                continue
            cell = entry.get("target_cell")
            if isinstance(cell, Sequence) and not isinstance(cell, str) and len(cell) == 2:
                cells[index, 0] = int(cell[0])
                cells[index, 1] = int(cell[1])
        return cells

    def fallback(self, annotation: Mapping[str, Any]) -> bool:
        """Whether the clip has no usable IGR annotation and has to run as a clean sample."""
        if not annotation:
            return True
        if not annotation.get("gate_enabled", False):
            return True
        return int(annotation.get("n_detected_frames", 0) or 0) <= 0

    def video_path(self, vid: str, record: Mapping[str, Any]) -> Path:
        """Path of the clip file, from the record, the configured roots or the environment."""
        for key in VIDEO_KEYS:
            value = record.get(key)
            if isinstance(value, str) and value:
                candidate = _resolve(value)
                if candidate.is_file():
                    return candidate
        tried: list[Path] = []
        for root in self._data_roots():
            for subdir in self.video_subdirs:
                candidate = root / subdir / f"{vid}.mp4" if subdir else root / f"{vid}.mp4"
                if candidate.is_file():
                    return candidate
                tried.append(candidate)
        shown = ", ".join(str(path) for path in tried[:8])
        raise FileNotFoundError(f"{self.corpus} clip {vid} not found; tried {len(tried)} paths: {shown}")

    def _data_roots(self) -> list[Path]:
        """Directories searched for the clips, in order and without duplicates."""
        roots: list[Path] = []
        if self.video_dir is not None:
            roots.append(self.video_dir)
        for name in self.video_roots:
            value = os.environ.get(name)
            if value:
                roots.append(_resolve(value))
        root_variable = os.environ.get(EVEWORLD_ROOT_VARIABLE)
        if root_variable:
            roots.append(_resolve(root_variable))
        roots.append(self.root)
        unique: list[Path] = []
        for root in roots:
            if root not in unique:
                unique.append(root)
        return unique

    def _apply_transform(self, sample: dict[str, Any], index: int) -> dict[str, Any]:
        """Apply the optional transform; a returned mapping is merged into the item."""
        if self.transform is None:
            return sample
        if self._transform_rng:
            result = self.transform(sample, np.random.default_rng(self.seed + index))
        else:
            result = self.transform(sample)
        if isinstance(result, Mapping):
            merged = dict(sample)
            merged.update(result)
            return merged
        return sample

    def _warn(self, message: str) -> None:
        """Log ``message`` once per dataset instance."""
        if message not in self._warned:
            self._warned.add(message)
            logger.warning(message)


class DreamGenDataset(_ClipDataset):
    """DreamGen/GR1 clips: 93 frames at 480 x 768, 16 FPS, target tracked on a ``24 x 30 x 48`` grid."""

    corpus = "DreamGen"
    default_split = "data/splits/dreamgen/train.txt"
    default_metadata = "data/metadata/dreamgenbench"
    video_roots = ("EVE_VIDEO_ROOT", "GR1_DATA_ROOT")
    video_subdirs = ("gr1", "", "video", "raw_data")

    def __init__(
        self,
        root: str | os.PathLike[str] = ".",
        split: str | os.PathLike[str] | None = None,
        metadata_dir: str | os.PathLike[str] | None = None,
        *,
        num_frames: int = 93,
        resolution: tuple[int, int] = (768, 480),
        fps: float = 16.0,
        with_weight_map: bool = True,
        weight_map_dir: str | os.PathLike[str] | None = None,
        video_dir: str | os.PathLike[str] | None = None,
        seed: int = DEFAULT_SEED,
        transform: Any = None,
    ) -> None:
        super().__init__(
            root,
            split,
            metadata_dir,
            num_frames=num_frames,
            resolution=resolution,
            fps=fps,
            with_weight_map=with_weight_map,
            weight_map_dir=weight_map_dir,
            video_dir=video_dir,
            seed=seed,
            transform=transform,
        )


class AgiBotDataset(_ClipDataset):
    """AgiBot clips: 93 frames at 640 x 480, 16 FPS, target tracked on a ``24 x 30 x 40`` grid."""

    corpus = "AgiBot"
    default_split = "data/splits/agibot/train.txt"
    default_metadata = "data/metadata/agibot"
    video_roots = ("EVE_VIDEO_ROOT", "GAGI_ROOT")
    video_subdirs = ("agibot_ewm_clean", "", "video", "raw_data")

    def __init__(
        self,
        root: str | os.PathLike[str] = ".",
        split: str | os.PathLike[str] | None = None,
        metadata_dir: str | os.PathLike[str] | None = None,
        *,
        num_frames: int = 93,
        resolution: tuple[int, int] = (640, 480),
        fps: float = 16.0,
        with_weight_map: bool = True,
        weight_map_dir: str | os.PathLike[str] | None = None,
        video_dir: str | os.PathLike[str] | None = None,
        seed: int = DEFAULT_SEED,
        transform: Any = None,
    ) -> None:
        super().__init__(
            root,
            split,
            metadata_dir,
            num_frames=num_frames,
            resolution=resolution,
            fps=fps,
            with_weight_map=with_weight_map,
            weight_map_dir=weight_map_dir,
            video_dir=video_dir,
            seed=seed,
            transform=transform,
        )


def collate_fn(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate dataset items into a batch, keeping the union of their keys.

    Tensors and numeric arrays are stacked, booleans become a ``bool`` tensor, numeric scalars
    become a tensor of the batch size, lists and tuples are kept as lists of per-item lists, and
    anything else (strings, mappings) stays a list. Keys whose values are all ``None`` are dropped.

    Args:
        batch: Items, each a mapping of the dataset keys.

    Returns:
        dict: The batched mapping; empty for an empty batch.
    """
    if not batch:
        return {}
    keys: list[str] = []
    for item in batch:
        for key in item:
            if key not in keys:
                keys.append(key)
    collated: dict[str, Any] = {}
    for key in keys:
        values = [item.get(key) for item in batch]
        if all(value is None for value in values):
            continue
        first = next(value for value in values if value is not None)
        if isinstance(first, torch.Tensor):
            collated[key] = torch.stack([torch.as_tensor(value) for value in values])
        elif isinstance(first, np.ndarray):
            collated[key] = torch.from_numpy(np.stack([np.asarray(value) for value in values]))
        elif isinstance(first, (bool, np.bool_)):
            collated[key] = torch.as_tensor([bool(value) for value in values], dtype=torch.bool)
        elif isinstance(first, (int, float, np.integer, np.floating)):
            collated[key] = torch.as_tensor([value for value in values])
        elif isinstance(first, (list, tuple)):
            collated[key] = [value for value in values]
        else:
            collated[key] = list(values)
    return collated
