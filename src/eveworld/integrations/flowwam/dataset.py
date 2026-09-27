"""RoboTwin manipulation-transfer clips of the FlowWAM integration.

A RoboTwin episode is a 121-frame, 24 FPS, 640x480 clean demonstration annotated on the
``16`` px cell grid the IGR weight map lives on. The FlowWAM protocol trains on the first
29 frames of an episode, so an item is the clean video of one episode truncated (or
last-frame padded) to ``num_frames``. Every item has the following keys:

===================  ============================================  =================================
key                  shape / type                                  meaning
===================  ============================================  =================================
``video``            ``(C, T, H, W)`` float32 in ``[-1, 1]``       clean clip
``caption``          ``str``                                       the episode prompt
``instruction``      ``str``                                       instruction the IGR parser reads
``weight_map``       ``(T_lat, H_lat, W_lat)`` float32, unit mean  IGR map of the cached corruption
``tia_cells``        ``(T_lat, 2)`` int64                          target cell ``(row, col)``; ``-1``
                                                                   where the target was not detected
``sample_id``        ``str``                                       episode id, e.g.
                                                                   ``task_place_phone_episode_45``
``fallback``         ``bool``                                      no usable IGR annotation
===================  ============================================  =================================

The split file of the protocol is the *held-out* manifest, not the training list:
``data/splits/robotwin/heldout.txt`` lists the 250 evaluation episodes, episodes ``45-49`` of
the 50 RoboTwin tasks, while the FlowWAM arm trains on the same 50 tasks with episodes ``0-44``
(2,250 episodes). The dataset derives that training pool from the manifest - every task of the
manifest, every episode below the lowest held-out episode of that task - so a released run
trains on exactly the complement of the file it evaluates with, and
:attr:`RoboTwinDataset.held_out` keeps the manifest rows it was derived from. ``dev.txt`` is
the episode-45 slice of the same range and is disjoint from both.

Geometry is taken from :class:`~eveworld.integrations.flowwam.model.FlowWAMConfig` and never
from the record: the ``latent_grid`` of a prepared record describes the whole 121-frame
episode (``31x30x40``) while an item covers ``num_frames`` (``8x30x40`` for the 29-frame
protocol clip), so reading it would misalign the weight map and the TIA cells with the clip.
The weight map is flat (all ones) when no cache is configured or the cache cannot be used;
the disturbance of ``alg:igr_construction`` is applied on the fly by
:class:`eveworld.integrations.gigaworld.hooks.IGRCollator`, which rewrites ``video``,
``weight_map``, ``fallback`` and ``events`` of each item between the dataset and
:func:`collate_fn`.

The released recipe keeps the gate off on RoboTwin - its annotations carry no
``gate_enabled`` flag at all - so ``tia_cells`` are read from ``per_lat_frame`` directly,
and a record with no detectable cell marks the item ``fallback``. The flow stream of the
recipe conditions on the flow-codec images of the robot-only render, which this dataset
cannot produce (RAFT and the reversible codec are third-party GPU modules): items carry no
``flow_video`` key, :class:`~eveworld.integrations.flowwam.model.FlowWAMModel` falls back to
the blank flow stream, and a caller that owns the codecs can inject the images through
``transform``.
"""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from eveworld.data.transforms.latent import resize_weight_map
from eveworld.data.transforms.video import frames_to_tensor, load_video
from eveworld.integrations.flowwam.model import CELL_SIZE, _as_config, latent_frames
from eveworld.methods.igr.weight_map import BACKGROUND, DISTURBED, normalize_unit_mean
from eveworld.utils.io import read_json, repo_root
from eveworld.utils.logging import get_logger

__all__ = ["RoboTwinDataset", "collate_fn"]

logger = get_logger(__name__)

DEFAULT_SEED = 42
EVEWORLD_ROOT_VARIABLE = "EVEWORLD_ROOT"
VIDEO_KEYS = ("video", "video_path", "path", "clip")
CAPTION_KEYS = ("prompt", "caption", "text", "instruction")
INSTRUCTION_KEYS = ("instruction", "prompt", "text", "caption")
VALUE_TOLERANCE = 1e-3
EPISODE_PATTERN = re.compile(r"^task_(?P<task>.+)_episode_(?P<number>\d+)$")


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
    return all(
        any(abs(float(value) - float(level)) <= tolerance for level in levels) for value in values
    )


class RoboTwinDataset(Dataset):
    """RoboTwin clips: 640x480, 24 FPS clean demonstrations annotated on the ``16`` px grid.

    Args:
        config: Run configuration or :class:`FlowWAMConfig` filling the geometry arguments
            left at ``None``; ``None`` uses the RoboTwin preset.
        root: Data root holding the ``<task>/<collection>/video/`` trees, resolved against
            the repository root.
        split: Held-out manifest listing the evaluation episodes, one id per line; the
            training pool is derived from it - the same tasks, with the episodes below the
            lowest held-out episode of each task. ``None`` uses
            ``data/splits/robotwin/heldout.txt``.
        metadata_dir: Directory of ``<vid>.json`` records; ``None`` uses
            ``data/metadata/robotwin``. A missing directory is warned about rather than
            raised on, because the annotations are optional and every item falls back to
            the flat weight map without them.
        num_frames: Frames of a clip; ``None`` takes ``model.num_frames`` of ``config``.
        resolution: ``(width, height)`` the clips are resized to; ``None`` takes the
            configured resolution.
        fps: Frame rate the clips are resampled to; ``None`` takes the configured rate.
        with_weight_map: Whether the cached IGR weight maps are read at all.
        weight_map_dir: Directory of ``<vid>.npy`` weight maps; ``None`` means flat unit maps.
        video_dir: Explicit directory of the clip trees, tried before the environment roots.
        seed: Base seed of the per-item generator handed to ``transform``.
        transform: Optional callable applied to the item; it receives the item, plus a
            ``numpy.random.Generator`` when it takes a second argument, and may either mutate
            the item or return a mapping merged into it.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If the split file holds no episode id, or derives an empty training pool.
    """

    corpus = "RoboTwin"
    default_split = "data/splits/robotwin/heldout.txt"
    default_metadata = "data/metadata/robotwin"
    video_roots = ("ROBOTWIN_DATA_ROOT",)
    video_subdirs = ("", "video", "raw_data")

    def __init__(
        self,
        config: Any = None,
        root: str | os.PathLike[str] = ".",
        split: str | os.PathLike[str] | None = None,
        metadata_dir: str | os.PathLike[str] | None = None,
        *,
        num_frames: int | None = None,
        resolution: tuple[int, int] | None = None,
        fps: float | None = None,
        with_weight_map: bool = True,
        weight_map_dir: str | os.PathLike[str] | None = None,
        video_dir: str | os.PathLike[str] | None = None,
        seed: int = DEFAULT_SEED,
        transform: Any = None,
    ) -> None:
        overrides: dict[str, Any] = {}
        if num_frames is not None:
            overrides["num_frames"] = int(num_frames)
        if resolution is not None:
            overrides["resolution"] = tuple(int(value) for value in resolution)
        if fps is not None:
            overrides["fps"] = float(fps)
        geometry = _as_config(config, **overrides)
        self.root = _resolve(root)
        self.split_path = _resolve(split if split is not None else self.default_split)
        if not self.split_path.is_file():
            raise FileNotFoundError(f"{self.corpus} split file not found: {self.split_path}")
        metadata = _resolve(metadata_dir if metadata_dir is not None else self.default_metadata)
        if not metadata.is_dir():
            logger.warning(
                "%s metadata directory not found: %s; the items fall back to the flat weight "
                "map and get no TIA annotation",
                self.corpus,
                metadata,
            )
        self.metadata_dir = metadata
        self.held_out = self._read_split(self.split_path)
        self.ids = self._training_ids(self.held_out)
        self.num_frames = int(geometry.num_frames)
        self.resolution = (int(geometry.width), int(geometry.height))
        self.fps = float(geometry.fps)
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
        """Episode ids of a split file, skipping blank lines and ``#`` headers."""
        ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not ids:
            raise ValueError(f"split file {path} holds no episode id")
        return ids

    @staticmethod
    def _training_ids(held_out: Sequence[str]) -> list[str]:
        """The training pool of a held-out manifest, as episode ids in manifest order.

        Every task of the manifest contributes its episodes below the lowest held-out episode
        of that task, in ascending order: the 250 held-out episodes of RoboTwin (episodes
        ``45-49`` of 50 tasks) derive the 2,250 training episodes ``0-44`` of the same tasks.
        Rows that are not an episode id are skipped, so a hand-edited manifest does not fail
        the run over a stray line.

        Args:
            held_out: Rows of the held-out manifest.

        Returns:
            The training episode ids.

        Raises:
            ValueError: If the manifest derives no training episode at all.
        """
        lowest: dict[str, int] = {}
        for vid in held_out:
            match = EPISODE_PATTERN.match(vid)
            if match is None:
                logger.warning("split row %r is not an episode id; skipped", vid)
                continue
            task = match.group("task")
            number = int(match.group("number"))
            if task not in lowest or number < lowest[task]:
                lowest[task] = number
        ids = [
            f"task_{task}_episode_{number}"
            for task, limit in lowest.items()
            for number in range(limit)
        ]
        if not ids:
            raise ValueError(
                "the split file derives no training episode: every task it lists holds out its "
                "episode 0, so there is nothing below the held-out range to train on"
            )
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
        frames = load_video(self.video_path(vid, record), size=self.resolution, out_fps=self.fps)
        cells = self.tia_cells(record)
        sample: dict[str, Any] = {
            "video": frames_to_tensor(_fix_length(frames, self.num_frames)),
            "caption": _text(record, CAPTION_KEYS),
            "instruction": _text(record, INSTRUCTION_KEYS),
            "weight_map": self.weight_map(vid),
            "tia_cells": cells,
            "sample_id": vid,
            "fallback": self.fallback(record, cells),
        }
        return self._apply_transform(sample, index)

    @property
    def grid(self) -> tuple[int, int, int]:
        """``(T_lat, H_lat, W_lat)`` cell grid the weight map and the TIA cells live on.

        The grid is derived from ``num_frames`` and ``resolution``, not from the
        ``latent_grid`` of a record, which covers the whole 121-frame episode.
        """
        return (
            latent_frames(self.num_frames),
            self.resolution[1] // CELL_SIZE,
            self.resolution[0] // CELL_SIZE,
        )

    def annotation(self, vid: str) -> Mapping[str, Any]:
        """Metadata record of one episode; an empty mapping when it is missing or unreadable."""
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

    def weight_map(self, vid: str) -> np.ndarray:
        """Unit-mean IGR weight map ``(T_lat, H_lat, W_lat)`` of one clip, flat without a cache.

        A cached map may cover the whole episode (``31`` latent frames); the first ``T_lat``
        frames are kept, a shorter map is padded with the flat level.
        """
        if self.with_weight_map and self.weight_map_dir is not None:
            cached = self._read_cached_map(self.weight_map_dir / f"{vid}.npy", self.grid)
            if cached is not None:
                return cached
        return np.ones(self.grid, dtype=np.float32)

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
        if array.shape[0] > grid[0]:
            array = array[: grid[0]]
        elif array.shape[0] < grid[0]:
            pad = np.ones((grid[0] - array.shape[0],) + array.shape[1:], dtype=array.dtype)
            array = np.concatenate([array, pad], axis=0)
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
        """Target cells ``(T_lat, 2)`` int64 of the tracked instance, ``-1`` where undetected.

        The cells live on the ``(H_lat, W_lat)`` cell grid of the weight map; the TIA token
        grid is half of it. RoboTwin records carry no gate flag, so ``per_lat_frame`` is read
        directly and a row whose ``detected`` is ``False`` stays ``-1``.
        """
        cells = np.full((self.grid[0], 2), -1, dtype=np.int64)
        entries = annotation.get("per_lat_frame")
        if not isinstance(entries, Sequence) or isinstance(entries, str):
            return cells
        for index, entry in enumerate(entries[: self.grid[0]]):
            if not isinstance(entry, Mapping):
                continue
            if "detected" in entry and not entry.get("detected"):
                continue
            cell = entry.get("target_cell")
            if not isinstance(cell, Sequence) or isinstance(cell, str) or len(cell) != 2:
                continue
            if cell[0] is None or cell[1] is None:
                continue
            try:
                cells[index, 0] = int(cell[0])
                cells[index, 1] = int(cell[1])
            except (TypeError, ValueError):
                continue
        return cells

    def fallback(self, annotation: Mapping[str, Any], cells: np.ndarray | None = None) -> bool:
        """Whether the clip has no usable IGR annotation and has to run as a clean sample.

        A record is usable when at least one latent frame carries both cell coordinates;
        this mirrors the ``(cells >= 0).all(axis=1).any()`` test of the IGR transform.
        """
        if not annotation:
            return True
        if cells is None:
            cells = self.tia_cells(annotation)
        return not bool((cells >= 0).all(axis=1).any())

    def video_path(self, vid: str, record: Mapping[str, Any]) -> Path:
        """Path of the clean clip, from the record, the configured roots or the data tree.

        The RoboTwin tree stores the clip as ``<task>/<collection>/video/episode<N>.mp4``
        with a collection directory that is not fixed, so every collection under the task
        is tried. Relative record paths resolve against the repository root.
        """
        for key in VIDEO_KEYS:
            value = record.get(key)
            if isinstance(value, str) and value:
                candidate = _resolve(value)
                if candidate.is_file():
                    return candidate
        tried: list[Path] = []
        parsed = EPISODE_PATTERN.match(vid)
        for root in self._data_roots():
            for subdir in self.video_subdirs:
                candidate = root / subdir / f"{vid}.mp4" if subdir else root / f"{vid}.mp4"
                if candidate.is_file():
                    return candidate
                tried.append(candidate)
            if parsed is not None:
                for candidate in self._collection_candidates(
                    root, parsed.group("task"), parsed.group("number")
                ):
                    if candidate.is_file():
                        return candidate
                    tried.append(candidate)
        shown = ", ".join(str(path) for path in tried[:8])
        raise FileNotFoundError(f"{self.corpus} clip {vid} not found; tried {len(tried)} paths: {shown}")

    @staticmethod
    def _collection_candidates(root: Path, task: str, number: str) -> list[Path]:
        """Clip candidates of one task under a root, one per collection directory."""
        task_dir = root / task
        candidates: list[Path] = []
        if task_dir.is_dir():
            candidates.extend(sorted(task_dir.glob(f"*/video/episode{number}.mp4")))
            candidates.append(task_dir / "video" / f"episode{number}.mp4")
        return candidates

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
