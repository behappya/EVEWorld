#!/usr/bin/env python3
"""Build the AgiBot split and per-clip metadata under ``data/``.

The script reads the AgiBot clip tree pointed to by ``--data-root`` (the
``AGIBOT_DATA_ROOT`` environment variable) and writes three things:

``<output-dir>/splits/agibot/train.txt``
    The 777 training clips of the cross-distribution arm behind the EWMBench row
    of paper Table 4: a ``#`` comment header, one blank line and then one
    ``<task>_<episode>_<segment>`` id per line.
``<output-dir>/metadata/agibot/<vid>.json``
    Prompt, latent geometry, the per-timestamp expected instance count and the
    IGR annotation of ``docs/data_preparation.md``, whose keys sit at the top
    level of the record so the file is the annotation itself. AgiBot annotations
    add a ``state_cells`` key to every ``per_lat_frame`` entry.
``<output-dir>/metadata/agibot/_leakage_guard.json``
    The audit trail of the 21-episode EWMBench leakage guard.

The guard drops every clip whose episode id belongs to the EWMBench test set
before the split is written, and the written split is checked once more against
the holdout list, so a clip the EWMBench row scores can never reach training.
The 21 episode ids are the ones the EWMBench release ships; ``--ewmbench-root``
(the ``EWMBENCH_DATA_ROOT`` environment variable) points at that checkout when
it is available, and its ``GT/<task>/<episode>/`` layout is read for them.
Without the checkout the built-in list of the release is used, and the audit
records which source was read.

Upstream layout under ``--data-root``::

    <task>_<episode>_<segment>.mp4        93 frames at 480x640, 16 FPS
    <task>_<episode>_<segment>.txt        the sub-action instruction
    <task>_<episode>_<segment>.json       optional detection annotation sidecar

The clips may also sit one level down (``<root>/agibot_ewm_clean/``), as the
extraction of the upstream project lays them out. The ``*_trans.mp4`` preview
duplicates of the packed dataset are skipped. An annotation sidecar is accepted
next to the clip or under ``<root>/t4g_anno/`` and ``<root>/annotations/``; it
may carry ``state_cells`` next to the object ``box`` of each frame. A clip whose
annotation carries no object track is written with ``gate_enabled`` false and
null cells, the state the IGR corruption builder skips over.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

NUM_FRAMES = 93
FPS = 16
WIDTH, HEIGHT = 640, 480
N_LAT, H_LAT, W_LAT = 24, 30, 40
SAMPLE_COUNT = 24
TRAIN_CLIP_COUNT = 777
PREVIEW_SUFFIX = "_trans"
ANNOTATION_DIRS = ("t4g_anno", "annotations", "anno")
PROMPT_KEYS = ("prompt", "text", "instruction", "caption", "action_text", "task")
TARGET_KEYS = ("target_name", "target", "mover", "object")
CONTAINER_KEYS = ("b_name", "destination", "receiver", "container", "dest")
ARRIVAL_KEYS = ("t_arrival", "arrival", "interaction_start")
FRAME_KEYS = ("target_box", "box", "bbox", "xyxy")
STATE_KEYS = ("state_cells", "state_box", "state", "part_box")
SPLIT_HEADER = (
    "# AgiBot training split.",
    "#",
    "# The 777 clips of the cross-distribution arm behind the EWMBench row of",
    "# paper Table 4: one <task>_<episode>_<segment> id per line, 93 frames at",
    "# 480 x 640, 16 FPS. The clips are the AgiBot release minus the 21 test",
    "# episodes of EWMBench, so no episode the EWMBench row scores is ever",
    "# trained on; the audit of the guard is metadata/agibot/_leakage_guard.json",
    "# and scripts/prepare/prepare_agibot.py regenerates the file from a data root.",
)
# The 21 episodes of the EWMBench test set, the guard of eveworld/agibot/w9_extract_agibot.py.
HOLDOUT_EPISODES = frozenset(
    {
        "649524", "649559", "650191", "651464", "664600", "681186", "766602",
        "773025", "773496", "743247", "743964", "744776", "798615", "798749",
        "807480", "787136", "789120", "791059", "808158", "824748", "834014",
    }
)


@dataclass
class Clip:
    """One AgiBot clip: its id, instruction, split and optional annotation."""

    vid: str
    prompt: str
    split: str
    category: str = ""
    video: Path | None = None
    sidecar: dict[str, Any] = field(default_factory=dict)


def layout_help(document: str, marker: str = "Upstream layout under ``--data-root``::") -> str:
    """The indented layout block of the module docstring, used as the ``--help`` epilog."""
    body = document.split(marker, 1)[1].lstrip("\n")
    return body.split("\n\n", 1)[0].rstrip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the AgiBot split and per-clip metadata under data/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=layout_help(__doc__),
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        default=os.environ.get("AGIBOT_DATA_ROOT"),
        help="AgiBot clip tree of the training set (default: $AGIBOT_DATA_ROOT)",
    )
    data.add_argument(
        "--ewmbench-root",
        default=os.environ.get("EWMBENCH_DATA_ROOT"),
        help="EWMBench checkout holding the GT/ test episodes (default: $EWMBENCH_DATA_ROOT)",
    )
    data.add_argument(
        "--output-dir",
        default=None,
        help="directory the splits and metadata are written to (default: <repo>/data)",
    )
    data.add_argument("--limit", type=int, default=None, help="cap the number of clips, for a smoke run")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--dry-run", action="store_true", help="print the plan without writing anything")
    return parser.parse_args(argv)


def repo_root() -> Path:
    """Repository root: ``eveworld.utils.io.repo_root`` when importable, else the pyproject walk."""
    try:
        from eveworld.utils.io import repo_root as _library_root
    except ImportError:
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").is_file():
                return parent
        return Path.cwd()
    return Path(_library_root())


def first_key(record: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first present key of ``keys`` in ``record``, or ``None``."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def load_json(path: Path) -> Any:
    """Parse a JSON or JSONL file, choosing the reader by suffix."""
    with open(path, "r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def find_file(root: Path, names: Sequence[str], itself: bool = False) -> Path | None:
    """First existing path among ``root/<name>`` and, unless ``itself``, ``root/*/<name>``."""
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
        if not itself and root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / name).is_file():
                    return child / name
    return None


def holdout_episodes(ewmbench_root: Path | None) -> tuple[frozenset[str], str]:
    """The EWMBench test episode ids and the source they were read from.

    The built-in list of the release is always part of the result, so the guard
    stays in force when no EWMBench checkout is available to read.
    """
    if ewmbench_root is None or not ewmbench_root.is_dir():
        return HOLDOUT_EPISODES, "built-in EWMBench release list"
    listed = find_file(ewmbench_root, ("holdout.txt", "test_episodes.txt"), itself=True)
    if listed is not None:
        ids = {line.strip() for line in listed.read_text(encoding="utf-8").splitlines() if line.strip()}
        return HOLDOUT_EPISODES | ids, str(listed)
    gt_root = ewmbench_root / "GT"
    if gt_root.is_dir():
        ids = {
            episode.name
            for task in sorted(gt_root.iterdir())
            if task.is_dir()
            for episode in sorted(task.iterdir())
            if episode.is_dir()
        }
        if ids:
            return HOLDOUT_EPISODES | ids, str(gt_root)
    return HOLDOUT_EPISODES, "built-in EWMBench release list"


def clip_episode(vid: str, holdout: frozenset[str]) -> str | None:
    """The holdout episode id a clip belongs to, or ``None`` when the clip is clear."""
    for token in vid.split("_"):
        if token in holdout:
            return token
    return None


def load_annotation(root: Path, video: Path) -> dict[str, Any]:
    """The detection annotation of one clip: sidecar first, then the annotation directories."""
    sidecar = video.with_suffix(".json")
    candidates = [sidecar] + [root / name / f"{video.stem}.json" for name in ANNOTATION_DIRS]
    for candidate in candidates:
        if candidate.is_file():
            payload = load_json(candidate)
            if not isinstance(payload, dict):
                raise ValueError(f"{candidate} does not hold a JSON object")
            return payload
    return {}


def clip_root(root: Path) -> Path:
    """The directory holding the clips: ``--data-root`` itself or its single clip subdirectory."""
    if any(path.suffix == ".mp4" for path in root.iterdir() if path.is_file()):
        return root
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(path.suffix == ".mp4" for path in child.iterdir() if path.is_file()):
            return child
    return root


def load_clips(root: Path, holdout: frozenset[str]) -> tuple[list[Clip], list[str], int]:
    """Discover the AgiBot clips, split into the training set and the guarded-out holdout."""
    directory = clip_root(root)
    clips: list[Clip] = []
    guarded: list[str] = []
    previews = 0
    for video in sorted(directory.rglob("*.mp4")):
        stem = video.stem
        if stem.endswith(PREVIEW_SUFFIX):
            previews += 1
            continue
        episode = clip_episode(stem, holdout)
        if episode is not None:
            guarded.append(stem)
            continue
        sidecar = load_annotation(root, video)
        prompt = first_key(sidecar, PROMPT_KEYS)
        if prompt is None:
            text = video.with_suffix(".txt")
            prompt = text.read_text(encoding="utf-8").strip() if text.is_file() else None
        if prompt is None:
            raise ValueError(
                f"{video} has no instruction: expected {video.with_suffix('.txt').name} "
                f"or an annotation carrying a prompt key"
            )
        clips.append(Clip(vid=stem, prompt=str(prompt), split="train", video=video, sidecar=sidecar))
    if not clips:
        raise FileNotFoundError(f"no clip found below {directory}")
    return clips, sorted(guarded), previews


def sample_timestamps(num_frames: int = NUM_FRAMES, count: int = SAMPLE_COUNT) -> list[int]:
    """Round sampling of ``count`` timestamps over ``num_frames``, the MLR protocol axis."""
    if count == 1:
        return [0]
    step = (num_frames - 1) / (count - 1)
    return [int(round(index * step)) for index in range(count)]


def to_cell(box: Sequence[float], shape: tuple[int, int] = (HEIGHT, WIDTH)) -> list[int]:
    """Centre of a pixel ``xyxy`` box as a ``(row, col)`` cell on the latent grid."""
    height, width = shape
    row = (box[1] + box[3]) / 2.0 * H_LAT / height
    col = (box[0] + box[2]) / 2.0 * W_LAT / width
    return [min(H_LAT - 1, max(0, int(row))), min(W_LAT - 1, max(0, int(col)))]


def to_cells(value: Any, shape: tuple[int, int] = (HEIGHT, WIDTH)) -> list[list[int]]:
    """Normalise an annotation cell or box list into latent ``(row, col)`` cells."""
    if not isinstance(value, list):
        return []
    cells: list[list[int]] = []
    for entry in value:
        if isinstance(entry, dict):
            box = first_key(entry, FRAME_KEYS)
        else:
            box = entry
        if isinstance(box, list) and len(box) == 4:
            cells.append(to_cell(box, shape))
        elif isinstance(box, list) and len(box) == 2:
            cells.append([int(box[0]), int(box[1])])
    return cells


def annotation_frames(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-frame annotation entries of a sidecar, in clip order; empty without a track."""
    frames = sidecar.get("frames") or sidecar.get("per_frame") or []
    if not isinstance(frames, list):
        return []
    return [frame for frame in frames if isinstance(frame, dict)]


def build_per_lat_frame(sidecar: dict[str, Any], timestamps: Sequence[int]) -> list[dict[str, Any]]:
    """Sample the annotation onto the ``n_lat`` latent timestamps of the IGR annotation."""
    frames = annotation_frames(sidecar)
    entries: list[dict[str, Any]] = []
    for timestamp in timestamps:
        frame = frames[timestamp] if timestamp < len(frames) else {}
        box = first_key(frame, FRAME_KEYS)
        entries.append(
            {
                "target_cell": to_cell(box) if box else None,
                "b_cells": to_cells(frame.get("b_cells")),
                "detected": bool(box),
                "state_cells": to_cells(first_key(frame, STATE_KEYS)),
            }
        )
    return entries


def build_igr_annotation(clip: Clip, timestamps: Sequence[int]) -> dict[str, Any]:
    """Assemble the IGR annotation of ``docs/data_preparation.md`` for one clip."""
    sidecar = clip.sidecar
    per_lat_frame = build_per_lat_frame(sidecar, timestamps)
    target_cell_0 = per_lat_frame[0]["target_cell"] if per_lat_frame else None
    detected = [entry for entry in per_lat_frame if entry["detected"]]
    target_name = first_key(sidecar, TARGET_KEYS)
    gate_enabled = bool(detected) and target_name is not None
    if gate_enabled:
        gate_reason = "target tracked over the interaction window"
    elif target_name is None:
        gate_reason = "no target name in the upstream annotation"
    else:
        gate_reason = "no target track in the upstream annotation"
    arrival = first_key(sidecar, ARRIVAL_KEYS)
    return {
        "vid": clip.vid,
        "prompt": clip.prompt,
        "target_name": str(target_name) if target_name is not None else None,
        "b_name": str(first_key(sidecar, CONTAINER_KEYS) or "") or None,
        "gate_enabled": gate_enabled,
        "gate_reason": gate_reason,
        "t_arrival": int(arrival) if arrival is not None else None,
        "n_lat": N_LAT,
        "H_lat": H_LAT,
        "W_lat": W_LAT,
        "target_cell_0": target_cell_0,
        "inventory_cells": [target_cell_0] if target_cell_0 else [],
        "distractor_cells": to_cells(sidecar.get("distractor_cells")),
        "a_cells": to_cells(sidecar.get("a_cells")),
        "n_inventory": 1 if target_cell_0 else 0,
        "n_detected_frames": len(detected),
        "per_lat_frame": per_lat_frame,
    }


def build_record(clip: Clip) -> dict[str, Any]:
    """Metadata record for one clip: prompt, geometry, expected counts and the IGR annotation."""
    timestamps = sample_timestamps()
    # One object per sampled timestamp is the reading the recipe uses for the
    # single-arm AgiBot corpus; the count itself is not fixed by the paper.
    counts = [1] * len(timestamps)
    record: dict[str, Any] = {
        "vid": clip.vid,
        "split": clip.split,
        "prompt": clip.prompt,
        "category": clip.category,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "latent_grid": [N_LAT, H_LAT, W_LAT],
        "timestamps": timestamps,
        "expected_counts": counts,
    }
    record.update(build_igr_annotation(clip, timestamps))
    return record


def write_split(path: Path, ids: Sequence[str], dry_run: bool) -> None:
    """Write the ``#`` header, one blank line and then one clip id per line."""
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{line}\n" for line in SPLIT_HEADER) + "\n" + "".join(f"{vid}\n" for vid in ids)
    path.write_text(text, encoding="utf-8")


def write_metadata(directory: Path, records: Sequence[dict[str, Any]], dry_run: bool) -> None:
    """Write one ``<vid>.json`` per record through the atomic writer of the library."""
    if dry_run:
        return
    from eveworld.utils.io import write_json

    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_json(record, directory / f"{record['vid']}.json")


def write_guard_audit(
    path: Path,
    holdout: frozenset[str],
    source: str,
    guarded: Sequence[str],
    dry_run: bool,
) -> None:
    """Record which holdout episodes the guard used and which clips it removed."""
    payload = {
        "protocol": "ewmbench_21_episode_leakage_guard",
        "holdout_episode_count": len(holdout),
        "holdout_episodes": sorted(holdout),
        "holdout_source": source,
        "excluded_clip_count": len(guarded),
        "excluded_clips": sorted(guarded),
        "written_split": "train.txt",
        "in_split_holdout_clips": [],
    }
    if dry_run:
        return
    from eveworld.utils.io import write_json

    write_json(payload, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: read the AgiBot tree, then write the split and the per-clip metadata."""
    args = parse_args(argv)
    if not args.data_root:
        print("error: --data-root is required (or export AGIBOT_DATA_ROOT)", file=sys.stderr)
        return 2
    data_root = Path(args.data_root).expanduser()
    if not data_root.is_dir():
        print(f"error: data root {data_root} is not a directory", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else repo_root() / "data"
    ewmbench_root = Path(args.ewmbench_root).expanduser() if args.ewmbench_root else None
    holdout, guard_source = holdout_episodes(ewmbench_root)

    clips, guarded, previews = load_clips(data_root, holdout)
    if args.limit is None and len(clips) != TRAIN_CLIP_COUNT:
        print(
            f"error: found {len(clips)} clips outside the EWMBench holdout, the recipe uses "
            f"{TRAIN_CLIP_COUNT}",
            file=sys.stderr,
        )
        return 1
    if args.limit is not None:
        clips = clips[: args.limit]
    leaked = [clip.vid for clip in clips if clip_episode(clip.vid, holdout) is not None]
    if leaked:
        shown = ", ".join(leaked[:5])
        print(
            f"error: {len(leaked)} clips of the split belong to the EWMBench holdout: {shown}",
            file=sys.stderr,
        )
        return 1

    annotated = sum(1 for clip in clips if clip.sidecar)
    print(f"data root:    {data_root}")
    print(f"output dir:   {output_dir}")
    print(
        f"clips:        {len(clips)} ({TRAIN_CLIP_COUNT} in the recipe, "
        f"{previews} preview duplicates skipped)"
    )
    print(f"guard:        {len(holdout)} EWMBench test episodes from {guard_source}")
    print(f"guard drops:  {len(guarded)} clips of this tree")
    print(f"annotations:  {annotated} clips carry an upstream annotation sidecar")
    print(f"timestamps:   {len(sample_timestamps())} per clip on a {N_LAT}x{H_LAT}x{W_LAT} latent grid")

    splits = output_dir / "splits" / "agibot"
    metadata = output_dir / "metadata" / "agibot"
    write_split(splits / "train.txt", [clip.vid for clip in clips], args.dry_run)
    records = [build_record(clip) for clip in clips]
    write_metadata(metadata, records, args.dry_run)
    write_guard_audit(metadata / "_leakage_guard.json", holdout, guard_source, guarded, args.dry_run)

    if args.dry_run:
        print("dry run: no file written")
        return 0
    print(f"wrote {splits}/train.txt, {len(records)} records and the guard audit under {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
