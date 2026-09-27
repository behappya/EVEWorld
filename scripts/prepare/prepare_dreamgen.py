#!/usr/bin/env python3
"""Build the DreamGen splits and per-clip metadata under ``data/``.

The script reads the DreamGen tree pointed to by ``--data-root`` (the ``DREAMGEN_DATA_ROOT``
environment variable) and writes:

``<output-dir>/splits/dreamgen/train.txt``
    The 92 GR1 fine-tuning clips behind the GigaWorld-0 arms of paper Table 1: a ``#``
    comment header, one blank line and then one 5-digit clip stem per line.
``<output-dir>/splits/dreamgen/val.txt``
    The 12 clips held out of the 92-clip post-training set for checkpoint selection and
    protocol development. They are excluded from the reported DreamGenBench numbers, which
    come from ``test.txt``.
``<output-dir>/splits/dreamgen/test.txt``
    The 126 DreamGenBench evaluation request ids, one per line, in manifest order.
``<output-dir>/metadata/dreamgenbench/target_queries.json``
    The prompt, target, source and destination of every request, plus the eligibility flag
    the shared eligible set ``U_63`` is built from.
``<output-dir>/metadata/dreamgenbench/eligible_ids.json``
    ``U_63``, the 63 requests eligible for Model Laziness Rate and for both
    instruction-following judges (coverage 50.00% of the 126 prompts).
``<output-dir>/metadata/dreamgenbench/<vid>.json``
    Prompt, latent geometry, the per-timestamp expected instance count and, for the 104
    post-training clips, the IGR annotation of ``docs/data_preparation.md``, whose keys sit
    at the top level of the record so the file is the annotation itself.

The clips below ``gr1/`` are read in stem order: the first 92 become the training split and the
next 12 the held-out validation split. That ordering is the manifest order of the release and is
not fixed by the paper.

A request is eligible when its parsed instruction names both a source and a destination. Parsing
is the one of ``eveworld.data.parsers.instruction_parser``, except that the two static-scene
templates ``Wait until the {T} stops moving in the scene.`` and ``Hold position and do not move
the {T}.`` are read with their object slot, which is the object the benchmark annotations name.
Targets keep the spelling of the prompt; source and destination surfaces are joined with
underscores (``top shelf`` -> ``top_shelf``).

Upstream layout under ``--data-root``::

    gr1/<vid>.mp4                 93 frames at 480x768, 16 FPS
    gr1/<vid>.json                sidecar: "prompt" plus an optional object track
    dreamgenbench/prompts.jsonl   126 records {"id": ..., "prompt": ..., "category": ...}

``prompts.json`` is accepted as well, and a record may carry ``prompt``/``text``/``instruction``/
``caption`` and ``id``/``vid``/``request_id``. A request without an id takes the next free id
after the GR1 block, so the order of the file is the manifest order the ids follow. The upstream
categories are ``gr1_env`` (29), ``gr1_object`` (50) and ``gr1_behavior`` (47).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

NUM_FRAMES = 93
FPS = 16
WIDTH, HEIGHT = 768, 480
N_LAT, H_LAT, W_LAT = 24, 30, 48
SAMPLE_COUNT = 24
TRAIN_CLIP_COUNT = 92
VAL_CLIP_COUNT = 12
BENCHMARK_PROMPT_COUNT = 126
ELIGIBLE_COUNT = 63
POST_TRAINING_COUNT = TRAIN_CLIP_COUNT + VAL_CLIP_COUNT
FIRST_BENCHMARK_ID = 105
TRAINING_SPLITS = ("train", "val")
TARGET_KEYS = ("target_name", "target", "mover", "object")
CONTAINER_KEYS = ("b_name", "destination", "receiver", "container")
ARRIVAL_KEYS = ("t_arrival", "arrival", "interaction_start")
FRAME_KEYS = ("target_box", "box", "bbox", "xyxy")
PROMPT_KEYS = ("prompt", "text", "instruction", "caption", "task")
ID_KEYS = ("id", "vid", "video_id", "request_id", "clip_id")
CATEGORY_KEYS = ("category", "group", "subset", "split_name")
# The library parser reads these two static-scene templates as a target of "… stops moving"
# and "position"; the DreamGenBench annotations name the object of the template instead.
STATIC_SCENE_TARGETS = (
    re.compile(r"wait until the (?P<target>.+?) stops moving in the scene\b", re.IGNORECASE),
    re.compile(r"hold position and do not move the (?P<target>.+?)\.?\s*$", re.IGNORECASE),
)
TRAIN_HEADER = (
    "# DreamGen / GR1 fine-tuning split - training clips.",
    "#",
    "# 92 clips of 93 frames at 480 x 768, 16 FPS. This is the post-training set of",
    "# the GigaWorld-0 arms (Table 1, Table 6); the DreamGenBench evaluation",
    "# split is test.txt. One 5-digit clip stem per line: the stems are manifest ids,",
    "# not media, and scripts/prepare/prepare_dreamgen.py regenerates the file from a",
    "# data root.",
)
VAL_HEADER = (
    "# DreamGen / GR1 fine-tuning split - held-out validation clips.",
    "#",
    "# 12 clips held out of the 92-clip post-training set. They are used for",
    "# checkpoint selection and protocol development and are excluded from the",
    "# reported DreamGenBench numbers, which come from test.txt. One 5-digit clip",
    "# stem per line; scripts/prepare/prepare_dreamgen.py regenerates the file.",
)
TEST_HEADER = (
    "# DreamGenBench evaluation split.",
    "#",
    "# The 126 evaluation prompts of the benchmark, one 5-digit clip stem per line.",
    "# 63 of the 126 clips form the shared eligible set U_63 used by Model Laziness",
    "# Rate and by both instruction-following judges (coverage 50.00%); the eligible",
    "# ids are listed in data/metadata/dreamgenbench/eligible_ids.json. The stems are",
    "# manifest ids, not media; scripts/prepare/prepare_dreamgen.py regenerates the",
    "# file from a data root.",
)


@dataclass
class Clip:
    """One clip of the DreamGen tree: its id, instruction, split and optional sidecar."""

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
        description="Build the DreamGen splits and per-clip metadata under data/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=layout_help(__doc__),
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        default=os.environ.get("DREAMGEN_DATA_ROOT"),
        help="DreamGen tree holding gr1/ and dreamgenbench/ (default: $DREAMGEN_DATA_ROOT)",
    )
    data.add_argument(
        "--output-dir",
        default=None,
        help="directory the splits and metadata are written to (default: <repo>/data)",
    )
    data.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of clips and requests per split, for a smoke run",
    )
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


def phrase(text: Any) -> str | None:
    """A parsed surface with single spaces, as the parser returns it; ``None`` when empty."""
    return " ".join(str(text).split()) if text else None


def snake_case(text: Any) -> str | None:
    """A scene surface as an id: spaces become underscores; ``None`` when empty."""
    return "_".join(str(text).split()) if text else None


def parse_prompt(prompt: str) -> tuple[str | None, str | None, str | None]:
    """Target and snake-case source/destination of one instruction.

    The two static-scene templates are read with their object slot before the library parser
    sees the prompt, because their object is what the DreamGenBench annotations name.
    """
    for pattern in STATIC_SCENE_TARGETS:
        match = pattern.search(prompt)
        if match is not None:
            return phrase(match.group("target")), None, None
    from eveworld.data.parsers.instruction_parser import parse_instruction

    parsed = parse_instruction(prompt)
    return phrase(parsed.target), snake_case(parsed.source), snake_case(parsed.destination)


def load_prompt_table(root: Path) -> list[dict[str, Any]]:
    """Read the benchmark prompt table in manifest order, naming the requests without an id."""
    path = find_file(root, ("prompts.jsonl", "prompts.json"))
    if path is None:
        raise FileNotFoundError(f"no prompts.jsonl or prompts.json below {root}")
    payload = load_json(path)
    if isinstance(payload, dict):
        records = [dict(record, id=key) for key, record in payload.items() if isinstance(record, dict)]
    else:
        records = list(payload)
    table: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path} holds a record that is not an object: {record!r}")
        prompt = first_key(record, PROMPT_KEYS)
        if prompt is None:
            raise ValueError(f"{path} holds a record without a prompt: {record!r}")
        record_id = first_key(record, ID_KEYS)
        if record_id is None:
            record_id = f"{FIRST_BENCHMARK_ID + index:05d}"
        table.append(
            {
                "id": str(record_id),
                "prompt": str(prompt),
                "category": str(first_key(record, CATEGORY_KEYS) or ""),
            }
        )
    ids = [record["id"] for record in table]
    duplicated = sorted({vid for vid in ids if ids.count(vid) > 1})
    if duplicated:
        raise ValueError(f"{path} holds duplicated request ids: {duplicated[:5]}")
    return table


def build_target_queries(
    table: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The parsed surfaces of the benchmark table and the ids of the eligible set it holds."""
    queries: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for record in table:
        target, source, destination = parse_prompt(record["prompt"])
        queries[record["id"]] = {
            "prompt": record["prompt"],
            "target": target,
            "source": source,
            "destination": destination,
            "eligible": bool(source and destination),
        }
        if source and destination:
            eligible.append(record["id"])
    return queries, eligible


def load_training_clips(root: Path) -> list[Clip]:
    """Discover the GR1 clips and their instructions below ``gr1/``, in stem order."""
    gr1_root = root / "gr1"
    if not gr1_root.is_dir():
        gr1_root = root
    clips: list[Clip] = []
    for video in sorted(gr1_root.rglob("*.mp4")):
        sidecar_path = video.with_suffix(".json")
        sidecar = load_json(sidecar_path) if sidecar_path.is_file() else {}
        if not isinstance(sidecar, dict):
            sidecar = {}
        prompt = first_key(sidecar, PROMPT_KEYS)
        if prompt is None:
            text = video.with_suffix(".txt")
            prompt = text.read_text(encoding="utf-8").strip() if text.is_file() else None
        if prompt is None:
            raise ValueError(
                f"{video} has no instruction: expected {sidecar_path.name} with a prompt key "
                f"or {video.with_suffix('.txt').name}"
            )
        clips.append(
            Clip(
                vid=video.stem,
                prompt=str(prompt),
                split="train",
                category=str(first_key(sidecar, CATEGORY_KEYS) or ""),
                video=video,
                sidecar=sidecar,
            )
        )
    if not clips:
        raise FileNotFoundError(f"no clip found below {gr1_root}")
    return clips


def split_clips(clips: Sequence[Clip], limit: int | None) -> tuple[list[Clip], list[Clip]]:
    """The 92 training clips and the 12 held-out validation clips, each capped by ``--limit``."""
    train = list(clips[:TRAIN_CLIP_COUNT])
    val = list(clips[TRAIN_CLIP_COUNT:POST_TRAINING_COUNT])
    if limit is not None:
        train = train[:limit]
        val = val[:limit]
    return (
        [replace(clip, split="train") for clip in train],
        [replace(clip, split="val") for clip in val],
    )


def benchmark_clips(table: Sequence[dict[str, Any]]) -> list[Clip]:
    """One evaluation clip per request of the prompt table, in manifest order."""
    return [
        Clip(vid=record["id"], prompt=record["prompt"], split="test", category=record["category"])
        for record in table
    ]


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
    """Per-frame object boxes of a sidecar, in clip order; empty when it carries no track."""
    frames = sidecar.get("frames") or sidecar.get("per_frame") or []
    if not isinstance(frames, list):
        return []
    return [frame for frame in frames if isinstance(frame, dict)]


def build_per_lat_frame(sidecar: dict[str, Any], timestamps: Sequence[int]) -> list[dict[str, Any]]:
    """Sample the sidecar track onto the ``n_lat`` latent timestamps of the IGR annotation."""
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
            }
        )
    return entries


def build_igr_annotation(clip: Clip, timestamps: Sequence[int]) -> dict[str, Any]:
    """Assemble the IGR annotation of ``docs/data_preparation.md`` for one post-training clip."""
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
    # One object per sampled timestamp is the reading the recipe uses for the GR1 corpus;
    # the count itself is not fixed by the paper, and the list is aligned with `timestamps`.
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
    if clip.split in TRAINING_SPLITS:
        record.update(build_igr_annotation(clip, timestamps))
    return record


def write_split(path: Path, ids: Sequence[str], header: Sequence[str], dry_run: bool) -> None:
    """Write the ``#`` header, one blank line and then one clip id per line."""
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{line}\n" for line in header) + "\n" + "".join(f"{vid}\n" for vid in ids)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any, dry_run: bool) -> None:
    """Write one JSON document through the atomic writer of the library."""
    if dry_run:
        return
    from eveworld.utils.io import write_json as _write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(payload, path)


def write_metadata(directory: Path, records: Sequence[dict[str, Any]], dry_run: bool) -> None:
    """Write one ``<vid>.json`` per record through the atomic writer of the library."""
    if dry_run:
        return
    from eveworld.utils.io import write_json as _write_json

    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        _write_json(record, directory / f"{record['vid']}.json")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: read the DreamGen tree, then write the splits and the per-clip metadata."""
    args = parse_args(argv)
    if not args.data_root:
        print("error: --data-root is required (or export DREAMGEN_DATA_ROOT)", file=sys.stderr)
        return 2
    data_root = Path(args.data_root).expanduser()
    if not data_root.is_dir():
        print(f"error: data root {data_root} is not a directory", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else repo_root() / "data"
    try:
        return prepare(args, data_root, output_dir)
    except ImportError as error:
        print(f"error: the eveworld package is not importable: {error}", file=sys.stderr)
        return 2


def prepare(args: argparse.Namespace, data_root: Path, output_dir: Path) -> int:
    """Read the tree, check the recipe counts and write the three splits and the metadata."""
    clips = load_training_clips(data_root)
    train, val = split_clips(clips, args.limit)
    table = load_prompt_table(data_root)
    if args.limit is not None:
        table = table[: args.limit]
    queries, eligible = build_target_queries(table)
    benchmark = benchmark_clips(table)
    if args.limit is None:
        if len(clips) != POST_TRAINING_COUNT:
            print(
                f"error: found {len(clips)} clips below the data root, the recipe uses "
                f"{POST_TRAINING_COUNT} ({TRAIN_CLIP_COUNT} train and {VAL_CLIP_COUNT} validation)",
                file=sys.stderr,
            )
            return 1
        if len(table) != BENCHMARK_PROMPT_COUNT:
            print(
                f"error: found {len(table)} benchmark prompts, DreamGenBench has "
                f"{BENCHMARK_PROMPT_COUNT}",
                file=sys.stderr,
            )
            return 1
        if len(eligible) != ELIGIBLE_COUNT:
            print(
                f"error: the eligible set holds {len(eligible)} ids, U_63 holds {ELIGIBLE_COUNT}",
                file=sys.stderr,
            )
            return 1

    coverage = round(100.0 * len(eligible) / len(queries), 2) if queries else 0.0
    print(f"data root:    {data_root}")
    print(f"output dir:   {output_dir}")
    print(f"train clips:  {len(train)} of {len(clips)} GR1 clips ({TRAIN_CLIP_COUNT} in the recipe)")
    print(f"val clips:    {len(val)} ({VAL_CLIP_COUNT} held out for checkpoint selection)")
    print(f"benchmark:    {len(benchmark)} prompts ({BENCHMARK_PROMPT_COUNT} in DreamGenBench)")
    print(f"eligible:     {len(eligible)} of {len(table)} (coverage {coverage:.2f}%)")
    print(f"timestamps:   {len(sample_timestamps())} per clip on a {N_LAT}x{H_LAT}x{W_LAT} latent grid")

    splits = output_dir / "splits" / "dreamgen"
    metadata = output_dir / "metadata" / "dreamgenbench"
    write_split(splits / "train.txt", [clip.vid for clip in train], TRAIN_HEADER, args.dry_run)
    write_split(splits / "val.txt", [clip.vid for clip in val], VAL_HEADER, args.dry_run)
    write_split(splits / "test.txt", [clip.vid for clip in benchmark], TEST_HEADER, args.dry_run)
    write_json(metadata / "target_queries.json", queries, args.dry_run)
    write_json(
        metadata / "eligible_ids.json",
        {
            "eligible": eligible,
            "num_eligible": len(eligible),
            "num_total": len(table),
            "coverage": coverage,
        },
        args.dry_run,
    )
    records = [build_record(clip) for clip in train + val + benchmark]
    write_metadata(metadata, records, args.dry_run)

    if args.dry_run:
        print("dry run: no file written")
        return 0
    print(f"wrote {splits}/train.txt, val.txt, test.txt and {len(records)} records under {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
