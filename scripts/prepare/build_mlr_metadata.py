#!/usr/bin/env python3
"""Build the WorldArena MLR metadata and the WorldArena splits under ``data/``.

The script reads the frozen WorldArena 1.0 manifest that ``--data-root``
(``WORLDARENA_DATA_ROOT``) points to and writes:

``<output-dir>/metadata/worldarena/parsed_targets.json``
    The manipulated object and the manipulator of all 1,000 requests, keyed by the request
    id of the manifest position.
``<output-dir>/metadata/worldarena/eligible_ids.json``
    The requests that are eligible for Model Laziness Rate, with the counts and the
    coverage of the published protocol.
``<output-dir>/splits/worldarena/train.txt``
    The 16 requests that freeze the detector thresholds and the occlusion rule before the
    final evaluation.
``<output-dir>/splits/worldarena/eval.txt``
    The full manifest, in manifest order.

WorldArena 1.0 is an evaluation-only suite, so ``train.txt`` is a protocol-calibration
slice rather than a training corpus: the requests it lists are evaluated with the same
frozen protocol as the rest, and nothing trains on them.

A request is eligible for MLR when its instruction names the object it manipulates and a
manipulator exists to be lazy. The frozen manifest renders every instruction from one of a
small set of frames; the frame that matches a prompt fixes the object and, where the text
determines it, the manipulator. Prompts outside the object frames are robot navigation,
collective tidying or scene inspection: they carry no object and are not eligible. The
coverage denominator is the number of requests that name a manipulator, which is what the
published coverage is measured against.

Upstream layout under ``--data-root``::

    <manifest>.json  a list of 1,000 entries in manifest order, each entry holding the
                     instruction of the request plus the image and the ground-truth paths
                     of its rendered scene. Recognised names: worldarena_summary.json,
                     summary.json, manifest.json, track1_it2v.json, prompts.jsonl,
                     prompts.json.

The instruction of an entry is the string under its ``prompt`` key, or the first string of
a list of prompts. The image and ground-truth paths are not read, because the metadata
describes the instruction rather than the rendered scene.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

REQUEST_COUNT = 1000
ELIGIBLE_COUNT = 157
RESOLVED_COUNT = 817
CALIBRATION_STRIDE = 63
CALIBRATION_OFFSET = 31
CALIBRATION_COUNT = 16
SUMMARY_NAMES = (
    "worldarena_summary.json",
    "summary.json",
    "manifest.json",
    "track1_it2v.json",
    "prompts.jsonl",
    "prompts.json",
)
PROMPT_KEYS = ("prompt", "instruction", "text")
BOTH_GRIPPERS = "both_grippers"
ROBOT_BASE = "robot_base"
NO_ARM = None
# 133 requests act with a single arm and the instruction does not say which one: the manifest
# generator sampled the arm, and the same verb with the same object appears on both sides.
# An unnamed arm is recorded as both, the union the detector already works with, so the
# resolved count stays the one of the published protocol.
SINGLE_GRIPPER = "single_gripper"
DEFAULT_MOVER = BOTH_GRIPPERS
# The instruction frames of the frozen manifest. Each frame marks one noun phrase as the
# manipulated object and the manipulator the text attributes it to. Frames are tried in
# order and the first match wins, so the object frames are matched before the navigation,
# collective and scene frames below.
TARGET_FRAMES = (
    (re.compile(r"^Hand the (?P<target>.+?) from the left gripper to the right one\.$"), BOTH_GRIPPERS),
    (re.compile(r"^Move the (?P<target>.+?) into .+\.$"), SINGLE_GRIPPER),
    (re.compile(r"^Slide the (?P<target>.+?) to the right edge of .+\.$"), SINGLE_GRIPPER),
    (re.compile(r"^Put the (?P<target>.+?) down on .+\.$"), SINGLE_GRIPPER),
    (re.compile(r"^Lift the (?P<target>.+?) above .+ and hold it there\.$"), SINGLE_GRIPPER),
    (re.compile(r"^Pick up the (?P<target>.+?) and place it on .+\.$"), SINGLE_GRIPPER),
    (re.compile(r"^Place the (?P<target>.+?) next to .+\.$"), SINGLE_GRIPPER),
    (re.compile(r"^Rotate the (?P<target>.+?) so that its handle faces up\.$"), SINGLE_GRIPPER),
    (re.compile(r"^Watch the (?P<target>.+?) until it comes to rest\.$"), NO_ARM),
    (re.compile(r"^Wait for the (?P<target>.+?) to stop moving\.$"), NO_ARM),
    (re.compile(r"^Keep the (?P<target>.+?) inside the frame until the scene settles\.$"), NO_ARM),
    (re.compile(r"^Do not disturb the (?P<target>.+)\.$"), NO_ARM),
)
# Frames without an object: the mobile base moves itself, the whole scene is tended with
# both arms, or nothing is manipulated at all.
ROBOT_BASE_RE = re.compile(r"^(?:Advance|Back|Drive|Shift|Turn|Move|Rotate) the robot base\b")
NO_ARM_RE = re.compile(
    r"^(?:Keep|Wait|Do not|Let|Observe|Inspect|Leave|Watch|Repeat|Follow|Nothing|Open)\b" r"|^(?:Pick|Take|Grab) (?:up )?(?:whichever|whatever|any)\b"
)
TWO_HAND_RE = re.compile(r"^(?:Hand|Clear|Rearrange|Tidy|Arrange|Line|Push|Pair|Sort|Stack|Collect|Group)\b" r"|^Move everything\b")
TRAIN_HEADER = (
    "# WorldArena 1.0 protocol calibration manifest.",
    "#",
    "# WorldArena 1.0 is an evaluation-only suite: it carries no training corpus, so",
    "# this file selects the 16 requests that were used to freeze the detector",
    "# thresholds and the occlusion rule before the final evaluation (see",
    "# configs/eval/mlr/worldarena.yaml). The remaining requests are evaluated with",
    "# that frozen protocol and are never trained on.",
)
EVAL_HEADER = (
    "# WorldArena 1.0 evaluation manifest.",
    "#",
    "# The frozen WorldArena 1.0 manifest, in manifest order. The requests eligible",
    "# for Model Laziness Rate are listed in data/metadata/worldarena/eligible_ids.json",
    "# and the parsed target and mover of every request in",
    "# data/metadata/worldarena/parsed_targets.json.",
    "# scripts/prepare/build_mlr_metadata.py regenerates the file from a data root.",
)


def layout_help(document: str, marker: str = "Upstream layout under ``--data-root``::") -> str:
    """The indented layout block of the module docstring, used as the ``--help`` epilog."""
    body = document.split(marker, 1)[1].lstrip("\n")
    return body.split("\n\n", 1)[0].rstrip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the WorldArena MLR metadata and splits under data/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=layout_help(__doc__),
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        default=os.environ.get("WORLDARENA_DATA_ROOT"),
        help="WorldArena tree holding the frozen manifest (default: $WORLDARENA_DATA_ROOT)",
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
        help="cap the number of requests read from the manifest, for a smoke run",
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


def manifest_path(root: Path) -> Path:
    """The frozen manifest below ``--data-root``, chosen from the recognised names."""
    for name in SUMMARY_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    found = sorted(path for path in root.rglob("*") if path.is_file() and path.name in SUMMARY_NAMES)
    if found:
        return found[0]
    raise FileNotFoundError(f"no WorldArena manifest below {root}: expected one of {', '.join(SUMMARY_NAMES)}")


def load_requests(path: Path) -> list[Any]:
    """The manifest entries of a JSON array or a JSONL file."""
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} holds a {type(payload).__name__}, expected a list of requests")
    return payload


def item_prompt(item: Any, index: int) -> str:
    """The instruction of one manifest entry; a one-element prompt list is flattened."""
    if not isinstance(item, dict):
        raise ValueError(f"manifest entry {index + 1} is a {type(item).__name__}, expected an object")
    value = first_key(item, PROMPT_KEYS)
    while isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest entry {index + 1} holds no prompt: keys {sorted(item)}")
    return value.strip()


def request_id(index: int) -> str:
    """The request id of the manifest position ``index``, counting from one."""
    return f"fixed_scene_task_episode{index + 1}"


def frame(prompt: str) -> tuple[str | None, str | None]:
    """The object ``prompt`` manipulates and the manipulator its text attributes to it.

    The mover is ``SINGLE_GRIPPER`` when one arm acts and the text does not say which one,
    and ``None`` when no manipulator acts at all.
    """
    for pattern, mover in TARGET_FRAMES:
        match = pattern.match(prompt)
        if match is not None:
            return match.group("target"), mover
    if ROBOT_BASE_RE.match(prompt):
        return None, ROBOT_BASE
    if NO_ARM_RE.match(prompt):
        return None, NO_ARM
    if TWO_HAND_RE.match(prompt):
        return None, BOTH_GRIPPERS
    return None, SINGLE_GRIPPER


def unresolved_count(prompts: Sequence[str]) -> int:
    """The requests that act with a single arm the instruction does not name."""
    return sum(1 for prompt in prompts if frame(prompt)[1] == SINGLE_GRIPPER)


def parse_request(prompt: str) -> tuple[str | None, str | None]:
    """The (target, mover) of one request, with the unnamed arm falling back to both."""
    target, mover = frame(prompt)
    return target, DEFAULT_MOVER if mover == SINGLE_GRIPPER else mover


def build_records(prompts: Sequence[str]) -> dict[str, dict[str, Any]]:
    """The parsed target and mover of every request, keyed by request id in manifest order."""
    records: dict[str, dict[str, Any]] = {}
    for index, prompt in enumerate(prompts):
        target, mover = parse_request(prompt)
        records[request_id(index)] = {"prompt": prompt, "target": target, "mover": mover}
    return records


def eligible_ids(records: dict[str, dict[str, Any]]) -> list[str]:
    """The requests eligible for MLR: an object to manipulate and a manipulator to move it."""
    return [rid for rid, record in records.items() if record["target"] and record["mover"]]


def build_eligibility(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The ``eligible_ids.json`` document: the eligible requests, the counts and the coverage."""
    eligible = eligible_ids(records)
    resolved = sum(1 for record in records.values() if record["mover"])
    coverage = round(100.0 * len(eligible) / resolved, 2) if resolved else 0.0
    return {
        "eligible": eligible,
        "num_eligible": len(eligible),
        "num_total": len(records),
        "num_resolved": resolved,
        "coverage": coverage,
    }


def calibration_ids(ids: Sequence[str]) -> list[str]:
    """The calibration slice: every 63rd request from the 32nd, spread over the manifest."""
    return list(ids[CALIBRATION_OFFSET::CALIBRATION_STRIDE])


def write_split(path: Path, ids: Sequence[str], header: Sequence[str], dry_run: bool) -> None:
    """Write the ``#`` header, one blank line and then one request id per line."""
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{line}\n" for line in header) + "\n" + "".join(f"{rid}\n" for rid in ids)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any, dry_run: bool) -> None:
    """Write one JSON document through the atomic writer of the library."""
    if dry_run:
        return
    from eveworld.utils.io import write_json as _write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(payload, path)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: read the frozen manifest, then write the metadata and the two splits."""
    args = parse_args(argv)
    if not args.data_root:
        print("error: --data-root is required (or export WORLDARENA_DATA_ROOT)", file=sys.stderr)
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
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def prepare(args: argparse.Namespace, data_root: Path, output_dir: Path) -> int:
    """Read the manifest, check the protocol counts and write the metadata and the splits."""
    manifest = manifest_path(data_root)
    requests = load_requests(manifest)
    if args.limit is not None:
        requests = requests[: args.limit]
    prompts = [item_prompt(item, index) for index, item in enumerate(requests)]
    records = build_records(prompts)
    eligibility = build_eligibility(records)
    ids = list(records)
    train = calibration_ids(ids)
    if args.limit is None:
        if len(requests) != REQUEST_COUNT:
            print(
                f"error: the manifest holds {len(requests)} requests, the protocol uses " f"{REQUEST_COUNT}: {manifest}",
                file=sys.stderr,
            )
            return 1
        if eligibility["num_eligible"] != ELIGIBLE_COUNT:
            print(
                f"error: parsed {eligibility['num_eligible']} eligible requests, the protocol " f"publishes {ELIGIBLE_COUNT}",
                file=sys.stderr,
            )
            return 1
        if eligibility["num_resolved"] != RESOLVED_COUNT:
            print(
                f"error: parsed {eligibility['num_resolved']} requests with a manipulator, the " f"protocol publishes {RESOLVED_COUNT}",
                file=sys.stderr,
            )
            return 1
        if len(train) != CALIBRATION_COUNT:
            print(
                f"error: the calibration slice holds {len(train)} requests, the protocol uses " f"{CALIBRATION_COUNT}",
                file=sys.stderr,
            )
            return 1

    movers = Counter(record["mover"] for record in records.values())
    unresolved = unresolved_count(prompts)
    targets = sum(1 for record in records.values() if record["target"])
    print(f"data root:    {data_root}")
    print(f"manifest:     {manifest} ({len(requests)} requests)")
    print(f"output dir:   {output_dir}")
    print(f"targets:      {targets} requests name a manipulated object, {len(records) - targets} do not")
    print(f"movers:       {BOTH_GRIPPERS} {movers[BOTH_GRIPPERS]}, {ROBOT_BASE} {movers[ROBOT_BASE]}, " f"none {movers[None]}")
    print(f"unresolved:   {unresolved} requests act with a single arm the instruction does not name, " f"recorded as {DEFAULT_MOVER}")
    print(
        f"eligible:     {eligibility['num_eligible']} of {eligibility['num_resolved']} requests "
        f"with a manipulator (coverage {eligibility['coverage']:.2f}%)"
    )
    print(f"calibration:  {len(train)} requests, every {CALIBRATION_STRIDE}rd manifest row from " f"row {CALIBRATION_OFFSET + 1}")

    metadata = output_dir / "metadata" / "worldarena"
    splits = output_dir / "splits" / "worldarena"
    write_json(metadata / "parsed_targets.json", records, args.dry_run)
    write_json(metadata / "eligible_ids.json", eligibility, args.dry_run)
    write_split(splits / "train.txt", train, TRAIN_HEADER, args.dry_run)
    write_split(splits / "eval.txt", ids, EVAL_HEADER, args.dry_run)

    if args.dry_run:
        print("dry run: no file written")
        return 0
    print(f"wrote {metadata}/parsed_targets.json, {metadata}/eligible_ids.json, " f"{splits}/train.txt and {splits}/eval.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
