#!/usr/bin/env python3
"""Build the RoboTwin splits and per-clip metadata under ``data/``.

The script reads the RoboTwin tree pointed to by ``--data-root`` (the
``ROBOTWIN_DATA_ROOT`` environment variable) and writes:

``<output-dir>/splits/robotwin/heldout.txt``
    The 250 held-out evaluation episodes of paper Table 5, one row per line, ordered by
    task and then by episode: the 50 tasks with their episodes 45-49.
``<output-dir>/splits/robotwin/dev.txt``
    The episode-45 slice of the held-out range: 50 rows, one per task, used to screen
    checkpoints while the reported numbers use all 250 rows.
``<output-dir>/metadata/robotwin/<id>.json``
    Prompt, latent geometry and the per-timestamp expected instance count of every
    held-out episode.

The FlowWAM recipe reads 121 frames at 640 x 480 and 24 FPS, sampled onto a 24-step
latent grid. The 250 held-out episodes are never part of the 2,250 training episodes,
episodes 0-44 of the same 50 tasks; the script counts them and reports the number
instead of writing them, because no split file lists them.

The rows follow the task order of the release rather than the alphabetical order of the
task directories, and that order is not fixed by the paper. ``TASK_ORDER`` holds it and a
data root that holds a different set of tasks is rejected.

Upstream layout under ``--data-root``::

    <task>/aloha-agilex_clean_50/instructions/episode<N>.json                 {"seen": [...]}
    <task>/aloha-agilex_clean_50/video/episode<N>.mp4                         generated demo
    <task>/aloha-agilex_clean_50/robot_only/video/head_camera/episode<N>.mp4  flow conditioning

The instruction of an episode is the first entry of ``seen``, or of ``unseen`` when the
task has none. The collection directory (``aloha-agilex_clean_50``) is discovered, not
assumed, so a data root that tags it differently is read as well.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

NUM_FRAMES = 121
FPS = 24
WIDTH, HEIGHT = 640, 480
N_LAT, H_LAT, W_LAT = 31, 30, 40
SAMPLE_COUNT = 24
TASK_COUNT = 50
HELDOUT_EPISODES = (45, 46, 47, 48, 49)
DEV_EPISODE = 45
TRAINING_EPISODE_COUNT = 45
HELDOUT_COUNT = TASK_COUNT * len(HELDOUT_EPISODES)
DEV_COUNT = TASK_COUNT
TRAINING_POOL_COUNT = TASK_COUNT * TRAINING_EPISODE_COUNT
INSTRUCTION_KEYS = ("seen", "unseen")
PROMPT_KEYS = ("instruction", "prompt", "text")
INSTRUCTION_PATTERN = re.compile(r"^episode(?P<number>\d+)\.json$")
# The task order of the split rows, the order the 50 tasks are registered in upstream. It is
# not the alphabetical order of the directories (``pick_apple_mark`` follows
# ``pick_dual_bottles``) and the paper does not fix it.
TASK_ORDER = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "pick_apple_mark",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
HELDOUT_HEADER = (
    "# RoboTwin held-out evaluation split.",
    "#",
    "# 50 tasks x episodes 45-49 = 250 held-out episodes, one row per line, ordered",
    "# by task and then by episode. The rows are evaluated with all 250 episodes at",
    "# 480 x 640, 29 frames at 24 FPS (Table 5).",
    "#",
    "# The training pool of the FlowWAM arm is the same 50 tasks with episodes 0-44",
    "# (2,250 episodes), so no held-out episode is ever trained on. The episode-45",
    "# slice of this range is the disjoint dev split in dev.txt.",
    "# scripts/prepare/prepare_robotwin.py regenerates the file from a data root.",
)
DEV_HEADER = (
    "# RoboTwin checkpoint-screening dev split.",
    "#",
    "# The episode-45 slice of the held-out range: 50 tasks x episode 45 = 50 rows,",
    "# one row per task. This slice is disjoint from the 200 remaining held-out rows",
    "# (episodes 46-49), which form the confirmation set, and disjoint from the",
    "# training pool (episodes 0-44); it is used only to screen checkpoints, while",
    "# the reported numbers use all 250 rows of heldout.txt.",
    "# scripts/prepare/prepare_robotwin.py regenerates the file from a data root.",
)


@dataclass
class Episode:
    """One episode of the RoboTwin tree: its task, number, instruction and media."""

    task: str
    number: int
    prompt: str
    instruction: Path
    video: Path
    robot_video: Path

    @property
    def vid(self) -> str:
        """The split id of the episode."""
        return f"task_{self.task}_episode_{self.number}"


def layout_help(document: str, marker: str = "Upstream layout under ``--data-root``::") -> str:
    """The indented layout block of the module docstring, used as the ``--help`` epilog."""
    body = document.split(marker, 1)[1].lstrip("\n")
    return body.split("\n\n", 1)[0].rstrip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RoboTwin splits and per-clip metadata under data/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=layout_help(__doc__),
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        default=os.environ.get("ROBOTWIN_DATA_ROOT"),
        help="RoboTwin tree holding one directory per task (default: $ROBOTWIN_DATA_ROOT)",
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
        help="cap the number of rows written to each split, for a smoke run",
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


def instruction_prompt(path: Path) -> str:
    """The instruction of an episode: the first ``seen`` entry, else the first ``unseen`` one."""
    payload = load_json(path)
    if isinstance(payload, dict):
        for key in INSTRUCTION_KEYS:
            values = payload.get(key)
            if isinstance(values, list) and values:
                return str(values[0])
            if isinstance(values, str) and values:
                return values
        prompt = first_key(payload, PROMPT_KEYS)
        if prompt is not None:
            return str(prompt)
    if isinstance(payload, list) and payload:
        return str(payload[0])
    raise ValueError(f"{path} holds no instruction: expected a seen/unseen list")


def load_episodes(root: Path) -> dict[str, dict[int, Episode]]:
    """Every task below ``--data-root`` with the episodes its instruction files name."""
    tasks: dict[str, dict[int, Episode]] = {}
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        episodes: dict[int, Episode] = {}
        for instruction in sorted(task_dir.rglob("episode*.json")):
            if instruction.parent.name != "instructions":
                continue
            match = INSTRUCTION_PATTERN.match(instruction.name)
            if match is None:
                continue
            number = int(match.group("number"))
            collection = instruction.parent.parent
            episodes[number] = Episode(
                task=task_dir.name,
                number=number,
                prompt=instruction_prompt(instruction),
                instruction=instruction,
                video=collection / "video" / f"episode{number}.mp4",
                robot_video=collection / "robot_only" / "video" / "head_camera" / f"episode{number}.mp4",
            )
        if episodes:
            tasks[task_dir.name] = episodes
    if not tasks:
        raise FileNotFoundError(f"no task directory with an instructions tree below {root}")
    return tasks


def ordered_tasks(tasks: dict[str, dict[int, Episode]]) -> tuple[list[str], list[str]]:
    """The tasks of the recipe in manifest order and the directories the recipe does not know."""
    known = [task for task in TASK_ORDER if task in tasks]
    unknown = sorted(task for task in tasks if task not in TASK_ORDER)
    return known, unknown


def select_rows(
    tasks: dict[str, dict[int, Episode]],
    names: Sequence[str],
    episodes: Sequence[int],
) -> list[Episode]:
    """The episodes of ``names`` in task order, one entry per task and episode number."""
    return [tasks[task][number] for task in names for number in episodes if number in tasks[task]]


def incomplete_tasks(tasks: dict[str, dict[int, Episode]], names: Sequence[str]) -> list[str]:
    """The tasks of ``names`` that miss at least one of the held-out episode numbers."""
    return [task for task in names if any(number not in tasks[task] for number in HELDOUT_EPISODES)]


def missing_media(episodes: Sequence[Episode]) -> list[str]:
    """The media files of ``episodes`` that are not on disk, as ``<id>: <path>`` strings."""
    missing: list[str] = []
    for episode in episodes:
        for path in (episode.video, episode.robot_video):
            if not path.is_file():
                missing.append(f"{episode.vid}: {path}")
    return missing


def sample_timestamps(num_frames: int = NUM_FRAMES, count: int = SAMPLE_COUNT) -> list[int]:
    """Round sampling of ``count`` timestamps over ``num_frames``, the MLR protocol axis."""
    if count == 1:
        return [0]
    step = (num_frames - 1) / (count - 1)
    return [int(round(index * step)) for index in range(count)]


def build_record(episode: Episode) -> dict[str, Any]:
    """Metadata record for one held-out episode: prompt, geometry and expected counts."""
    timestamps = sample_timestamps()
    # One instance per sampled timestamp is the reading the recipe uses for the robot-only
    # clips; the count itself is not fixed by the paper, and the list is aligned with
    # `timestamps`, which is what the evaluator pairs it with.
    counts = [1] * len(timestamps)
    return {
        "vid": episode.vid,
        "split": "heldout",
        "prompt": episode.prompt,
        "category": episode.task,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "latent_grid": [N_LAT, H_LAT, W_LAT],
        "timestamps": timestamps,
        "expected_counts": counts,
    }


def write_split(path: Path, ids: Sequence[str], header: Sequence[str], dry_run: bool) -> None:
    """Write the ``#`` header, one blank line and then one episode id per line."""
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
    """Entry point: read the RoboTwin tree, then write the two splits and the metadata."""
    args = parse_args(argv)
    if not args.data_root:
        print("error: --data-root is required (or export ROBOTWIN_DATA_ROOT)", file=sys.stderr)
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
    """Read the tree, check the recipe counts and write the two splits and the metadata."""
    tasks = load_episodes(data_root)
    names, unknown = ordered_tasks(tasks)
    heldout = select_rows(tasks, names, HELDOUT_EPISODES)
    dev = select_rows(tasks, names, (DEV_EPISODE,))
    training_pool = sum(
        1 for task in names for number in tasks[task] if number < DEV_EPISODE
    )
    if args.limit is not None:
        heldout = heldout[: args.limit]
        dev = dev[: args.limit]
    if args.limit is None:
        if unknown:
            print(
                f"error: the data root holds {len(unknown)} task directories outside the recipe: "
                f"{unknown[:5]}",
                file=sys.stderr,
            )
            return 1
        if len(names) != TASK_COUNT:
            missing = [task for task in TASK_ORDER if task not in tasks]
            print(
                f"error: found {len(names)} tasks below the data root, the recipe uses {TASK_COUNT}: "
                f"missing {missing[:5]}",
                file=sys.stderr,
            )
            return 1
        if len(heldout) != HELDOUT_COUNT:
            print(
                f"error: found {len(heldout)} held-out episodes, the recipe uses {HELDOUT_COUNT} "
                f"(episodes {HELDOUT_EPISODES[0]}-{HELDOUT_EPISODES[-1]} of {TASK_COUNT} tasks): "
                f"incomplete {incomplete_tasks(tasks, names)[:5]}",
                file=sys.stderr,
            )
            return 1
        if len(dev) != DEV_COUNT:
            print(
                f"error: found {len(dev)} dev episodes, the recipe uses {DEV_COUNT} (episode "
                f"{DEV_EPISODE} of {TASK_COUNT} tasks)",
                file=sys.stderr,
            )
            return 1
    missing = missing_media(heldout)
    if missing:
        print(f"error: {len(missing)} held-out media files are missing, e.g. {missing[0]}", file=sys.stderr)
        return 1

    print(f"data root:    {data_root}")
    print(f"output dir:   {output_dir}")
    print(f"tasks:        {len(names)} of {TASK_COUNT} in the recipe")
    print(f"heldout:      {len(heldout)} rows ({HELDOUT_COUNT} in the recipe, episodes "
          f"{HELDOUT_EPISODES[0]}-{HELDOUT_EPISODES[-1]})")
    print(f"dev:          {len(dev)} rows ({DEV_COUNT} in the recipe, episode {DEV_EPISODE})")
    print(f"training:     {training_pool} episodes of episodes 0-{DEV_EPISODE - 1} (not written)")
    print(f"timestamps:   {len(sample_timestamps())} per clip on a {N_LAT}x{H_LAT}x{W_LAT} latent grid")

    splits = output_dir / "splits" / "robotwin"
    metadata = output_dir / "metadata" / "robotwin"
    write_split(splits / "heldout.txt", [episode.vid for episode in heldout], HELDOUT_HEADER, args.dry_run)
    write_split(splits / "dev.txt", [episode.vid for episode in dev], DEV_HEADER, args.dry_run)
    records = [build_record(episode) for episode in heldout]
    write_metadata(metadata, records, args.dry_run)

    if args.dry_run:
        print("dry run: no file written")
        return 0
    print(f"wrote {splits}/heldout.txt, dev.txt and {len(records)} records under {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
