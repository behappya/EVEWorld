#!/usr/bin/env python3
"""Prepare randomized, paired videos for a blinded human process audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def load_group(path: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("error") or row.get("parsed_ok") == "0":
                continue
            name = Path(row["video_path"]).name
            if name in rows:
                raise ValueError(f"duplicate video basename {name} in {path}")
            rows[name] = row
    return rows


def metric(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else 0.0


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", nargs="+", required=True)
    parser.add_argument("--method", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    if len(args.control) != len(args.method):
        raise ValueError("control/method CSV group counts must match")

    rng = random.Random(args.seed)
    pairs: list[dict] = []
    for group, (control_path, method_path) in enumerate(zip(args.control, args.method)):
        control = load_group(control_path)
        method = load_group(method_path)
        if set(control) != set(method):
            raise ValueError(
                f"group {group} does not pair exactly: "
                f"control_only={sorted(set(control) - set(method))}, "
                f"method_only={sorted(set(method) - set(control))}"
            )
        for name in sorted(control):
            pairs.append(
                {
                    "group": group,
                    "name": name,
                    "control": control[name],
                    "method": method[name],
                }
            )

    rng.shuffle(pairs)
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    review_rows = []
    private_rows = []
    for index, pair in enumerate(pairs, start=1):
        pair_id = f"pair_{index:03d}"
        method_side = rng.choice(("A", "B"))
        side_rows = {
            method_side: pair["method"],
            "B" if method_side == "A" else "A": pair["control"],
        }
        side_paths = {}
        for side in ("A", "B"):
            source = Path(side_rows[side]["video_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            link = videos_dir / f"{pair_id}_{side}.mp4"
            replace_symlink(link, source)
            side_paths[side] = str(link)

        control = pair["control"]
        method = pair["method"]
        completeness_delta = metric(method, "process_completeness") - metric(
            control, "process_completeness"
        )
        severity_delta = metric(method, "laziness_severity") - metric(
            control, "laziness_severity"
        )
        review_rows.append(
            {
                "pair_id": pair_id,
                "prompt": control["prompt"],
                "video_A": side_paths["A"],
                "video_B": side_paths["B"],
                "more_complete_process_A_B_tie": "",
                "unacceptable_quality_A_B_both_none": "",
                "notes": "",
            }
        )
        private_rows.append(
            {
                "pair_id": pair_id,
                "method_side": method_side,
                "source_name": pair["name"],
                "group": pair["group"],
                "control_path": control["video_path"],
                "method_path": method["video_path"],
                "qwen_control_completeness": metric(control, "process_completeness"),
                "qwen_method_completeness": metric(method, "process_completeness"),
                "qwen_completeness_delta": completeness_delta,
                "qwen_control_severity": metric(control, "laziness_severity"),
                "qwen_method_severity": metric(method, "laziness_severity"),
                "qwen_severity_delta": severity_delta,
                "qwen_control_evidence": control.get("evidence", ""),
                "qwen_method_evidence": method.get("evidence", ""),
            }
        )

    review_path = out_dir / "review_form.csv"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    key_path = out_dir / "private_key.json"
    key_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "control_csvs": args.control,
                "method_csvs": args.method,
                "pairs": private_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"prepared {len(pairs)} pairs in {out_dir}")
    print(f"blind review form: {review_path}")
    print(f"do not inspect before review: {key_path}")


if __name__ == "__main__":
    main()
