#!/usr/bin/env python3
"""Run three resumable Gemini-IF repeats for the uniform negative-route set."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def read_manifest_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def error_count(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(bool(row.get("error")) for row in csv.DictReader(handle))


def row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_once(args: argparse.Namespace, output_dir: Path, rerun_errors: bool) -> None:
    command = [
        args.python,
        str(args.eval_script),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--metrics",
        "qwen_if",
        "--concurrency",
        str(args.concurrency),
        "--frame-count",
        "49",
        "--jpeg-quality",
        "85",
        "--temperature",
        "0",
        "--thinking-level",
        "low",
        "--no-include-thoughts",
        "--model-timeout",
        "1200",
        "--model-max-tokens",
        "32000",
        "--limit",
        str(args.expected),
        "--resume",
    ]
    if rerun_errors:
        command.append("--rerun-errors")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-error-passes", type=int, default=4)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.eval_script = args.eval_script.resolve()
    args.expected = read_manifest_count(args.manifest)
    if args.expected != 1048:
        raise RuntimeError(f"expected 1048 manifest rows, found {args.expected}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    report = {"model": args.model, "expected": args.expected, "repeats": {}}
    for repeat in range(1, args.repeats + 1):
        output_dir = args.output_root / f"repeat{repeat:02d}"
        print(f"[serial] repeat {repeat}/{args.repeats} -> {output_dir}", flush=True)
        run_once(args, output_dir, rerun_errors=False)
        csv_path = output_dir / "qwen_if.csv"
        for error_pass in range(1, args.max_error_passes + 1):
            errors = error_count(csv_path)
            if errors == 0:
                break
            print(f"[serial] repeat {repeat}: retry pass {error_pass}, errors={errors}", flush=True)
            run_once(args, output_dir, rerun_errors=True)
        rows = row_count(csv_path)
        errors = error_count(csv_path)
        report["repeats"][f"repeat{repeat:02d}"] = {"rows": rows, "errors": errors}
        (args.output_root / "serial_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if rows != args.expected or errors != 0:
            raise RuntimeError(f"repeat {repeat} incomplete: rows={rows}, errors={errors}")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
