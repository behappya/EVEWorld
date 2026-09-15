#!/usr/bin/env python3
"""Build disjoint FlowWAM checkpoint-selection and confirmation manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--dev-episode", type=int, default=45)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())
    if len(rows) != 250:
        raise ValueError(f"expected 250 source rows, got {len(rows)}")
    request_ids = [str(row["request_id"]) for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("source manifest contains duplicate request_id values")

    dev = [row for row in rows if int(row["episode"]) == args.dev_episode]
    test = [row for row in rows if int(row["episode"]) != args.dev_episode]
    if len(dev) != 50 or len(test) != 200:
        raise ValueError(f"expected a 50/200 split, got {len(dev)}/{len(test)}")
    if {row["task"] for row in dev} != {row["task"] for row in rows}:
        raise ValueError("development manifest does not cover every task")

    write_json(args.dev_output, dev)
    write_json(args.test_output, test)
    print(f"wrote dev={len(dev)} to {args.dev_output}")
    print(f"wrote test={len(test)} to {args.test_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
