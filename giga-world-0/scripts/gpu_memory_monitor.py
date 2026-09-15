#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample nvidia-smi GPU memory usage to CSV and rolling peak JSON.")
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--peak-json", type=Path, required=True)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--label", default="gpu_memory_monitor")
    return parser.parse_args()


def now_fields() -> tuple[float, str]:
    ts = time.time()
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return ts, iso


def query_gpu_memory() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    rows: list[dict[str, Any]] = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 5:
            continue
        index, uuid, name, used, total = [field.strip() for field in row[:5]]
        rows.append(
            {
                "gpu_index": index,
                "gpu_uuid": uuid,
                "gpu_name": name,
                "memory_used_mib": int(float(used)),
                "memory_total_mib": int(float(total)),
            }
        )
    return rows


def cuda_visible_filter() -> set[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw or raw.lower() in {"all", "none", "void"}:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def filter_visible_rows(rows: list[dict[str, Any]], visible: set[str] | None) -> list[dict[str, Any]]:
    if visible is None:
        return rows
    return [row for row in rows if row["gpu_index"] in visible or row["gpu_uuid"] in visible]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    samples_csv = args.samples_csv.expanduser().resolve()
    peak_json = args.peak_json.expanduser().resolve()
    samples_csv.parent.mkdir(parents=True, exist_ok=True)
    peak_json.parent.mkdir(parents=True, exist_ok=True)

    started_ts, started_iso = now_fields()
    peak: dict[str, Any] = {
        "label": args.label,
        "started_at_unix": started_ts,
        "started_at_iso": started_iso,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "last_sample_at_unix": None,
        "last_sample_at_iso": None,
        "sample_count": 0,
        "peak_memory_used_mib": None,
        "peak_memory_used_gib": None,
        "peak_gpu_index": None,
        "peak_gpu_uuid": None,
        "peak_gpu_name": None,
        "peak_timestamp_unix": None,
        "peak_timestamp_iso": None,
        "latest": [],
    }
    atomic_write_json(peak_json, peak)

    with samples_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "timestamp_unix",
                "timestamp_iso",
                "label",
                "gpu_index",
                "gpu_uuid",
                "gpu_name",
                "memory_used_mib",
                "memory_total_mib",
            ],
        )
        writer.writeheader()
        fh.flush()

        while True:
            ts, iso = now_fields()
            rows = filter_visible_rows(query_gpu_memory(), cuda_visible_filter())
            for row in rows:
                writer.writerow({"timestamp_unix": ts, "timestamp_iso": iso, "label": args.label, **row})
            fh.flush()

            peak["last_sample_at_unix"] = ts
            peak["last_sample_at_iso"] = iso
            peak["sample_count"] = int(peak["sample_count"]) + len(rows)
            peak["latest"] = rows
            for row in rows:
                used_mib = int(row["memory_used_mib"])
                current_peak = peak["peak_memory_used_mib"]
                if current_peak is None or used_mib > int(current_peak):
                    peak["peak_memory_used_mib"] = used_mib
                    peak["peak_memory_used_gib"] = round(used_mib / 1024.0, 4)
                    peak["peak_gpu_index"] = row["gpu_index"]
                    peak["peak_gpu_uuid"] = row["gpu_uuid"]
                    peak["peak_gpu_name"] = row["gpu_name"]
                    peak["peak_timestamp_unix"] = ts
                    peak["peak_timestamp_iso"] = iso
            atomic_write_json(peak_json, peak)
            time.sleep(max(args.interval_sec, 0.1))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(0)
