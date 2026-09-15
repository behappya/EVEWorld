#!/usr/bin/env python3
"""Build and evaluate the uniform negative-route manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2


GAGI = Path("/data/datasets/gagi")
IT2V = GAGI / "gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json"
DEFAULT_ROOT = GAGI / "eve_v2_outputs/failure_routes_reval_v1"
EXPECTED_ROUTE_PAIRS = {
    "physicslatent": 92,
    "eag": 16,
    "lad_lora": 32,
    "causal_frontier": 16,
    "ich_d": 368,
}
EXPECTED_FRAMES = {
    "physicslatent": 93,
    "eag": 125,
    "lad_lora": 93,
    "causal_frontier": 93,
    "ich_d": 93,
}
EXPECTED_WIDTH = {
    # The legacy PhysicsLatent encoder wrote 776x480; the MLR examiner
    # deterministically center-crops/resizes it to the common 768x480 input.
    "physicslatent": 776,
    "eag": 768,
    "lad_lora": 768,
    "causal_frontier": 768,
    "ich_d": 768,
}


@dataclass(frozen=True)
class PairSource:
    route: str
    display_name: str
    seed: int
    method_dirs: tuple[Path, ...]
    control_dirs: tuple[Path, ...]
    expected_pairs: int
    filename_kind: str = "request_index"


def load_it2v() -> dict[int, dict[str, Any]]:
    rows = json.loads(IT2V.read_text(encoding="utf-8"))
    result = {int(str(row["request_id"]).split("_", 1)[0]): row for row in rows}
    if len(result) != 92:
        raise RuntimeError(f"expected 92 canonical prompts, found {len(result)}")
    return result


def load_it2v_maps() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return maps for generated names using request index or source filename."""
    rows = json.loads(IT2V.read_text(encoding="utf-8"))
    by_request = {}
    by_source = {}
    for row in rows:
        request_id = str(row["request_id"])
        request_index = int(request_id.split("_", 1)[0])
        source_name = str(row["source_file_name"])
        source_index = int(Path(source_name).stem)
        by_request[request_index] = row
        by_source[source_index] = row
    if len(by_request) != 92 or len(by_source) != 92:
        raise RuntimeError("canonical prompt maps are not one-to-one")
    return by_request, by_source


def direct_mp4s(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".mp4")


def indexed_videos(directories: Iterable[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for directory in directories:
        for path in direct_mp4s(directory):
            try:
                index = int(path.name.split("_", 1)[0])
            except ValueError as exc:
                raise ValueError(f"video lacks numeric request prefix: {path}") from exc
            if index in result:
                raise RuntimeError(f"duplicate prompt index {index}: {result[index]} and {path}")
            result[index] = path.resolve()
    return result


def pair_sources() -> list[PairSource]:
    dreamgen = GAGI / "gr1_dreamgen_eval/dreamgenbench_video_dirs"
    eve = GAGI / "eve_outputs"
    eve_v2 = GAGI / "eve_v2_outputs"
    frontier = eve / "frontier/generation_train8_onesided50_noguard_20260716"

    sources = [
        PairSource(
            route="physicslatent",
            display_name="PhysicsLatent",
            seed=0,
            method_dirs=(dreamgen / "physlatent_aux_step500_full92",),
            control_dirs=(dreamgen / "pretrain_dreamgen_8gpu_5p8s_full_20260627_162849",),
            expected_pairs=92,
        ),
        PairSource(
            route="eag",
            display_name="EAG sampling guidance",
            seed=6666,
            method_dirs=(eve / "eag_eval/eag_w0.03_seed6666_f125/generated_only",),
            control_dirs=(eve / "eag_eval/baseline_seed6666_f125/generated_only",),
            expected_pairs=16,
        ),
    ]

    for seed in (1234, 6666):
        sources.append(
            PairSource(
                route="lad_lora",
                display_name="LAD-LoRA step50",
                seed=seed,
                method_dirs=(eve / f"lad_lora_eval_gradbal_v2/main_s50_seed{seed}/generated_only",),
                control_dirs=(eve / f"lad_lora_eval_gradbal_v2/control_s50_seed{seed}/generated_only",),
                expected_pairs=16,
            )
        )
        sources.append(
            PairSource(
                route="causal_frontier",
                display_name="Causal Frontier one-sided50 no-guard",
                seed=seed,
                method_dirs=tuple(sorted(frontier.glob(f"one50_seed{seed}_q*/generated_only"))),
                control_dirs=(frontier / f"eval_merged/control50_seed{seed}",),
                expected_pairs=8,
                filename_kind="source_file",
            )
        )

    for seed in (42, 314, 777, 999):
        sources.append(
            PairSource(
                route="ich_d",
                display_name="ICH-D step350",
                seed=seed,
                method_dirs=(eve_v2 / f"selfcase/pool_ichD_s350_f93/seed{seed}_f93/generated_only",),
                control_dirs=(eve_v2 / f"selfcase/pool_pretrain_f93/seed{seed}_f93/generated_only",),
                expected_pairs=92,
            )
        )
    return sources


def build_rows() -> list[dict[str, Any]]:
    prompts, prompts_by_source = load_it2v_maps()
    rows: list[dict[str, Any]] = []
    for source in pair_sources():
        method_raw = indexed_videos(source.method_dirs)
        control_raw = indexed_videos(source.control_dirs)
        if source.filename_kind == "source_file":
            method = {}
            control = {}
            for source_index, path in method_raw.items():
                meta = prompts_by_source.get(source_index)
                if meta is None:
                    raise RuntimeError(f"unknown source_file_name {source_index}")
                request_index = int(str(meta["request_id"]).split("_", 1)[0])
                if request_index in method:
                    raise RuntimeError(f"duplicate mapped request index {request_index}")
                method[request_index] = path
            for source_index, path in control_raw.items():
                meta = prompts_by_source.get(source_index)
                if meta is None:
                    raise RuntimeError(f"unknown source_file_name {source_index}")
                request_index = int(str(meta["request_id"]).split("_", 1)[0])
                if request_index in control:
                    raise RuntimeError(f"duplicate mapped request index {request_index}")
                control[request_index] = path
        else:
            method, control = method_raw, control_raw
        if set(method) != set(control):
            raise RuntimeError(
                f"{source.route}/seed{source.seed} pair mismatch: "
                f"method-only={sorted(set(method) - set(control))}, "
                f"control-only={sorted(set(control) - set(method))}"
            )
        if len(method) != source.expected_pairs:
            raise RuntimeError(
                f"{source.route}/seed{source.seed}: expected {source.expected_pairs} pairs, "
                f"found {len(method)}"
            )
        for index in sorted(method):
            if index not in prompts:
                raise RuntimeError(f"unknown canonical prompt index {index}")
            meta = prompts[index]
            pair_id = f"{source.route}/seed{source.seed}/{index}"
            for arm, path in (("control", control[index]), ("method", method[index])):
                rows.append(
                    {
                        "key": f"{pair_id}/{arm}",
                        "pair_id": pair_id,
                        "route": source.route,
                        "route_name": source.display_name,
                        "arm": arm,
                        "model": f"{source.route}__{arm}",
                        "inference_seed": source.seed,
                        "split": f"{source.route}_seed{source.seed}",
                        "index": index,
                        "request_id": str(meta["request_id"]),
                        "prompt": str(meta["prompt"]),
                        "video_path": str(path),
                        "expected_frames": EXPECTED_FRAMES[source.route],
                    }
                )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_video(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["video_path"])
    result = {"key": row["key"], "video_path": str(path), "error": None}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open video")
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        ok_first, first = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames - 1))
        ok_last, last = cap.read()
        if not ok_first or first is None or not ok_last or last is None:
            raise RuntimeError("first/last frame decode failed")
        if frames != int(row["expected_frames"]):
            raise RuntimeError(f"expected {row['expected_frames']} frames, found {frames}")
        expected_width = EXPECTED_WIDTH[row["route"]]
        if (width, height) != (expected_width, 480):
            raise RuntimeError(f"expected {expected_width}x480, found {width}x{height}")
        if abs(fps - 16.0) > 0.1:
            raise RuntimeError(f"expected 16 fps, found {fps:.4f}")
        result.update(frames=frames, width=width, height=height, fps=fps)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cap.release()
    return result


def route_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["route"]][row["arm"]] += 1
    return {route: dict(arms) for route, arms in sorted(counts.items())}


def command_build(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifests = root / "manifests"
    rows = build_rows()
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("manifest contains duplicate keys")
    by_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_pair[row["pair_id"]].add(row["arm"])
    bad_pairs = {pair: sorted(arms) for pair, arms in by_pair.items() if arms != {"control", "method"}}
    if bad_pairs:
        raise RuntimeError(f"unmatched pairs: {bad_pairs}")
    if len(rows) != 1048 or len(by_pair) != 524:
        raise RuntimeError(f"expected 1048 videos/524 pairs, found {len(rows)}/{len(by_pair)}")

    node_a = [
        row
        for row in rows
        if row["route"] == "physicslatent"
        or (row["route"] == "ich_d" and row["inference_seed"] in {42, 777})
    ]
    node_b = [row for row in rows if row not in node_a]
    if (len(node_a), len(node_b)) != (552, 496):
        raise RuntimeError(f"unexpected node split: {len(node_a)}, {len(node_b)}")

    canary: list[dict[str, Any]] = []
    for route in EXPECTED_ROUTE_PAIRS:
        for arm in ("control", "method"):
            canary.append(next(row for row in rows if row["route"] == route and row["arm"] == arm))

    write_jsonl(manifests / "all.jsonl", rows)
    write_jsonl(manifests / "node_a.jsonl", node_a)
    write_jsonl(manifests / "node_b.jsonl", node_b)
    write_jsonl(manifests / "gemini_canary.jsonl", canary)

    print(f"[build] auditing {len(rows)} videos", flush=True)
    audit_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.audit_workers) as pool:
        futures = [pool.submit(audit_video, row) for row in rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            audit_records.append(future.result())
            if completed % 100 == 0:
                print(f"[build] media {completed}/{len(rows)}", flush=True)
    audit_records.sort(key=lambda row: row["key"])
    errors = [row for row in audit_records if row["error"]]
    audit = {
        "ready": not errors,
        "records": len(rows),
        "pairs": len(by_pair),
        "node_a": len(node_a),
        "node_b": len(node_b),
        "route_counts": route_counts(rows),
        "media_errors": errors,
        "media_records": audit_records,
    }
    write_json(root / "media_audit.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "media_records"}, indent=2))
    if errors:
        raise SystemExit(1)


def write_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(path, rows)


def command_mlr_shard(args: argparse.Namespace) -> None:
    from t4g_exam_v2 import exam_video_v2
    from t4g_gdino import GDinoLocator

    rows = load_jsonl(args.manifest)
    rows = rows[args.shard_index :: args.num_shards]
    locator = GDinoLocator(device="cuda")
    records: list[dict[str, Any]] = []
    print(f"[mlr] shard {args.shard_index}/{args.num_shards}: {len(rows)} videos", flush=True)
    for index, row in enumerate(rows, start=1):
        record = {
            key: row[key]
            for key in (
                "key",
                "pair_id",
                "route",
                "route_name",
                "arm",
                "model",
                "inference_seed",
                "split",
                "index",
                "request_id",
                "video_path",
            )
        }
        record["error"] = None
        try:
            record.update(exam_video_v2(locator, row["video_path"], row["prompt"]))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if index % 5 == 0 or index == len(rows):
            write_shard(args.out, records)
            print(f"[mlr] shard {args.shard_index}: {index}/{len(rows)}", flush=True)


def summarize_mlr(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["route"], row["arm"])].append(row)
    summary: dict[str, Any] = {}
    for (route, arm), rows in sorted(groups.items()):
        valid = [row for row in rows if not row.get("error")]
        eligible = [row for row in valid if not row.get("undetectable")]
        summary.setdefault(route, {})[arm] = {
            "records": len(rows),
            "errors": len(rows) - len(valid),
            "undetectable": len(valid) - len(eligible),
            "eligible": len(eligible),
            "dup_events": sum(bool(row.get("dup")) for row in eligible),
            "dup_rate": sum(bool(row.get("dup")) for row in eligible) / len(eligible) if eligible else None,
            "vanish_events": sum(bool(row.get("vanish")) for row in eligible),
            "vanish_rate": sum(bool(row.get("vanish")) for row in eligible) / len(eligible) if eligible else None,
        }
    return summary


def merge_parts(manifest: Path, part_paths: list[Path], out_dir: Path) -> None:
    expected = {row["key"] for row in load_jsonl(manifest)}
    records: list[dict[str, Any]] = []
    for path in part_paths:
        if not path.is_file():
            raise RuntimeError(f"missing shard output: {path}")
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    actual = [row["key"] for row in records]
    if len(actual) != len(set(actual)):
        raise RuntimeError("duplicate MLR record keys")
    if set(actual) != expected:
        raise RuntimeError(
            f"MLR coverage mismatch: missing={len(expected - set(actual))}, extra={len(set(actual) - expected)}"
        )
    records.sort(key=lambda row: row["key"])
    summary = summarize_mlr(records)
    summary["records"] = len(records)
    summary["errors"] = sum(bool(row.get("error")) for row in records)
    write_json(out_dir / "records.json", records)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["errors"]:
        raise SystemExit(1)


def command_mlr_worker(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    processes: list[tuple[int, subprocess.Popen[Any], Any]] = []
    for shard in range(args.num_gpus):
        log_handle = (out_dir / f"shard{shard}.log").open("w", encoding="utf-8")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [
                args.python,
                str(script),
                "mlr-shard",
                "--manifest",
                str(args.manifest),
                "--shard-index",
                str(shard),
                "--num-shards",
                str(args.num_gpus),
                "--out",
                str(out_dir / f"part{shard}.json"),
            ],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((shard, process, log_handle))
    print(f"[worker] started {len(processes)} GPU shards", flush=True)
    while any(process.poll() is None for _, process, _ in processes):
        counts = []
        for shard, process, _ in processes:
            path = out_dir / f"part{shard}.json"
            count = len(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else 0
            state = "run" if process.poll() is None else f"rc{process.returncode}"
            counts.append(f"g{shard}:{count}:{state}")
        print("[worker] " + " ".join(counts), flush=True)
        time.sleep(args.poll_seconds)
    return_codes = []
    for shard, process, log_handle in processes:
        return_codes.append(process.wait())
        log_handle.close()
    if any(return_codes):
        raise SystemExit(max(return_codes))
    merge_parts(
        args.manifest,
        [out_dir / f"part{shard}.json" for shard in range(args.num_gpus)],
        out_dir,
    )


def command_merge_mlr(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    records: list[dict[str, Any]] = []
    for path in args.records:
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    temporary_dir = args.out_dir.resolve()
    temporary_dir.mkdir(parents=True, exist_ok=True)
    combined = temporary_dir / "combined_parts.json"
    write_json(combined, records)
    merge_parts(manifest, [combined], temporary_dir)


def percentile(values: list[float], quantile: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    position = (len(clean) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def relative_percent(method: float, control: float) -> float | None:
    if control == 0:
        return None
    return (method - control) / control * 100.0


def arrow_percent(value: float | None) -> str:
    if value is None:
        return "undefined"
    if abs(value) < 0.05:
        return "0.0%"
    return ("↑" if value > 0 else "↓") + f"{abs(value):.1f}%"


def load_gemini_repeat(path: Path, expected: set[str]) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError(
            f"Gemini coverage mismatch at {path}: rows={len(keys)}, "
            f"unique={len(set(keys))}, missing={len(expected - set(keys))}"
        )
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise RuntimeError(f"Gemini errors remain at {path}: {len(errors)}")
    return {row["key"]: int(float(row["prediction"])) for row in rows}


def bootstrap_route(
    pairs: list[dict[str, Any]], repeats: int, samples: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    if_changes: list[float] = []
    mlr_changes: list[float] = []
    if_pp: list[float] = []
    mlr_pp: list[float] = []
    for _ in range(samples):
        drawn = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        control_if = sum(sum(pair["gemini_control"]) for pair in drawn) / (len(drawn) * repeats)
        method_if = sum(sum(pair["gemini_method"]) for pair in drawn) / (len(drawn) * repeats)
        if_pp.append((method_if - control_if) * 100.0)
        change = relative_percent(method_if, control_if)
        if change is not None:
            if_changes.append(change)

        control_eligible = [pair["mlr_control"] for pair in drawn if not pair["mlr_control"]["undetectable"]]
        method_eligible = [pair["mlr_method"] for pair in drawn if not pair["mlr_method"]["undetectable"]]
        if control_eligible and method_eligible:
            control_mlr = sum(bool(row["dup"]) for row in control_eligible) / len(control_eligible)
            method_mlr = sum(bool(row["dup"]) for row in method_eligible) / len(method_eligible)
            mlr_pp.append((method_mlr - control_mlr) * 100.0)
            change = relative_percent(method_mlr, control_mlr)
            if change is not None:
                mlr_changes.append(change)

    return {
        "samples": samples,
        "relative_if_change_ci95": [percentile(if_changes, 0.025), percentile(if_changes, 0.975)],
        "if_change_pp_ci95": [percentile(if_pp, 0.025), percentile(if_pp, 0.975)],
        "relative_mlr_change_ci95": [percentile(mlr_changes, 0.025), percentile(mlr_changes, 0.975)],
        "mlr_change_pp_ci95": [percentile(mlr_pp, 0.025), percentile(mlr_pp, 0.975)],
        "relative_mlr_defined_samples": len(mlr_changes),
    }


def command_aggregate(args: argparse.Namespace) -> None:
    manifest_rows = load_jsonl(args.manifest)
    manifest = {row["key"]: row for row in manifest_rows}
    expected = set(manifest)
    mlr_records = json.loads(args.mlr_records.read_text(encoding="utf-8"))
    mlr = {row["key"]: row for row in mlr_records}
    if set(mlr) != expected or len(mlr_records) != len(mlr):
        raise RuntimeError("MLR records do not exactly cover the frozen manifest")
    mlr_errors = [row for row in mlr_records if row.get("error")]
    if mlr_errors:
        raise RuntimeError(f"MLR errors remain: {len(mlr_errors)}")

    repeat_paths = sorted(args.gemini_root.glob("repeat*/qwen_if.csv"))
    if len(repeat_paths) != 3:
        raise RuntimeError(f"expected 3 Gemini repeats, found {len(repeat_paths)}")
    gemini_repeats = [load_gemini_repeat(path, expected) for path in repeat_paths]

    routes = sorted({row["route"] for row in manifest_rows})
    final: dict[str, Any] = {
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "records": len(manifest_rows),
            "pairs": len(manifest_rows) // 2,
            "gemini_model": "gemini-3.6-flash",
            "gemini_repeats": len(repeat_paths),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_unit": "matched pair_id; Gemini repeats averaged within pair",
        },
        "routes": {},
    }
    csv_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    for route_index, route in enumerate(routes):
        route_rows = [row for row in manifest_rows if row["route"] == route]
        pair_ids = sorted({row["pair_id"] for row in route_rows})
        pairs: list[dict[str, Any]] = []
        for pair_id in pair_ids:
            pair_rows = {row["arm"]: row for row in route_rows if row["pair_id"] == pair_id}
            if set(pair_rows) != {"control", "method"}:
                raise RuntimeError(f"unmatched aggregate pair: {pair_id}")
            pairs.append(
                {
                    "pair_id": pair_id,
                    "gemini_control": [repeat[pair_rows["control"]["key"]] for repeat in gemini_repeats],
                    "gemini_method": [repeat[pair_rows["method"]["key"]] for repeat in gemini_repeats],
                    "mlr_control": mlr[pair_rows["control"]["key"]],
                    "mlr_method": mlr[pair_rows["method"]["key"]],
                }
            )

        arm_summary: dict[str, Any] = {}
        for arm in ("control", "method"):
            arm_rows = [row for row in route_rows if row["arm"] == arm]
            repeat_scores = [
                sum(repeat[row["key"]] for row in arm_rows) / len(arm_rows)
                for repeat in gemini_repeats
            ]
            eligible = [mlr[row["key"]] for row in arm_rows if not mlr[row["key"]]["undetectable"]]
            arm_summary[arm] = {
                "records": len(arm_rows),
                "gemini_if": sum(repeat_scores) / len(repeat_scores),
                "gemini_if_repeat_scores": repeat_scores,
                "gemini_if_repeat_std": statistics.stdev(repeat_scores),
                "mlr_eligible": len(eligible),
                "mlr_undetectable": len(arm_rows) - len(eligible),
                "mlr_dup_events": sum(bool(row["dup"]) for row in eligible),
                "mlr": sum(bool(row["dup"]) for row in eligible) / len(eligible) if eligible else None,
                "vanish_events": sum(bool(row["vanish"]) for row in eligible),
                "vanish_rate": sum(bool(row["vanish"]) for row in eligible) / len(eligible) if eligible else None,
            }

        control = arm_summary["control"]
        method = arm_summary["method"]
        if_relative = relative_percent(method["gemini_if"], control["gemini_if"])
        mlr_relative = relative_percent(method["mlr"], control["mlr"])
        comparison = {
            "relative_instruction_fidelity_change_percent": if_relative,
            "instruction_fidelity_change_pp": (method["gemini_if"] - control["gemini_if"]) * 100.0,
            "relative_process_laziness_change_percent": mlr_relative,
            "process_laziness_change_pp": (method["mlr"] - control["mlr"]) * 100.0,
            "display_instruction_fidelity_change": arrow_percent(if_relative),
            "display_process_laziness_change": arrow_percent(mlr_relative),
        }
        comparison["paired_bootstrap"] = bootstrap_route(
            pairs,
            repeats=len(gemini_repeats),
            samples=args.bootstrap_samples,
            seed=args.seed + route_index,
        )
        route_name = route_rows[0]["route_name"]
        final["routes"][route] = {
            "route_name": route_name,
            "pairs": len(pairs),
            "control": control,
            "method": method,
            "comparison": comparison,
        }
        csv_rows.append(
            {
                "route": route_name,
                "pairs": len(pairs),
                "control_mlr": control["mlr"],
                "method_mlr": method["mlr"],
                "control_dup_events_eligible": f"{control['mlr_dup_events']}/{control['mlr_eligible']}",
                "method_dup_events_eligible": f"{method['mlr_dup_events']}/{method['mlr_eligible']}",
                "relative_process_laziness_change_percent": mlr_relative,
                "control_gemini_if": control["gemini_if"],
                "method_gemini_if": method["gemini_if"],
                "relative_instruction_fidelity_change_percent": if_relative,
                "gemini_control_repeat_std": control["gemini_if_repeat_std"],
                "gemini_method_repeat_std": method["gemini_if_repeat_std"],
            }
        )
        compact_rows.append(
            {
                "route": route_name,
                "relative_process_laziness_change": comparison["display_process_laziness_change"],
                "relative_instruction_fidelity_change": comparison["display_instruction_fidelity_change"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "final_summary.json", final)
    for filename, rows in (("table.csv", csv_rows), ("table_compact.csv", compact_rows)):
        with (args.out_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--audit-workers", type=int, default=24)
    build.set_defaults(func=command_build)

    shard = subparsers.add_parser("mlr-shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--num-shards", type=int, required=True)
    shard.add_argument("--out", type=Path, required=True)
    shard.set_defaults(func=command_mlr_shard)

    worker = subparsers.add_parser("mlr-worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--out-dir", type=Path, required=True)
    worker.add_argument("--num-gpus", type=int, default=8)
    worker.add_argument("--python", default=sys.executable)
    worker.add_argument("--poll-seconds", type=int, default=30)
    worker.set_defaults(func=command_mlr_worker)

    merge = subparsers.add_parser("merge-mlr")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--records", type=Path, action="append", required=True)
    merge.add_argument("--out-dir", type=Path, required=True)
    merge.set_defaults(func=command_merge_mlr)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--mlr-records", type=Path, required=True)
    aggregate.add_argument("--gemini-root", type=Path, required=True)
    aggregate.add_argument("--out-dir", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10000)
    aggregate.add_argument("--seed", type=int, default=20260901)
    aggregate.set_defaults(func=command_aggregate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
