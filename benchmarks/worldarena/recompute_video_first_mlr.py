#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("pretrain", "round0", "t4g_wmapA_pre_seed42_s250")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = events / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - margin, center + margin]


def exact_mcnemar(discordant_a: int, discordant_b: int) -> float:
    discordant = discordant_a + discordant_b
    tail = min(discordant_a, discordant_b)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def video_first_event(counts: list[int]) -> bool:
    initial_count = counts[0]
    consecutive = 0
    for count in counts[1:]:
        consecutive = consecutive + 1 if count >= initial_count + 1 else 0
        if consecutive >= 2:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute WorldArena 1.0 MLR using each generated video's first frame"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--eveworld-model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eveworld_model = args.eveworld_model
    if eveworld_model is None and tuple(args.models) == DEFAULT_MODELS:
        eveworld_model = DEFAULT_MODELS[-1]
    source = json.loads(args.input.read_text(encoding="utf-8"))
    frame_count = len(next(record["counts"] for record in source["records"] if record["counts"]))
    records_by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in source["records"]:
        if record["model"] in args.models and record["counts"]:
            records_by_model[record["model"]][record["request_id"]] = record

    common_ids = sorted(
        request_id
        for request_id in set.intersection(
            *(set(records_by_model[model]) for model in args.models)
        )
        if all(records_by_model[model][request_id]["counts"][0] > 0 for model in args.models)
    )

    event_ids: dict[str, set[str]] = {}
    summary: dict[str, Any] = {}
    output_records = []
    for model in args.models:
        model_events = {
            request_id
            for request_id in common_ids
            if video_first_event(records_by_model[model][request_id]["counts"])
        }
        event_ids[model] = model_events
        events = len(model_events)
        summary[model] = {
            "events": events,
            "eligible": len(common_ids),
            "mlr": events / len(common_ids),
            "wilson_95": wilson_interval(events, len(common_ids)),
        }
        for request_id in common_ids:
            source_record = records_by_model[model][request_id]
            output_records.append(
                {
                    "model": model,
                    "request_id": request_id,
                    "mover": source_record["mover"],
                    "initial_count": source_record["counts"][0],
                    "counts": source_record["counts"],
                    "mlr_event": request_id in model_events,
                }
            )

    comparisons = {}
    if eveworld_model:
        if eveworld_model not in args.models:
            raise ValueError("--eveworld-model must be included in --models")
        comparison_prefix = "eveworld" if eveworld_model == DEFAULT_MODELS[-1] else eveworld_model
        for baseline in (model for model in args.models if model != eveworld_model):
            baseline_only = len(event_ids[baseline] - event_ids[eveworld_model])
            eveworld_only = len(event_ids[eveworld_model] - event_ids[baseline])
            baseline_mlr = summary[baseline]["mlr"]
            relative_reduction = (
                1 - summary[eveworld_model]["mlr"] / baseline_mlr
                if baseline_mlr > 0
                else None
            )
            comparisons[f"{comparison_prefix}_vs_{baseline}"] = {
                "relative_reduction": relative_reduction,
                "baseline_only_events": baseline_only,
                "eveworld_only_events": eveworld_only,
                "exact_mcnemar_two_sided_p": exact_mcnemar(baseline_only, eveworld_only),
            }

    payload = {
        "metadata": {
            "protocol": "worldarena1_mlr_video_first_v1",
            "source": str(args.input),
            "inventory_anchor": "first uniformly sampled frame of each generated video",
            "selected_prompt_count": source["metadata"]["selected_prompt_count"],
            "eligibility": "target detected in the generated first frame for every requested model",
            "frame_count": frame_count,
            "consecutive_frames": 2,
            "models": list(args.models),
        },
        "summary": summary,
        "comparisons": comparisons,
        "records": output_records,
    }
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
