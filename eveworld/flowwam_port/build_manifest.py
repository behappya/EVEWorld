#!/usr/bin/env python3
"""FlowWAM_WorldArena 640 档 episode manifest 构建。

扫描 640_extracted/<task>/aloha-agilex_clean_50/，每条 episode 输出:
  task, episode, video, robot_only_video, hdf5, instruction(取 seen[0]),
  instructions_path, target_asset({A} 资产 ID), target_name(资产 ID -> 物体名),
  arm({a}), frames, width, height
供 IGR 定位(GDINO prompt 用 target_name)与训练数据管线消费。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2


def asset_to_name(asset: str | None) -> str | None:
    """'020_hammer/base0' -> 'hammer'; '071_can/can3' -> 'can'。"""
    if not asset:
        return None
    stem = asset.split("/")[0]
    m = re.match(r"^\d+_(.+)$", stem)
    name = (m.group(1) if m else stem).replace("_", " ").strip()
    return name or None


def probe_video(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/datasets/gagi/flowwam/data_worldarena/640_extracted")
    ap.add_argument("--output", default="/data/datasets/gagi/flowwam/igr/manifest_640.json")
    ap.add_argument("--probe-videos", action="store_true", help="逐条解码探测帧数(慢)")
    args = ap.parse_args()

    root = Path(args.root)
    rows, problems = [], []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        base = task_dir / "aloha-agilex_clean_50"
        if not base.is_dir():
            problems.append(f"{task_dir.name}: missing aloha-agilex_clean_50")
            continue
        scene_info = {}
        si_path = base / "scene_info.json"
        if si_path.exists():
            scene_info = json.loads(si_path.read_text())
        for vid in sorted((base / "video").glob("episode*.mp4"),
                          key=lambda p: int(re.search(r"\d+", p.stem).group())):
            ep = vid.stem  # episode0
            info = scene_info.get(f"episode_{ep.replace('episode', '')}", {}).get("info", {})
            target_asset = info.get("{A}")
            arm = info.get("{a}")
            inst_path = base / "instructions" / f"{ep}.json"
            instruction = None
            if inst_path.exists():
                d = json.loads(inst_path.read_text())
                seen = d.get("seen") or []
                instruction = seen[0] if seen else d.get("unseen", [None])[0]
            # scene_info 为空时从任务名兜底 (handover_block -> "block")
            if not target_asset:
                tail = task_dir.name.split("_")[-1]
                fallback = {"block": "block"}.get(tail, tail)
                target_fallback = fallback.replace("_", " ")
            row = {
                "task": task_dir.name,
                "episode": ep,
                "video": str(vid),
                "robot_only_video": str(base / "robot_only" / "video" / f"{ep}.mp4"),
                "hdf5": str(base / "data" / f"{ep}.hdf5"),
                "instructions_path": str(inst_path),
                "instruction": instruction,
                "target_asset": target_asset,
                "target_name": asset_to_name(target_asset) or (target_fallback if not target_asset else None),
                "arm": arm,
            }
            if args.probe_videos:
                row["frames"], row["width"], row["height"] = probe_video(vid)
            if instruction is None:
                problems.append(f"{task_dir.name}/{ep}: no instruction")
            rows.append(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    n_target = sum(1 for r in rows if r["target_name"])
    tasks = sorted({r["task"] for r in rows})
    print(f"episodes={len(rows)} tasks={len(tasks)} with_target_name={n_target}")
    print(f"problems={len(problems)}")
    for p in problems[:10]:
        print("  !", p)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
