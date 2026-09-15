#!/usr/bin/env python3
"""Build the 50-task x episode45--49 FlowWAM fidelity manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def first_frame(video: Path, out: Path) -> None:
    if out.is_file() and out.stat().st_size > 100:
        return
    import cv2

    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read first frame: {video}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        raise RuntimeError(f"cannot write first frame: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", nargs="+", type=int, default=[45, 46, 47, 48, 49])
    args = ap.parse_args()
    tasks = sorted(p.name for p in args.data_root.iterdir() if p.is_dir())
    if len(tasks) != 50:
        raise RuntimeError(f"expected 50 task directories, got {len(tasks)}")
    rows = []
    frame_root = args.out.parent / (args.out.stem + "_first_frames")
    for task in tasks:
        base = args.data_root / task / "aloha-agilex_clean_50"
        for episode in args.episodes:
            ep = str(episode)
            video = base / "video" / f"episode{ep}.mp4"
            robot = base / "robot_only" / "video" / "head_camera" / f"episode{ep}.mp4"
            instruction = base / "instructions" / f"episode{ep}.json"
            missing = [str(p) for p in (video, robot, instruction) if not p.is_file()]
            if missing:
                raise FileNotFoundError("; ".join(missing))
            payload = json.loads(instruction.read_text(encoding="utf-8"))
            seen = payload.get("seen")
            if not isinstance(seen, list) or not seen or not isinstance(seen[0], str):
                raise RuntimeError(f"invalid seen instruction: {instruction}")
            image = frame_root / f"{task}__episode{ep}.png"
            first_frame(video, image)
            rows.append({
                "request_id": f"{task}__episode{ep}",
                "prompt": seen[0],
                "image": str(image),
                "gt_video": str(video),
                "robot_only_video": str(robot),
                "task": task,
                "episode": episode,
            })
    if len(rows) != 250 or len({r["request_id"] for r in rows}) != 250:
        raise RuntimeError(f"manifest cardinality error: {len(rows)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "tasks": len(tasks), "episodes": args.episodes,
                      "rows": len(rows), "manifest": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
