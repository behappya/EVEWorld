#!/usr/bin/env python3
"""Tier table and frame arithmetic of the long-horizon length sweep.

PBench rollouts are generated at seven robot-length tiers that stretch the
generation horizon from 3.8 s to 29.8 s at 16 FPS; the same sweep prices the
horizon, since the length-scaling table reports latency, throughput, memory
and GPU cost per tier. This script is the arithmetic behind the sweep: for
every tier it prints the duration, the fit of the frame count to the 4-frame
temporal stride of the video VAE and the latent length that follows, so the
tier geometry can be re-derived without a GPU. Only the standard library is
used.

The tier list is the published one, 61, 93, 157, 253, 317, 413 and 477 frames
(``configs/eval/pbench.yaml``, ``docs/evaluation.md``). The latent length
follows the backbone helper ``latent_frames``
(``src/eveworld/integrations/gigaworld/model.py``): ``1 + (frames - 1) // 4``.

Run it:

```bash
python experiments/analysis/long_horizon/lengths.py
```
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Sequence

__all__ = ["main", "latent_frames", "seconds"]

FPS = 16.0
TEMPORAL_STRIDE = 4
HORIZON_TIERS = (61, 93, 157, 253, 317, 413, 477)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fps",
        type=float,
        default=FPS,
        help="frame rate the durations are derived from (default: %(default)s)",
    )
    parser.add_argument("--out", default=None, help="optional file for the markdown table")
    return parser.parse_args(argv)


def seconds(frames: int, fps: float) -> float:
    """Duration of a clip of ``frames`` frames at ``fps`` frames per second."""
    return frames / fps


def latent_frames(frames: int, stride: int = TEMPORAL_STRIDE) -> int:
    """Latent length of a clip, ``1 + (frames - 1) // stride``."""
    return 1 + (frames - 1) // stride


def _table(tiers: Sequence[int], fps: float) -> str:
    lines = ["| Frames | Seconds | (frames - 1) % 4 | Latent frames |"]
    lines.append("|---:|---:|---:|---:|")
    for frames in tiers:
        lines.append(
            f"| {frames} | {seconds(frames, fps):.1f} "
            f"| {(frames - 1) % TEMPORAL_STRIDE} | {latent_frames(frames)} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise SystemExit(f"--fps must be a positive number, got {args.fps}")
    tiers = HORIZON_TIERS
    aligned = sum(1 for frames in tiers if (frames - 1) % TEMPORAL_STRIDE == 0)
    lines = [
        f"# Long-horizon length tiers: {tiers[0]}-{tiers[-1]} frames at {args.fps:g} FPS",
        "",
        _table(tiers, args.fps),
        "",
        f"tiers: {len(tiers)}",
        f"durations: {seconds(tiers[0], args.fps):.1f} to {seconds(tiers[-1], args.fps):.1f} s",
        f"latent frames: {latent_frames(tiers[0])} to {latent_frames(tiers[-1])}",
        f"stride alignment: {aligned}/{len(tiers)} tiers satisfy (frames - 1) % {TEMPORAL_STRIDE} == 0",
    ]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
