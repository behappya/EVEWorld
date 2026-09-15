"""Shared manifest selection and image preparation for held-out generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from giga_datasets import image_utils


def _row_id(row: dict) -> str:
    for key in ("source_file_name", "file_name", "request_id", "id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError(f"row has no stable id: {row}")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty split manifest: {path}")
    return rows


def select_manifest_rows(
    data_path: Path,
    split_manifest: Path,
    expected_split: str,
    limit: int,
) -> list[dict]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{data_path} must contain a JSON list")
    by_id: dict[str, dict] = {}
    for row in data:
        key = _row_id(row)
        if key in by_id:
            raise ValueError(f"duplicate data id {key!r} in {data_path}")
        by_id[key] = row

    split = _read_jsonl(split_manifest)
    split = split[:limit] if limit > 0 else split
    output = []
    seen = set()
    for item in split:
        key = _row_id(item)
        if key in seen:
            raise ValueError(f"duplicate manifest id {key!r}")
        seen.add(key)
        if str(item.get("split", "")) != expected_split:
            raise ValueError(
                f"manifest row {key!r} has split={item.get('split')!r}, "
                f"expected {expected_split!r}"
            )
        if key not in by_id:
            raise KeyError(f"{key!r} is absent from {data_path}")
        row = dict(by_id[key])
        row["manifest_id"] = key
        row["packed_index"] = item.get("packed_index")
        output.append(row)
    return output


def clean_output_id(row: dict) -> str:
    stem = Path(str(row["manifest_id"])).stem
    prompt = re.sub(r"[^A-Za-z0-9]+", "_", str(row["prompt"]).strip()).strip("_")
    return f"{stem}_{prompt[:120]}"


def load_input_image(row: dict, data_path: Path, width: int, height: int) -> Image.Image:
    image_path = Path(str(row["image"]))
    if not image_path.is_file():
        image_path = data_path.parent / image_path
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    dst_width, dst_height = image_utils.get_image_size(
        (image_width, image_height), (width, height), mode="area", multiple=16
    )
    if float(dst_height) / image_height < float(dst_width) / image_width:
        new_height = int(round(float(dst_width) / image_width * image_height))
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / image_height * image_width))
    resized = F.resize(image, (new_height, new_width), InterpolationMode.BILINEAR)
    return F.crop(
        resized,
        (new_height - dst_height) // 2,
        (new_width - dst_width) // 2,
        dst_height,
        dst_width,
    )
