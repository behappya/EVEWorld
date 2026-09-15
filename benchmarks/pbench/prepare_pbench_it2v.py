import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA PBench parquet rows into GigaWorld-0 image-to-video JSON input."
    )
    parser.add_argument(
        "--parquet",
        default="/home/jovyan/gagibench/pbench_raw/data/pbench.parquet",
        help="Path to nvidia/PBench data/pbench.parquet.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/jovyan/gagibench/pbench/giga_input",
        help="Directory where images, JSON input, and metadata will be written.",
    )
    parser.add_argument(
        "--subset",
        default="robot",
        help="PBench id prefix to export, for example robot/av/human/industry/physics/common. Use all for all rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum rows to export after subset filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Rewrite extracted conditioning images even when the target files already exist.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path. Defaults to pbench_<subset>_it2v.json under output-root.",
    )
    return parser.parse_args()


def parse_qa_pairs(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return raw


def extract_image(condition_image: dict, image_path: Path, overwrite: bool) -> None:
    if image_path.exists() and not overwrite:
        return

    if not isinstance(condition_image, dict) or not condition_image.get("bytes"):
        raise ValueError(f"Unsupported condition_image payload for {image_path}")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(io.BytesIO(condition_image["bytes"])).convert("RGB")
    image.save(image_path, format="JPEG", quality=95)


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    image_dir = output_root / "images"
    subset = args.subset.strip()

    df = pd.read_parquet(parquet_path)
    if subset and subset != "all":
        df = df[df["id"].astype(str).str.startswith(f"{subset}_")].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    if df.empty:
        raise SystemExit(f"No PBench rows matched subset={subset!r} limit={args.limit}")

    output_root.mkdir(parents=True, exist_ok=True)
    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
    else:
        suffix = subset if subset else "all"
        if args.limit and args.limit > 0:
            suffix = f"{suffix}_first{args.limit}"
        output_json = output_root / f"pbench_{suffix}_it2v.json"
    metadata_jsonl = output_json.with_suffix(".metadata.jsonl")

    records = []
    metadata_rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        pbench_id = str(row["id"])
        image_rel = Path("images") / f"{pbench_id}.jpg"
        image_abs = image_dir / f"{pbench_id}.jpg"
        extract_image(row["condition_image"], image_abs, args.overwrite_images)

        qa_pairs = parse_qa_pairs(row["qa_pairs"])
        item = {
            "prompt": row["text_prompt"],
            "image": image_rel.as_posix(),
            "pbench_id": pbench_id,
            "pbench_index": idx,
            "qa_pairs": qa_pairs,
        }
        records.append(item)
        metadata_rows.append(
            {
                "output_mp4": f"{idx}.mp4",
                "pbench_id": pbench_id,
                "image": image_rel.as_posix(),
                "prompt": row["text_prompt"],
                "qa_pairs": qa_pairs,
            }
        )

    output_json.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    with metadata_jsonl.open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"parquet:       {parquet_path}")
    print(f"subset:        {subset}")
    print(f"rows:          {len(records)}")
    print(f"images:        {image_dir}")
    print(f"giga json:     {output_json}")
    print(f"metadata jsonl:{metadata_jsonl}")


if __name__ == "__main__":
    main()
