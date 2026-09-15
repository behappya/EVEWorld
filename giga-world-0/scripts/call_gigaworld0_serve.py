import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a running GigaWorld-0 HTTP service.")
    parser.add_argument("--url", required=True, help="Base URL, for example http://127.0.0.1:8000")
    parser.add_argument("--prompt", default=None, help="Prompt for a single request.")
    parser.add_argument("--image-path", default=None, help="Shared filesystem image path for a single request.")
    parser.add_argument("--request-id", default=None, help="Optional output id for a single request.")
    parser.add_argument("--data-path", default=None, help="GigaWorld JSON input file with prompt/image rows.")
    parser.add_argument("--save-dir", required=True, help="Shared filesystem output directory for generated mp4 files.")
    parser.add_argument("--limit", type=int, default=0, help="Number of rows to send from --data-path. 0 means all rows.")
    parser.add_argument("--offset", type=int, default=0, help="Start row for --data-path.")
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--summary-path", default=None, help="Optional JSON summary path. Defaults to SAVE_DIR/call_summary.json.")
    return parser.parse_args()


def normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        raise SystemExit("Empty --url. Set URL=http://HOST:PORT or pass --url http://HOST:PORT directly.")
    if "://" not in url:
        url = f"http://{url}"
    return url


def request_id_for_row(index: int, row: dict) -> str:
    for key in ("pbench_id", "id", "request_id"):
        if row.get(key):
            return str(row[key])
    image = row.get("image") or row.get("image_path") or f"sample_{index:06d}"
    return Path(str(image)).stem or f"sample_{index:06d}"


def resolve_image_path(data_path: Path, row: dict) -> str:
    image = row.get("image_path") or row.get("image")
    if not image:
        raise ValueError("row is missing image/image_path")
    image_path = Path(str(image)).expanduser()
    if not image_path.is_absolute():
        image_path = data_path.parent / image_path
    return str(image_path.resolve())


def build_payload(args: argparse.Namespace, prompt: str, image_path: str, request_id: str) -> dict:
    return {
        "prompt": prompt,
        "image_path": image_path,
        "save_dir": str(Path(args.save_dir).expanduser().resolve()),
        "request_id": request_id,
        "num_inference_steps": args.num_inference_steps,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
    }


def send_one(url: str, payload: dict, timeout: float) -> dict:
    response = requests.post(f"{url}/generate", json=payload, timeout=timeout)
    try:
        result = response.json()
    except Exception:
        result = {"ok": False, "status_code": response.status_code, "text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"request failed HTTP {response.status_code}: {result}")
    return result


def nested_number(data: dict, path: tuple[str, ...]) -> float | None:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def max_nested(results: list[dict], path: tuple[str, ...]) -> float | None:
    values = [value for result in results if (value := nested_number(result, path)) is not None]
    return max(values) if values else None


def write_summary(
    path: Path,
    *,
    url: str,
    health: dict,
    save_dir: Path,
    results_path: Path,
    results: list[dict],
    wall_time_sec: float,
) -> None:
    elapsed_values = [float(result["elapsed_sec"]) for result in results if isinstance(result.get("elapsed_sec"), (int, float))]
    ok_count = sum(1 for result in results if result.get("ok") is True)
    summary = {
        "url": url,
        "save_dir": str(save_dir),
        "results_path": str(results_path),
        "request_count": len(results),
        "ok_count": ok_count,
        "wall_time_sec": round(wall_time_sec, 3),
        "model_elapsed_total_sec": round(sum(elapsed_values), 3) if elapsed_values else None,
        "model_elapsed_mean_sec": round(statistics.mean(elapsed_values), 3) if elapsed_values else None,
        "model_elapsed_median_sec": round(statistics.median(elapsed_values), 3) if elapsed_values else None,
        "model_elapsed_min_sec": round(min(elapsed_values), 3) if elapsed_values else None,
        "model_elapsed_max_sec": round(max(elapsed_values), 3) if elapsed_values else None,
        "cuda_memory": {
            "max_request_torch_allocated_mib": max_nested(results, ("cuda_memory", "after", "max_memory_allocated_mib")),
            "max_request_torch_reserved_mib": max_nested(results, ("cuda_memory", "after", "max_memory_reserved_mib")),
            "max_after_cuda_mem_used_mib": max_nested(results, ("cuda_memory", "after", "cuda_mem_used_mib")),
            "note": "These values are aggregated from service responses. The kjob GPU monitor records device-level nvidia-smi peaks separately.",
        },
        "health": health,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary json: {path}", flush=True)


def main() -> None:
    args = parse_args()
    url = normalize_url(args.url)
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "call_results.jsonl"
    summary_path = Path(args.summary_path).expanduser().resolve() if args.summary_path else save_dir / "call_summary.json"

    health = requests.get(f"{url}/health", timeout=30).json()
    print(json.dumps({"health": health}, ensure_ascii=False), flush=True)

    requests_to_send = []
    if args.data_path:
        data_path = Path(args.data_path).expanduser().resolve()
        data = json.load(open(data_path, "r"))
        start = args.offset
        end = None if args.limit <= 0 else start + args.limit
        for index, row in enumerate(data[start:end], start=start):
            prompt = row.get("prompt")
            if not prompt:
                raise ValueError(f"row {index} is missing prompt")
            image_path = resolve_image_path(data_path, row)
            requests_to_send.append(build_payload(args, prompt, image_path, request_id_for_row(index, row)))
    else:
        if not args.prompt or not args.image_path:
            raise SystemExit("Use either --data-path or both --prompt and --image-path.")
        image_path = str(Path(args.image_path).expanduser().resolve())
        request_id = args.request_id or Path(image_path).stem
        requests_to_send.append(build_payload(args, args.prompt, image_path, request_id))

    print(f"sending {len(requests_to_send)} request(s) to {url}", flush=True)
    run_start = time.time()
    results = []
    with results_path.open("a", encoding="utf-8") as f:
        for index, payload in enumerate(requests_to_send):
            print(f"[{index + 1}/{len(requests_to_send)}] request_id={payload['request_id']}", flush=True)
            result = send_one(url, payload, timeout=args.timeout)
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)

    print(f"results jsonl: {results_path}", flush=True)
    write_summary(
        summary_path,
        url=url,
        health=health,
        save_dir=save_dir,
        results_path=results_path,
        results=results,
        wall_time_sec=time.time() - run_start,
    )


if __name__ == "__main__":
    main()
