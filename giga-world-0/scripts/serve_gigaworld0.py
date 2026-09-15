import argparse
import os
import socket
import threading
import time
import traceback
from pathlib import Path

import imageio
import torch
from accelerate.utils import set_seed
from flask import Flask, jsonify, request
from giga_datasets import image_utils
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from giga_models import GigaWorld0Pipeline


NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GigaWorld-0 image-to-video inference over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transformer-model-path", required=True)
    parser.add_argument("--text-encoder-model-path", required=True)
    parser.add_argument("--vae-model-path", required=True)
    parser.add_argument("--default-save-dir", default="/data/datasets/gagi/giga_world_0_outputs/gigaworld0_serving/results")
    parser.add_argument("--default-num-inference-steps", type=int, default=30)
    parser.add_argument("--default-fps", type=int, default=16)
    parser.add_argument("--default-num-frames", type=int, default=61)
    parser.add_argument("--default-height", type=int, default=480)
    parser.add_argument("--default-width", type=int, default=640)
    parser.add_argument("--default-seed", type=int, default=6666)
    return parser.parse_args()


def local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        return "unknown"


def clean_request_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned or str(int(time.time() * 1000))


def cuda_info() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            info["cuda_mem_free"] = free_bytes
            info["cuda_mem_total"] = total_bytes
        except RuntimeError:
            pass
    return info


def bytes_to_mib(value: int) -> float:
    return round(value / 1024 / 1024, 3)


def cuda_memory_snapshot(device: str | int | torch.device) -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    try:
        torch.cuda.synchronize(device)
    except RuntimeError:
        pass

    free_bytes = None
    total_bytes = None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except RuntimeError:
        pass

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    snapshot = {
        "cuda_available": True,
        "device": str(device),
        "memory_allocated_bytes": allocated,
        "memory_reserved_bytes": reserved,
        "max_memory_allocated_bytes": max_allocated,
        "max_memory_reserved_bytes": max_reserved,
        "memory_allocated_mib": bytes_to_mib(allocated),
        "memory_reserved_mib": bytes_to_mib(reserved),
        "max_memory_allocated_mib": bytes_to_mib(max_allocated),
        "max_memory_reserved_mib": bytes_to_mib(max_reserved),
    }
    if free_bytes is not None and total_bytes is not None:
        used_bytes = total_bytes - free_bytes
        snapshot.update(
            {
                "cuda_mem_free_bytes": free_bytes,
                "cuda_mem_total_bytes": total_bytes,
                "cuda_mem_used_bytes": used_bytes,
                "cuda_mem_free_mib": bytes_to_mib(free_bytes),
                "cuda_mem_total_mib": bytes_to_mib(total_bytes),
                "cuda_mem_used_mib": bytes_to_mib(used_bytes),
            }
        )
    return snapshot


class GigaWorld0Server:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.pipe = None
        self.load_pipeline()

    def load_pipeline(self) -> None:
        print(f"[gw0-serve] loading pipeline on {self.args.device}", flush=True)
        torch.cuda.set_device(self.args.device)
        pipe = GigaWorld0Pipeline.from_pretrained(
            transformer_model_path=self.args.transformer_model_path,
            text_encoder_model_path=self.args.text_encoder_model_path,
            vae_model_path=self.args.vae_model_path,
        )
        print(f"[gw0-serve] moving pipeline to {self.args.device}", flush=True)
        pipe.to(self.args.device)
        self.pipe = pipe
        print("[gw0-serve] pipeline ready", flush=True)

    def health(self) -> dict:
        return {
            "ok": True,
            "loaded": self.pipe is not None,
            "hostname": socket.gethostname(),
            "local_ip": local_ip(),
            "uptime_sec": round(time.time() - self.started_at, 3),
            "device": self.args.device,
            **cuda_info(),
        }

    def _prepare_image(self, image_path: str, height: int, width: int) -> Image.Image:
        path = Path(image_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"image_path does not exist: {image_path}")
        image = Image.open(path).convert("RGB")
        image_width, image_height = image.width, image.height
        dst_width, dst_height = image_utils.get_image_size(
            (image_width, image_height),
            (width, height),
            mode="area",
            multiple=16,
        )
        if float(dst_height) / image_height < float(dst_width) / image_width:
            new_height = int(round(float(dst_width) / image_width * image_height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / image_height * image_width))
        if dst_width > new_width or dst_height > new_height:
            raise ValueError(f"invalid resize/crop target: dst={dst_width}x{dst_height} new={new_width}x{new_height}")
        x1 = (new_width - dst_width) // 2
        y1 = (new_height - dst_height) // 2
        input_image = F.resize(image, (new_height, new_width), InterpolationMode.BILINEAR)
        return F.crop(input_image, y1, x1, dst_height, dst_width)

    def generate(self, payload: dict) -> dict:
        prompt = payload.get("prompt")
        image_path = payload.get("image_path")
        if not prompt or not isinstance(prompt, str):
            raise ValueError("payload.prompt must be a non-empty string")
        if not image_path or not isinstance(image_path, str):
            raise ValueError("payload.image_path must be a non-empty shared filesystem path")

        save_dir = Path(payload.get("save_dir") or self.args.default_save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)

        request_id = clean_request_id(str(payload.get("request_id") or Path(image_path).stem))
        fps = int(payload.get("fps", self.args.default_fps))
        num_frames = int(payload.get("num_frames", self.args.default_num_frames))
        height = int(payload.get("height", self.args.default_height))
        width = int(payload.get("width", self.args.default_width))
        seed = int(payload.get("seed", self.args.default_seed))
        num_inference_steps = int(payload.get("num_inference_steps", self.args.default_num_inference_steps))
        negative_prompt = payload.get("negative_prompt") or NEGATIVE_PROMPT

        print(
            f"[gw0-serve] request={request_id} steps={num_inference_steps} "
            f"frames={num_frames} fps={fps} size={height}x{width} image={image_path}",
            flush=True,
        )
        start = time.time()
        with self.lock:
            cuda_before = {}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.args.device)
                cuda_before = cuda_memory_snapshot(self.args.device)
            set_seed(seed)
            input_image = self._prepare_image(image_path, height=height, width=width)
            output_images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                num_inference_steps=num_inference_steps,
                fps=fps,
                num_frames=num_frames,
                height=input_image.height,
                width=input_image.width,
                seed=seed,
            )[0]
            cuda_after = cuda_memory_snapshot(self.args.device)

        vis_images = []
        for frame in output_images:
            vis_images.append(image_utils.concat_images_grid([input_image, frame], cols=2, pad=2))

        video_path = save_dir / f"{request_id}.mp4"
        imageio.mimsave(video_path, vis_images, fps=fps)
        elapsed = time.time() - start
        print(f"[gw0-serve] request={request_id} saved={video_path} elapsed={elapsed:.2f}s", flush=True)
        return {
            "ok": True,
            "request_id": request_id,
            "video_path": str(video_path),
            "elapsed_sec": round(elapsed, 3),
            "num_inference_steps": num_inference_steps,
            "fps": fps,
            "num_frames": num_frames,
            "height": input_image.height,
            "width": input_image.width,
            "seed": seed,
            "cuda_memory": {
                "before": cuda_before,
                "after": cuda_after,
                "note": "PyTorch peak stats are reset immediately before this request; nvidia-smi device-level peaks are recorded by the kjob GPU monitor when enabled.",
            },
        }


def create_app(server: GigaWorld0Server) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(server.health())

    @app.post("/generate")
    def generate():
        try:
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            result = server.generate(payload)
            return jsonify(result)
        except Exception as exc:  # noqa: BLE001
            print("[gw0-serve] request failed", flush=True)
            traceback.print_exc()
            return jsonify({"ok": False, "error": str(exc), "error_type": type(exc).__name__}), 500

    return app


def main() -> None:
    args = parse_args()
    if os.environ.get("ALLOW_LOCAL_RUN") != "1" and socket.gethostname().startswith("coder-workspace-"):
        raise SystemExit(
            "Refusing to load GigaWorld-0 on the workspace host. "
            "Submit the service with ./scripts/launch_gigaworld0_serve_kjob.sh, "
            "or set ALLOW_LOCAL_RUN=1 if this is intentional."
        )
    print("============================================", flush=True)
    print("GigaWorld-0 HTTP serving", flush=True)
    print(f"Host:                    {socket.gethostname()}", flush=True)
    print(f"Local IP:                {local_ip()}", flush=True)
    print(f"Serving URL:             http://{local_ip()}:{args.port}", flush=True)
    print(f"Bind:                    {args.host}:{args.port}", flush=True)
    print(f"Device:                  {args.device}", flush=True)
    print(f"Transformer model path:  {args.transformer_model_path}", flush=True)
    print(f"Text encoder model path: {args.text_encoder_model_path}", flush=True)
    print(f"VAE model path:          {args.vae_model_path}", flush=True)
    print(f"Default save dir:        {args.default_save_dir}", flush=True)
    print("============================================", flush=True)
    server = GigaWorld0Server(args)
    app = create_app(server)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
