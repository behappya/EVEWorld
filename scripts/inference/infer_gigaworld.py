#!/usr/bin/env python3
"""Generate clips with a trained EVEWorld run on GigaWorld-0.

::

    python scripts/inference/infer_gigaworld.py \
        --config configs/paper/gigaworld/dreamgen/eveworld.yaml \
        --prompt-file data/metadata/dreamgenbench/target_queries.json \
        --output-dir outputs/dreamgen_eveworld/inference

``--prompt-file`` defaults to ``data.split`` of the configuration. Every request of that list
is generated with the sampling loop of the released pipeline and written as one MP4::

    <output-dir>/generated_only/<request_id>.mp4

``--num-steps`` and ``--cfg-scale`` default to ``inference.num_steps`` and
``inference.cfg_scale`` of the configuration, ``--checkpoint`` replaces ``model.checkpoint``,
either with a released backbone or with the ``checkpoint-<step>.pt`` of a fine-tuned run, and
``--seed`` replaces ``seed``; the run therefore records the effective generation settings
on its command line. ``--shard-index`` and ``--num-shards`` split the requests of the file over
several processes through ``eveworld.utils.distributed.shard_range``, so a sharded run covers
every request exactly once.

The prompt file is read in whichever of these shapes it has:

* a JSON object mapping a request id to its prompt, or to a record carrying a ``prompt`` key,
  the shape of the per-request metadata files;
* a JSON list of prompts, or of records carrying ``request_id`` and ``prompt``;
* JSON Lines, one such record or one prompt per line;
* plain text, one request per line, either ``request_id<TAB>prompt`` or a bare line.

A bare line without whitespace names the request id, the shape of the split files, and its
prompt is read from ``data.metadata`` of the configuration when that directory carries one;
without one the id stays its own prompt. Lines with whitespace are prompts and are numbered
in file order. ``--dry-run`` prints the plan and the first requests and then stops without
loading weights, writing files or touching a device.

The script loads the released backbone with ``GigaWorldModel``, so a run needs the checkpoint
of ``model.checkpoint`` below ``EVEWORLD_CHECKPOINT_ROOT`` and a GPU for anything beyond a
smoke check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

FRAME_DIR = "generated_only"
PREVIEW = 3

LAYOUT = """\
output layout:

    <output-dir>/generated_only/<request_id>.mp4    one clip per request of the prompt file

prompt file shapes: a JSON object of request id -> prompt, a JSON list, JSON Lines, or plain
text, one request per line: `request_id<TAB>prompt`, or a bare request id whose prompt comes
from `data.metadata` of the configuration.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the inference script."""
    parser = argparse.ArgumentParser(
        description="Generate clips with a trained EVEWorld run on GigaWorld-0.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    run = parser.add_argument_group("run")
    run.add_argument(
        "--config",
        default=os.environ.get("EVEWORLD_CONFIG"),
        help=(
            "run configuration, e.g. configs/paper/gigaworld/dreamgen/eveworld.yaml "
            "(default: $EVEWORLD_CONFIG)"
        ),
    )
    run.add_argument("--prompt-file", help="requests to generate, one per clip (default: data.split)")
    run.add_argument(
        "--checkpoint",
        help="override model.checkpoint: a release name, or the .pt of a fine-tuned run",
    )
    run.add_argument("--output-dir", help="directory of the generated clips")
    generation = parser.add_argument_group("generation")
    generation.add_argument("--num-steps", type=int, help="override inference.num_steps")
    generation.add_argument("--cfg-scale", type=float, help="override inference.cfg_scale")
    generation.add_argument("--seed", type=int, help="override seed of the configuration")
    generation.add_argument("--limit", type=int, help="generate at most this many requests")
    generation.add_argument("--shard-index", type=int, default=0, help="index of this shard")
    generation.add_argument("--num-shards", type=int, default=1, help="number of shards")
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", help="torch device of the run, e.g. cuda:0 or cpu")
    runtime.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the requests, without loading weights or writing clips",
    )
    return parser.parse_args(argv)


def repo_root() -> Path:
    """Repository root: ``eveworld.utils.io.repo_root`` when importable, else the pyproject walk."""
    try:
        from eveworld.utils.io import repo_root as _library_root
    except ImportError:
        pass
    else:
        return Path(_library_root())
    path = Path(__file__).resolve()
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def request_id_of(record: Any, index: int) -> str:
    """Request id of a record: its own id when it carries one, else its position in the file."""
    if isinstance(record, dict):
        for key in ("request_id", "request", "id", "name"):
            value = record.get(key)
            if value not in (None, ""):
                return str(value)
    return f"{index:04d}"


def prompt_of(record: Any) -> str:
    """Prompt of a record, raising when the record carries none."""
    if isinstance(record, str) and record.strip():
        return record.strip()
    if isinstance(record, dict):
        for key in ("prompt", "text", "instruction"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        nested = record.get("target")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    raise ValueError(f"no prompt in request {record!r}")


@dataclass
class Request:
    """One clip to generate.

    Attributes:
        request_id: Identifier of the request, the stem of its MP4 and its key in the evaluation.
        prompt: Prompt the clip is generated from.
    """

    request_id: str
    prompt: str


def requests_from_document(document: Any) -> list[Request]:
    """Requests of a parsed JSON prompt document, either a mapping or a list."""
    if isinstance(document, dict):
        return [Request(str(key), prompt_of(value)) for key, value in document.items()]
    if isinstance(document, list):
        return [
            Request(request_id_of(item, index), prompt_of(item))
            for index, item in enumerate(document, 1)
        ]
    raise ValueError(f"prompt document is a {type(document).__name__}, expected an object or a list")


def read_requests(path: Path) -> list[Request]:
    """Requests of a prompt file: a JSON document, JSON Lines, or one request per line.

    A line without a tab and without whitespace names a request id, the shape of the split
    files, and starts out as its own prompt; ``metadata_prompts`` then replaces it with the
    prompt the metadata of the configuration carries for that id.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"prompt file {path} is empty")
    if stripped[0] in "[{":
        try:
            document = json.loads(stripped)
        except json.JSONDecodeError:
            document = None
        if document is not None:
            return requests_from_document(document)
    requests: list[Request] = []
    for index, line in enumerate(text.splitlines(), 1):
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        try:
            record: Any = json.loads(row)
        except json.JSONDecodeError:
            record = row
        if isinstance(record, str) and "\t" in record:
            request_id, _, prompt = record.partition("\t")
            requests.append(Request(request_id.strip(), prompt.strip()))
        elif isinstance(record, str) and len(record.split()) == 1:
            requests.append(Request(record, record))
        else:
            requests.append(Request(request_id_of(record, index), prompt_of(record)))
    if not requests:
        raise ValueError(f"prompt file {path} lists no requests")
    return requests


def metadata_prompts(metadata: Path, ids: Sequence[str]) -> dict[str, str]:
    """Prompts of ``ids`` in a metadata directory: one document per clip, or keyed by id.

    ``prepare_*`` writes both shapes, so both are read; documents that carry no prompt for a
    wanted id are skipped, and unreadable documents are ignored instead of failing the run.
    """
    index: dict[str, str] = {}
    wanted = set(ids)
    if not wanted or not metadata.is_dir():
        return index
    for path in sorted(metadata.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, dict):
            continue
        records = {path.stem: document} if path.stem in wanted else {}
        records.update({key: value for key, value in document.items() if key in wanted})
        for request_id, value in records.items():
            if request_id in index:
                continue
            try:
                index[request_id] = prompt_of(value)
            except ValueError:
                continue
    return index


def with_metadata_prompts(requests: Sequence[Request], metadata: Path) -> list[Request]:
    """``requests`` with the prompt of every id-named request read from the metadata."""
    named = [request.request_id for request in requests if request.prompt == request.request_id]
    if not named:
        return list(requests)
    index = metadata_prompts(metadata, named)
    return [
        Request(request.request_id, index.get(request.request_id, request.prompt))
        for request in requests
    ]


@dataclass
class InferencePlan:
    """Everything a generation run needs, resolved from the configuration and the command line.

    Attributes:
        config_path: Configuration the run was resolved from, as named on the command line.
        config: Parsed configuration, with the command-line overrides applied.
        name: ``name`` of the configuration.
        output_dir: Directory the clips are written to, into ``generated_only/``.
        prompt_file: File the requests were read from.
        requests: Requests this shard generates.
        total_requests: Requests the prompt file lists, before sharding and ``--limit``.
        num_steps: Sampling steps per clip.
        cfg_scale: Classifier-free guidance scale.
        seed: Seed of the sampler.
        checkpoint: ``model.checkpoint`` after the overrides.
        fps: Frame rate of the generated clips.
        device: Device requested on the command line, or ``None`` for the model default.
        shard_index: Index of this shard.
        num_shards: Number of shards the requests were split over.
        dry_run: Whether the run stops after the plan.
    """

    config_path: Path
    config: Any
    name: str
    output_dir: Path
    prompt_file: Path
    requests: list[Request]
    total_requests: int
    num_steps: int
    cfg_scale: float
    seed: int
    checkpoint: str
    fps: float
    device: str | None
    shard_index: int
    num_shards: int
    dry_run: bool

    @property
    def frame_dir(self) -> Path:
        """Directory the clips of the run are written to."""
        return self.output_dir / FRAME_DIR


def load_config(config_path: Path, overrides: Sequence[str]) -> Any:
    """Parse ``config_path`` with ``overrides`` applied on top of it."""
    from eveworld.utils.config import load_config as _load_config

    return _load_config(config_path, overrides)


def cfg_value(config: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` out of a parsed configuration without raising on a missing key."""
    from eveworld.utils.config import cfg_get

    return cfg_get(config, key, default)


def build_plan(args: argparse.Namespace, config_path: Path) -> InferencePlan:
    """Resolve the configuration, the requests and the generation settings."""
    from eveworld.utils.config import resolve_path
    from eveworld.utils.distributed import shard_range

    root = repo_root()
    overrides: list[str] = []
    if args.num_steps is not None:
        overrides.append(f"inference.num_steps={int(args.num_steps)}")
    if args.cfg_scale is not None:
        overrides.append(f"inference.cfg_scale={float(args.cfg_scale)}")
    if args.seed is not None:
        overrides.append(f"seed={int(args.seed)}")
    if args.checkpoint is not None:
        overrides.append(f"model.checkpoint={args.checkpoint}")
    config = load_config(config_path, overrides)

    prompt_file = args.prompt_file or cfg_value(config, "data.split")
    if not prompt_file:
        raise ValueError("no requests to generate: pass --prompt-file or set data.split")
    prompt_file = resolve_path(str(prompt_file), root)
    if not prompt_file.is_file():
        raise FileNotFoundError(f"prompt file {prompt_file} does not exist")
    requests = read_requests(prompt_file)
    metadata = cfg_value(config, "data.metadata")
    if metadata:
        requests = with_metadata_prompts(requests, resolve_path(str(metadata), root))
        unresolved = sum(1 for request in requests if request.prompt == request.request_id)
        if unresolved:
            print(
                f"warning: {metadata} carries no prompt for {unresolved} of {len(requests)} "
                "requests, their request id is used as the prompt",
                file=sys.stderr,
            )
    total = len(requests)
    start, stop = shard_range(total, args.shard_index, args.num_shards)
    requests = requests[start:stop]
    if args.limit is not None:
        requests = requests[: args.limit]
    output_dir = resolve_path(args.output_dir, root) if args.output_dir else resolve_path(
        cfg_value(config, "output_dir", "outputs/run"), root
    ) / "inference"
    return InferencePlan(
        config_path=config_path,
        config=config,
        name=str(cfg_value(config, "name", config_path.stem)),
        output_dir=output_dir,
        prompt_file=prompt_file,
        requests=requests,
        total_requests=total,
        num_steps=int(cfg_value(config, "inference.num_steps", 0)),
        cfg_scale=float(cfg_value(config, "inference.cfg_scale", 1.0)),
        seed=int(cfg_value(config, "seed", 0)),
        checkpoint=str(cfg_value(config, "model.checkpoint", "-")),
        fps=float(cfg_value(config, "model.fps", 0.0)),
        device=args.device,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        dry_run=bool(args.dry_run),
    )


def print_plan(plan: InferencePlan) -> None:
    """Print the generation run, one aligned field per line."""
    config = plan.config
    resolution = cfg_value(config, "model.resolution", (0, 0))
    print(f"config:       {plan.config_path}")
    print(f"run:          {plan.name}")
    print(f"checkpoint:   {plan.checkpoint}")
    print(f"output dir:   {plan.frame_dir}")
    print(f"prompts:      {len(plan.requests)} of {plan.total_requests} requests from {plan.prompt_file}")
    print(f"shard:        {plan.shard_index} of {plan.num_shards}")
    print(
        f"sampling:     {plan.num_steps} steps, cfg {plan.cfg_scale:g}, "
        f"{int(cfg_value(config, 'model.num_frames', 0))} frames at {plan.fps:g} fps, "
        f"{int(resolution[0])}x{int(resolution[1])}, block "
        f"{int(cfg_value(config, 'model.block_index', 0))}, seed {plan.seed}"
    )
    print(f"device:       {plan.device or 'auto'}")
    for request in plan.requests[:PREVIEW]:
        prompt = request.prompt if len(request.prompt) <= 72 else f"{request.prompt[:69]}..."
        print(f"request:      {request.request_id} {prompt}")
    if len(plan.requests) > PREVIEW:
        print(f"request:      ... and {len(plan.requests) - PREVIEW} more")


def filter_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``kwargs`` whose names the signature of ``function`` accepts."""
    import inspect

    accepted = inspect.signature(function).parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in accepted.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in accepted and value is not None}


def generation_method(model: Any) -> Any:
    """Sampling callable of the model: ``inference``, or the names the integration may use."""
    for name in ("inference", "generate", "sample"):
        method = getattr(model, name, None)
        if callable(method):
            return method
    public = ", ".join(sorted(item for item in dir(model) if not item.startswith("_")))
    raise AttributeError(f"the model exposes no inference method; it carries {public}")


def as_frames(output: Any) -> Any:
    """Frames of a model output, i.e. a ``(T, H, W, 3)`` array, as ``uint8``."""
    import numpy as np

    if hasattr(output, "detach"):
        output = output.detach().to("cpu").numpy()
    if not isinstance(output, np.ndarray):
        raise RuntimeError(f"the model returned a {type(output).__name__}, expected video frames")
    if output.dtype != np.uint8:
        scaled = output.astype("float32")
        if scaled.size and scaled.max() <= 1.0 + 1e-6:
            scaled = scaled * 255.0
        output = np.clip(np.rint(scaled), 0, 255).astype("uint8")
    if output.ndim != 4 or output.shape[-1] != 3:
        raise RuntimeError(f"the model returned frames of shape {output.shape}, expected (T, H, W, 3)")
    return output


def write_video(frames: Any, path: Path, fps: float) -> None:
    """Write ``(T, H, W, 3)`` uint8 frames as an H.264 MP4 at ``fps``."""
    import av
    import numpy as np

    height, width = int(frames.shape[1]), int(frames.shape[2])
    frames = frames[:, : height - height % 2, : width - width % 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=round(fps) or 1)
        stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            packet = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            container.mux(stream.encode(packet))
        container.mux(stream.encode())


def generate(plan: InferencePlan) -> int:
    """Generate the clips of ``plan`` and return the process exit code."""
    from eveworld.integrations.gigaworld import GigaWorldConfig, GigaWorldModel
    from eveworld.utils.config import resolve_path
    from eveworld.utils.io import is_run_checkpoint

    model_config = GigaWorldConfig.from_config(plan.config)
    checkpoint = resolve_path(plan.checkpoint, repo_root())
    trainer = None
    if is_run_checkpoint(checkpoint):
        from eveworld.integrations.gigaworld import GigaWorldTrainer

        trainer = GigaWorldTrainer(
            config=plan.config, output_dir=plan.output_dir, device=plan.device
        )
        step = trainer.load_checkpoint(checkpoint)
        model = trainer.model.eval()
        print(f"model:        the run of step {step} at {checkpoint}")
    else:
        model = GigaWorldModel(config=model_config)
        if plan.device:
            model.to(plan.device)
    method = generation_method(model)
    frames = int(cfg_value(plan.config, "model.num_frames", model_config.num_frames))
    for index, request in enumerate(plan.requests, 1):
        output = method(
            **filter_kwargs(
                method,
                {
                    "prompt": request.prompt,
                    "num_frames": frames,
                    "num_steps": plan.num_steps,
                    "cfg_scale": plan.cfg_scale,
                    "seed": plan.seed,
                },
            )
        )
        path = plan.frame_dir / f"{request.request_id}.mp4"
        write_video(as_frames(output), path, plan.fps)
        print(f"[{index}/{len(plan.requests)}] {request.request_id} -> {path}")
    print(f"generated {len(plan.requests)} clips into {plan.frame_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: resolve the run, then generate the clips or print the plan."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(arguments)
    if not args.config:
        print("error: --config is required (or export EVEWORLD_CONFIG)", file=sys.stderr)
        return 2
    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        print(f"error: config file {config_path} does not exist", file=sys.stderr)
        return 2
    if args.num_steps is not None and args.num_steps < 1:
        print("error: --num-steps must be positive", file=sys.stderr)
        return 2
    if args.cfg_scale is not None and args.cfg_scale <= 0:
        print("error: --cfg-scale must be positive", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be positive", file=sys.stderr)
        return 2
    if args.num_shards < 1:
        print("error: --num-shards must be positive", file=sys.stderr)
        return 2
    if not 0 <= args.shard_index < args.num_shards:
        print(f"error: --shard-index must be in [0, {args.num_shards})", file=sys.stderr)
        return 2
    try:
        plan = build_plan(args, config_path)
        print_plan(plan)
        if plan.dry_run:
            print("dry run: no clip generated, no directory written")
            return 0
        return generate(plan)
    except ImportError as error:
        print(f"error: the eveworld package is not importable: {error}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
