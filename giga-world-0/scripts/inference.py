import json
import math
import os
import re
import statistics
import sys
import time
from typing import List, Literal, Optional

import imageio
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
from accelerate.utils import set_seed
from giga_datasets import image_utils
from giga_datasets import utils as gd_utils
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm

from giga_models import GigaWorld0Pipeline
from giga_models.acceleration import get_sequence_parallel_group, initialize_sequence_parallel_group
from giga_models.utils import find_free_port


PipelineVariant = Literal['standard', 'cic-transport']


def _ranked_json_path(path: str, dp_world_size: int, dp_rank: int) -> str:
    if dp_world_size == 1:
        return path
    stem, suffix = os.path.splitext(path)
    return f'{stem}.dp{dp_rank:03d}{suffix or ".json"}'


def _atomic_json_dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f'{path}.{os.getpid()}.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    os.replace(temporary, path)


def _json_safe_stats(stats: dict[str, float]) -> dict[str, Optional[float]]:
    return {
        key: value if math.isfinite(value) else None
        for key, value in stats.items()
    }


def clean_output_id(value: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip())
    cleaned = re.sub(r'_+', '_', cleaned).strip('._-')
    return cleaned or str(int(time.time() * 1000))


def output_id_for_row(index: int, row: dict) -> str:
    for key in ('request_id', 'pbench_id', 'id'):
        if row.get(key):
            return clean_output_id(str(row[key]))
    prompt = str(row.get('prompt') or 'sample').strip().rstrip('.')
    prompt = re.sub(r'[^A-Za-z0-9]+', '_', prompt)
    prompt = re.sub(r'_+', '_', prompt).strip('_')[:180].rstrip('_') or 'sample'
    return clean_output_id(f'{index}_{prompt}')


def _inference(
    device,
    data_path: str,
    save_dir: str,
    transformer_model_path: str,
    text_encoder_model_path: str = None,
    vae_model_path: str = None,
    lora_model_path: str = None,
    lora_fuse: bool = False,
    physics_latent_model_path: Optional[str] = None,
    physlatent_uncond_mode: str = 'shared',
    num_inference_steps: int = 30,
    fps: int = 16,
    num_frames: int = 61,
    height: int = 480,
    width: int = 640,
    seed: int = 6666,
    summary_path: Optional[str] = None,
    save_generated_only: bool = False,
    pipeline_variant: PipelineVariant = 'standard',
    transport_sentinel_path: Optional[str] = None,
    dp_world_size: int = 1,
    dp_rank: int = 0,
    process_index: int = 0,
    global_rank: int = 0,
):
    """Run inference on a split of the dataset using a single device
    (optionally as part of DP/SP setup).

    Args:
        device: Device string (e.g., 'cuda:0').
        data_path: Path to the JSON data file.
        save_dir: Directory to save results.
        transformer_model_path, text_encoder_model_path, vae_model_path: Model paths.
        lora_model_path: Optional LoRA weights.
        lora_fuse: Whether to fuse LoRA weights.
        num_inference_steps, fps, num_frames, height, width, seed: Generation parameters.
        dp_world_size, dp_rank: Data parallel world size and rank.
        process_index: Index for multi-process setups.
    """
    print(
        f'[gw0] global_rank={global_rank} process={process_index} '
        f'dp_rank={dp_rank}/{dp_world_size} using device={device} '
        f'pipeline_variant={pipeline_variant}',
        flush=True,
    )
    torch.cuda.set_device(device)
    enable_progress = os.environ.get('GW0_ENABLE_TQDM') == '1' or (process_index == 0 and sys.stderr.isatty())
    # Load the GigaWorld0 pipeline
    if pipeline_variant == 'cic-transport':
        if physics_latent_model_path:
            raise ValueError('CIC-Transport cannot be combined with PhysicsLatent')
        from eveworld.tia_transport.cic_transport_pipeline import (
            CICTransportGigaWorld0Pipeline,
        )

        print(
            f'[gw0] global_rank={global_rank} loading CIC-Transport pipeline...',
            flush=True,
        )
        pipe = CICTransportGigaWorld0Pipeline.from_pretrained(
            transformer_model_path=transformer_model_path,
            text_encoder_model_path=text_encoder_model_path,
            vae_model_path=vae_model_path,
            lora_model_path=lora_model_path,
            lora_fuse=lora_fuse,
        )
    elif physics_latent_model_path:
        from eveworld.alternatives.physlatent.pipeline import PhysLatentGigaWorld0Pipeline

        print(
            f'[gw0] process={process_index} loading PhysLatent pipeline '
            f'physics_latent={physics_latent_model_path} uncond_mode={physlatent_uncond_mode}...',
            flush=True,
        )
        pipe = PhysLatentGigaWorld0Pipeline.from_pretrained(
            transformer_model_path=transformer_model_path,
            text_encoder_model_path=text_encoder_model_path,
            vae_model_path=vae_model_path,
            lora_model_path=lora_model_path,
            lora_fuse=lora_fuse,
            physics_latent_model_path=physics_latent_model_path,
            physlatent_uncond_mode=physlatent_uncond_mode,
        )
    else:
        print(f'[gw0] process={process_index} loading pipeline...', flush=True)
        pipe = GigaWorld0Pipeline.from_pretrained(
            transformer_model_path=transformer_model_path,
            text_encoder_model_path=text_encoder_model_path,
            vae_model_path=vae_model_path,
            lora_model_path=lora_model_path,
            lora_fuse=lora_fuse,
        )
    if pipeline_variant == 'cic-transport':
        transformer = pipe.transformer
        if not getattr(transformer, '_cic_transport_enabled', False):
            raise RuntimeError('CIC-Transport pipeline loaded without an enabled adapter')
        load_stats = transformer.cic_transport.sentinel_stats()
        if load_stats['input_proj_norm'] <= 0 or load_stats['output_proj_norm'] <= 0:
            raise RuntimeError(f'CIC-Transport checkpoint has inactive weights: {load_stats}')
        sentinel = {
            'status': 'loaded',
            'pipeline_class': f'{type(pipe).__module__}.{type(pipe).__name__}',
            'transformer_class': f'{type(transformer).__module__}.{type(transformer).__name__}',
            'transformer_model_path': str(os.path.realpath(transformer_model_path)),
            'transport_config': transformer.cic_transport_config(),
            'stats': _json_safe_stats(load_stats),
            'global_rank': global_rank,
            'dp_rank': dp_rank,
            'dp_world_size': dp_world_size,
            'device': str(device),
        }
        print(f'[cic-transport-runtime] {json.dumps(sentinel, sort_keys=True)}', flush=True)
        if transport_sentinel_path:
            sentinel_path = _ranked_json_path(
                transport_sentinel_path, dp_world_size, dp_rank
            )
            _atomic_json_dump(sentinel_path, sentinel)
    if hasattr(pipe, 'set_progress_bar_config'):
        pipe.set_progress_bar_config(disable=not enable_progress)
    print(f'[gw0] process={process_index} pipeline loaded; moving to {device}...', flush=True)
    pipe.to(device)
    print(f'[gw0] process={process_index} pipeline ready on {device}', flush=True)
    # Load and split data for this process
    negative_prompt = 'The video captures a series of frames showing ugly scenes, static with no motion, motion blur, \
                    over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, \
                    underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, \
                    jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special \
                    effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and \
                    flickering. Overall, the video is of poor quality.'
    data_list = json.load(open(data_path, 'r'))
    data_list = gd_utils.split_data(data_list, dp_world_size, dp_rank)
    print(f'[gw0] process={process_index} loaded {len(data_list)} samples from {data_path}', flush=True)
    os.makedirs(save_dir, exist_ok=True)
    run_start = time.time()
    sample_timings = []
    # Inference loop
    for n in tqdm(range(len(data_list)), disable=not enable_progress):
        set_seed(seed)
        sample_start = time.time()
        data_dict = data_list[n]
        prompt = data_dict['prompt']
        image_path = data_dict['image']
        output_id = output_id_for_row(n, data_dict)
        print(f'[gw0] process={process_index} sample={n} output_id={output_id} image={image_path}', flush=True)
        if not os.path.exists(image_path):
            image_path = os.path.join(os.path.dirname(data_path), image_path)
        image = Image.open(image_path)
        image_width, image_height = image.width, image.height
        # Compute resize/crop to maintain aspect ratio and fit model input
        dst_width, dst_height = image_utils.get_image_size((image_width, image_height), (width, height), mode='area', multiple=16)
        if float(dst_height) / image_height < float(dst_width) / image_width:
            new_height = int(round(float(dst_width) / image_width * image_height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / image_height * image_width))
        assert dst_width <= new_width and dst_height <= new_height
        x1 = (new_width - dst_width) // 2
        y1 = (new_height - dst_height) // 2
        input_image = F.resize(image, (new_height, new_width), InterpolationMode.BILINEAR)
        input_image = F.crop(input_image, y1, x1, dst_height, dst_width)
        # Run the pipeline
        print(f'[gw0] process={process_index} sample={n} generating video...', flush=True)
        output_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            fps=fps,
            num_frames=num_frames,
            height=dst_height,
            width=dst_width,
            seed=seed,
        )[0]
        if pipeline_variant == 'cic-transport' and n == 0:
            runtime_stats = pipe.transformer.cic_transport.sentinel_stats()
            required_runtime = ('match_peak', 'match_entropy', 'update_rms', 'feature_rms')
            if not all(math.isfinite(runtime_stats[key]) for key in required_runtime):
                raise RuntimeError(
                    f'CIC-Transport adapter did not record finite runtime stats: {runtime_stats}'
                )
            if runtime_stats['update_rms'] <= 0:
                raise RuntimeError(
                    f'CIC-Transport adapter produced no runtime update: {runtime_stats}'
                )
            sentinel.update({'status': 'active', 'stats': _json_safe_stats(runtime_stats)})
            print(
                f'[cic-transport-runtime] {json.dumps(sentinel, sort_keys=True)}',
                flush=True,
            )
            if transport_sentinel_path:
                _atomic_json_dump(sentinel_path, sentinel)
        # Save results (only on main process)
        if process_index == 0:
            vis_images = []
            for k in range(len(output_images)):
                if save_generated_only:
                    vis_image = output_images[k]
                else:
                    if image is not None:
                        vis_image = [input_image, output_images[k]]
                    else:
                        vis_image = [output_images[k]]
                    vis_image = image_utils.concat_images_grid(vis_image, cols=2, pad=2)
                vis_images.append(vis_image)
            save_path = os.path.join(save_dir, f'{output_id}.mp4')
            imageio.mimsave(save_path, vis_images, fps=fps)
            elapsed = time.time() - sample_start
            sample_timings.append(
                {
                    'sample_index': n,
                    'output_id': output_id,
                    'video_path': save_path,
                    'elapsed_sec': round(elapsed, 3),
                }
            )
            print(f'[gw0] process={process_index} sample={n} saved {save_path} elapsed={elapsed:.2f}s', flush=True)

    if process_index == 0 and summary_path:
        ranked_summary_path = _ranked_json_path(summary_path, dp_world_size, dp_rank)
        elapsed_values = [item['elapsed_sec'] for item in sample_timings]
        summary = {
            'data_path': data_path,
            'save_dir': save_dir,
            'request_count': len(data_list),
            'generated_count': len(sample_timings),
            'wall_time_sec': round(time.time() - run_start, 3),
            'model_elapsed_total_sec': round(sum(elapsed_values), 3) if elapsed_values else None,
            'model_elapsed_mean_sec': round(statistics.mean(elapsed_values), 3) if elapsed_values else None,
            'model_elapsed_median_sec': round(statistics.median(elapsed_values), 3) if elapsed_values else None,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'physics_latent_model_path': physics_latent_model_path,
            'physlatent_uncond_mode': physlatent_uncond_mode if physics_latent_model_path else None,
            'pipeline_variant': pipeline_variant,
            'global_rank': global_rank,
            'dp_world_size': dp_world_size,
            'dp_rank': dp_rank,
            'save_generated_only': save_generated_only,
            'samples': sample_timings,
        }
        _atomic_json_dump(ranked_summary_path, summary)
        print(
            f'[gw0] global_rank={global_rank} summary saved {ranked_summary_path}',
            flush=True,
        )


def _inference_sp(rank, gpu_ids, sp_size, port, kwargs):
    """Worker function for sequence parallel (SP) inference.

    Initializes process group and runs _inference.
    Args:
        rank: Process rank.
        gpu_ids: List of GPU IDs.
        sp_size: Sequence parallel group size.
        port: TCP port for distributed init.
        kwargs: Arguments for _inference.
    """
    gpu_id = gpu_ids[rank]
    world_size = len(gpu_ids)
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=world_size,
        rank=rank,
        device_id=torch.device(device),
    )
    if sp_size == 1:
        sp_world_size = 1
        sp_rank = 0
    else:
        initialize_sequence_parallel_group(sp_size)
        sp_group = get_sequence_parallel_group()
        sp_world_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
    dp_world_size = world_size // sp_world_size
    dp_rank = rank // sp_world_size
    assert sp_size == sp_world_size
    try:
        _inference(
            device,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            process_index=sp_rank,
            global_rank=rank,
            **kwargs,
        )
    except BaseException:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    else:
        # DP ranks can own different sample counts. Keep the TCPStore alive until
        # every rank has finished writing its outputs and summaries.
        dist.barrier()
        dist.destroy_process_group()


def inference(
    data_path: str,
    save_dir: str,
    transformer_model_path: str,
    text_encoder_model_path: str = None,
    vae_model_path: str = None,
    lora_model_path: str = None,
    lora_fuse: bool = False,
    physics_latent_model_path: Optional[str] = None,
    physlatent_uncond_mode: str = 'shared',
    gpu_ids: List[int] = [0],
    num_inference_steps: int = 30,
    fps: int = 16,
    num_frames: int = 61,
    height: int = 480,
    width: int = 640,
    seed: int = 6666,
    summary_path: Optional[str] = None,
    save_generated_only: bool = False,
    pipeline_variant: PipelineVariant = 'standard',
    sequence_parallel_size: Optional[int] = None,
    transport_sentinel_path: Optional[str] = None,
):
    """Main entry point for inference.

    Handles single- and multi-GPU, launches processes as needed.
    Args:
        data_path: Path to JSON data file.
        save_dir: Directory to save results.
        transformer_model_path, text_encoder_model_path, vae_model_path: Model paths.
        lora_model_path: Optional LoRA weights.
        lora_fuse: Whether to fuse LoRA weights.
        gpu_ids: List of GPU IDs to use.
        num_inference_steps, fps, num_frames, height, width, seed: Generation parameters.
    """
    kwargs = dict(
        data_path=data_path,
        save_dir=save_dir,
        transformer_model_path=transformer_model_path,
        text_encoder_model_path=text_encoder_model_path,
        vae_model_path=vae_model_path,
        lora_model_path=lora_model_path,
        lora_fuse=lora_fuse,
        physics_latent_model_path=physics_latent_model_path,
        physlatent_uncond_mode=physlatent_uncond_mode,
        num_inference_steps=num_inference_steps,
        fps=fps,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
        summary_path=summary_path,
        save_generated_only=save_generated_only,
        pipeline_variant=pipeline_variant,
        transport_sentinel_path=transport_sentinel_path,
    )
    num_gpus = len(gpu_ids)
    assert num_gpus >= 1
    sp_size = num_gpus if sequence_parallel_size is None else sequence_parallel_size
    if sp_size < 1 or num_gpus % sp_size != 0:
        raise ValueError(
            f'sequence_parallel_size={sp_size} must be a positive divisor of '
            f'num_gpus={num_gpus}'
        )
    if pipeline_variant == 'cic-transport' and sp_size != 1:
        raise ValueError('CIC-Transport requires sequence_parallel_size=1')
    print(
        f'[gw0] launching inference with gpu_ids={gpu_ids} '
        f'pipeline_variant={pipeline_variant} sp_size={sp_size} '
        f'dp_size={num_gpus // sp_size}',
        flush=True,
    )
    if num_gpus == 1:
        _inference(f'cuda:{gpu_ids[0]}', **kwargs)
    else:
        port = find_free_port()
        mp.start_processes(
            _inference_sp,
            nprocs=num_gpus,
            args=(gpu_ids, sp_size, port, kwargs),
        )


if __name__ == '__main__':
    tyro.cli(inference)
