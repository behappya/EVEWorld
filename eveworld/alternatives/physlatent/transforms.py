from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from giga_datasets import video_utils
from giga_train import TRANSFORMS
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


@TRANSFORMS.register
class PhysLatentGigaWorld0Transform:
    """Baseline GigaWorld video transform for PhysLatent experiments.

    This mirrors ``GigaWorld0Transform`` but lives in this package so importing
    PhysLatent configs does not need to import the baseline trainer.
    """

    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        image_cfg: dict,
        fps: int = 16,
        random_crop: bool = True,
        phys_labels_path: str | None = None,
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.random_crop = random_crop
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])
        self.phys_labels = self._load_phys_labels(phys_labels_path)
        if self.phys_labels and self.random_crop:
            raise ValueError('phys_labels_path requires random_crop=False so image-space labels stay aligned')

    def __call__(self, data_dict):
        video = data_dict['video']
        video_length = len(video)
        sample_indexes = np.linspace(0, video_length - 1, self.num_frames, dtype=int)
        input_images = video_utils.sample_video(video, sample_indexes, method=2)
        input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()

        image_height = input_images.shape[2]
        image_width = input_images.shape[3]
        dst_width, dst_height = self.width, self.height

        if float(dst_height) / image_height < float(dst_width) / image_width:
            new_height = int(round(float(dst_width) / image_width * image_height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / image_height * image_width))

        if self.random_crop:
            x1 = random.randint(0, new_width - dst_width)
            y1 = random.randint(0, new_height - dst_height)
        else:
            x1 = max((new_width - dst_width) // 2, 0)
            y1 = max((new_height - dst_height) // 2, 0)

        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        input_images = input_images / 255.0
        input_images = self.normalize(input_images)

        ref_masks, ref_latent_masks = self.mask_generator.get_mask(input_images.shape[0])
        ref_masks = ref_masks[:, None, None, None]
        ref_latent_masks = ref_latent_masks[None, :, None, None]
        ref_images = copy.deepcopy(input_images)
        ref_images = ref_images * ref_masks

        output = dict(
            fps=self.fps,
            images=input_images,
            ref_images=ref_images,
            ref_masks=ref_latent_masks,
            prompt_embeds=data_dict['prompt_embeds'],
        )

        if 'data_index' in data_dict:
            data_index = int(data_dict['data_index'])
            output['data_index'] = torch.tensor(data_index, dtype=torch.long)
            self._add_phys_labels(output, data_index)
        if 'prompt' in data_dict:
            output['prompt'] = data_dict['prompt']
        return output

    @staticmethod
    def _load_phys_labels(phys_labels_path: str | None) -> dict[int, dict]:
        if not phys_labels_path:
            return {}
        path = Path(phys_labels_path)
        if not path.is_file():
            raise FileNotFoundError(f'PhysLatent pseudo-label file does not exist: {path}')
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        records = payload.get('records', payload)
        return {int(key): value for key, value in records.items()}

    def _add_phys_labels(self, output: dict, data_index: int) -> None:
        record = self.phys_labels.get(data_index)
        if not record:
            return
        tensor_specs = {
            'phys_state': torch.float32,
            'phys_goal': torch.float32,
            'phys_trajectory': torch.float32,
            'phys_state_mask': torch.float32,
            'phys_goal_mask': torch.float32,
            'phys_trajectory_mask': torch.float32,
            'phys_contact_mask': torch.float32,
            'phys_valid_mask': torch.float32,
            'phys_done_mask': torch.float32,
            'phys_terminal_mask': torch.float32,
            'phys_goal_reached': torch.float32,
            'phys_goal_reached_mask': torch.float32,
            'phys_release': torch.float32,
            'phys_release_mask': torch.float32,
            'phys_object_motion': torch.float32,
            'phys_object_motion_mask': torch.float32,
            'phys_label_quality': torch.float32,
            'phys_contact': torch.long,
            'phys_phase': torch.long,
            'phys_phase_mask': torch.float32,
            'phys_done_bin': torch.long,
        }
        for key, dtype in tensor_specs.items():
            if key in record:
                output[key] = torch.as_tensor(record[key], dtype=dtype)


class MaskGenerator:
    def __init__(self, max_ref_frames: int, factor: int = 8, start: int = 1):
        assert max_ref_frames > 0 and (max_ref_frames - 1) % factor == 0
        self.max_ref_frames = max_ref_frames
        self.factor = factor
        self.start = start
        self.max_ref_latents = 1 + (max_ref_frames - 1) // factor
        assert self.start <= self.max_ref_latents

    def get_mask(self, num_frames: int):
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_ref_frames

        num_latents = 1 + (num_frames - 1) // self.factor
        num_ref_latents = random.randint(self.start, self.max_ref_latents)
        if num_ref_latents > 0:
            num_ref_frames = 1 + (num_ref_latents - 1) * self.factor
        else:
            num_ref_frames = 0

        ref_masks = torch.zeros((num_frames,), dtype=torch.float32)
        ref_masks[:num_ref_frames] = 1

        ref_latent_masks = torch.zeros((num_latents,), dtype=torch.float32)
        ref_latent_masks[:num_ref_latents] = 1

        return ref_masks, ref_latent_masks
