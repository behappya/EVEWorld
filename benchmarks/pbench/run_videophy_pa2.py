#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VideoPhy PA-II inference with local compatibility patches.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def sanitize_for_json(value: Any) -> Any:
    if callable(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value if not callable(v)]
    return value


def patch_config_repr() -> None:
    # The VideoPhy mPLUG-Owl config copies attributes from its nested text config.
    # With newer transformers this can leave method objects in __dict__, and
    # transformers 4.31 tries to JSON dump the config while logging it.
    try:
        from mplug_owl_video.configuration_mplug_owl import MplugOwlConfig
    except Exception:
        return

    original_to_dict = MplugOwlConfig.to_dict

    def patched_to_dict(self):
        return sanitize_for_json(original_to_dict(self))

    MplugOwlConfig.to_dict = patched_to_dict
    PretrainedConfig.__repr__ = lambda self: f"{self.__class__.__name__} {sanitize_for_json(self.to_dict())}"


def get_entail(logits, input_ids, tokenizer):
    softmax = nn.Softmax(dim=2)
    logits = softmax(logits)
    token_id_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    token_id_no = tokenizer.encode("No", add_special_tokens=False)[0]
    entailment = []
    for j in range(len(logits)):
        for i in range(len(input_ids[j])):
            if input_ids[j][i] == tokenizer.pad_token_id:
                i = i - 1
                break
            if i == len(input_ids[j]) - 1:
                break
        score = logits[j][i][token_id_yes] / (logits[j][i][token_id_yes] + logits[j][i][token_id_no])
        entailment.append(score)
    return torch.stack(entailment)


def main() -> None:
    args = parse_args()
    patch_config_repr()

    from data_utils.xgpt3_dataset import MultiModalDataset
    from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
    from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
    from utils import batchify

    checkpoint = args.checkpoint
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    valid_data = MultiModalDataset(args.input_csv, tokenizer, processor, max_length=256, loss_objective="sequential")
    dataloader = DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True, collate_fn=batchify)

    model = MplugOwlForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to("cuda")
    print("Model Loaded", flush=True)
    model.eval()

    with torch.no_grad():
        for index, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    if value.dtype == torch.float:
                        value = value.bfloat16()
                    inputs[key] = value.to(model.device)
            outputs = model(
                pixel_values=inputs["pixel_values"],
                video_pixel_values=inputs["video_pixel_values"],
                labels=None,
                num_images=inputs["num_images"],
                num_videos=inputs["num_videos"],
                input_ids=inputs["input_ids"],
                non_padding_mask=inputs["non_padding_mask"],
                non_media_mask=inputs["non_media_mask"],
                prompt_mask=inputs["prompt_mask"],
            )
            entail_scores = get_entail(outputs["logits"], inputs["input_ids"], tokenizer)
            with output_csv.open("a", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                for row_index, score in enumerate(entail_scores):
                    writer.writerow([inputs["videopaths"][row_index], inputs["captions"][row_index], score.item()])
            print(f"Batch {index} Done", flush=True)


if __name__ == "__main__":
    main()
