#!/usr/bin/env python3
"""GroundingDINO 物体定位: prompt 解析物体名 -> 首帧框物体 -> 特征网格坐标。
供 t4g 探针/可视化用正确的物体位置当追踪起点(替代光流猜的"最大运动格")。"""
import re

import numpy as np

GDINO_PATH = '/data/datasets/gagi/ext_data/grounding_dino_base'


def parse_objects(prompt):
    """从模板 prompt 解析 X(被操作物, 该动) / A,B(源/目标容器, 该静)。
    模板: Use the {L/R} hand to pick up {X} from {A} to {B}。返回 dict。"""
    m = re.search(r'pick up (?:the )?(.+?) from (?:the )?(.+?) to (?:the )?(.+?)\.?\s*$', prompt, re.I)
    if m:
        return {'mover': m.group(1).strip(), 'src': m.group(2).strip(), 'tgt': m.group(3).strip()}
    # 兜底: 取 pick up 后第一个名词短语
    m2 = re.search(r'pick up (?:the )?(.+?)(?: from| to|\.|\s*$)', prompt, re.I)
    return {'mover': m2.group(1).strip() if m2 else prompt, 'src': None, 'tgt': None}


class GDinoLocator:
    def __init__(self, device='cuda'):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(GDINO_PATH)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_PATH).to(device).eval()
        self.device = device
        self.torch = torch

    def locate(self, frame_rgb, text, box_thr=0.20, text_thr=0.15):
        """frame_rgb: HxWx3 uint8; text: 物体名。返回 (cx, cy, box) 像素坐标, 失败返回 None。"""
        from PIL import Image
        torch = self.torch
        img = Image.fromarray(frame_rgb)
        # GDINO 文本需小写 + 句点结尾
        q = text.lower().strip().rstrip('.') + '.'
        inputs = self.proc(images=img, text=q, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs['input_ids'], threshold=box_thr, text_threshold=text_thr,
            target_sizes=[img.size[::-1]])[0]
        if len(res['boxes']) == 0:
            return None
        # 取分数最高的框
        i = int(res['scores'].argmax())
        x0, y0, x1, y1 = res['boxes'][i].tolist()
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1), (x0, y0, x1, y1), float(res['scores'][i]))


def pixel_to_grid(cx, cy, img_w, img_h, gw, gh):
    """像素中心 -> 特征网格 (gy, gx)。"""
    gx = int(np.clip(cx / img_w * gw, 0, gw - 1))
    gy = int(np.clip(cy / img_h * gh, 0, gh - 1))
    return gy, gx
