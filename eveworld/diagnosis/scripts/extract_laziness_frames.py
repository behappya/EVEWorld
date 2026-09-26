#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GigaWorld-0 偷懒案例抽帧 + 横向拼图工具（EVE Model-Laziness 可视化证据）。

从 side-by-side 评测 mp4（左=输入条件图，右=生成视频）的**右半（生成视频）**里，
抽 4-6 帧按时序从左往右拼成一张横向长图，用来一眼展示"偷懒"：
物体提前到位 / 瞬移 / 手还在空挥（终态早于必要操作）。

用哪个 python 跑（有 cv2 4.11 / PIL / numpy）：
    /home/jovyan/miniconda/envs/EVEWorld/bin/python extract_laziness_frames.py ...

三种模式：
  1) survey：均匀抽 N 帧（带帧号+秒标注）拼 survey 长图，先看整段轨迹再挑帧
       python extract_laziness_frames.py --survey 16 --video <mp4>
  2) id：从脚本内 REGISTRY 取某案例的 frames 列表，产出最终文件夹（复现用）
       python extract_laziness_frames.py --id gigaworld0/dreamgen_98_1
  3) video+frames：即时试某组帧（不进 registry），产出文件夹并打印可粘贴的 registry 片段
       python extract_laziness_frames.py --video <mp4> --frames 0,44,78,100,130,156
       python extract_laziness_frames.py --video <mp4> --times 0,2.75,4.9,6.25,8.1,9.75

输出文件夹 <out_root>/<task>/<model>/<bench>_<dur>/ 内含：
  - fN_idxXXX.png       每张抽出的右半帧（按顺序 + 帧号命名）
  - montage.png         横向拼接长图（可带指令标题 + 每帧秒数）
  - right_video.mp4     右半生成视频（官方 crop=iw/2:ih:iw/2:0；供核实时间戳代表性）
  - instruction.json/txt  指令、源 mp4、model/bench/dur/task、fps、总帧数、帧号、秒、note

批量复现：
    python extract_laziness_frames.py --list --id-prefix gigaworld0_pretrain/
    python extract_laziness_frames.py --all --id-prefix gigaworld0_pretrain/,gigaworld0_gr1_sft/
"""

import argparse
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# REGISTRY：挑好的时间戳写这里，方便复现。
# key = "<model>/<bench>_<dur>_<task>"，frames = 右半视频的帧号列表（4~6 个整数）。
# 用模式 3（--video --frames）试出满意组合后，把打印的片段粘贴到这里即可长期复现。
# ============================================================================
REGISTRY = {
    # 示例案例（frames 由 survey 后填入；见下方 note）
    "gigaworld0/dreamgen_98_1": {
        "video": "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_9p8s_full_20260627_140442/"
                 "1_Use_the_right_hand_to_pick_up_pink_peach_from_top_level_of_the_shelf_"
                 "to_bottom_level_of_the_shelf.mp4",
        "frames": [0, 20, 55, 90, 156],
        "note": "指令=将桃子从货架上层搬到下层。idx0 底层托盘为空; idx20(1.25s) 桃子已直接"
                "出现在底层托盘=目标终态, 无抓取无搬运(瞬移+提前完成); idx55(3.44s) 桃子仍在"
                "底层, 机械臂才从左侧靠近(终态早于动作); idx90(5.62s) 机械臂在桃子旁悬停未抓"
                "取(事后空挥); idx156(9.75s) 结束桃子仍在原位, 搬运从未真正发生。典型 Model "
                "Laziness: 终态远早于任何必要操作。",
    },
    "gigaworld0/dreamgen_58_1": {
        "video": "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_full_20260625_212933/"
                 "1_Use_the_right_hand_to_pick_up_pink_peach_from_top_level_of_the_shelf_"
                 "to_bottom_level_of_the_shelf.mp4",
        "frames": [0, 14, 28, 65, 92],
        "note": "同一指令(桃子上层->下层)的 5.8s 档。偷懒表现为物体凭空生成+数量不守恒幻觉: "
                "idx0(0.00s) 上层货架空、无桃; idx14(0.88s) 凭空冒出两个粉桃(夹爪旁+背景), "
                "非抓取产生; idx28(1.75s) 一个大桃直接出现在双爪之间, 背景仍漂另一个桃(多物体"
                "幻觉); idx65(4.06s) 桃悬于双爪间、背景重影桃仍在; idx92(5.75s) 终态一个桃落在"
                "底层托盘=目标位, 但右上人手里还拿着一个幻觉桃, 全程无'上层抓取->搬运->放置'的"
                "合法过程。典型 Model Laziness: 直接生成终态并伴随物体幻觉。",
    },

    # ---- 魔方任务(下层->上层), 同一 prompt 三个时长档, 偷懒模式各异 ----
    "gigaworld0/dreamgen_98_3": {
        "video": "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_9p8s_full_20260627_140442/"
                 "3_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_the_"
                 "wooden_shelf_to_top_level_of_the_wooden_shelf.mp4",
        "frames": [0, 16, 55, 78, 156],
        "note": "指令=魔方从下层搬到上层。瞬移+提前完成型: idx0 魔方在下层(起点正确); "
                "idx16(1.0s) 魔方已瞬移到上层目标位被右爪托着, 1秒内不可能真实完成下->上搬运"
                "(瞬移+提前完成); idx55(3.44s) 右上冒出第二个红魔方(幻觉); idx78(4.88s) 双魔"
                "方并存于上层区(数量不守恒最明显); idx156(9.75s) 收尾魔方在上层、下层空。",
    },
    "gigaworld0/dreamgen_58_3": {
        "video": "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_full_20260625_212933/"
                 "3_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_the_"
                 "wooden_shelf_to_top_level_of_the_wooden_shelf.mp4",
        "frames": [0, 19, 33, 65, 92],
        "note": "同指令 5.8s 档。终态闪现又消失+未完成型: idx0 起点就两个魔方(上层已有绿魔方"
                "=目标位竟被占, 下层也有一个), 起点异常; idx19(1.19s) 仍上下各一、双爪未动; "
                "idx33(2.06s) 上层魔方消失只剩下层中间一个(状态跳变); idx92(5.75s) 结束只剩下"
                "层一个、上层空=任务实际失败。终态一度出现又消失, 叠加数量不守恒。",
    },
    "gigaworld0/dreamgen_158_3": {
        "video": "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_15p8s_full_20260627_145209/"
                 "3_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_the_"
                 "wooden_shelf_to_top_level_of_the_wooden_shelf.mp4",
        "frames": [0, 51, 126, 190, 252],
        "note": "同指令 15.8s 档。全程双魔方幻觉型: idx0 起点就两个魔方(上层彩色=目标位已被占"
                "+下层绿), 真实起点应只有下层一个; idx51(3.19s) 上层魔方消失只剩下层绿; "
                "idx126(7.88s) 又变回上层彩色+下层绿并存; idx252(15.75s) 结束仍双魔方并存、上"
                "层魔方全程靠人手扶。数量始终不守恒, 无'下层抓取->搬到上层'的合法过程。",
    },

    # ---- 紫玻璃杯任务(蓝盘->青盘), 同一 prompt 三个时长档 ----
    # 三档共性: 起点蓝盘上根本没有紫杯(是个矮红杯), 紫杯直接瞬移生成在目标青盘上,
    # 全程与蓝盘红杯并存, 数量不守恒。起点错+瞬移到终态+提前完成。
    "gigaworld0/dreamgen_98_4": {
        "video": "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_9p8s_full_20260627_140442/"
                 "4_Use_the_left_hand_to_pick_up_tall_purple_glass_from_center_of_blue_"
                 "plate_to_center_of_teal_plate.mp4",
        "frames": [0, 14, 30, 50, 70],
        "note": "指令=左手把高紫玻璃杯从蓝盘搬到青盘。idx0(0.00s) 左蓝盘上是个矮红杯而非紫杯"
                "(起点就错, 紫杯不在蓝盘), 中青盘/右粉盘空; idx31(1.94s) 高紫杯凭空出现在中间"
                "青盘=目标终态, 从未在蓝盘出现也未被抓取搬运(瞬移+提前完成), 蓝盘红杯仍在; "
                "idx55(3.44s) 紫杯稳在青盘全程无'从蓝盘抓取'; idx156(9.75s) 结束紫杯在青盘、蓝"
                "盘红杯仍在, 两杯并存数量不守恒。",
    },
    "gigaworld0/dreamgen_58_4": {
        "video": "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_full_20260625_212933/"
                 "4_Use_the_left_hand_to_pick_up_tall_purple_glass_from_center_of_blue_"
                 "plate_to_center_of_teal_plate.mp4",
        "frames": [0, 8, 22, 55, 60],
        "note": "同指令 5.8s 档。起点错+瞬移+物体形变幻觉: idx0 左蓝盘矮红杯、无紫杯; "
                "idx33(2.06s) 高紫杯凭空出现在中间青盘(目标位), 同时右侧凭空冒出一个扭曲玻璃壶"
                "幻影, 蓝盘红杯还在; idx92(5.75s) 结束紫杯在青盘+右侧畸形玻璃器+蓝盘红杯三容器"
                "并存, 数量不守恒。紫杯直接生成在目标盘, 无抓取搬运。",
    },
    "gigaworld0/dreamgen_158_4": {
        "video": "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side/"
                 "gr1_dreamgen_8gpu_15p8s_full_20260627_145209/"
                 "4_Use_the_left_hand_to_pick_up_tall_purple_glass_from_center_of_blue_"
                 "plate_to_center_of_teal_plate.mp4",
        "frames": [0, 89, 177, 220, 252],
        "note": "同指令 15.8s 档(分辨率最高, 适合主图)。idx0 左蓝盘矮红杯、中青盘/右粉盘空, "
                "起点无紫杯; idx89(5.56s) 高紫杯已在中间青盘=目标位、右爪刚够到, 蓝盘红杯仍在"
                "(紫杯直接生成在终点, 无从蓝盘抓取搬运); idx177(11.06s) 紫杯稳在青盘双杯并存; "
                "idx252(15.75s) 结束仍紫杯在青盘+蓝盘红杯并存, 数量不守恒。",
    },
}


# ---- 本轮新增：DreamGen 任务按 task -> model -> duration 自动展开到 REGISTRY ----
# 如果某个自动展开案例后续要精修帧号，直接在上面的 REGISTRY 手写同 key；
# 下面用 setdefault，不会覆盖手写条目。
DREAMGEN_SIDE_BY_SIDE_ROOT = (
    "/data/datasets/gagi/gr1_dreamgen_eval/generated_side_by_side"
)

DREAMGEN_SWEEP_RUNS_3P8_5P8_7P8 = [
    ("gigaworld0_pretrain", "38",
     "pretrain_dreamgen_8gpu_3p8s_full_short_3p8_7p8"),
    ("gigaworld0_pretrain", "58",
     "pretrain_dreamgen_8gpu_5p8s_full_20260627_162849"),
    ("gigaworld0_pretrain", "78",
     "pretrain_dreamgen_8gpu_7p8s_full_short_3p8_7p8"),
    ("gigaworld0_gr1_sft", "38",
     "sft_dreamgen_8gpu_3p8s_full_short_3p8_7p8"),
    # 5.8s 的 GR1/SFT run 是早期目录名，没有显式 sft/5p8s。
    ("gigaworld0_gr1_sft", "58",
     "gr1_dreamgen_8gpu_full_20260625_212933"),
    ("gigaworld0_gr1_sft", "78",
     "sft_dreamgen_8gpu_7p8s_full_short_3p8_7p8"),
]

DREAMGEN_DEFAULT_FRAMES_BY_DUR = {
    "38": [0, 15, 30, 45, 60],
    "58": [0, 23, 46, 69, 92],
    "78": [0, 31, 62, 93, 124],
}

# 单个视频要精修帧号/说明时改这里即可。key 就是 REGISTRY key，
# 所以同一时长的不同视频也能分别改，不会互相影响。
# 例:
#     "gigaworld0_pretrain/dreamgen_58_86": {
#         "frames": [0, 12, 28, 52, 76],
#         "note": "clear orange cup 5.8s pretrain; 避开人手的针对性帧。",
#     },
DREAMGEN_CASE_OVERRIDES = {
    # task 001: pink peach top shelf -> bottom shelf
    # gigaworld0_pretrain/dreamgen_58_1，wkq可用，偷懒后不知道怎么办了
    "gigaworld0_gr1_sft/dreamgen_38_1": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_1": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_1": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_1": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_1": {"frames": [0, 23, 42, 72, 92]},
    "gigaworld0_pretrain/dreamgen_78_1": {"frames": [0, 31, 62, 93, 124]},

    # task 004: tall purple glass blue plate -> teal plate
    #wkq-可用，gigaworld0_pretrain/dreamgen_58_4，偷懒后不知道怎么办了
    "gigaworld0_gr1_sft/dreamgen_38_4": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_4": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_4": {"frames": [0, 23, 58, 69, 79,91]},
    "gigaworld0_pretrain/dreamgen_38_4": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_4": {"frames": [0, 23, 58, 69, 79,91]},
    "gigaworld0_pretrain/dreamgen_78_4": {"frames": [0, 31, 62, 93, 124]},

    # task 005: green apple bottom shelf -> top shelf
    #wkq 可用gigaworld0_gr1_sft/dreamgen_58_5
    "gigaworld0_gr1_sft/dreamgen_38_5": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_5": {"frames": [0, 23, 46, 50, 60,70]},
    "gigaworld0_gr1_sft/dreamgen_78_5": {"frames": [0, 20, 30,39, 40,124]},
    "gigaworld0_pretrain/dreamgen_38_5": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_5": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_5": {"frames": [0, 31, 62, 93, 124]},

    # task 008: green cucumber beige place mat -> small cyan plate 
    # 左右手错了
    "gigaworld0_gr1_sft/dreamgen_38_8": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_8": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_8": {"frames": [0, 15, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_8": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_8": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_8": {"frames": [0, 31, 62, 93, 124]},

    # task 014: rubik's cube bottom brown wooden shelf -> top shelf
    #wkq 可用，不知道怎么办了gigaworld0_gr1_sft/dreamgen_58_14
    "gigaworld0_gr1_sft/dreamgen_38_14": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_14": {"frames": [0, 35, 50, 60, 67,92]},
    "gigaworld0_gr1_sft/dreamgen_78_14": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_14": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_14": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_14": {"frames": [0, 31, 62, 93, 124]},

    # task 015: tangerine large pink plate -> small teal plate
    "gigaworld0_gr1_sft/dreamgen_38_15": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_15": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_15": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_15": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_15": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_15": {"frames": [0, 31, 62, 93, 124]},

    # task 016: red tomato upper black tray -> brown paper bag
    "gigaworld0_gr1_sft/dreamgen_38_16": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_16": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_16": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_16": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_16": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_16": {"frames": [0, 31, 62, 93, 124]},

    # task 018: rubik's cube bottom wooden shelf -> top wooden shelf
    "gigaworld0_gr1_sft/dreamgen_38_18": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_18": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_18": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_18": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_18": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_18": {"frames": [0, 31, 62, 93, 124]},

    # task 023: grapes white table left side -> bottom level of shelf
    "gigaworld0_gr1_sft/dreamgen_38_23": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_23": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_23": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_23": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_23": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_23": {"frames": [0, 31, 62, 93, 124]},

    # task 029: yellow mustard bottle tan table -> middle white shelf
    "gigaworld0_gr1_sft/dreamgen_38_29": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_29": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_29": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_29": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_29": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_29": {"frames": [0, 31, 62, 93, 124]},

    # task 036: green apple top wooden shelf -> bottom wooden shelf
    "gigaworld0_gr1_sft/dreamgen_38_36": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_36": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_36": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_36": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_36": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_36": {"frames": [0, 31, 62, 93, 124]},

    # task 042: red pepper black top shelf -> bottom white shelf
    "gigaworld0_gr1_sft/dreamgen_38_42": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_42": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_42": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_42": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_42": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_42": {"frames": [0, 31, 62, 93, 124]},

    # task 048: yellow mango right side of table -> white shelf
    "gigaworld0_gr1_sft/dreamgen_38_48": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_48": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_48": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_48": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_48": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_48": {"frames": [0, 31, 62, 93, 124]},

    # task 049: green apple bottom shelf -> top shelf
    "gigaworld0_gr1_sft/dreamgen_38_49": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_49": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_49": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_49": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_49": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_49": {"frames": [0, 31, 62, 93, 124]},

    # task 052: ruik's cube bottom dark three-tier shelf -> top shelf
    "gigaworld0_gr1_sft/dreamgen_38_52": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_52": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_52": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_52": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_52": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_52": {"frames": [0, 31, 62, 93, 124]},

    # task 056: peach top black shelf -> brown paper bag
    "gigaworld0_gr1_sft/dreamgen_38_56": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_56": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_56": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_56": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_56": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_56": {"frames": [0, 31, 62, 93, 124]},

    # task 066: clock lower left shelf -> center of table
    "gigaworld0_gr1_sft/dreamgen_38_66": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_66": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_66": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_66": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_66": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_66": {"frames": [0, 31, 62, 93, 124]},

    # task 069: rubik's cube top three-tier shelf -> bottom brown shelf
    "gigaworld0_gr1_sft/dreamgen_38_69": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_69": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_69": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_69": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_69": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_69": {"frames": [0, 31, 62, 93, 124]},

    # task 079: rubik's cube top wooden shelf -> bottom wooden shelf
    "gigaworld0_gr1_sft/dreamgen_38_79": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_79": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_79": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_79": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_79": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_79": {"frames": [0, 31, 62, 93, 124]},

    # task 080: yellow mustard bottle tan table -> middle white shelf
    "gigaworld0_gr1_sft/dreamgen_38_80": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_80": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_80": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_80": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_80": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_80": {"frames": [0, 31, 62, 93, 124]},

    # task 086：clear orange cup top shelf -> bottom shelf
    #wkq可用gigaworld0_pretrain/dreamgen_58_86
    #wkq可用gigaworld0_pretrain/dreamgen_38_86，38未完成，但是58先完成
    "gigaworld0_gr1_sft/dreamgen_38_86": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_86": {"frames": [0, 15, 22, 30, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_86": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_86": {"frames": [0, 7,10, 22, 39, 45]},
    "gigaworld0_pretrain/dreamgen_58_86": {"frames": [0, 10,15, 24, 30, 31]},
    "gigaworld0_pretrain/dreamgen_78_86": {"frames": [0, 31, 62, 93, 124]},

    # task 091: corn right side of table -> top two-tier wooden shelf
    "gigaworld0_gr1_sft/dreamgen_38_91": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_gr1_sft/dreamgen_58_91": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_gr1_sft/dreamgen_78_91": {"frames": [0, 31, 62, 93, 124]},
    "gigaworld0_pretrain/dreamgen_38_91": {"frames": [0, 15, 30, 45, 60]},
    "gigaworld0_pretrain/dreamgen_58_91": {"frames": [0, 23, 46, 69, 92]},
    "gigaworld0_pretrain/dreamgen_78_91": {"frames": [0, 31, 62, 93, 124]},
}

# 旧 tuple 写法也保留，已有笔记不用迁移；新修改优先用 DREAMGEN_CASE_OVERRIDES。
DREAMGEN_FRAME_OVERRIDES = {
}

DREAMGEN_LAZINESS_TASKS = [
    ("1",
     "1_Use_the_right_hand_to_pick_up_pink_peach_from_top_level_of_the_shelf_"
     "to_bottom_level_of_the_shelf",
     "重要: 执行生成; 桃子上层->下层。先用均匀保底帧, 后续可精修。"),
    ("4",
     "4_Use_the_left_hand_to_pick_up_tall_purple_glass_from_center_of_blue_"
     "plate_to_center_of_teal_plate",
     "重要: 紫杯蓝盘->青盘; 补 pretrain/GR1-SFT 的 3.8/5.8/7.8s 档。"),
    ("5",
     "5_Use_the_right_hand_to_pick_up_green_apple_from_bottom_shelf_to_top_shelf",
     "重要: 执行生成; 绿苹果下层货架->上层货架。按同一套 3.8/5.8/7.8s 抽帧逻辑生成。"),
    ("8",
     "8_Use_the_left_hand_to_pick_up_green_cucumber_from_from_the_beige_"
     "place_mat_to_to_the_small_cyan_plate",
     "重要: 执行生成; 黄瓜垫子->小青盘。先用均匀保底帧。"),
    ("14",
     "14_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_"
     "brown_wooden_shelf_to_top_level_of_brown_wooden_shelf",
     "重要: 执行生成; 魔方下层->上层。先用均匀保底帧。"),
    ("15",
     "15_Use_the_right_hand_to_pick_up_the_tangerine_from_from_the_large_"
     "pink_plate_to_to_the_small_teal_plate",
     "重要: 执行生成; 橘子大粉盘->小青盘。先用均匀保底帧。"),
    ("16",
     "16_Use_the_right_hand_to_pick_up_red_tomato_from_upper_black_tray_of_"
     "plastic_shelf_to_inside_of_brown_paper_bag",
     "有-后来消失了; 番茄上层黑托盘->纸袋。先用均匀保底帧。"),
    ("18",
     "18_Use_the_right_hand_to_pick_up_rubik_s_cube_from_bottom_level_of_the_"
     "wooden_shelf_to_top_level_of_the_wooden_shelf",
     "有, 但是没有影响, 继续放; 魔方下层->上层。先用均匀保底帧。"),
    ("23",
     "23_Use_the_right_hand_to_pick_up_grapes_from_white_table_left_side_to_"
     "bottom_level_of_shelf",
     "有, 但是没有影响, 继续放; 葡萄桌左侧->货架下层。先用均匀保底帧。"),
    ("29",
     "29_Use_the_left_hand_to_pick_up_yellow_mustard_bottle_from_tan_table_"
     "to_middle_level_of_white_shelf",
     "有, 但是没有影响, 继续放; 芥末瓶桌面->白架中层。先用均匀保底帧。"),
    ("36",
     "36_Use_the_right_hand_to_pick_up_green_apple_from_top_tier_of_wooden_"
     "shelf_to_bottom_tier_of_wooden_shelf",
     "有, 但是没有影响, 继续放; 绿苹果上层->下层。先用均匀保底帧。"),
    ("42",
     "42_Use_the_right_hand_to_pick_up_red_pepper_from_black_top_shelf_to_"
     "bottom_white_shelf",
     "重要: 执行生成; 红椒黑色上层架->白色下层架。先用均匀保底帧。"),
    ("48",
     "48_Use_the_right_hand_to_pick_up_yellow_mango_from_right_side_of_table_"
     "to_white_shelf",
     "有, 但是没有影响, 继续放; 芒果桌右侧->白架。先用均匀保底帧。"),
    ("49",
     "49_Use_the_right_hand_to_pick_up_green_apple_from_bottom_shelf_to_top_"
     "shelf",
     "有, 但是没有影响, 继续放; 绿苹果下层->上层。先用均匀保底帧。"),
    ("52",
     "52_Use_the_right_hand_to_pick_up_ruik_s_cube_from_bottom_dark_wooden_"
     "three_tier_shelf_to_top_dark_wooden_three_tier_shelf",
     "有, 但是没有影响, 继续放; ruik_s_cube 下层->上层。先用均匀保底帧。"),
    ("56",
     "56_Use_the_right_hand_to_pick_up_peach_from_top_black_shelf_to_inside_"
     "brown_paper_bag",
     "重要: 执行生成; 桃子黑架上层->纸袋。先用均匀保底帧。"),
    ("66",
     "66_Use_the_left_hand_to_pick_up_clock_from_lower_level_shelf_on_the_"
     "left_side_of_the_table_to_center_of_the_table",
     "不知道怎么办了; 时钟左侧低层架->桌中心。先用均匀保底帧。"),
    ("69",
     "69_Use_the_right_hand_to_pick_up_rubik_s_cube_from_from_the_top_of_the_"
     "three_tiered_wooden_shelf_to_to_the_bottom_of_the_brown_three_tiered_"
     "wooden_shelf",
     "魔方上层->下层; 先用均匀保底帧。"),
    ("79",
     "79_Use_the_right_hand_to_pick_up_rubik_s_cube_from_top_level_of_the_"
     "wooden_shelf_to_bottom_level_of_the_wooden_shelf",
     "魔方上层->下层; 先用均匀保底帧。"),
    ("80",
     "80_Use_the_left_hand_to_pick_up_yellow_mustard_bottle_from_tan_table_"
     "to_middle_level_of_white_shelf",
     "重要: 直接生成, 让另一个消失; 芥末瓶桌面->白架中层。"),
    ("86",
     "86_Use_the_right_hand_to_pick_up_clear_orange_cup_from_top_level_of_"
     "the_shelf_to_bottom_level_of_the_shelf",
     "重要: 又拿回来; 后续精修时尽量不要选到人手。先用均匀保底帧。"),
    ("91",
     "91_Use_the_right_hand_to_pick_up_corn_from_from_the_right_side_of_the_"
     "table_to_to_the_top_of_the_two_tiered_wooden_shelf",
     "不知道怎么办了; 玉米桌右侧->两层木架上层。先用均匀保底帧。"),
]


def dur_to_label(dur):
    """把内部 dur 编码(58/98/158)转成目录友好的 5p8s/9p8s/15p8s。"""
    s = str(dur)
    if len(s) >= 2 and s.isdigit():
        return f"{int(s[:-1])}p{s[-1]}s"
    return s


def task_to_dir(task):
    """任务序号放在最外层目录；数字任务补零方便排序。"""
    s = str(task)
    return f"{int(s):03d}" if s.isdigit() else s


def _register_dreamgen_sweep_cases():
    added = 0
    for task_id, stem, note in DREAMGEN_LAZINESS_TASKS:
        for model, dur, run_dir in DREAMGEN_SWEEP_RUNS_3P8_5P8_7P8:
            video = os.path.join(DREAMGEN_SIDE_BY_SIDE_ROOT, run_dir, stem + ".mp4")
            if not os.path.isfile(video):
                continue
            key = f"{model}/dreamgen_{dur}_{task_id}"
            if key not in REGISTRY:
                added += 1
            entry = {
                "video": video,
                "frames": DREAMGEN_FRAME_OVERRIDES.get(
                    (model, dur, task_id),
                    DREAMGEN_DEFAULT_FRAMES_BY_DUR[dur],
                ),
                "note": f"{note} [{model}, {dur_to_label(dur)}]",
            }
            entry.update(DREAMGEN_CASE_OVERRIDES.get(key, {}))
            REGISTRY.setdefault(key, entry)
    return added


# ---- 指令来源清单（按需惰性加载） ----
DREAMGEN_INPUT_JSON = ("/data/datasets/gagi/gr1_dreamgen_eval/"
                       "giga_input/gr1_dreamgen_it2v.json")
PBENCH_INPUT_JSON = "/home/jovyan/gagibench/pbench/giga_input/pbench_robot_it2v.json"

DEFAULT_OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
REGISTERED_SWEEP_CASES = _register_dreamgen_sweep_cases()


# ============================================================================
# 视频读取（cv2）
# ============================================================================
def open_video(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"视频不存在: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 打不开视频: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    return cap, w, h, n, fps


def read_right_half(cap, idx, w):
    """读第 idx 帧的右半（生成视频），返回 RGB ndarray。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"读取帧 {idx} 失败")
    right = frame[:, w // 2:, :]          # 列 [w//2 : w]，全高
    return cv2.cvtColor(right, cv2.COLOR_BGR2RGB)


# ============================================================================
# 命名推断：model / bench / dur / task
# ============================================================================
def infer_bench(video_path):
    base = os.path.basename(video_path)
    low = video_path.lower()
    if base.startswith("robot_") or "pbench" in low:
        return "pbench"
    return "dreamgen"


def infer_model(video_path):
    low = video_path.lower()
    if "pretrain_dreamgen" in low or "pretrain" in low:
        return "gigaworld0_pretrain"
    if "gr1_dreamgen" in low or "sft_dreamgen" in low or "gr1_pbench" in low:
        return "gigaworld0_gr1_sft"
    if "sft" in low:
        return "gigaworld0_gr1_sft"
    return "gigaworld0"


def infer_dur(video_path, n_frames, fps):
    m = re.search(r"(\d+)p(\d+)s", video_path)
    if m:
        return f"{m.group(1)}{m.group(2)}"       # 9p8s -> 98
    # 兜底：帧数/fps 估算秒，取整数+小数第一位
    sec = n_frames / (fps or 16.0)
    return f"{int(sec)}{int(round((sec - int(sec)) * 10))}"


def infer_task(video_path, bench):
    base = os.path.splitext(os.path.basename(video_path))[0]
    if bench == "pbench":
        m = re.search(r"robot_(\d+)", base)
        return str(int(m.group(1))) if m else base
    # dreamgen：文件名前导 N_
    m = re.match(r"(\d+)_", base)
    return m.group(1) if m else base


# ============================================================================
# 指令解析
# ============================================================================
_dg_cache = None
_pb_cache = None


def resolve_instruction(video_path, bench):
    """返回 (instruction_text, source_desc)。"""
    global _dg_cache, _pb_cache
    base = os.path.basename(video_path)
    stem = os.path.splitext(base)[0]

    if bench == "dreamgen":
        # 首选 json 按 request_id 查 prompt
        if _dg_cache is None and os.path.isfile(DREAMGEN_INPUT_JSON):
            try:
                _dg_cache = {d["request_id"]: d.get("prompt", "")
                             for d in json.load(open(DREAMGEN_INPUT_JSON))}
            except Exception:
                _dg_cache = {}
        if _dg_cache and stem in _dg_cache:
            return _dg_cache[stem], f"dreamgen_json[request_id={stem}]"
        # 兜底：去 N_ 前缀，_ -> 空格
        txt = re.sub(r"^\d+_", "", stem).replace("_", " ").strip()
        return txt, "dreamgen_filename_fallback"

    # pbench：文件名无指令，查 json 按 pbench_id / index
    if _pb_cache is None and os.path.isfile(PBENCH_INPUT_JSON):
        try:
            data = json.load(open(PBENCH_INPUT_JSON))
            _pb_cache = {"by_id": {d.get("pbench_id"): d.get("prompt", "") for d in data},
                         "list": data}
        except Exception:
            _pb_cache = {"by_id": {}, "list": []}
    m = re.search(r"robot_(\d+)", stem)
    if _pb_cache and m:
        rid = f"robot_{int(m.group(1)):03d}"
        if rid in _pb_cache["by_id"]:
            return _pb_cache["by_id"][rid], f"pbench_json[pbench_id={rid}]"
        idx = int(m.group(1))
        if 0 <= idx < len(_pb_cache["list"]):
            return _pb_cache["list"][idx].get("prompt", ""), f"pbench_json[index={idx}]"
    return stem, "pbench_stem_fallback"


# ============================================================================
# 右半视频复制（官方 crop 约定）：优先 imageio_ffmpeg，失败回退 cv2 VideoWriter
# ============================================================================
def export_right_video(src, dst, w, h, fps):
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, "-y", "-i", src, "-vf", "crop=iw/2:ih:iw/2:0",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-preset", "veryfast", "-crf", "18", dst]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
        if r.returncode == 0 and os.path.isfile(dst):
            return "imageio_ffmpeg crop=iw/2:ih:iw/2:0"
        sys.stderr.write(f"[warn] ffmpeg 失败，回退 cv2: {r.stderr.decode()[:200]}\n")
    except Exception as e:
        sys.stderr.write(f"[warn] imageio_ffmpeg 不可用({e})，回退 cv2 VideoWriter\n")

    # 回退：cv2 逐帧裁右半重写
    cap = cv2.VideoCapture(src)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(dst, fourcc, fps, (w - w // 2, h))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr[:, w // 2:, :])
    cap.release()
    vw.release()
    return "cv2_VideoWriter fallback"


# ============================================================================
# 拼图
# ============================================================================
def _font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_montage(frames_rgb, labels, out_path, caption="", sep=4,
                 caption_h=0, label_h=26):
    """等高横向拼接。labels 每帧下方标注（如 'f2 idx078 4.9s'）。caption 顶部指令。"""
    imgs = [Image.fromarray(f) for f in frames_rgb]
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    cap_h = caption_h if caption else 0
    total_w = sum(im.width for im in imgs) + sep * (len(imgs) - 1)
    canvas = Image.new("RGB", (total_w, cap_h + h + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if caption:
        draw.text((6, 4), caption, fill=(0, 0, 0), font=_font(18))
    x = 0
    for im, lab in zip(imgs, labels):
        canvas.paste(im, (x, cap_h))
        if lab:
            draw.text((x + 4, cap_h + h + 3), lab, fill=(0, 0, 0), font=_font(15))
        x += im.width + sep
    canvas.save(out_path)
    return canvas.size


# ============================================================================
# 模式实现
# ============================================================================
def run_survey(video, n, out_root):
    cap, w, h, nf, fps = open_video(video)
    bench = infer_bench(video)
    model = infer_model(video)
    dur = infer_dur(video, nf, fps)
    task = infer_task(video, bench)
    idxs = [int(round(i)) for i in np.linspace(0, nf - 1, n)]
    frames = [read_right_half(cap, i, w) for i in idxs]
    cap.release()
    labels = [f"idx{ix:03d} {ix/fps:.2f}s" for ix in idxs]
    instr, src = resolve_instruction(video, bench)
    d = os.path.join(out_root, "survey", task_to_dir(task), model,
                     f"{bench}_{dur_to_label(dur)}")
    os.makedirs(d, exist_ok=True)
    cap_txt = (f"[SURVEY {n}帧] task={task} {model} "
               f"{bench}_{dur_to_label(dur)} | {instr}")
    make_montage(frames, labels, os.path.join(d, "survey.png"),
                 caption=cap_txt, caption_h=26)
    print(f"[survey] {nf}帧@{fps:.1f}fps -> {os.path.join(d, 'survey.png')}")
    print(f"[survey] 指令({src}): {instr}")
    print(f"[survey] 均匀抽帧号: {idxs}")
    print(f"[survey] 看完这张图，挑 4~6 个偷懒帧，用 --video ... --frames a,b,c,d 试。")
    return d


def build_case(video, frames, out_root, model_override=None,
               bench_override=None, note="", no_caption=False, case_id=None,
               print_snippet=True):
    cap, w, h, nf, fps = open_video(video)
    bench = bench_override or infer_bench(video)
    model = model_override or infer_model(video)
    dur = infer_dur(video, nf, fps)
    task = infer_task(video, bench)

    if not (4 <= len(frames) <= 6):
        raise ValueError(f"frames 数必须 4~6，当前 {len(frames)}: {frames}")
    for ix in frames:
        if not (0 <= ix < nf):
            raise ValueError(f"帧号 {ix} 越界 [0,{nf-1}]")

    out_dir = os.path.join(out_root, task_to_dir(task), model,
                           f"{bench}_{dur_to_label(dur)}")
    os.makedirs(out_dir, exist_ok=True)

    # 抽帧存图
    imgs = []
    for k, ix in enumerate(frames):
        rgb = read_right_half(cap, ix, w)
        imgs.append(rgb)
        Image.fromarray(rgb).save(os.path.join(out_dir, f"f{k}_idx{ix:03d}.png"))
    cap.release()

    instr, src = resolve_instruction(video, bench)
    # 拼图
    labels = [f"f{k} idx{ix:03d} {ix/fps:.2f}s" for k, ix in enumerate(frames)]
    caption = ("" if no_caption else
               f"task={task} {model} {bench}_{dur_to_label(dur)} | {instr}")
    make_montage(imgs, labels, os.path.join(out_dir, "montage.png"),
                 caption=caption, caption_h=26 if caption else 0)

    # 右半视频
    right_mode = export_right_video(video, os.path.join(out_dir, "right_video.mp4"),
                                    w, h, fps)

    # instruction 元数据
    meta = {
        "case_id": case_id or f"{model}/{bench}_{dur}_{task}",
        "instruction": instr, "instruction_source": src,
        "source_video": video, "model": model, "bench": bench,
        "dur": dur, "task": task, "fps": fps, "num_frames": nf,
        "video_wh": [w, h], "right_half_x": w // 2,
        "selected_frames": list(frames),
        "selected_times_sec": [round(ix / fps, 3) for ix in frames],
        "output_layout": "<out_root>/<task>/<model>/<bench>_<dur>",
        "note": note, "right_video_mode": right_mode,
    }
    with open(os.path.join(out_dir, "instruction.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "instruction.txt"), "w") as f:
        f.write(instr + "\n")

    print(f"[case] -> {out_dir}")
    print(f"[case] 指令({src}): {instr}")
    print(f"[case] 帧号 {frames}  秒 {meta['selected_times_sec']}")
    print(f"[case] right_video: {right_mode}")
    # 打印可粘贴 registry 片段（支撑"试满意就留下"）
    if print_snippet:
        print("\n# ---- 可粘贴到 REGISTRY 的片段 ----")
        print(f'    "{model}/{bench}_{dur}_{task}": {{')
        print(f'        "video": "{video}",')
        print(f'        "frames": {list(frames)},')
        print(f'        "note": "{note}",')
        print(f'    }},')
    return out_dir


# ============================================================================
# CLI
# ============================================================================
def parse_frames(s):
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def select_registry_keys(id_prefix=""):
    prefixes = [p.strip() for p in str(id_prefix).split(",") if p.strip()]
    keys = sorted(REGISTRY, key=_registry_sort_key)
    if not prefixes:
        return keys
    return [k for k in keys if any(k.startswith(p) for p in prefixes)]


def _registry_sort_key(key):
    model, rest = key.split("/", 1) if "/" in key else ("", key)
    m = re.match(r"([a-zA-Z0-9]+)_(\d+)_(\d+)$", rest)
    if m:
        bench, dur, task = m.groups()
        task_key = int(task) if task.isdigit() else task
        dur_key = int(dur) if dur.isdigit() else dur
        return (task_key, model, bench, dur_key)
    return (999999, model, rest, 999999)


def run_registry_id(case_id, args, print_snippet=True):
    e = REGISTRY[case_id]
    model = args.model or (case_id.split("/")[0] if "/" in case_id else "")
    return build_case(e["video"], e["frames"], args.out_root,
                      model_override=model or None,
                      bench_override=args.bench or None,
                      note=e.get("note", ""), no_caption=args.no_caption,
                      case_id=case_id, print_snippet=print_snippet)


def main():
    ap = argparse.ArgumentParser(description="GigaWorld-0 偷懒抽帧+拼图")
    ap.add_argument("--survey", type=int, default=0,
                    help="survey 模式：均匀抽 N 帧（如 16）")
    ap.add_argument("--id", type=str, default="",
                    help="id 模式：REGISTRY 的 key，如 gigaworld0/dreamgen_98_1")
    ap.add_argument("--all", action="store_true",
                    help="批量跑 REGISTRY；可配 --id-prefix 限定前缀")
    ap.add_argument("--list", action="store_true",
                    help="列出 REGISTRY key；可配 --id-prefix 限定前缀")
    ap.add_argument("--id-prefix", type=str, default="",
                    help="逗号分隔 key 前缀，如 gigaworld0_pretrain/,gigaworld0_gr1_sft/")
    ap.add_argument("--video", type=str, default="", help="video 模式：mp4 路径")
    ap.add_argument("--frames", type=str, default="", help="逗号分隔帧号，如 0,44,78,100")
    ap.add_argument("--times", type=str, default="", help="逗号分隔秒（按 fps 转帧）")
    ap.add_argument("--model", type=str, default="", help="覆盖 model 名（跨模型阶段用）")
    ap.add_argument("--bench", type=str, default="", choices=["", "dreamgen", "pbench"])
    ap.add_argument("--note", type=str, default="", help="偷懒描述")
    ap.add_argument("--no-caption", action="store_true", help="拼图不加指令标题")
    ap.add_argument("--out-root", type=str, default=DEFAULT_OUT_ROOT)
    a = ap.parse_args()

    if a.survey > 0:
        if not a.video:
            ap.error("--survey 需要 --video")
        run_survey(a.video, a.survey, a.out_root)
        return 0

    if a.list:
        keys = select_registry_keys(a.id_prefix)
        print(f"[list] {len(keys)} cases")
        for k in keys:
            print(k)
        return 0

    if a.all:
        keys = select_registry_keys(a.id_prefix)
        if not keys:
            ap.error(f"--all 未匹配到 REGISTRY key: {a.id_prefix}")
        print(f"[all] running {len(keys)} cases")
        for i, key in enumerate(keys, 1):
            print(f"\n[all] {i}/{len(keys)} {key}")
            run_registry_id(key, a, print_snippet=False)
        print(f"\n[all] done: {len(keys)} cases -> {a.out_root}")
        return 0

    if a.id:
        if a.id not in REGISTRY:
            ap.error(f"REGISTRY 无此 id: {a.id}；已有: {list(REGISTRY)}")
        if not REGISTRY[a.id].get("frames"):
            ap.error(f"{a.id} 的 frames 为空，请先 survey 挑帧后填入 REGISTRY")
        run_registry_id(a.id, a)
        return 0

    if a.video:
        frames = None
        if a.frames:
            frames = parse_frames(a.frames)
        elif a.times:
            _, w, h, nf, fps = open_video(a.video)
            frames = [int(round(float(t) * fps)) for t in a.times.split(",")]
        if not frames:
            ap.error("--video 需配 --frames 或 --times")
        build_case(a.video, frames, a.out_root,
                   model_override=a.model or None, bench_override=a.bench or None,
                   note=a.note, no_caption=a.no_caption)
        return 0

    ap.error("需指定 --survey N | --id KEY | --video PATH(--frames/--times)")


if __name__ == "__main__":
    sys.exit(main())
