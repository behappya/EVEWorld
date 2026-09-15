#!/usr/bin/env python3
"""EWMBench caption 预跑(零 GPU): 对已完成布局转换的模型, 直接经 @12 endpoint
生成 <model>_caption_responses.json 缓存, 与官方 caption_reference 的 key 规则一致。
等 GPU 评测链跑到 semantics 时命中缓存跳过。
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/jovyan/gagibench/EWMBench")
from EWMBench.caption import inference_api, prepare_prompt  # noqa: E402
import json_repair  # noqa: E402

LAYOUT = "/data/datasets/gagi/eve_v2_outputs/ewmbench_gen/eval_layout"
SAVE = "/data/datasets/gagi/eve_v2_outputs/ewmbench_eval"
EP = os.environ.get("QWEN_BASE", "http://127.0.0.1:8000/v1")


def one(args):
    key, vdir = args
    try:
        return key, json_repair.loads(inference_api(EP, vdir, prepare_prompt(vdir)))
    except Exception as e:  # noqa: BLE001
        return key, "Error: " + str(e)


def main(models):
    for m in models:
        out_dir = f"{SAVE}/{m}"
        os.makedirs(out_dir, exist_ok=True)
        out = f"{out_dir}/{m}_caption_responses.json"
        if os.path.exists(out):
            print(f"[skip] {m}")
            continue
        jobs = []
        for vdir in sorted(glob.glob(f"{LAYOUT}/{m}_dataset/*/*/*/video")):
            parts = vdir.split("/")
            task, ep, trial = parts[-4], parts[-3], parts[-2]
            jobs.append((f"{m}_dataset_{task}_{ep}_{trial}", vdir))
        if len(jobs) != 63:
            print(f"[warn] {m}: {len(jobs)} 目录(期望 63)")
        res = {}
        with ThreadPoolExecutor(64) as ex:
            for k, v in ex.map(one, jobs):
                res[k] = v
        bad = sum(1 for v in res.values() if not isinstance(v, dict))
        json.dump(res, open(out, "w"), indent=4)
        print(f"[ok] {m}: {len(res)} 条, 非dict {bad}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
