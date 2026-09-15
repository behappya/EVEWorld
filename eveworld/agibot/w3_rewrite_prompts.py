#!/usr/bin/env python3
"""WMB 适配 D3b:用 Qwen3.6 endpoint 把 trainset 原生指令改写为 WMB 句式。

WMB robotics 官方指令风格: "The robotic arm places the carrot into the metal bowl."
(第三人称、现在时、以 The robotic arm 开头、一句话)
改写后覆盖 trainset_v1/<name>.txt;原文备份到 <name>.txt.orig。
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

BASE = os.environ.get("QWEN_BASE", "http://127.0.0.1:8000/v1")
MODEL = "Qwen/Qwen3.6-35B-A3B"
TRAIN = "/data/datasets/gagi/wmb_adapt/trainset_v1"

SYS = (
    "Rewrite the given robot task command into one English sentence in this exact style: "
    "third person, present tense, starting with 'The robotic arm', describing the action. "
    "Keep all object/color/location details. Output ONLY the rewritten sentence."
)

client = OpenAI(base_url=BASE, api_key="EMPTY")


def rewrite(path):
    orig = open(path).read().strip()
    if os.path.exists(path + ".orig"):
        return path, "skip"
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=80,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": orig}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            out = (r.choices[0].message.content or "").strip().strip('"')
            if out.lower().startswith("the robotic arm") and 4 < len(out.split()) < 40:
                with open(path + ".orig", "w") as f:
                    f.write(orig)
                with open(path, "w") as f:
                    f.write(out)
                return path, "ok"
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return path, "FAIL"


def main():
    files = sorted(glob.glob(f"{TRAIN}/*.txt"))
    files = [f for f in files if not f.endswith(".orig")]
    print(f"待改写: {len(files)}")
    stats = {"ok": 0, "skip": 0, "FAIL": 0}
    with ThreadPoolExecutor(32) as ex:
        for p, st in ex.map(rewrite, files):
            stats[st] += 1
            if st == "FAIL":
                print("FAIL:", p)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
