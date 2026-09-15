#!/usr/bin/env python3
"""AgiBot 配方 checkpoint -> probe 模型目录 (方案 Phase 7, 仿 w7_make_probe)。

probe = checkpoint 的 transformer (非 ema, 与 ewm_vanilla probe 同口径) +
pretrain 底座的 vae/text_encoder。
用法: python agi_make_probe.py --run agi_apre_wmaponly_s800 --step 50
产出: agibot_ewm_apre/probes/agi_apre_<arm>_s<step>/{transformer,vae,text_encoder}
"""
import argparse
import glob
import os
import re

GAGI = '/data/datasets/gagi'
APRE = f'{GAGI}/eve_v2_outputs/agibot_ewm_apre'          # 训练 checkpoint 所在
PROBE_ROOT = f'{GAGI}/agibot_ewm_apre/probes'           # 生成 harness 预期 probe 位置
PRETRAIN = f'{GAGI}/giga_world_0_video_pretrain'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='agi_apre_wmaponly_s800',
                    help='训练 run 名 (定位 experiments 与 probe 命名 arm)')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--ema', action='store_true', help='用 transformer_ema 而非 transformer')
    a = ap.parse_args()

    pats = glob.glob(f'{APRE}/experiments/**/checkpoint_*_step_{a.step}', recursive=True)
    pats = [p for p in pats if os.path.isdir(p)]
    if not pats:
        raise SystemExit(f'未找到 step {a.step} checkpoint: {APRE}/experiments')
    ckpt = sorted(pats)[-1]
    tf_name = 'transformer_ema' if a.ema else 'transformer'
    tf = os.path.join(ckpt, tf_name)
    if not os.path.isfile(os.path.join(tf, 'config.json')):
        raise SystemExit(f'checkpoint 缺 {tf_name}/config.json: {tf}')

    arm = re.sub(r'_s\d+$', '', a.run)                      # agi_apre_wmaponly
    # 生成/评测 harness 认 ewm_apre_* 前缀 (kjob_ewmbench_8gpu_serial.sh case)
    arm = re.sub(r'^agi_apre', 'ewm_apre', arm)             # ewm_apre_wmaponly
    probe_name = f'{arm}_s{a.step}'                         # ewm_apre_wmaponly_s50
    probe = f'{PROBE_ROOT}/{probe_name}'
    os.makedirs(probe, exist_ok=True)
    for name, src in (('transformer', tf),
                      ('vae', f'{PRETRAIN}/vae'),
                      ('text_encoder', f'{PRETRAIN}/text_encoder')):
        dst = os.path.join(probe, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f'probe OK: {probe}\n  transformer -> {tf}')
    print(f'  生成用: MODELS={probe_name}')


if __name__ == '__main__':
    main()
