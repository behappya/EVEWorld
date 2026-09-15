#!/usr/bin/env python3
"""Validate A-pre-noaug against the legacy A-pre runtime before submission."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


GAGI = Path('/data/datasets/gagi')
REPO = Path(__file__).resolve().parents[3]
WORKSPACE = REPO.parent
GOLDEN_RUNTIME = (
    GAGI / 'eve_v2_outputs/t4g_joint_wmapA_pre/runtime_configs/t4g_wmapA_pre150.json'
)
OUTPUT_ROOT = GAGI / 'eve_v2_outputs/t4g_joint_wmapA_pre_noaug'
PACKED = GAGI / 'gr1_finetune_data/packed_data'
PROBE = GAGI / 'eve_v2_outputs/track4gen_probe'
ANNO = PROBE / 't4g_anno'
ASSETS = PROBE / 'aug_assets'
WMAP = PROBE / 'weightmap_cache'
PRETRAIN = GAGI / 'giga_world_0_video_pretrain'


def data_size(part: str) -> int:
    with (PACKED / part / 'config.json').open(encoding='utf-8') as f:
        return int(json.load(f)['data_size'])


def numeric_stems(root: Path, suffix: str) -> set[str]:
    return {p.stem for p in root.glob(f'*{suffix}') if p.stem.isdigit()}


def assert_single_variable_config() -> None:
    for path in (str(WORKSPACE), str(REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from eveworld.pipeline.t4g_apre_noaug_config import config

    with GOLDEN_RUNTIME.open(encoding='utf-8') as f:
        expected = json.load(f)
    expected['project_dir'] = str(OUTPUT_ROOT / 'experiments')
    expected['launch'].pop('executable', None)
    expected['dataloaders']['train']['transform']['p_aug'] = 0.0
    expected['train']['max_steps'] = 300

    if config != expected:
        raise AssertionError(
            'A-pre-noaug drifted beyond project_dir, p_aug, and requested max_steps'
        )

    baseline = copy.deepcopy(expected)
    baseline['project_dir'] = str(GAGI / 'eve_v2_outputs/t4g_joint_wmapA_pre/experiments')
    baseline['dataloaders']['train']['transform']['p_aug'] = 0.5
    baseline['train']['max_steps'] = 150
    with GOLDEN_RUNTIME.open(encoding='utf-8') as f:
        golden = json.load(f)
    golden['launch'].pop('executable', None)
    if baseline != golden:
        raise AssertionError('preflight normalization no longer reproduces the legacy runtime')


def main() -> None:
    required_dirs = (PACKED, ANNO, ASSETS, WMAP, PRETRAIN / 'transformer', PRETRAIN / 'vae')
    for root in required_dirs:
        if not root.is_dir():
            raise FileNotFoundError(root)
    if not GOLDEN_RUNTIME.is_file():
        raise FileNotFoundError(GOLDEN_RUNTIME)

    assert_single_variable_config()

    packed_sizes = {part: data_size(part) for part in ('labels', 'videos', 'prompts')}
    if set(packed_sizes.values()) != {92}:
        raise AssertionError(f'legacy packed data is not 92 samples: {packed_sizes}')

    anno_ids = numeric_stems(ANNO, '.json')
    asset_ids = numeric_stems(ASSETS, '.npz')
    wmap_ids = numeric_stems(WMAP, '.npy')
    if len(anno_ids) != 92 or anno_ids != asset_ids or anno_ids != wmap_ids:
        raise AssertionError(
            f'legacy artifact mismatch: anno={len(anno_ids)} assets={len(asset_ids)} '
            f'wmap={len(wmap_ids)}'
        )

    with (ANNO / '_packidx2vid.json').open(encoding='utf-8') as f:
        idx2vid = json.load(f)
    if len(idx2vid) != 92 or set(idx2vid.values()) != anno_ids:
        raise AssertionError('legacy idx2vid does not match the 92 annotation IDs')

    values: set[float] = set()
    for vid in sorted(wmap_ids):
        wm = np.load(WMAP / f'{vid}.npy')
        if wm.shape != (24, 30, 48) or not np.isfinite(wm).all():
            raise AssertionError(f'vid{vid}: invalid legacy weightmap')
        values.update(float(x) for x in np.unique(wm))
    expected_values = {0.5, 2.0, 3.0, 4.0, 6.0}
    if values != expected_values:
        raise AssertionError(f'legacy weightmap levels changed: {sorted(values)}')

    print(json.dumps({
        'status': 'PASS',
        'comparison': 'legacy A-pre plus no augmentation and requested 300-step budget',
        'samples': 92,
        'p_aug': 0.0,
        'packed_sizes': packed_sizes,
        'weightmap_values': sorted(values),
        'base': str(PRETRAIN / 'transformer'),
        'output': str(OUTPUT_ROOT),
    }, indent=2))


if __name__ == '__main__':
    main()
