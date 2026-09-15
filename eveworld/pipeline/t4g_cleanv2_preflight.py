#!/usr/bin/env python3
"""Validate every artifact used by A-pre-clean-v4-u3 before GPU training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GAGI = Path('/data/datasets/gagi')
PROBE = GAGI / 'eve_v2_outputs/track4gen_probe'


def numeric_ids(root: Path, suffix: str) -> set[str]:
    return {p.stem for p in root.glob(f'*{suffix}') if p.stem.isdigit()}


def data_size(packed: Path, part: str) -> int:
    with (packed / part / 'config.json').open(encoding='utf-8') as f:
        return int(json.load(f)['data_size'])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path,
                    default=GAGI / 'gr1_finetune_data/raw_data_t4g_nohuman_v2')
    ap.add_argument('--packed', type=Path,
                    default=GAGI / 'gr1_finetune_data/packed_data_t4g_nohuman_v2')
    ap.add_argument('--anno', type=Path, default=PROBE / 't4g_anno_nohuman_v2')
    ap.add_argument('--assets', type=Path,
                    default=PROBE / 'aug_assets_nohuman_v2_cleanframe')
    ap.add_argument('--old-assets', type=Path, default=PROBE / 'aug_assets_nohuman_v2')
    ap.add_argument('--wmap', type=Path,
                    default=PROBE / 'weightmap_cache_uniform3_armfix_v4_nohuman_v2')
    ap.add_argument('--expected', type=int, default=91)
    args = ap.parse_args()

    for root in (args.raw, args.packed, args.anno, args.assets, args.wmap):
        if not root.is_dir():
            raise FileNotFoundError(root)

    raw_mp4 = numeric_ids(args.raw, '.mp4')
    raw_txt = numeric_ids(args.raw, '.txt')
    anno_ids = numeric_ids(args.anno, '.json')
    asset_ids = numeric_ids(args.assets, '.npz')
    wmap_ids = numeric_ids(args.wmap, '.npy')
    expected_ids = raw_mp4
    groups = {
        'raw_txt': raw_txt,
        'anno': anno_ids,
        'assets': asset_ids,
        'wmap': wmap_ids,
    }
    if len(expected_ids) != args.expected or '32' in expected_ids:
        raise AssertionError(f'raw ids invalid: count={len(expected_ids)} has32={"32" in expected_ids}')
    for name, ids in groups.items():
        if ids != expected_ids:
            raise AssertionError(
                f'{name} ids differ: missing={sorted(expected_ids - ids)} extra={sorted(ids - expected_ids)}')

    with (args.anno / '_packidx2vid.json').open(encoding='utf-8') as f:
        idx2vid = json.load(f)
    lex_ids = sorted(expected_ids)
    if list(idx2vid) != [str(i) for i in range(args.expected)]:
        raise AssertionError('idx2vid keys are not contiguous 0..90')
    if list(idx2vid.values()) != lex_ids:
        raise AssertionError('idx2vid values do not match pack_data.py lexicographic order')

    packed_sizes = {part: data_size(args.packed, part) for part in ('labels', 'videos', 'prompts')}
    if set(packed_sizes.values()) != {args.expected}:
        raise AssertionError(f'packed sizes invalid: {packed_sizes}')

    wmap_union: set[float] = set()
    coverage = []
    for vid in lex_ids:
        with (args.anno / f'{vid}.json').open(encoding='utf-8') as f:
            anno = json.load(f)
        if len(anno.get('per_lat_frame', [])) != 24:
            raise AssertionError(f'vid{vid}: per_lat_frame length is not 24')

        with np.load(args.assets / f'{vid}.npz') as asset:
            if asset['zones'].shape != (24, 30, 48):
                raise AssertionError(f'vid{vid}: invalid zones shape {asset["zones"].shape}')
            if vid in {'2', '89'} and not {'patch', 'box'}.issubset(asset.files):
                raise AssertionError(f'vid{vid}: rebuilt asset lacks patch/box')

        wm = np.load(args.wmap / f'{vid}.npy')
        if wm.shape != (24, 30, 48) or not np.isfinite(wm).all():
            raise AssertionError(f'vid{vid}: invalid weightmap shape or values')
        values = {float(x) for x in np.unique(wm)}
        if not values.issubset({1.0, 3.0}):
            raise AssertionError(f'vid{vid}: invalid weightmap levels {sorted(values)}')
        wmap_union |= values
        coverage.append(float((wm > 1.0).mean()))

    if wmap_union != {1.0, 3.0}:
        raise AssertionError(f'global weightmap values are {sorted(wmap_union)}')
    w16 = np.load(args.wmap / '16.npy')
    if not np.all(w16[:, 0:9, 0:18] == 1.0):
        raise AssertionError('vid16 background-person exclusion region is not all 1x')

    for vid in ('2', '89'):
        with np.load(args.old_assets / f'{vid}.npz') as old, np.load(args.assets / f'{vid}.npz') as new:
            same_patch = ('patch' in old.files and 'patch' in new.files and
                          np.array_equal(old['patch'], new['patch']))
            same_box = ('box' in old.files and 'box' in new.files and
                        np.array_equal(old['box'], new['box']))
            if same_patch and same_box:
                raise AssertionError(f'vid{vid}: patch and box still match stale pre-trim asset')

    print(json.dumps({
        'status': 'PASS',
        'samples': args.expected,
        'packed_sizes': packed_sizes,
        'weightmap_values': sorted(wmap_union),
        'coverage_min_median_max': [min(coverage), float(np.median(coverage)), max(coverage)],
        'vid32_present': False,
        'repatched_assets': [2, 89],
    }, indent=2))


if __name__ == '__main__':
    main()
