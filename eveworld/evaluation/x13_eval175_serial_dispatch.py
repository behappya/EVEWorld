#!/usr/bin/env python3
"""Run EVAL-175 model groups sequentially while x9 parallelizes each group."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eveworld.evaluation.x9_eval175_dispatch import MODELS, SPLITS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-base', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--splits', nargs='*', default=None)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', default='42')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument(
        '--x9-script',
        default=str(Path(__file__).with_name('x9_eval175_dispatch.py')),
    )
    args = parser.parse_args()

    splits = args.splits or SPLITS
    max_batch_size = 8 // len(splits)
    if not 1 <= args.batch_size <= max_batch_size:
        raise SystemExit(
            f'batch-size must be in [1, {max_batch_size}] for {len(splits)} splits'
        )
    if len(set(args.models)) != len(args.models):
        raise SystemExit('duplicate model names are not allowed')
    unknown = [model for model in args.models if model not in MODELS]
    if unknown:
        raise SystemExit(f'unknown models: {unknown}')

    batches = [
        args.models[offset : offset + args.batch_size]
        for offset in range(0, len(args.models), args.batch_size)
    ]
    print(
        f'[x13] models={len(args.models)} batches={len(batches)} '
        f'batch_size={args.batch_size} splits={splits}',
        flush=True,
    )
    for index, batch in enumerate(batches, start=1):
        command = [
            args.python,
            args.x9_script,
            '--out-base',
            args.out_base,
            '--python',
            args.python,
            '--seed',
            args.seed,
            '--models',
            *batch,
            '--splits',
            *splits,
        ]
        print(f'[x13] batch {index}/{len(batches)} start: {batch}', flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f'batch {index}/{len(batches)} failed with rc={result.returncode}: {batch}'
            )
        print(f'[x13] batch {index}/{len(batches)} complete: {batch}', flush=True)
    print('[x13] all batches complete', flush=True)


if __name__ == '__main__':
    main()
