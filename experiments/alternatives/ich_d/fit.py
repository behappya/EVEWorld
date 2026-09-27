#!/usr/bin/env python3
"""Fit the inter-frame consistency detection head to cached clip features.

The head is a frozen logistic probe over a 16-dimensional descriptor of a
denoised clip, taken from four Transformer blocks: novelty statistics from
blocks 10, 12 and 16, and the consistency statistics of the block where the
temporal adapter is attached. The probe output gates a zero-initialised
residual, so a clip that reads as a repeat of its predecessor is pushed away
from the repeated content. The probe itself is fitted offline and then frozen:
training never updates it.

Rows for every cached noise level are merged before fitting, which matches the
deployment distribution where the noise level varies from step to step. The
per-feature mean and standard deviation are stored with the weights, so the
runtime head can reproduce the exact logits.

Inputs are a ``.npz`` cache with one feature matrix per block and noise level,
``X_block10_<sigma>``, ``X_block12_<sigma>``, ``X_block16_<sigma>`` and
``X_cic_<sigma>``, a binary label vector ``y`` (one for a repeated clip, zero for
a background clip) and an optional ``meta`` JSON string naming the consistency
block. Outputs are the frozen probe parameters and a report of the fit: the
training AUC, the separation between the two class means and the Brier score of
the reconstructed labels.

Run ``python fit.py --cache cic_cell_features.npz --out ich_d_frozen_lr.npz``,
adding ``--loo`` to also report a leave-one-out AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np

__all__ = ["leave_one_out_scores", "main", "rank_auc"]

BLOCKS = ("block10", "block12", "block16")
DEFAULT_SIGMAS = (0.2, 0.4)
CIC_BLOCK = "block22"
NUM_DIM = 16
LOO_REFERENCE = 0.8261


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True, help="npz cache of per-block clip features")
    parser.add_argument("--out", default=None, help="npz file for the frozen probe parameters")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=list(DEFAULT_SIGMAS),
        help="noise levels whose cached rows are merged",
    )
    parser.add_argument(
        "--loo",
        action="store_true",
        help="also refit once per row and report the leave-one-out AUC",
    )
    return parser.parse_args(argv)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Logistic link, evaluated in the numerically stable form."""
    values = np.asarray(logits, dtype=float)
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def rank_auc(positives: np.ndarray, negatives: np.ndarray) -> float:
    """Area under the ROC curve from midranks, i.e. the Mann-Whitney statistic.

    The value is the probability that a randomly drawn positive scores above a
    randomly drawn negative, with ties counting half; ``0.5`` is chance.
    """
    pos = np.asarray(positives, dtype=float).reshape(-1)
    neg = np.asarray(negatives, dtype=float).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both classes are required to score the probe")
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ordered = values[order]
    start = 0
    while start < ordered.size:
        stop = start
        while stop + 1 < ordered.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _feature_matrix(cache, sigma: float, blocks: Sequence[str] = BLOCKS) -> np.ndarray:
    """One cached feature matrix per block at one noise level, stacked column-wise."""
    columns = []
    for block in tuple(blocks) + ("cic",):
        key = f"X_{block}_{sigma:g}"
        if key not in cache:
            raise KeyError(f"the cache has no {key!r} entry")
        columns.append(np.asarray(cache[key], dtype=float))
    matrix = np.hstack(columns)
    if matrix.shape[1] != NUM_DIM:
        raise ValueError(f"expected {NUM_DIM} feature columns, got {matrix.shape[1]}")
    return matrix


def leave_one_out_scores(matrix: np.ndarray, labels: np.ndarray, *, max_iter: int) -> np.ndarray:
    """Refit the probe without each row and return the held-out scores.

    Standardisation is refitted on every fold as well, so a row never
    contributes to the statistics it is scored with.
    """
    from sklearn.linear_model import LogisticRegression

    scores = np.empty(labels.shape[0], dtype=float)
    for index in range(labels.shape[0]):
        keep = np.arange(labels.shape[0]) != index
        mean = matrix[keep].mean(axis=0)
        scale = matrix[keep].std(axis=0) + 1e-8
        model = LogisticRegression(max_iter=max_iter)
        model.fit((matrix[keep] - mean) / scale, labels[keep])
        z = (matrix[index] - mean) / scale
        scores[index] = _sigmoid(z @ model.coef_.ravel() + model.intercept_[0])
    return scores


def _report(name: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    """Train AUC, class-mean separation and Brier score of one row selection."""
    brier = float(np.mean((scores - labels) ** 2))
    auc = rank_auc(scores[labels == 1], scores[labels == 0])
    mean_dup = float(scores[labels == 1].mean())
    mean_bg = float(scores[labels == 0].mean())
    print(
        f"{name:<12} train AUC {auc:.4f}  Brier {brier:.4f}  "
        f"mean dup {mean_dup:.4f}  mean background {mean_bg:.4f}  "
        f"separation {mean_dup - mean_bg:+.4f}"
    )
    return {
        "train_auc": auc,
        "brier": brier,
        "mean_dup": mean_dup,
        "mean_background": mean_bg,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Fit the frozen probe to a cached feature file and report the fit."""
    args = _parse_args(argv)
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print(
            "scikit-learn is required to fit the probe; install the evaluation extras with "
            'pip install -e ".[eval]"',
            file=sys.stderr,
        )
        return 1

    try:
        cache = np.load(args.cache, allow_pickle=True)
    except OSError as error:
        print(f"cannot read {args.cache}: {error}", file=sys.stderr)
        return 1
    if "y" not in cache:
        print(f"{args.cache} has no 'y' label vector", file=sys.stderr)
        return 1
    labels_one = np.asarray(cache["y"]).astype(int).reshape(-1)
    meta = None
    if "meta" in cache:
        meta = json.loads(str(cache["meta"]))
        if meta.get("cic_block") != CIC_BLOCK:
            print(
                f"the cache was built on {meta.get('cic_block')!r}, "
                f"expected {CIC_BLOCK!r}",
                file=sys.stderr,
            )
            return 1

    matrices = {sigma: _feature_matrix(cache, sigma) for sigma in args.sigmas}
    for sigma, matrix in matrices.items():
        if matrix.shape[0] != labels_one.shape[0]:
            print(
                f"X_{BLOCKS[0]}_{sigma:g} has {matrix.shape[0]} rows, "
                f"y has {labels_one.shape[0]}",
                file=sys.stderr,
            )
            return 1
    matrix = np.vstack([matrices[sigma] for sigma in args.sigmas])
    labels = np.concatenate([labels_one] * len(args.sigmas))

    cases = meta.get("n_cases") if meta else None
    print(
        f"cache {args.cache}: {cases if cases is not None else 'unknown'} cases, "
        f"{labels_one.shape[0]} rows per noise level "
        f"({int(labels_one.sum())} duplicate / {int((labels_one == 0).sum())} background), "
        f"{matrix.shape[0]} rows merged over sigma {list(args.sigmas)}, {matrix.shape[1]} dimensions"
    )

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0) + 1e-8
    standardised = (matrix - mean) / scale
    model = LogisticRegression(max_iter=1000)
    model.fit(standardised, labels)
    weights = model.coef_.ravel().astype(np.float64)
    bias = model.intercept_.astype(np.float64)

    scores = _sigmoid(standardised @ weights + bias[0] if bias.size == 1 else standardised @ weights + bias)
    probabilities = model.predict_proba(standardised)[:, 1]
    if not np.allclose(scores, probabilities, atol=1e-10):
        print("the evaluated probe does not reproduce the fitted probabilities", file=sys.stderr)
        return 1
    print("the stored weights reproduce sklearn predict_proba to 1e-10")

    stats = {"merged": _report("merged", scores, labels)}
    for sigma in args.sigmas:
        count = labels_one.shape[0]
        offset = count * list(args.sigmas).index(sigma)
        stats[f"{sigma:g}"] = _report(
            f"sigma {sigma:g}", scores[offset : offset + count], labels[offset : offset + count]
        )

    if args.loo:
        held_out = leave_one_out_scores(standardised, labels, max_iter=1000)
        loo_auc = rank_auc(held_out[labels == 1], held_out[labels == 0])
        print(f"leave-one-out AUC {loo_auc:.4f} (16-dimensional reference at sigma 0.4: {LOO_REFERENCE:.4f})")
        stats["leave_one_out"] = {"auc": loo_auc}

    if args.out:
        payload = {
            "w": weights,
            "b": bias,
            "mu": mean,
            "sd": scale,
            "meta": np.array(
                json.dumps(
                    {
                        "cache": str(args.cache),
                        "sigmas": [float(sigma) for sigma in args.sigmas],
                        "feature_order": list(BLOCKS) + [CIC_BLOCK],
                        "n_cases": cases,
                        "n_rows_merged": int(labels.shape[0]),
                        "stats": stats,
                    }
                )
            ),
        }
        np.savez(args.out, **payload)
        print(f"frozen probe written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
