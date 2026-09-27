#!/usr/bin/env python3
"""Causal-frontier weighting for the block-sequential variant of the model.

The variant trains and generates over committed four-latent blocks instead of
denoising the whole clip at once. A training step samples one frontier, feeds
the committed prefix clean and detached with a mask channel set to one, noises
only the active block, and computes the denoising loss on the active tokens
alone. Generation walks the same blocks in order: the next block is denoised
while conditioned on the clean prefix, then committed without overlap or
correction.

This module holds the two functions the variant is built from: the frontier
weighting used to mask the loss (:func:`frontier_weights`) and the loss module
that applies it (:class:`FrontierLoss`). The block geometry follows the GigaWorld-0
contract: latent blocks of ``BLOCK_SIZE`` frames anchored at zero, with the first
``condition_latents`` frames of a clip always belonging to the committed history.

Run ``python loss.py`` to print the frontier layout of a clip, and
``python loss.py --demo`` to score a synthetic reconstruction at every frontier.
"""

from __future__ import annotations

import argparse
import sys

try:  # torch is required to build the loss, but not to read its layout
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without a torch install
    torch = None
    nn = None

_ModuleBase = nn.Module if nn is not None else object

__all__ = ["BLOCK_SIZE", "HISTORY_SIGMA", "FrontierLoss", "frontier_weights"]

BLOCK_SIZE = 4
HISTORY_SIGMA = 1e-4

if torch is not None:

    def frontier_weights(
        t: int, num_frames: int, block_size: int = BLOCK_SIZE
    ) -> "torch.Tensor":
        """Weight map ``(num_frames,)`` of the active block that starts at ``t``.

        The block spans ``[t, min(t + block_size, num_frames))``; the committed
        prefix before ``t`` and the not-yet-committed frames after the block
        carry weight zero, so a loss term built from this map touches the active
        block only. The frame index ``t`` is the first frame of the block that is
        being reconstructed, i.e. ``1`` for a clip whose first frame is the
        condition and ``4`` for the second block of a four-frame layout.

        Args:
            t: First frame of the active block, in ``[0, num_frames)``.
            num_frames: Number of latent frames in the clip.
            block_size: Latent frames per block; four follows the released
                variant, ``24`` latent frames per 93-frame clip.

        Returns:
            Float32 tensor of shape ``(num_frames,)`` with ones inside the block.
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if not 0 <= t < num_frames:
            raise ValueError(f"t must be in [0, {num_frames}), got {t}")
        weights = torch.zeros(int(num_frames), dtype=torch.float32)
        weights[int(t) : min(int(t) + int(block_size), int(num_frames))] = 1.0
        return weights


class FrontierLoss(_ModuleBase):
    """Prefix + active-block denoising loss with a frontier mask.

    The committed prefix is detached before it enters the model input, so a
    training step never rewrites frames that an earlier frontier already
    produced; the loss itself is taken on the active block only and normalised
    by the number of active tokens, which keeps its scale independent of how
    late the sampled frontier sits in the clip.

    Args:
        block_size: Latent frames per block.
        condition_latents: Latent frames of the clip that are always committed.
        history_sigma: Noise level reported for the committed prefix; the prefix
            itself is clean, and the value only marks the level at which a
            committed block was produced.
        sigma_data: EDM data standard deviation used by the loss weight.
    """

    def __init__(
        self,
        block_size: int = BLOCK_SIZE,
        condition_latents: int = 1,
        history_sigma: float = HISTORY_SIGMA,
        sigma_data: float = 1.0,
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if condition_latents <= 0:
            raise ValueError(f"condition_latents must be positive, got {condition_latents}")
        if condition_latents >= block_size:
            raise ValueError("condition_latents must be smaller than block_size")
        if history_sigma <= 0:
            raise ValueError(f"history_sigma must be positive, got {history_sigma}")
        if sigma_data <= 0:
            raise ValueError(f"sigma_data must be positive, got {sigma_data}")
        self.block_size = int(block_size)
        self.condition_latents = int(condition_latents)
        self.history_sigma = float(history_sigma)
        self.sigma_data = float(sigma_data)

    def frontier_slices(self, num_frames: int) -> list[tuple[int, int]]:
        """Active-block ranges of every frontier, in commit order.

        The slices are ``[condition_latents:block_size]`` for the first block and
        ``[k * block_size:(k + 1) * block_size]`` afterwards, with the final block
        truncated at the end of the clip. A clip of 24 latent frames and a block
        size of four therefore has the active ranges ``[1:4], [4:8], ..., [20:24]``.
        """
        if num_frames <= self.condition_latents:
            return []
        slices = [(self.condition_latents, min(self.block_size, int(num_frames)))]
        start = self.block_size
        while start < num_frames:
            slices.append((start, min(start + self.block_size, int(num_frames))))
            start += self.block_size
        return [block for block in slices if block[0] < block[1]]

    def weights(self, num_frames: int, frontier_index: int) -> "torch.Tensor":
        """Frontier weight map ``(num_frames,)`` for one sampled frontier."""
        blocks = self.frontier_slices(num_frames)
        if not blocks:
            raise ValueError(f"num_frames={num_frames} leaves no active block")
        start, stop = blocks[int(frontier_index)]
        weights = torch.zeros(int(num_frames), dtype=torch.float32)
        weights[start:stop] = 1.0
        return weights

    def prepare_input(
        self,
        clean: "torch.Tensor",
        sigma: float,
        frontier_index: int,
        noise: "torch.Tensor | None" = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Build the model input, target and condition mask of one frontier.

        Args:
            clean: Clean latents ``(B, C, T, h, w)`` of the full clip.
            sigma: Noise level used to corrupt the active block.
            frontier_index: Index into :meth:`frontier_slices`.
            noise: Optional ``(B, C, block, h, w)`` noise for the active block; a
                fresh draw is used when omitted.

        Returns:
            The tuple ``(model_input, target, condition_mask)``. ``model_input`` is
            ``(B, C + 1, stop, h, w)``: the detached clean prefix followed by the
            preconditioned noisy active block, with the condition mask as an extra
            channel that is one on the committed frames and zero on the active
            block. ``target`` is ``(B, C, stop, h, w)`` and holds the clean prefix
            plus the clean active block, so ``model_input`` and ``target`` share a
            shape. Both stop at the end of the active block: frames after the
            frontier never enter the computation.
        """
        if torch is None:  # pragma: no cover - only reachable without torch
            raise RuntimeError("FrontierLoss.prepare_input requires torch")
        if clean.ndim != 5:
            raise ValueError(f"clean latents must be (B, C, T, h, w), got {tuple(clean.shape)}")
        blocks = self.frontier_slices(clean.shape[2])
        if not 0 <= frontier_index < len(blocks):
            raise ValueError(
                f"frontier_index must index one of {len(blocks)} frontiers, got {frontier_index}"
            )
        start, stop = blocks[int(frontier_index)]
        batch, channels = clean.shape[:2]
        history = clean[:, :, :start, :, :].detach()
        active = clean[:, :, start:stop, :, :]
        sigma_tensor = torch.as_tensor(float(sigma), device=clean.device, dtype=clean.dtype)
        if float(sigma) <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if noise is None:
            noise = torch.randn_like(active)
        if noise.shape != active.shape:
            raise ValueError(f"noise {tuple(noise.shape)} does not match active {tuple(active.shape)}")
        noisy_active = active + noise * sigma_tensor
        scaled_active = noisy_active / torch.sqrt(sigma_tensor.square() + self.sigma_data**2)
        model_input = torch.cat([history, scaled_active], dim=2)
        target = torch.cat([history, active], dim=2)
        condition_mask = torch.zeros(
            (batch, 1, stop, 1, 1), device=clean.device, dtype=clean.dtype
        )
        condition_mask[:, :, :start] = 1.0
        model_input = torch.cat([model_input, condition_mask.expand_as(model_input[:, :1])], dim=1)
        return model_input, target, condition_mask

    def forward(
        self,
        denoised: "torch.Tensor",
        target: "torch.Tensor",
        condition_mask: "torch.Tensor | None" = None,
        sigma: "torch.Tensor | float" = 1.0,
    ) -> "torch.Tensor":
        """EDM-weighted reconstruction error on the active block.

        ``denoised`` is the model's clean-latent prediction and ``target`` the
        clean latents, both ``(B, C, T, h, w)``. The error is weighted by
        ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2`` and averaged over
        the tokens where ``condition_mask`` is zero, i.e. over the active block
        only; a mask of ones everywhere (a clip with no committed prefix) scores
        the whole tensor.
        """
        if torch is None:  # pragma: no cover - only reachable without torch
            raise RuntimeError("FrontierLoss.forward requires torch")
        if denoised.shape != target.shape:
            raise ValueError(
                f"denoised {tuple(denoised.shape)} and target {tuple(target.shape)} differ in shape"
            )
        batch = denoised.shape[0]
        sigma_tensor = torch.as_tensor(sigma, device=denoised.device, dtype=denoised.dtype)
        if sigma_tensor.ndim == 0:
            sigma_tensor = sigma_tensor.expand(batch)
        if sigma_tensor.shape != (batch,):
            raise ValueError(f"sigma must be scalar or shape ({batch},), got {tuple(sigma_tensor.shape)}")
        if bool(torch.any(sigma_tensor <= 0)):
            raise ValueError("sigma must be positive")
        sigma_view = sigma_tensor.reshape((batch,) + (1,) * (denoised.ndim - 1))
        weight = (sigma_view.square() + self.sigma_data**2) / (sigma_view * self.sigma_data).square()
        if condition_mask is None:
            active = torch.ones(
                (batch, 1, denoised.shape[2], 1, 1), device=denoised.device, dtype=denoised.dtype
            )
        else:
            if condition_mask.shape[0] != batch:
                raise ValueError("condition_mask must cover the batch dimension")
            active = 1.0 - condition_mask.to(device=denoised.device, dtype=denoised.dtype)
        active = active.expand_as(denoised)
        per_element = weight * (denoised.float() - target.float()).square()
        numerator = (per_element * active).flatten(1).sum(dim=1)
        denominator = active.flatten(1).sum(dim=1)
        if bool(torch.any(denominator <= 0)):
            raise ValueError("the condition mask leaves no active token in at least one sample")
        return (numerator / denominator).mean()


def _parse_args(argv: "list[str] | None") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=24, help="latent frames in the synthetic clip")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="latent frames per block")
    parser.add_argument("--sigma", type=float, default=0.5, help="noise level of the active block")
    parser.add_argument("--frontier", type=int, default=None, help="score a single frontier index")
    parser.add_argument("--seed", type=int, default=0, help="seed of the synthetic batch")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="score a synthetic reconstruction that degrades along the rollout",
    )
    return parser.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    """Print the frontier layout, and with ``--demo`` the loss at every frontier."""
    args = _parse_args(argv)
    if torch is None:  # pragma: no cover - only reachable without torch
        print("torch is required for this script", file=sys.stderr)
        return 1
    loss = FrontierLoss(block_size=args.block_size)
    blocks = loss.frontier_slices(args.frames)
    if not blocks:
        print(f"clip of {args.frames} frames leaves no active block", file=sys.stderr)
        return 1
    print(f"clip: {args.frames} latent frames, block size {args.block_size}, condition latents "
          f"{loss.condition_latents}, history sigma {loss.history_sigma}")
    print(f"{'frontier':>8}  {'active block':>14}  {'tokens':>6}  {'loss':>10}")
    indices = [args.frontier] if args.frontier is not None else list(range(len(blocks)))
    if args.demo:
        torch.manual_seed(args.seed)
        clean = torch.randn(1, 4, args.frames, 4, 6)
    for index in indices:
        if not 0 <= index < len(blocks):
            print(f"frontier {index} out of range (0..{len(blocks) - 1})", file=sys.stderr)
            return 1
        start, stop = blocks[index]
        token_count = stop - start
        score = float("nan")
        if args.demo:
            torch.manual_seed(args.seed + index)
            model_input, target, condition_mask = loss.prepare_input(
                clean, args.sigma, index
            )
            stage = (index + 1) / len(blocks)
            prediction_error = 0.05 * stage * torch.randn_like(target)
            denoised = target + prediction_error
            score = float(loss(denoised, target, condition_mask, args.sigma))
        print(f"{index:>8}  {f'[{start}, {stop})':>14}  {token_count:>6}  {score:>10.4f}")
    if args.demo:
        print("the demo reconstruction degrades with the frontier, so the loss grows along the rollout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
