"""Pure PyTorch mechanics for the EVE-Frontier MVP.

The MVP deliberately stops at prefix + active-block factorization.  It does
not change the backbone attention mask: future tokens are absent from the
forward input.  This module is therefore useful both for CPU mechanism tests
and as the small adapter layer a trainer can call around the existing EDM
model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FrontierSlice:
    """One active frontier and its committed prefix in latent time."""

    index: int
    history: slice
    active: slice
    end: int

    @property
    def history_length(self) -> int:
        return self.history.stop - self.history.start

    @property
    def active_length(self) -> int:
        return self.active.stop - self.active.start


@dataclass(frozen=True)
class FrontierConfig:
    """MVP shape and EDM settings.

    ``condition_latents`` is one for the first-frame latent used by the
    existing GigaWorld-0 image-to-video contract.  Block boundaries remain
    anchored at zero, so a 24-latent clip with block size four has active
    ranges ``[1:4], [4:8], ..., [20:24]``.
    """

    block_size: int = 4
    condition_latents: int = 1
    history_sigma: float = 1e-4
    sigma_data: float = 1.0
    use_flow: bool = True

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.condition_latents <= 0:
            raise ValueError("condition_latents must be positive")
        if self.history_sigma <= 0:
            raise ValueError("history_sigma must be positive")
        if self.sigma_data <= 0:
            raise ValueError("sigma_data must be positive")


@dataclass
class FrontierBatch:
    """Prepared prefix + active tensors for one random frontier."""

    frontier: FrontierSlice
    model_input: torch.Tensor
    target: torch.Tensor
    raw_noisy_active: torch.Tensor
    sigma: torch.Tensor
    timesteps: torch.Tensor
    condition_mask: torch.Tensor
    active_mask: torch.Tensor

    @property
    def input_length(self) -> int:
        return self.model_input.shape[2]

    def active_target(self) -> torch.Tensor:
        return self.target[:, :, self.frontier.active, :, :]

    def active_model_input(self) -> torch.Tensor:
        start = self.frontier.active.start
        return self.model_input[:, :, start : self.frontier.active.stop, :, :]


def latent_frames_from_pixels(num_frames: int, temporal_scale: int = 4) -> int:
    """Match GigaWorld-0's ``1 + (T-1)//scale`` temporal VAE shape."""

    if num_frames <= 0 or temporal_scale <= 0:
        raise ValueError("num_frames and temporal_scale must be positive")
    if (num_frames - 1) % temporal_scale:
        raise ValueError(f"num_frames={num_frames} is not aligned to temporal_scale={temporal_scale}")
    return 1 + (num_frames - 1) // temporal_scale


def pixel_frames_from_latents(num_latents: int, temporal_scale: int = 4) -> int:
    """Inverse of :func:`latent_frames_from_pixels` for aligned clips."""

    if num_latents <= 0 or temporal_scale <= 0:
        raise ValueError("num_latents and temporal_scale must be positive")
    return 1 + (num_latents - 1) * temporal_scale


def frontier_slices(
    num_latents: int,
    block_size: int = 4,
    condition_latents: int = 1,
) -> tuple[FrontierSlice, ...]:
    """Enumerate every valid frontier, including a shorter final block."""

    if num_latents <= condition_latents:
        raise ValueError("the clip must contain at least one latent after the committed condition")
    if block_size <= 0 or condition_latents <= 0:
        raise ValueError("block_size and condition_latents must be positive")

    output = []
    block_start = 0
    index = 0
    while block_start < num_latents:
        active_start = max(block_start, condition_latents)
        active_stop = min(block_start + block_size, num_latents)
        if active_start < active_stop:
            output.append(
                FrontierSlice(
                    index=index,
                    history=slice(0, active_start),
                    active=slice(active_start, active_stop),
                    end=active_stop,
                )
            )
            index += 1
        block_start += block_size
    if not output:
        raise ValueError("no active frontier exists")
    return tuple(output)


def sample_frontier(
    num_latents: int,
    block_size: int = 4,
    condition_latents: int = 1,
    generator: torch.Generator | None = None,
) -> FrontierSlice:
    """Sample a frontier uniformly over frontier *stages*, not tokens."""

    candidates = frontier_slices(num_latents, block_size, condition_latents)
    choice = torch.randint(len(candidates), (), generator=generator).item()
    return candidates[choice]


def fixed_sigma_frontier_batches(
    clean_latents: torch.Tensor,
    sigma: float = 0.5,
    config: FrontierConfig = FrontierConfig(),
    seed: int = 0,
) -> tuple[FrontierBatch, ...]:
    """Build every frontier with deterministic noise at one fixed sigma.

    This is the diagnostic counterpart to random training.  Repeating it with
    the same seed produces bitwise-identical inputs, making per-frontier
    reconstruction curves meaningful instead of mixing sigma/noise variance
    into the comparison.
    """

    if clean_latents.ndim != 5:
        raise ValueError(f"clean_latents must be (B,C,T,H,W), got {clean_latents.shape}")
    generator = torch.Generator(device=clean_latents.device).manual_seed(seed)
    output = []
    for frontier in frontier_slices(clean_latents.shape[2], config.block_size, config.condition_latents):
        shape = (
            clean_latents.shape[0],
            clean_latents.shape[1],
            frontier.active_length,
            clean_latents.shape[3],
            clean_latents.shape[4],
        )
        noise = torch.randn(shape, generator=generator, device=clean_latents.device, dtype=clean_latents.dtype)
        output.append(prepare_frontier_batch(clean_latents, frontier, sigma=sigma, noise=noise, config=config))
    return tuple(output)


def fixed_sigma_frontier_report(
    clean_latents: torch.Tensor,
    predict_fn,
    sigma: float = 0.5,
    config: FrontierConfig = FrontierConfig(),
    seed: int = 0,
) -> tuple[dict[str, object], ...]:
    """Score all fixed-sigma frontiers using a caller-provided model.

    ``predict_fn`` receives one :class:`FrontierBatch` and must return the raw
    backbone prediction with the same shape as ``batch.model_input``.  The
    helper intentionally does not load a model or checkpoint; the same code is
    usable by a CPU toy test and by a GPU checkpoint evaluator.
    """

    report = []
    for batch in fixed_sigma_frontier_batches(clean_latents, sigma=sigma, config=config, seed=seed):
        prediction = predict_fn(batch)
        score = frontier_score_from_prediction(batch, prediction, config)
        report.append(
            {
                "frontier": batch.frontier.index,
                "active_start": batch.frontier.active.start,
                "active_stop": batch.frontier.active.stop,
                "sigma": float(sigma),
                "score_mean": float(score.detach().mean().cpu()),
                "score_min": float(score.detach().min().cpu()),
                "score_max": float(score.detach().max().cpu()),
            }
        )
    return tuple(report)


def _as_batch_sigma(sigma: torch.Tensor | float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = torch.as_tensor(sigma, device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.expand(batch_size)
    elif value.ndim == 1 and value.shape[0] == 1:
        value = value.expand(batch_size)
    elif value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"sigma must be scalar or shape ({batch_size},), got {tuple(value.shape)}")
    if torch.any(value <= 0):
        raise ValueError("active sigma must be positive")
    return value


def edm_precondition_input(noisy: torch.Tensor, sigma: torch.Tensor, use_flow: bool = True, sigma_data: float = 1.0) -> torch.Tensor:
    """Apply the same EDM input preconditioning used by GW-0."""

    shape = (sigma.shape[0],) + (1,) * (noisy.ndim - 1)
    sigma_view = sigma.reshape(shape)
    if use_flow:
        c_in = 1.0 - sigma_view / (1.0 + sigma_view)
    else:
        c_in = 1.0 / torch.sqrt(sigma_view.square() + sigma_data**2)
    return noisy * c_in


def edm_denoise(
    noisy: torch.Tensor,
    model_prediction: torch.Tensor,
    sigma: torch.Tensor,
    use_flow: bool = True,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    """Convert the backbone output to an x0 prediction for active tokens."""

    shape = (sigma.shape[0],) + (1,) * (noisy.ndim - 1)
    sigma_view = sigma.reshape(shape)
    if use_flow:
        c_skip = 1.0 / (1.0 + sigma_view)
        c_out = -sigma_view / (1.0 + sigma_view)
    else:
        c_skip = sigma_data**2 / (sigma_view.square() + sigma_data**2)
        c_out = sigma_view * sigma_data / torch.sqrt(sigma_view.square() + sigma_data**2)
    return c_skip * noisy + c_out * model_prediction


def masked_edm_loss(
    denoised: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    """Compute EDM loss only on masked temporal tokens.

    The denominator counts only active elements per sample, so selecting a
    later frontier does not silently change the loss scale.  A full tensor may
    be passed for diagnostics; history and future gradients are exactly zero
    when their mask entries are zero.
    """

    if denoised.shape != target.shape:
        raise ValueError(f"denoised/target shapes differ: {denoised.shape} vs {target.shape}")
    if loss_mask.shape[0] != denoised.shape[0] or loss_mask.shape[2] != denoised.shape[2]:
        raise ValueError("loss_mask must cover the batch and temporal dimensions")
    batch_size = denoised.shape[0]
    sigma = _as_batch_sigma(sigma, batch_size, denoised.device, denoised.dtype)
    sigma_view = sigma.reshape((batch_size,) + (1,) * (denoised.ndim - 1))
    weight = (sigma_view.square() + sigma_data**2) / (sigma_view * sigma_data).square()
    mask = loss_mask.to(device=denoised.device, dtype=denoised.dtype)
    while mask.ndim < denoised.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(denoised)
    per_element = weight * (denoised.float() - target.float()).square()
    numerator = (per_element * mask.float()).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1)
    if torch.any(denominator <= 0):
        raise ValueError("loss_mask must select at least one element per sample")
    return numerator / denominator


def prepare_frontier_candidate_batch(
    clean_latents: torch.Tensor,
    frontier: FrontierSlice,
    active_target: torch.Tensor,
    sigma: torch.Tensor | float,
    noise: torch.Tensor | None = None,
    config: FrontierConfig = FrontierConfig(),
) -> FrontierBatch:
    """Prepare one candidate under a fixed committed history.

    ``active_target`` can be the immediate-next block or a later block from the
    same clip.  Only that candidate is noised and concatenated to the history;
    the rest of the clip never enters the model input.  Keeping this operation
    explicit is important for skip contrast: next and future candidates can
    share exactly the same history, sigma, and noise.
    """

    if clean_latents.ndim != 5:
        raise ValueError(f"clean_latents must be (B,C,T,H,W), got {clean_latents.shape}")
    batch_size, _, num_latents, _, _ = clean_latents.shape
    expected = frontier_slices(num_latents, config.block_size, config.condition_latents)
    if frontier not in expected:
        raise ValueError("frontier does not match this clip/config")
    expected_active_shape = (
        batch_size,
        clean_latents.shape[1],
        frontier.active_length,
        clean_latents.shape[3],
        clean_latents.shape[4],
    )
    if active_target.shape != expected_active_shape:
        raise ValueError(f"active_target shape {active_target.shape} != expected {expected_active_shape}")
    active_target = active_target.to(device=clean_latents.device, dtype=clean_latents.dtype)
    sigma = _as_batch_sigma(sigma, batch_size, clean_latents.device, clean_latents.dtype)
    if noise is None:
        noise = torch.randn_like(active_target)
    if noise.shape != active_target.shape:
        raise ValueError(f"noise shape {noise.shape} does not match active target {active_target.shape}")
    raw_noisy_active = active_target + noise * sigma.reshape((batch_size, 1, 1, 1, 1))
    input_active = edm_precondition_input(raw_noisy_active, sigma, config.use_flow, config.sigma_data)
    history = clean_latents[:, :, frontier.history, :, :].detach()
    model_input = torch.cat([history, input_active], dim=2)
    target = torch.cat([history, active_target], dim=2)

    history_sigma = torch.full_like(sigma, config.history_sigma)
    history_timestep = history_sigma / (1.0 + history_sigma) if config.use_flow else 0.25 * history_sigma.log()
    active_timestep = sigma / (1.0 + sigma) if config.use_flow else 0.25 * sigma.log()
    timesteps = torch.cat(
        [
            history_timestep.reshape(batch_size, 1, 1).expand(-1, -1, frontier.history_length),
            active_timestep.reshape(batch_size, 1, 1).expand(-1, -1, frontier.active_length),
        ],
        dim=2,
    )[..., None, None]
    condition_mask = torch.zeros((batch_size, 1, frontier.end, 1, 1), device=clean_latents.device, dtype=clean_latents.dtype)
    condition_mask[:, :, : frontier.history_length, :, :] = 1
    active_mask = 1.0 - condition_mask
    return FrontierBatch(
        frontier=frontier,
        model_input=model_input,
        target=target,
        raw_noisy_active=raw_noisy_active,
        sigma=sigma,
        timesteps=timesteps,
        condition_mask=condition_mask,
        active_mask=active_mask,
    )


def prepare_frontier_batch(
    clean_latents: torch.Tensor,
    frontier: FrontierSlice,
    sigma: torch.Tensor | float,
    noise: torch.Tensor | None = None,
    config: FrontierConfig = FrontierConfig(),
) -> FrontierBatch:
    """Create the ordinary immediate-next prefix + active batch."""

    active_target = clean_latents[:, :, frontier.active, :, :]
    return prepare_frontier_candidate_batch(
        clean_latents,
        frontier,
        active_target=active_target,
        sigma=sigma,
        noise=noise,
        config=config,
    )


def prepare_frontier_sampling_inputs(
    committed: torch.Tensor,
    scaled_active: torch.Tensor,
    active_timestep: torch.Tensor | float,
    config: FrontierConfig = FrontierConfig(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the exact transformer inputs for one sequential sampling step.

    ``scaled_active`` is the scheduler-preconditioned current block.  The
    committed prefix is already clean and is therefore concatenated without
    further scaling, matching :func:`prepare_frontier_candidate_batch` during
    training.  Future tokens are not accepted by this interface.
    """

    if committed.ndim != 5 or scaled_active.ndim != 5:
        raise ValueError("committed and scaled_active must be (B,C,T,H,W)")
    if committed.shape[:2] + committed.shape[3:] != scaled_active.shape[:2] + scaled_active.shape[3:]:
        raise ValueError(
            f"committed/active batch, channel, or spatial shapes differ: "
            f"{committed.shape} vs {scaled_active.shape}"
        )
    if committed.shape[2] < config.condition_latents:
        raise ValueError("committed history is shorter than the required condition")
    if scaled_active.shape[2] <= 0:
        raise ValueError("scaled_active must contain at least one temporal token")

    batch_size = committed.shape[0]
    timestep = torch.as_tensor(active_timestep, device=scaled_active.device, dtype=scaled_active.dtype)
    if timestep.ndim == 0:
        timestep = timestep.expand(batch_size)
    elif timestep.ndim == 1 and timestep.shape[0] == 1:
        timestep = timestep.expand(batch_size)
    elif timestep.ndim != 1 or timestep.shape[0] != batch_size:
        raise ValueError(f"active_timestep must be scalar or shape ({batch_size},), got {tuple(timestep.shape)}")

    clean_history = committed.detach()
    latent_input = torch.cat([clean_history, scaled_active], dim=2)
    history_length = clean_history.shape[2]
    total_length = latent_input.shape[2]
    condition_mask = torch.zeros(
        (batch_size, 1, total_length, 1, 1),
        device=latent_input.device,
        dtype=latent_input.dtype,
    )
    condition_mask[:, :, :history_length] = 1
    input_mask = condition_mask.expand(-1, -1, -1, latent_input.shape[-2], latent_input.shape[-1])
    model_input = torch.cat([latent_input, input_mask], dim=1)

    history_sigma = torch.full(
        (batch_size,),
        config.history_sigma,
        device=latent_input.device,
        dtype=latent_input.dtype,
    )
    if config.use_flow:
        history_timestep = history_sigma / (1.0 + history_sigma)
    else:
        history_timestep = 0.25 * history_sigma.log()
    timesteps = torch.cat(
        [
            history_timestep[:, None, None].expand(-1, -1, history_length),
            timestep[:, None, None].expand(-1, -1, scaled_active.shape[2]),
        ],
        dim=2,
    )[..., None, None]
    return model_input, timesteps, condition_mask


def boundary_commit_guard(
    committed: torch.Tensor,
    active: torch.Tensor,
    strength: float = 0.0,
    velocity_scale: float = 0.5,
    decay: float = 1.0,
) -> torch.Tensor:
    """Align a new block to committed boundary motion without rewriting history.

    The target for the first active token is a short velocity extrapolation
    from the final two committed tokens.  The corresponding latent-space
    correction decays across the active block, preserving its interior while
    suppressing a hard boundary jump.
    """

    if committed.ndim != 5 or active.ndim != 5:
        raise ValueError("committed and active must be (B,C,T,H,W)")
    if committed.shape[:2] + committed.shape[3:] != active.shape[:2] + active.shape[3:]:
        raise ValueError(f"committed/active shapes are incompatible: {committed.shape} vs {active.shape}")
    if committed.shape[2] < 1 or active.shape[2] < 1:
        raise ValueError("committed and active must contain temporal tokens")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if velocity_scale < 0:
        raise ValueError("velocity_scale must be non-negative")
    if decay <= 0:
        raise ValueError("decay must be positive")
    if strength == 0:
        return active.detach().clone()

    history = committed.detach().to(device=active.device, dtype=active.dtype)
    anchor = history[:, :, -1]
    if history.shape[2] >= 2:
        anchor = anchor + velocity_scale * (history[:, :, -1] - history[:, :, -2])
    correction = anchor - active[:, :, 0]
    positions = torch.arange(active.shape[2], device=active.device, dtype=torch.float32)
    weights = (strength * torch.exp(-positions / decay)).to(active.dtype)
    return active.detach().clone() + correction.unsqueeze(2) * weights.view(1, 1, -1, 1, 1)


def future_candidate_target(
    clean_latents: torch.Tensor,
    frontier: FrontierSlice,
    config: FrontierConfig = FrontierConfig(),
    offset_blocks: int = 1,
) -> torch.Tensor | None:
    """Return a later block cropped to the next block's length.

    The final frontier has no later block and returns ``None``.  For a shorter
    first/last block, the later block is cropped so both candidates have an
    identical tensor shape and can be scored with the same noise draw.
    """

    if clean_latents.ndim != 5:
        raise ValueError(f"clean_latents must be (B,C,T,H,W), got {clean_latents.shape}")
    if offset_blocks <= 0:
        raise ValueError("offset_blocks must be positive")
    candidates = frontier_slices(clean_latents.shape[2], config.block_size, config.condition_latents)
    if frontier not in candidates:
        raise ValueError("frontier does not match this clip/config")
    target_index = frontier.index + offset_blocks
    if target_index >= len(candidates):
        return None
    source = candidates[target_index]
    start = source.active.start
    stop = start + frontier.active_length
    if stop > clean_latents.shape[2]:
        return None
    return clean_latents[:, :, start:stop, :, :]


def terminal_candidate_target(
    clean_latents: torch.Tensor,
    frontier: FrontierSlice,
    config: FrontierConfig = FrontierConfig(),
) -> torch.Tensor | None:
    """Use the final-state block as a direct process-skipping negative.

    The candidate has the same length as the immediate next block and is
    drawn from the end of the clip.  It is valid only when it lies strictly
    after the active target, so no frontier is contrasted with itself.
    """

    if clean_latents.ndim != 5:
        raise ValueError(f"clean_latents must be (B,C,T,H,W), got {clean_latents.shape}")
    candidates = frontier_slices(clean_latents.shape[2], config.block_size, config.condition_latents)
    if frontier not in candidates:
        raise ValueError("frontier does not match this clip/config")
    start = clean_latents.shape[2] - frontier.active_length
    if start < frontier.active.stop:
        return None
    return clean_latents[:, :, start:, :, :]


def frontier_score_from_prediction(
    batch: FrontierBatch,
    model_prediction: torch.Tensor,
    config: FrontierConfig = FrontierConfig(),
) -> torch.Tensor:
    """Return one EDM energy per sample for a candidate frontier.

    This is deliberately unreduced so two candidates can be compared under the
    same history/noise draw.  ``frontier_loss_from_prediction`` remains the
    usual mean-reduced training loss.
    """

    if model_prediction.shape != batch.model_input.shape:
        raise ValueError(f"prediction shape {model_prediction.shape} must equal model input {batch.model_input.shape}")
    active_start = batch.frontier.history_length
    active_prediction = model_prediction[:, :, active_start:, :, :]
    active_denoised = edm_denoise(
        batch.raw_noisy_active,
        active_prediction,
        batch.sigma,
        config.use_flow,
        config.sigma_data,
    )
    return masked_edm_loss(
        active_denoised,
        batch.active_target(),
        torch.ones_like(batch.active_mask[:, :, active_start:, :, :]),
        batch.sigma,
        config.sigma_data,
    )


def reference_centered_skip_loss(
    model_next: torch.Tensor,
    model_future: torch.Tensor,
    base_next: torch.Tensor,
    base_future: torch.Tensor,
    margin: float = 0.0,
    reduction: str = "mean",
    detach_model_future: bool = False,
) -> torch.Tensor:
    """Rank the immediate next block above a later/terminal candidate.

    Energies are EDM reconstruction errors, so lower is better.  The frozen
    base difference is subtracted from the trainable model difference:

    ``centered = (E_theta(next)-E_theta(future))
                 - (E_base(next)-E_base(future))``.

    A positive centered value means the trained model is *less* next-preferring
    than its base reference.  The hinge therefore penalizes
    ``centered + margin > 0``.  Base scores are detached by construction so
    the reference cannot receive gradients.  ``detach_model_future`` keeps the
    same contrastive value while using the future score only as an anchor.  In
    that mode the objective can win only by improving the immediate next block,
    not by deliberately making a later block harder to reconstruct.
    """

    future_anchor = model_future.detach() if detach_model_future else model_future
    values = (model_next - future_anchor) - (base_next.detach() - base_future.detach())
    if values.ndim == 0:
        values = values.reshape(1)
    for name, value in (
        ("model_next", model_next),
        ("model_future", model_future),
        ("base_next", base_next),
        ("base_future", base_future),
    ):
        if value.shape != values.shape:
            raise ValueError(f"{name} shape {value.shape} does not match score shape {values.shape}")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    loss = F.relu(values + margin)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def frontier_loss_from_prediction(
    batch: FrontierBatch,
    model_prediction: torch.Tensor,
    config: FrontierConfig = FrontierConfig(),
) -> torch.Tensor:
    """Turn a prefix + active backbone prediction into active-only EDM loss."""

    return frontier_score_from_prediction(batch, model_prediction, config).mean()


class CommittedHistory:
    """Immutable committed prefix for a sequential frontier rollout."""

    def __init__(self, committed: torch.Tensor, total_latents: int, config: FrontierConfig):
        if committed.ndim != 5:
            raise ValueError("committed must be (B,C,T,H,W)")
        self._committed = committed.detach().clone()
        self._snapshot = self._committed.clone()
        self.total_latents = total_latents
        self.config = config

    @classmethod
    def from_first_latent(cls, latents: torch.Tensor, config: FrontierConfig = FrontierConfig()) -> "CommittedHistory":
        if latents.shape[2] <= config.condition_latents:
            raise ValueError("latents must contain an active block after the condition")
        return cls(latents[:, :, : config.condition_latents, :, :], latents.shape[2], config)

    @property
    def committed(self) -> torch.Tensor:
        return self._committed.clone()

    @property
    def next_frontier(self) -> FrontierSlice | None:
        for candidate in frontier_slices(self.total_latents, self.config.block_size, self.config.condition_latents):
            if candidate.history_length == self._committed.shape[2]:
                return candidate
        return None

    @property
    def complete(self) -> bool:
        return self._committed.shape[2] == self.total_latents

    def commit(self, active_block: torch.Tensor) -> "CommittedHistory":
        frontier = self.next_frontier
        if frontier is None:
            raise RuntimeError("all latent blocks are already committed")
        expected_shape = self._committed.shape[:2] + (frontier.active_length,) + self._committed.shape[3:]
        if active_block.shape != expected_shape:
            raise ValueError(f"active block shape {active_block.shape} != expected {expected_shape}")
        if not torch.equal(self._committed, self._snapshot):
            raise RuntimeError("committed history was mutated after the previous commit")
        return CommittedHistory(torch.cat([self._committed, active_block.detach().clone()], dim=2), self.total_latents, self.config)


__all__ = [
    "CommittedHistory",
    "boundary_commit_guard",
    "FrontierBatch",
    "FrontierConfig",
    "FrontierSlice",
    "edm_denoise",
    "edm_precondition_input",
    "future_candidate_target",
    "fixed_sigma_frontier_batches",
    "fixed_sigma_frontier_report",
    "frontier_score_from_prediction",
    "frontier_loss_from_prediction",
    "frontier_slices",
    "latent_frames_from_pixels",
    "masked_edm_loss",
    "pixel_frames_from_latents",
    "prepare_frontier_candidate_batch",
    "prepare_frontier_batch",
    "prepare_frontier_sampling_inputs",
    "reference_centered_skip_loss",
    "sample_frontier",
    "terminal_candidate_target",
]
