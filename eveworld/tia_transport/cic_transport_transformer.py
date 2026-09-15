"""CIC-guided local feature transport for GigaWorld-0.

The module is inserted after the correspondence-selected Transformer block. It
softly transports a low-rank identity summary from frame t-1 to matching cells
in frame t, then feeds the transported residual back through a zero-initialized
projection. Consequently, enabling the module on a baseline checkpoint is an
identity operation before training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from giga_models.acceleration import get_sequence_parallel_group
from giga_models.models.diffusion.giga_world_0.transformer_giga_world_0 import (
    GigaWorld0Transformer3DModel,
)
from giga_models.utils import load_state_dict


TRANSPORT_CONFIG_NAME = "cic_transport_config.json"


class CICTransportAdapter(nn.Module):
    """Adjacent-frame local matching followed by zero-init residual feedback."""

    def __init__(
        self,
        channels: int,
        rank: int = 64,
        window_radius: int = 3,
        temperature: float = 0.07,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if rank <= 0 or rank > channels:
            raise ValueError(f"rank must be in [1, {channels}], got {rank}")
        if window_radius < 0:
            raise ValueError("window_radius must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")

        self.channels = int(channels)
        self.rank = int(rank)
        self.window_radius = int(window_radius)
        self.temperature = float(temperature)
        self.residual_scale = float(residual_scale)
        self.input_proj = nn.Linear(channels, rank, bias=False)
        self.output_proj = nn.Linear(rank, channels, bias=False)
        nn.init.zeros_(self.output_proj.weight)
        self._last_stats: dict[str, torch.Tensor] = {}

    @property
    def kernel_size(self) -> int:
        return 2 * self.window_radius + 1

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, rank={self.rank}, "
            f"window={self.kernel_size}x{self.kernel_size}, "
            f"temperature={self.temperature}, residual_scale={self.residual_scale}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected (B,T,H,W,D), got {tuple(x.shape)}")
        if x.shape[-1] != self.channels:
            raise ValueError(f"expected D={self.channels}, got {x.shape[-1]}")
        if x.shape[1] <= 1:
            return x
        if get_sequence_parallel_group() is not None:
            raise RuntimeError("CIC-Transport does not support sequence parallelism")

        batch, frames, height, width, _ = x.shape
        pairs = batch * (frames - 1)
        locations = height * width
        kernel = self.kernel_size
        candidates = kernel * kernel

        low = self.input_proj(x)
        prev_value = low[:, :-1].permute(0, 1, 4, 2, 3).reshape(
            pairs, self.rank, height, width
        )
        curr_value = low[:, 1:].permute(0, 1, 4, 2, 3).reshape(
            pairs, self.rank, height, width
        )
        prev_norm = F.normalize(prev_value.float(), dim=1, eps=1e-6)
        curr_norm = F.normalize(curr_value.float(), dim=1, eps=1e-6)

        curr_patches = F.unfold(
            curr_norm,
            kernel_size=kernel,
            padding=self.window_radius,
        ).reshape(pairs, self.rank, candidates, locations)
        prev_queries = prev_norm.flatten(2)
        logits = torch.einsum("nrl,nrkl->nkl", prev_queries, curr_patches)
        logits = logits / self.temperature

        valid = F.unfold(
            torch.ones(1, 1, height, width, device=x.device, dtype=torch.float32),
            kernel_size=kernel,
            padding=self.window_radius,
        ).reshape(1, candidates, locations) > 0
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=1)

        prev_flat = prev_value.float().flatten(2)
        contributions = prev_flat.unsqueeze(2) * attention.unsqueeze(1)
        transported = F.fold(
            contributions.reshape(pairs, self.rank * candidates, locations),
            output_size=(height, width),
            kernel_size=kernel,
            padding=self.window_radius,
        )
        mass = F.fold(
            attention,
            output_size=(height, width),
            kernel_size=kernel,
            padding=self.window_radius,
        ).clamp_min_(1e-6)
        transported = transported / mass
        delta = transported - curr_value.float()
        delta = delta.permute(0, 2, 3, 1).to(dtype=x.dtype)
        update = self.residual_scale * torch.tanh(self.output_proj(delta))
        update = update.reshape(batch, frames - 1, height, width, self.channels)

        with torch.no_grad():
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=1)
            self._last_stats = {
                "match_peak": attention.max(dim=1).values.detach().mean(),
                "match_entropy": entropy.detach().mean(),
                "update_rms": update.detach().float().square().mean().sqrt(),
                "feature_rms": x[:, 1:].detach().float().square().mean().sqrt(),
            }
        return torch.cat((x[:, :1], x[:, 1:] + update), dim=1)

    @torch.no_grad()
    def sentinel_stats(self) -> dict[str, float]:
        stats = {
            "input_proj_norm": self.input_proj.weight.detach().float().norm().item(),
            "output_proj_norm": self.output_proj.weight.detach().float().norm().item(),
        }
        for key in ("match_peak", "match_entropy", "update_rms", "feature_rms"):
            value = self._last_stats.get(key)
            stats[key] = float("nan") if value is None else float(value.item())
        feature_rms = stats["feature_rms"]
        stats["relative_update"] = (
            stats["update_rms"] / feature_rms
            if math.isfinite(feature_rms) and feature_rms > 0
            else float("nan")
        )
        return stats


class CICTransportGigaWorld0Transformer3DModel(GigaWorld0Transformer3DModel):
    """GigaWorld-0 with one CIC-Transport adapter after a selected block."""

    def enable_cic_transport(
        self,
        after_block: str = "block22",
        rank: int = 64,
        window_radius: int = 3,
        temperature: float = 0.07,
        residual_scale: float = 0.1,
        init_seed: int = 20260808,
    ) -> "CICTransportGigaWorld0Transformer3DModel":
        if getattr(self, "_cic_transport_enabled", False):
            raise RuntimeError("CIC-Transport is already enabled")
        if after_block not in self.blocks:
            raise KeyError(f"unknown transport block: {after_block}")

        # Do not consume the training RNG. This keeps the paired control and
        # transport runs aligned in data order, augmentation, and diffusion noise.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            adapter = CICTransportAdapter(
                channels=self.config.model_channels,
                rank=rank,
                window_radius=window_radius,
                temperature=temperature,
                residual_scale=residual_scale,
            )
        reference = next(self.parameters())
        adapter.to(device=reference.device, dtype=reference.dtype)
        self.cic_transport = adapter
        self.cic_transport_after_block = str(after_block)
        self.cic_transport_init_seed = int(init_seed)

        def _transport_hook(_module, _inputs, output):
            return self.cic_transport(output)

        self._cic_transport_hook_handle = self.blocks[after_block].register_forward_hook(
            _transport_hook
        )
        self._cic_transport_enabled = True
        return self

    def cic_transport_config(self) -> dict[str, int | float | str]:
        if not getattr(self, "_cic_transport_enabled", False):
            raise RuntimeError("CIC-Transport is not enabled")
        adapter = self.cic_transport
        return {
            "after_block": self.cic_transport_after_block,
            "rank": adapter.rank,
            "window_radius": adapter.window_radius,
            "temperature": adapter.temperature,
            "residual_scale": adapter.residual_scale,
            "init_seed": self.cic_transport_init_seed,
        }

    def save_config(self, save_directory, *args, **kwargs):
        super().save_config(save_directory, *args, **kwargs)
        if getattr(self, "_cic_transport_enabled", False):
            path = Path(save_directory) / TRANSPORT_CONFIG_NAME
            path.write_text(
                json.dumps(self.cic_transport_config(), indent=2),
                encoding="utf-8",
            )

    @classmethod
    def from_pretrained_base(
        cls,
        pretrained_model_name_or_path,
        **transport_config,
    ) -> "CICTransportGigaWorld0Transformer3DModel":
        config = cls.load_config(pretrained_model_name_or_path)
        # Diffusers' regular from_pretrained loader preserves the caller's RNG.
        # Keep that contract so the paired run receives exactly the same data,
        # augmentation, VAE-sampling and diffusion-noise streams as control.
        with torch.random.fork_rng(devices=[]):
            model = cls.from_config(config)
            model.enable_cic_transport(**transport_config)
        state_dict = load_state_dict(pretrained_model_name_or_path)
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.endswith("_extra_state")
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        bad_missing = [key for key in missing if not key.startswith("cic_transport.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"base load mismatch: bad_missing={bad_missing[:8]} "
                f"unexpected={unexpected[:8]}"
            )
        return model

    @classmethod
    def from_pretrained_transport(
        cls,
        pretrained_model_name_or_path,
    ) -> "CICTransportGigaWorld0Transformer3DModel":
        sidecar = Path(pretrained_model_name_or_path) / TRANSPORT_CONFIG_NAME
        if not sidecar.is_file():
            raise FileNotFoundError(f"missing CIC-Transport sidecar: {sidecar}")
        transport_config = json.loads(sidecar.read_text(encoding="utf-8"))
        config = cls.load_config(pretrained_model_name_or_path)
        with torch.random.fork_rng(devices=[]):
            model = cls.from_config(config)
            model.enable_cic_transport(**transport_config)
        state_dict = load_state_dict(pretrained_model_name_or_path)
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.endswith("_extra_state")
        }
        model.load_state_dict(state_dict, strict=True)
        return model
