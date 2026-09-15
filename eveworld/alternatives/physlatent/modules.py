from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Read from a dict-like or attribute-style config object."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass
class PhysicsLatentConfig:
    enabled: bool = True
    num_tokens: int = 8
    prompt_dim: int = 1024
    latent_channels: int = 16
    token_dim: int = 1024
    hidden_dim: int = 1024
    num_contact_classes: int = 4
    num_phase_classes: int = 8
    num_traj_points: int = 8
    num_attention_heads: int = 8
    spatial_pool_size: tuple[int, int] = (15, 24)
    dropout: float = 0.0
    train_backbone: bool = False
    use_stop_token: bool = False

    @classmethod
    def from_config(cls, config: Any) -> 'PhysicsLatentConfig':
        spatial_pool_size = cfg_get(config, 'spatial_pool_size', cls.spatial_pool_size)
        if isinstance(spatial_pool_size, int):
            spatial_pool_size = (spatial_pool_size, spatial_pool_size)
        spatial_pool_size = tuple(int(x) for x in spatial_pool_size)
        return cls(
            enabled=bool(cfg_get(config, 'enabled', cls.enabled)),
            num_tokens=int(cfg_get(config, 'num_tokens', cls.num_tokens)),
            prompt_dim=int(cfg_get(config, 'prompt_dim', cls.prompt_dim)),
            latent_channels=int(cfg_get(config, 'latent_channels', cls.latent_channels)),
            token_dim=int(cfg_get(config, 'token_dim', cls.token_dim)),
            hidden_dim=int(cfg_get(config, 'hidden_dim', cls.hidden_dim)),
            num_contact_classes=int(cfg_get(config, 'num_contact_classes', cls.num_contact_classes)),
            num_phase_classes=int(cfg_get(config, 'num_phase_classes', cls.num_phase_classes)),
            num_traj_points=int(cfg_get(config, 'num_traj_points', cls.num_traj_points)),
            num_attention_heads=int(cfg_get(config, 'num_attention_heads', cls.num_attention_heads)),
            spatial_pool_size=spatial_pool_size,
            dropout=float(cfg_get(config, 'dropout', cls.dropout)),
            train_backbone=bool(cfg_get(config, 'train_backbone', cls.train_backbone)),
            use_stop_token=bool(cfg_get(config, 'use_stop_token', cls.use_stop_token)),
        )


class PhysicsLatentEncoder(nn.Module):
    """Generate physics-aware condition tokens for GigaWorld cross attention.

    Reference VAE latents provide visual state, prompt embeddings provide task
    semantics, and learned queries attend over pooled spatial latent tokens plus
    prompt tokens. This keeps the adapter lightweight while giving physics
    tokens access to object layout instead of only a global latent mean.
    """

    def __init__(
        self,
        num_tokens: int = 8,
        prompt_dim: int = 1024,
        latent_channels: int = 16,
        token_dim: int = 1024,
        hidden_dim: int = 1024,
        num_contact_classes: int = 4,
        num_phase_classes: int = 8,
        num_traj_points: int = 8,
        num_attention_heads: int = 8,
        spatial_pool_size: tuple[int, int] = (15, 24),
        dropout: float = 0.0,
        use_stop_token: bool = False,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_traj_points = num_traj_points
        self.num_phase_classes = num_phase_classes
        self.use_stop_token = use_stop_token
        self.spatial_pool_size = spatial_pool_size

        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent_channels),
            nn.Linear(latent_channels, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.token_queries = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
        self.token_type = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
        self.temporal_bin_embed = nn.Parameter(torch.zeros(num_tokens, token_dim))
        self.context_norm = nn.LayerNorm(token_dim)
        self.query_norm = nn.LayerNorm(token_dim)
        self.context_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )

        self.state_head = nn.Linear(token_dim, 4)
        self.goal_head = nn.Linear(token_dim, 4)
        self.contact_head = nn.Linear(token_dim, num_contact_classes)
        self.traj_head = nn.Linear(token_dim, 2)
        self.phase_head = nn.Linear(token_dim, num_phase_classes)
        self.done_head = nn.Linear(token_dim, 1)
        self.goal_reached_head = nn.Linear(token_dim, 1)
        self.release_head = nn.Linear(token_dim, 1)
        self.object_motion_head = nn.Linear(token_dim, 1)
        if self.use_stop_token:
            self.stop_query = nn.Parameter(torch.randn(1, token_dim) * 0.02)
            self.stop_mlp = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, token_dim),
            )

    def forward(
        self,
        ref_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        ref_masks: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build physics latent tokens.

        Args:
            ref_latents: Reference latents with shape ``(B, C, T, H, W)``.
            prompt_embeds: T5 embeddings with shape ``(B, L, D)``.
            ref_masks: Optional latent mask with shape broadcastable to
                ``(B, 1, T, 1, 1)``. When present, only reference timesteps
                contribute to the visual summary.
            return_aux: Return auxiliary head predictions for optional losses.
        """
        prompt_mask = self._prompt_mask(prompt_embeds)
        prompt_summary = self._masked_prompt_mean(prompt_embeds, prompt_mask)
        latent_summary = self._masked_latent_mean(ref_latents, ref_masks)
        prompt_tokens = self.prompt_proj(prompt_embeds)
        latent_tokens = self._spatial_latent_tokens(ref_latents, ref_masks)

        fused = self.prompt_proj(prompt_summary) + self.latent_proj(latent_summary)
        fused = self.fuse(fused)
        tokens = (
            self.token_queries.unsqueeze(0)
            + self.token_type.unsqueeze(0)
            + self.temporal_bin_embed.unsqueeze(0)
            + fused.unsqueeze(1)
        )
        context = torch.cat([prompt_tokens, latent_tokens], dim=1)
        context = self.context_norm(context)
        key_padding_mask = torch.cat(
            [
                ~prompt_mask.squeeze(-1),
                torch.zeros(
                    latent_tokens.shape[:2],
                    dtype=torch.bool,
                    device=latent_tokens.device,
                ),
            ],
            dim=1,
        )
        attn_tokens, _ = self.context_attn(
            query=self.query_norm(tokens),
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + attn_tokens
        tokens = tokens + self.token_mlp(tokens)

        output_tokens = tokens
        if self.use_stop_token:
            stop_context = self.stop_query.unsqueeze(0) + fused.unsqueeze(1) + tokens.mean(dim=1, keepdim=True)
            stop_token = stop_context + self.stop_mlp(stop_context)
            output_tokens = torch.cat([tokens, stop_token], dim=1)

        if not return_aux:
            return output_tokens

        pooled = tokens.mean(dim=1)
        aux = {
            'state': self.state_head(pooled),
            'goal': self.goal_head(pooled),
            'contact_logits': self.contact_head(tokens),
            'trajectory': self.traj_head(tokens),
            'phase_logits': self.phase_head(tokens),
            'done_logits': self.done_head(tokens).squeeze(-1),
            'goal_reached_logits': self.goal_reached_head(tokens).squeeze(-1),
            'release_logits': self.release_head(tokens).squeeze(-1),
            'object_motion': self.object_motion_head(tokens).squeeze(-1),
        }
        return output_tokens, aux

    @staticmethod
    def _prompt_mask(prompt_embeds: torch.Tensor) -> torch.Tensor:
        # Packed prompt embeddings are zero padded by T5TextEncoder.
        return prompt_embeds.abs().sum(dim=-1, keepdim=True) > 0

    @staticmethod
    def _masked_prompt_mean(prompt_embeds: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = PhysicsLatentEncoder._prompt_mask(prompt_embeds)
        denom = mask.sum(dim=1).clamp(min=1).to(prompt_embeds.dtype)
        return (prompt_embeds * mask.to(prompt_embeds.dtype)).sum(dim=1) / denom

    @staticmethod
    def _masked_latent_mean(ref_latents: torch.Tensor, ref_masks: torch.Tensor | None) -> torch.Tensor:
        if ref_masks is None:
            return ref_latents.mean(dim=(2, 3, 4))

        mask = ref_masks.to(device=ref_latents.device, dtype=ref_latents.dtype)
        while mask.ndim < ref_latents.ndim:
            mask = mask.unsqueeze(-1)
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        weighted = ref_latents * mask
        denom = mask.expand_as(ref_latents).sum(dim=(2, 3, 4)).clamp(min=1)
        return weighted.sum(dim=(2, 3, 4)) / denom

    def _spatial_latent_tokens(
        self,
        ref_latents: torch.Tensor,
        ref_masks: torch.Tensor | None,
    ) -> torch.Tensor:
        if ref_masks is None:
            spatial = ref_latents.mean(dim=2)
        else:
            mask = ref_masks.to(device=ref_latents.device, dtype=ref_latents.dtype)
            while mask.ndim < ref_latents.ndim:
                mask = mask.unsqueeze(-1)
            if mask.shape[1] != 1:
                mask = mask[:, :1]
            weighted = ref_latents * mask
            denom = mask.sum(dim=2).clamp(min=1)
            spatial = weighted.sum(dim=2) / denom

        spatial = F.adaptive_avg_pool2d(spatial, self.spatial_pool_size)
        spatial = spatial.flatten(2).transpose(1, 2)
        return self.latent_proj(spatial)


def append_physics_tokens(prompt_embeds: torch.Tensor, physics_tokens: torch.Tensor) -> torch.Tensor:
    """Append physics tokens to text tokens along the cross-attention axis."""
    if prompt_embeds.ndim != 3 or physics_tokens.ndim != 3:
        raise ValueError('prompt_embeds and physics_tokens must be rank-3 tensors')
    if prompt_embeds.shape[0] != physics_tokens.shape[0]:
        raise ValueError('batch size mismatch between prompt and physics tokens')
    if prompt_embeds.shape[-1] != physics_tokens.shape[-1]:
        raise ValueError('embedding dimension mismatch between prompt and physics tokens')
    return torch.cat([prompt_embeds, physics_tokens], dim=1)
