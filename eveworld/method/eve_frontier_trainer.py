"""GigaTrain runner for the EVE-Frontier prefix + active-block MVP."""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any

import torch

from eveworld.alternatives.physlatent.modules import cfg_get
from eveworld.alternatives.physlatent.trainer import PhysicsLatentGigaWorld0Trainer

from .eve_frontier_loss import (
    FrontierConfig,
    frontier_score_from_prediction,
    future_candidate_target,
    prepare_frontier_batch,
    prepare_frontier_candidate_batch,
    reference_centered_skip_loss,
    sample_frontier,
    terminal_candidate_target,
)


class EveFrontierTrainer(PhysicsLatentGigaWorld0Trainer):
    """Adapt the existing GW-0 trainer without changing the backbone.

    Each training example presents only the committed prefix and one noisy
    active block.  The inherited VAE, LoRA wrapping, DeepSpeed setup, and
    prompt/data path remain unchanged; only the temporal factorization and
    active-only EDM objective are replaced.
    """

    def get_models(self, model_config: Any):
        frontier_config = cfg_get(model_config, "frontier", {})
        self.frontier_config = FrontierConfig(
            block_size=int(cfg_get(frontier_config, "block_size", 4)),
            condition_latents=int(cfg_get(frontier_config, "condition_latents", 1)),
            history_sigma=float(cfg_get(frontier_config, "history_sigma", 1e-4)),
            sigma_data=float(cfg_get(frontier_config, "sigma_data", 1.0)),
            use_flow=bool(cfg_get(frontier_config, "use_flow", True)),
        )
        self.frontier_skip_weight = float(cfg_get(frontier_config, "skip_weight", 0.0))
        self.frontier_skip_margin = float(cfg_get(frontier_config, "skip_margin", 0.0))
        self.frontier_skip_offset = int(cfg_get(frontier_config, "skip_future_offset", 1))
        self.frontier_skip_negative_mode = str(
            cfg_get(frontier_config, "skip_negative_mode", "adjacent")
        )
        self.frontier_skip_detach_future = bool(
            cfg_get(frontier_config, "skip_detach_future", False)
        )
        self.frontier_skip_diagnostic = bool(cfg_get(frontier_config, "skip_diagnostic", False))
        if self.frontier_skip_weight < 0:
            raise ValueError("frontier.skip_weight must be non-negative")
        if self.frontier_skip_margin < 0:
            raise ValueError("frontier.skip_margin must be non-negative")
        if self.frontier_skip_offset <= 0:
            raise ValueError("frontier.skip_future_offset must be positive")
        if self.frontier_skip_negative_mode not in {"adjacent", "terminal"}:
            raise ValueError("frontier.skip_negative_mode must be 'adjacent' or 'terminal'")
        configured_train_mode = cfg_get(model_config, "train_mode", "full")
        if self.frontier_skip_weight > 0 and configured_train_mode != "lora":
            raise ValueError(
                "reference-centered skip loss currently requires train_mode='lora' "
                "so the frozen base can be evaluated by disabling the adapter"
            )
        models = super().get_models(model_config)
        return models

    def _forward_frontier_model(
        self,
        transformer,
        batch,
        prompt_embeds: torch.Tensor,
        padding_mask: torch.Tensor,
        fps: Any,
    ) -> torch.Tensor:
        """Run one prefix + active candidate under the GW-0 input contract."""

        # The extra clean-history channel follows the latent spatial grid
        # (60x96 for 480x768 video), while ``padding_mask`` keeps the baseline
        # trainer's original image-space contract.
        input_masks = batch.condition_mask.repeat(
            1, 1, 1, batch.model_input.shape[-2], batch.model_input.shape[-1]
        )
        model_input = torch.cat([batch.model_input, input_masks], dim=1).to(self.dtype)
        if self.train_mode == "lora":
            model_input.requires_grad_(True)
        return transformer(
            x=model_input,
            timesteps=batch.timesteps.to(self.dtype),
            crossattn_emb=prompt_embeds,
            padding_mask=padding_mask,
            fps=fps,
        )

    @contextmanager
    def _frozen_base_adapter(self):
        """Temporarily evaluate the pretrained base with LoRA disabled."""

        # The trainer wraps the whole ModuleDict in DeepSpeed, so unwrap that
        # container before selecting the transformer.  A second ``module``
        # unwrap handles an engine around the transformer itself.
        model_container = getattr(self.model, "module", self.model)
        module = model_container["transformer"]
        module = getattr(module, "module", module)
        peft_model = getattr(module, "model", module)
        disable_adapter = getattr(peft_model, "disable_adapter", None)
        base_model = getattr(peft_model, "base_model", peft_model)
        disable_layers = getattr(peft_model, "disable_adapters", None)
        enable_layers = getattr(peft_model, "enable_adapters", None)
        if disable_layers is None or enable_layers is None:
            disable_layers = getattr(base_model, "disable_adapter_layers", None)
            enable_layers = getattr(base_model, "enable_adapter_layers", None)
        if disable_adapter is None and (disable_layers is None or enable_layers is None):
            raise RuntimeError(
                "skip reference requested but transformer has no PEFT adapter-disable API "
                f"(module={type(module).__name__}, reference={type(peft_model).__name__})"
            )
        was_training = module.training
        module.eval()
        try:
            if disable_adapter is not None:
                with disable_adapter():
                    yield
            else:
                disable_layers()
                try:
                    yield
                finally:
                    enable_layers()
        finally:
            module.train(was_training)

    def _log_skip_diagnostics(
        self,
        frontier,
        next_score: torch.Tensor,
        future_score: torch.Tensor,
        base_next_score: torch.Tensor,
        base_future_score: torch.Tensor,
    ) -> None:
        """Log preference gaps without adding metrics to the training loss."""

        if not self.is_main_process:
            return
        raw_gap = (next_score.detach() - future_score.detach()).mean()
        base_gap = (base_next_score.detach() - base_future_score.detach()).mean()
        centered_per_sample = (
            next_score.detach()
            - future_score.detach()
            - (base_next_score.detach() - base_future_score.detach())
        )
        centered_gap = centered_per_sample.mean()
        violation_rate = ((centered_per_sample + self.frontier_skip_margin) > 0).float().mean()
        self.logger.info(
            "[FrontierSkip] step=%s frontier=%s raw_gap=%.6f base_gap=%.6f centered_gap=%.6f hinge_active=%.3f",
            getattr(self, "_cur_step", "?"),
            frontier.index,
            raw_gap.item(),
            base_gap.item(),
            centered_gap.item(),
            violation_rate.item(),
        )

    def forward_step(self, batch_dict: dict[str, Any]):
        transformer = functools.partial(self.model, "transformer")
        images = batch_dict["images"]
        prompt_embeds = batch_dict["prompt_embeds"].to(self.dtype)
        batch_size = images.shape[0]
        padding_mask = torch.zeros(
            (batch_size, 1, images.shape[-2], images.shape[-1]),
            dtype=self.dtype,
            device=self.device,
        )
        fps = batch_dict["fps"][0]

        clean_latents = self.forward_vae(images)
        # EDM samples every frontier, including the terminal block.  A terminal
        # frontier simply contributes no anti-skip pair, keeping terminal
        # modeling independent from non-terminal ranking supervision.
        frontier = sample_frontier(
            clean_latents.shape[2],
            block_size=self.frontier_config.block_size,
            condition_latents=self.frontier_config.condition_latents,
        )
        sigma = self.edm_loss.sde.sample_sigma(batch_size).to(device=clean_latents.device, dtype=clean_latents.dtype)
        noise = torch.randn(
            batch_size,
            clean_latents.shape[1],
            frontier.active_length,
            clean_latents.shape[-2],
            clean_latents.shape[-1],
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )
        frontier_batch = prepare_frontier_batch(
            clean_latents,
            frontier,
            sigma=sigma,
            noise=noise,
            config=self.frontier_config,
        )

        model_prediction = self._forward_frontier_model(
            transformer, frontier_batch, prompt_embeds, padding_mask, fps
        )
        next_score = frontier_score_from_prediction(frontier_batch, model_prediction.float(), self.frontier_config)
        losses = {"frontier_edm": next_score.mean()}

        if self.frontier_skip_weight <= 0 and not self.frontier_skip_diagnostic:
            return losses

        if self.frontier_skip_negative_mode == "terminal":
            future_target = terminal_candidate_target(clean_latents, frontier, self.frontier_config)
        else:
            future_target = future_candidate_target(
                clean_latents,
                frontier,
                config=self.frontier_config,
                offset_blocks=self.frontier_skip_offset,
            )
        if future_target is None:
            future_target = future_candidate_target(
                clean_latents, frontier, config=self.frontier_config, offset_blocks=1
            )
        if future_target is None:
            # This can only happen for a one-frontier clip; keep the EDM path
            # valid instead of inventing a terminal negative.
            losses["frontier_skip"] = next_score.sum() * 0.0
            return losses

        future_batch = prepare_frontier_candidate_batch(
            clean_latents,
            frontier,
            active_target=future_target,
            sigma=sigma,
            noise=noise,
            config=self.frontier_config,
        )
        if self.frontier_skip_detach_future:
            with torch.no_grad():
                future_prediction = self._forward_frontier_model(
                    transformer, future_batch, prompt_embeds, padding_mask, fps
                )
        else:
            future_prediction = self._forward_frontier_model(
                transformer, future_batch, prompt_embeds, padding_mask, fps
            )
        future_score = frontier_score_from_prediction(
            future_batch, future_prediction.float(), self.frontier_config
        )

        # The reference is the same pretrained transformer with its LoRA
        # adapter disabled.  It is evaluated under no_grad and is not part of
        # the optimization graph.
        with torch.no_grad():
            with self._frozen_base_adapter():
                base_next_prediction = self._forward_frontier_model(
                    transformer, frontier_batch, prompt_embeds, padding_mask, fps
                )
                base_future_prediction = self._forward_frontier_model(
                    transformer, future_batch, prompt_embeds, padding_mask, fps
                )
                base_next_score = frontier_score_from_prediction(
                    frontier_batch, base_next_prediction.float(), self.frontier_config
                )
                base_future_score = frontier_score_from_prediction(
                    future_batch, base_future_prediction.float(), self.frontier_config
                )

        self._log_skip_diagnostics(
            frontier,
            next_score,
            future_score,
            base_next_score,
            base_future_score,
        )

        if self.frontier_skip_weight > 0:
            losses["frontier_skip"] = self.frontier_skip_weight * reference_centered_skip_loss(
                next_score,
                future_score,
                base_next_score,
                base_future_score,
                margin=self.frontier_skip_margin,
                detach_model_future=self.frontier_skip_detach_future,
            )
        else:
            # Keep diagnostics out of the optimization objective.
            losses["frontier_skip"] = next_score.sum() * 0.0
        return losses


__all__ = ["EveFrontierTrainer"]
