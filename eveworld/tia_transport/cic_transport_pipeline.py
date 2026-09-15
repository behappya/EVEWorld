"""Inference loaders for CIC-Transport checkpoints."""

from __future__ import annotations

import torch
from diffusers.models import AutoencoderKLWan

from giga_models.exports.transformer_engine import apply_fp8_autowrap
from giga_models.models.diffusion.giga_world_0 import T5TextEncoder
from giga_models.pipelines.diffusion.giga_world_0.pipeline_giga_world_0 import (
    GigaWorld0Pipeline,
    _diffusers_weight_format_kwargs,
)
from giga_models.schedulers import EDMRESMultistepScheduler

from eveworld.method.pipeline_eag import EAGGigaWorld0Pipeline

from .cic_transport_transformer import (
    CICTransportGigaWorld0Transformer3DModel,
)


class _CICTransportLoaderMixin:
    @classmethod
    def from_pretrained(
        cls,
        transformer_model_path,
        text_encoder_model_path,
        vae_model_path,
        lora_model_path=None,
        lora_fuse=False,
        fp8_eval=True,
        **kwargs,
    ):
        if lora_model_path is not None:
            raise ValueError("CIC-Transport paired experiment does not use LoRA")
        if kwargs.get("physics_latent_model_path") is not None:
            raise ValueError("CIC-Transport paired experiment does not use PhysicsLatent")
        transformer = (
            CICTransportGigaWorld0Transformer3DModel.from_pretrained_transport(
                transformer_model_path
            )
        )
        transformer.to(torch.bfloat16)
        text_encoder = T5TextEncoder(text_encoder_model_path)
        vae = AutoencoderKLWan.from_pretrained(
            vae_model_path,
            **_diffusers_weight_format_kwargs(vae_model_path),
        )
        vae.to(torch.bfloat16)
        scheduler = EDMRESMultistepScheduler(
            prediction_type="rf",
            solver_order=1,
            final_sigmas_type="sigma_min",
            sigma_data=1.0,
        )
        if transformer.fp8:
            apply_fp8_autowrap(transformer, use_autocast_during_eval=fp8_eval)
        return cls(
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )


class CICTransportGigaWorld0Pipeline(_CICTransportLoaderMixin, GigaWorld0Pipeline):
    pass


class CICTransportEAGGigaWorld0Pipeline(
    _CICTransportLoaderMixin, EAGGigaWorld0Pipeline
):
    pass
