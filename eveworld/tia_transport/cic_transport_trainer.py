"""Historical EVEWorld trainer with an isolated CIC-Transport transformer."""

from __future__ import annotations

import json
import os

import torch
from diffusers.models import AutoencoderKLWan
from giga_train import ModuleDict

from giga_models.nn import EDMLoss

from eveworld.pipeline.t4g_corr_trainer import (
    DEFAULT_ANNO_DIR,
    ParamDisplacementProbe,
    _load_annos,
    _parse_lambdas,
)
from eveworld.pipeline.t4g_joint_trainer import T4GJointTrainer

from .cic_transport_transformer import (
    CICTransportGigaWorld0Transformer3DModel,
)


class CICTransportJointTrainer(T4GJointTrainer):
    """Full EVEWorld objective; the transformer is the only changed component."""

    def get_models(self, model_config):
        model = {}
        vae_dtype = self.dtype
        vae = AutoencoderKLWan.from_pretrained(model_config.vae_model_path)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(self.device, dtype=vae_dtype)
        self.latents_std = (1.0 / torch.tensor(self.vae.config.latents_std)).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(self.device, dtype=vae_dtype)

        self.train_mode = model_config.get("train_mode", "full")
        assert self.train_mode == "full", "CIC-Transport paired run requires full training"
        assert self.mixed_precision != "fp8", "CIC-Transport is not validated with fp8 training"
        transport_config = {
            "after_block": model_config.get("cic_transport_after_block", "block22"),
            "rank": int(model_config.get("cic_transport_rank", 64)),
            "window_radius": int(model_config.get("cic_transport_window_radius", 3)),
            "temperature": float(model_config.get("cic_transport_temperature", 0.07)),
            "residual_scale": float(model_config.get("cic_transport_residual_scale", 0.1)),
            "init_seed": int(model_config.get("cic_transport_init_seed", 20260808)),
        }
        transformer = CICTransportGigaWorld0Transformer3DModel.from_pretrained_base(
            model_config.transformer_model_path,
            **transport_config,
        )
        model["transformer"] = transformer
        self.edm_loss = EDMLoss(
            sigma_method=3,
            p_mean=0.0,
            p_std=1.0,
            use_flow=True,
            sigma_data=1.0,
        )
        model = ModuleDict(model)
        model.to(self.dtype)
        model.train()

        self.id_block = os.environ.get(
            "T4G_ID_BLOCK", model_config.get("t4g_id_block", "block22")
        )
        self.change_block = os.environ.get(
            "T4G_CHANGE_BLOCK", model_config.get("t4g_change_block", "block25")
        )
        if self.id_block != transport_config["after_block"]:
            raise ValueError(
                f"CIC block {self.id_block} must equal transport insertion "
                f"{transport_config['after_block']}"
            )
        self.sigma_lo = float(
            os.environ.get("T4G_SIGMA_LO", model_config.get("t4g_sigma_lo", 0.2))
        )
        self.sigma_hi = float(
            os.environ.get("T4G_SIGMA_HI", model_config.get("t4g_sigma_hi", 0.5))
        )
        self.warmup_steps = int(
            os.environ.get("T4G_WARMUP", model_config.get("t4g_warmup", 20))
        )
        self.lambda_base = _parse_lambdas(
            os.environ.get(
                "T4G_LAMBDAS", model_config.get("t4g_lambdas", "0.5,0.4")
            )
        )
        self.tau = float(
            os.environ.get("T4G_TAU", model_config.get("t4g_tau", 0.07))
        )
        self.tol = float(
            os.environ.get("T4G_TOL", model_config.get("t4g_tol", 0.2))
        )
        self.win_r_id = int(
            os.environ.get("T4G_WIN_R_ID", model_config.get("t4g_win_r_id", 3))
        )
        self.obj_excl_r = int(
            os.environ.get("T4G_OBJ_EXCL_R", model_config.get("t4g_obj_excl_r", 1))
        )
        anno_dir = os.environ.get(
            "T4G_ANNO_DIR", model_config.get("t4g_anno_dir", DEFAULT_ANNO_DIR)
        )
        idx2vid_path = os.environ.get(
            "T4G_IDX2VID",
            model_config.get(
                "t4g_idx2vid", os.path.join(anno_dir, "_packidx2vid.json")
            ),
        )
        self.annos = _load_annos(anno_dir)
        self.idx2vid = {}
        if os.path.exists(idx2vid_path):
            self.idx2vid = {
                int(key): str(value)
                for key, value in json.load(open(idx2vid_path)).items()
            }
        self.checkpoint_start_step = int(
            os.environ.get(
                "T4G_CHECKPOINT_START_STEP",
                self.kwargs.get("checkpoint_start_step", 0),
            )
        )
        self.param_probe = ParamDisplacementProbe(transformer)
        self._hook_done = False
        self._id_block_out = None
        self._change_block_out = None
        self._stat_buffer = []
        self._logged_step = -1

        self.w_paste = float(
            os.environ.get("T4G_W_PASTE", model_config.get("t4g_w_paste", 4.0))
        )
        levels = os.environ.get(
            "T4G_REGION_LEVELS",
            model_config.get("t4g_region_levels", "0.5,2.0,3.0,4.0,6.0"),
        )
        names = os.environ.get(
            "T4G_REGION_NAMES",
            model_config.get(
                "t4g_region_names", "bg0.5x,empty2x,gripB3x,obj4x,trans6x"
            ),
        )
        self.region_levels = tuple(float(value) for value in str(levels).split(","))
        self.region_names = tuple(value.strip() for value in str(names).split(","))
        if len(self.region_levels) != len(self.region_names) or not self.region_levels:
            raise ValueError("region levels and names must have equal non-zero length")
        expected_samples = int(
            os.environ.get(
                "T4G_EXPECTED_SAMPLES", model_config.get("t4g_expected_samples", 0)
            )
        )
        if expected_samples:
            anno_vids = set(self.annos)
            mapped_vids = set(self.idx2vid.values())
            if len(self.annos) != expected_samples or anno_vids != mapped_vids:
                raise ValueError("annotation/index mapping failed strict sample check")
        self._aug_stat_buffer = []
        self._transport_logged_step = -1

        self.print(
            "[cic-transport] %s | zero-init output; CIC hook remains at %s | "
            "annos=%d idx2vid=%d"
            % (transport_config, self.id_block, len(self.annos), len(self.idx2vid))
        )
        self.print(
            "[t4g] id=%s change=%s sigma=[%.2f,%.2f] warmup=%d "
            "lambda=(%.3f,%.3f) tau=%.3f tol=%.2f"
            % (
                self.id_block,
                self.change_block,
                self.sigma_lo,
                self.sigma_hi,
                self.warmup_steps,
                self.lambda_base[0],
                self.lambda_base[1],
                self.tau,
                self.tol,
            )
        )
        self.print(
            "[t4g-joint] w_paste=%.1f regions=%s"
            % (
                self.w_paste,
                ",".join(
                    f"{name}:{value:g}x"
                    for name, value in zip(self.region_names, self.region_levels)
                ),
            )
        )
        return model

    def print_step(self):
        super().print_step()
        if not self.is_main_process or self.cur_step % self.log_interval != 0:
            return
        if self.cur_step == self._transport_logged_step:
            return
        self._transport_logged_step = self.cur_step
        model = self.accelerator.unwrap_model(
            self.model, keep_torch_compile=False
        )["transformer"]
        stats = model.cic_transport.sentinel_stats()
        msg = (
            "[cic-transport-sentinel] step=%d in_norm=%.4e out_norm=%.4e "
            "peak=%.4f entropy=%.4f update_rms=%.4e rel_update=%.4e"
            % (
                self.cur_step,
                stats["input_proj_norm"],
                stats["output_proj_norm"],
                stats["match_peak"],
                stats["match_entropy"],
                stats["update_rms"],
                stats["relative_update"],
            )
        )
        self.logger.info(msg)
        print(msg, flush=True)
