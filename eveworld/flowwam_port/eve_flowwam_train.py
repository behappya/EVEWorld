#!/usr/bin/env python3
"""EVE x FlowWAM controlled-variant training entry point.

Variant switch:
  --arm control : legacy matched control; no corruption/TIA but retains the
                  historical spatial weighting for backward compatibility
  --arm sft     : strict uniform-loss SFT without IGR or TIA
  --arm igr     : IGR duplicate restoration and spatial weighting only
  --arm tia     : TIA only, with uniform video loss
  --arm eve     : IGR + TIA

与 FlowWAM 训练循环的差异: 砍掉 action expert 分支（action_loss_weight
恒 0）; swanlab 换 stdout; batch=1/卡（视频窗口训练）。
"""
from __future__ import annotations

import faulthandler
faulthandler.enable()  # SIGSEGV 时打印各线程 Python 栈（8 rank 段错误排障）

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
FLOWWAM_ROOT = os.environ.get("FLOWWAM_ROOT", "/home/jovyan/FlowWAM")
for _p in (FLOWWAM_ROOT, os.path.join(FLOWWAM_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flow_action_train import FlowActionTrainingModule  # noqa: E402
from diffsynth.pipelines.wan_video_dual_stream import (  # noqa: E402
    model_fn_wan_video_dual_stream,
)
from diffsynth.trainers.utils import ModelLogger  # noqa: E402

from eve_flowwam_dataset import EVEFlowWAMWindowDataset  # noqa: E402
from tia_adapter import TIAAdapter, TIAInjection, tia_infonce_loss  # noqa: E402

TOKEN_GH, TOKEN_GW = 15, 20


def collate_b1(batch):
    assert len(batch) == 1, "本训练固定 per-GPU batch=1"
    return batch[0]


class EVEFlowWAMTrainingModule(FlowActionTrainingModule):
    def __init__(self, *, arm: str, l_star: int = 12, tia_loss_weight: float = 0.1,
                 tia_rank: int = 64, tia_window_radius: int = 3, **kw):
        kw.setdefault("action_loss_weight", 0.0)
        # action expert 仅被构造不参与训练; 'text' 模式要求 text_context_dim>0,
        # 换 'state_token' 避免无意义的构造断言
        kw.setdefault("proprio_mode", "state_token")
        super().__init__(**kw)
        assert arm in ("control", "sft", "igr", "tia", "eve")
        self.arm = arm
        self.igr_enabled = arm in ("igr", "eve")
        self.tia_enabled = arm in ("tia", "eve")
        self.spatial_weighting_enabled = arm in ("control", "igr", "eve")
        self.l_star = int(l_star)
        self.tia_loss_weight = float(tia_loss_weight)
        # action expert 不参与训练/损失, 冻结以免进 optimizer
        self.action_expert.requires_grad_(False)
        if self.tia_enabled:
            self.tia_adapter = TIAAdapter(
                channels=int(self.pipe.dit.dim), rank=tia_rank,
                window_radius=tia_window_radius)
        else:
            self.tia_adapter = None

    # ------------------------------------------------------------------
    def forward_preprocess(self, data):
        dev, dt = self.pipe.device, self.pipe.torch_dtype
        clean_frames = data["tiled_rgb_video"][0]
        corr_frames = data["corrupted_rgb_video"][0]
        flow_frames = data["flow_video"][0]
        num_frames = len(clean_frames)
        w0, h0 = clean_frames[0].size
        h, w, num_frames = self.pipe.check_resize_height_width(h0, w0, num_frames)

        self.pipe.load_models_to_device(["text_encoder"])
        context = self.pipe.prompter.encode_prompt(
            data["video_prompt"], positive=True, device=dev)
        self.pipe.load_models_to_device(["vae"])

        def enc(frames):
            vid = self.pipe.preprocess_video([f.resize((w, h)) for f in frames])
            z = self.pipe.vae.encode(vid, device=dev).to(dtype=dt, device=dev)
            first = self.pipe.preprocess_image(frames[0].resize((w, h))).transpose(0, 1)
            fz = self.pipe.vae.encode([first], device=dev).to(dtype=dt, device=dev)
            z[:, :, 0:1] = fz
            return z

        z_clean = enc(clean_frames)
        pasted = bool(data["meta"].get("pasted"))
        z_corr = enc(corr_frames) if (self.igr_enabled and pasted) else z_clean
        z_flow = enc(flow_frames)

        weight = torch.from_numpy(np.asarray(data["weight_win"], np.float32))
        weight = weight.unsqueeze(0).unsqueeze(0).to(device=dev)  # (1,1,T,30,40)

        return {
            "rgb_clean_latents": z_clean,
            "rgb_corr_latents": z_corr,
            "flow_input_latents": z_flow,
            "rgb_noise": torch.randn_like(z_clean),
            "flow_noise": torch.randn_like(z_flow),
            "weight_win": weight,
            "tia_cells": data["tia_cells"],
            "context": context,
            "pasted": pasted,
        }

    # ------------------------------------------------------------------
    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        z_clean = inputs["rgb_clean_latents"]
        z_corr = inputs["rgb_corr_latents"]
        z_flow = inputs["flow_input_latents"]
        B = z_clean.shape[0]

        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (B,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device)

        if self.ref_aug_strength > 0:
            aug = random.random() * self.ref_aug_strength
            noise0 = torch.randn_like(z_corr[:, :, :1])
            z_corr = z_corr.clone(); z_corr[:, :, :1] = z_corr[:, :, :1] + aug * noise0
            z_flow = z_flow.clone(); z_flow[:, :, :1] = z_flow[:, :, :1] + aug * torch.randn_like(z_flow[:, :, :1])

        rgb_noisy = self.pipe.scheduler.add_noise(z_corr, inputs["rgb_noise"], timestep)
        flow_noisy = self.pipe.scheduler.add_noise(z_flow, inputs["flow_noise"], timestep)
        rgb_noisy[:, :, :1] = z_corr[:, :, :1]
        flow_noisy[:, :, :1] = z_flow[:, :, :1]

        # IGR 核心: 去噪目标指向【干净】latent（eq:pipeline 的 flow-matching 形式）
        rgb_target = self.pipe.scheduler.training_target(z_clean, inputs["rgb_noise"], timestep)
        flow_target = self.pipe.scheduler.training_target(z_flow, inputs["flow_noise"], timestep)

        T_lat = z_clean.shape[2]
        inj_ctx = (TIAInjection(self.tia_adapter, self.l_star, (T_lat, TOKEN_GH, TOKEN_GW))
                   if self.tia_enabled else None)

        def run_model():
            return model_fn_wan_video_dual_stream(
                dit=self.pipe.dit, flow_stream=self.flow_stream,
                latents=rgb_noisy, flow_latents=flow_noisy,
                timestep=timestep, context=inputs["context"],
                fuse_vae_embedding_in_latents=True,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            )

        if inj_ctx is not None:
            with inj_ctx:
                rgb_pred, flow_pred = run_model()
            post_feat = inj_ctx.post_features
        else:
            rgb_pred, flow_pred = run_model()
            post_feat = None

        ff = torch.ones(1, 1, T_lat, 1, 1, device=rgb_pred.device, dtype=torch.float32)
        ff[:, :, :1] = 0.0

        w_map = inputs["weight_win"]                       # (1,1,T,30,40)
        if self.spatial_weighting_enabled:
            w_norm = w_map / w_map[:, :, 1:].mean().clamp(min=1e-6)
        else:
            w_norm = torch.ones_like(w_map)
        rgb_se = ((rgb_pred.float() - rgb_target.float()) ** 2) * ff * w_norm
        rgb_ps = rgb_se.sum(dim=(1, 2, 3, 4)) / (ff * w_norm).expand_as(rgb_se).sum(
            dim=(1, 2, 3, 4)).clamp(min=1)

        flow_se = ((flow_pred.float() - flow_target.float()) ** 2) * ff
        if self.flow_motion_boost > 0:
            with torch.no_grad():
                deviationv = (z_flow - z_flow[:, :, :1]).abs().mean(dim=1, keepdim=True)
                dmax = deviationv.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)
                mb = 1.0 + self.flow_motion_boost * (deviationv / dmax)
            flow_se = flow_se * mb
            flow_denom = (mb * ff).expand_as(flow_se).sum(dim=(1, 2, 3, 4)).clamp(min=1)
        else:
            flow_denom = ff.expand_as(flow_se).sum(dim=(1, 2, 3, 4)).clamp(min=1)
        flow_ps = flow_se.sum(dim=(1, 2, 3, 4)) / flow_denom

        wl = self.flow_loss_weight
        loss_video_ps = (1.0 - wl) * rgb_ps + wl * flow_ps
        if self.loss_timestep_weighting:
            vw = self._timestep_weights(self.pipe.scheduler, timestep)
            loss_video = (loss_video_ps * vw).mean() if vw is not None else loss_video_ps.mean()
        else:
            loss_video = loss_video_ps.mean()

        loss_tia = torch.zeros((), device=loss_video.device)
        if post_feat is not None:
            loss_tia = tia_infonce_loss(post_feat, inputs["tia_cells"])
        loss = loss_video + self.tia_loss_weight * loss_tia

        return {
            "loss": loss,
            "loss_video": loss_video.detach(),
            "loss_rgb": rgb_ps.mean().detach(),
            "loss_flow": flow_ps.mean().detach(),
            "loss_tia": loss_tia.detach(),
            "pasted": float(inputs["pasted"]),
        }


def launch(dataset, model, logger, args):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from torch.optim.lr_scheduler import LambdaLR

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.95), foreach=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    nproc = max(int(accelerator.num_processes), 1)
    # 本集群 CPU 张量集合通信会段错误(实测 DDP verify 与 accelerate 的
    # RNG broadcast 均崩) -> dataloader 不给 accelerate 包, 用 DistributedSampler
    # 纯本地分片; 数据集本身按 index 确定性播种, 无需跨 rank RNG 同步。
    from torch.utils.data.distributed import DistributedSampler
    sampler = (DistributedSampler(dataset, num_replicas=nproc,
                                  rank=accelerator.process_index,
                                  shuffle=True, seed=1234)
               if nproc > 1 else None)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_b1,
                            num_workers=0, sampler=sampler,
                            shuffle=(sampler is None))
    steps_per_epoch = max(math.ceil(len(dataloader) / args.gradient_accumulation_steps), 1)
    total_steps = args.lr_max_steps or steps_per_epoch * args.num_epochs
    warm = max(int(total_steps * 0.05), 1)

    def lr_fn(s):
        if s < warm:
            return s / warm
        p = (s - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    scheduler = LambdaLR(optimizer, lr_fn)
    # 不用 DDP 包模型: pipe 内 T5/VAE 离载在 CPU、DiT 在 GPU 的混设备参数会让
    # DDP 的 _verify_param_shape_across_processes 段错误(NCCL 对 CPU 张量 allgather)。
    # 可训练参数仅 ~60M -> backward 后手动 all-reduce 平均梯度, 语义等价 DDP。
    model = model.to(accelerator.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    import torch.distributed as dist
    multi = accelerator.num_processes > 1

    gstep = 0
    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for data in dataloader:
            out = model(data)
            accelerator.backward(out["loss"])
            if multi and dist.is_initialized():
                for p in trainable:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            gstep += 1
            if accelerator.is_main_process and gstep % args.log_every == 0:
                m = {k: (float(v) if not torch.is_tensor(v) else float(v)) for k, v in out.items()}
                sent = ""
                if model.tia_adapter is not None:
                    s = model.tia_adapter.sentinel_stats()
                    sent = (f" tia[out_norm={s['output_proj_norm']:.4f}"
                            f" rel_up={s['relative_update']:.2e}]")
                print(f"[{args.arm}] ep{epoch} step{gstep}/{total_steps} "
                      f"loss={m['loss']:.4f} video={m['loss_video']:.4f} "
                      f"tia={m['loss_tia']:.4f} pasted={m['pasted']:.0f}"
                      f" lr={scheduler.get_last_lr()[0]:.2e}{sent}", flush=True)
            # save_model 内含 wait_for_everyone() barrier -> 必须全 rank 调用
            # (只 rank0 调会与其他 rank 的 all_reduce 互锁, v3 实锤死在 step200)
            if gstep % args.save_steps == 0:
                logger.save_model(accelerator, model, f"step-{gstep}.safetensors")
        logger.save_model(accelerator, model, f"epoch-{epoch}.safetensors")
    logger.save_model(accelerator, model, "final.safetensors")
    accelerator.wait_for_everyone()
    print("TRAIN_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", choices=["control", "sft", "igr", "tia", "eve"],
        required=True,
    )
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--models-root", default="/data/datasets/gagi/flowwam/models")
    ap.add_argument("--resume-checkpoint", default="/data/datasets/gagi/flowwam/checkpoints/flowwam_worldarena_stage1.safetensors")
    ap.add_argument("--t-lat-win", type=int, default=8)
    ap.add_argument("--paste-prob", type=float, default=0.5)
    ap.add_argument("--flow-mode", choices=["full_scene","robot_only"], default="full_scene")
    ap.add_argument("--l-star", type=int, default=12)
    ap.add_argument("--tia-loss-weight", type=float, default=0.1)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--lr-max-steps", type=int, default=0)
    ap.add_argument("--samples-per-epoch", type=int, default=None)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--heldout-per-task", type=int, default=5)
    ap.add_argument("--overfit-n", type=int, default=0, help=">0: 只用前 n 条 episode 过拟合冒烟")
    ap.add_argument("--full-offset", choices=["on", "off"], default="off",
                    help="on: 训练窗口起点覆盖全episode(分块滚动推理同分布)")
    ap.add_argument("--init-arm-ckpt", default="",
                    help="从已训臂 ckpt 续训(如冠军 final.safetensors)")
    args = ap.parse_args()

    dataset = EVEFlowWAMWindowDataset(
        split="train", heldout_per_task=args.heldout_per_task,
        t_lat_win=args.t_lat_win, igr_paste=(args.arm in ("igr", "eve")),
        paste_prob=args.paste_prob, flow_mode=args.flow_mode,
        samples_per_epoch=args.samples_per_epoch,
        full_offset=(args.full_offset == "on"))
    if args.overfit_n > 0:
        dataset.episodes = dataset.episodes[: args.overfit_n]
        dataset.samples_per_epoch = min(dataset.samples_per_epoch, args.overfit_n * 8)

    # 8 rank 并发加载 stage1(10.1G)+基座 曾触发 CPU 内存峰值段错误(SIGSEGV);
    # 节点内文件锁串行化重加载阶段, 削峰
    import fcntl
    import gc
    lock_f = open("/tmp/eve_flowwam_init.lock", "w")
    fcntl.flock(lock_f, fcntl.LOCK_EX)
    try:
        model = _build_model(args)
        if args.init_arm_ckpt:
            from diffsynth.models.utils import load_state_dict as _lsd
            state = _lsd(args.init_arm_ckpt)
            # 保存时剥掉了 pipe.dit. 前缀; flow_stream./tia_adapter. 为模块顶层属性
            remap = {}
            for k, v in state.items():
                if k.startswith(("flow_stream.", "tia_adapter.")):
                    remap[k] = v
                else:
                    remap["pipe.dit." + k] = v
            missing, unexpected = model.load_state_dict(remap, strict=False)
            n_hit = len(remap) - len(unexpected)
            print(f"[InitArm] {args.init_arm_ckpt}: 命中 {n_hit}/{len(remap)} keys "
                  f"(unexpected={len(unexpected)})", flush=True)
            assert n_hit > 0, "init-arm-ckpt 未命中任何参数, 前缀映射错误"
        gc.collect()
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt="pipe.dit.")
    os.makedirs(args.output_path, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_path, "train_args.json"), "w"), indent=1)
    launch(dataset, model, logger, args)


def _build_model(args):
    return EVEFlowWAMTrainingModule(
        arm=args.arm, l_star=args.l_star, tia_loss_weight=args.tia_loss_weight,
        model_paths=json.dumps([
            [f"{args.models_root}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
             f"{args.models_root}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
             f"{args.models_root}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"],
            f"{args.models_root}/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
            f"{args.models_root}/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        ]),
        # tokenizer 走 pipeline 默认 ./models/Wan-AI/Wan2.1-T2V-1.3B 相对路径,
        # 运行 cwd 必须含 models -> flowwam/models 的符号链接（kjob 脚本保证）
        lora_base_model="dit",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=args.lora_rank,
        resume_checkpoint=args.resume_checkpoint,
        use_gradient_checkpointing=True,
        num_frames=4 * (args.t_lat_win - 1) + 1,
    )


if __name__ == "__main__":
    main()
