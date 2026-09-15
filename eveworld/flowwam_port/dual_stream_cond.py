#!/usr/bin/env python3
"""光流条件版双流 forward（action/HDF5 驱动推理用）。

由官方 model_fn_wan_video_dual_stream 源码在导入时代码生成:
唯一改动 = flow 流 per-token timestep 全为 0（配合外部把 flow latent
钉为干净编码, 即对光流做与首帧 prefix 同机制的 teacher-forcing）。
在 ds 模块命名空间内 exec, 因此 TIAInjection 对 _dual_stream_block_fn
的 monkeypatch 依旧生效。
"""
from __future__ import annotations

import inspect

import diffsynth.pipelines.wan_video_dual_stream as ds

_FLOW_TPT_ORIG = """            flow_tpt = torch.cat([
                torch.zeros(1, flow_spatial, dtype=latents.dtype,
                            device=latents.device),
                torch.ones(flow_temporal - 1, flow_spatial, dtype=latents.dtype,
                           device=latents.device) * ts_b,
            ]).flatten()"""

_FLOW_TPT_COND = """            flow_tpt = torch.zeros(
                flow_temporal * flow_spatial, dtype=latents.dtype,
                device=latents.device)  # 光流全帧干净条件: t=0"""

_src = inspect.getsource(ds.model_fn_wan_video_dual_stream)
assert _FLOW_TPT_ORIG in _src, "上游 model_fn 源码已变化, 请重新核对 flow_tpt 段"
_src = _src.replace(_FLOW_TPT_ORIG, _FLOW_TPT_COND)
_src = _src.replace("def model_fn_wan_video_dual_stream(",
                    "def model_fn_dual_stream_flowcond(")
exec(compile(_src, "<dual_stream_cond codegen>", "exec"), ds.__dict__)

model_fn_dual_stream_flowcond = ds.model_fn_dual_stream_flowcond
