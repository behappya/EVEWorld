#!/usr/bin/env python3
"""CPU smoke tests for the isolated CIC-Transport experiment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from giga_models.models.diffusion.giga_world_0.transformer_giga_world_0 import (
    GigaWorld0Transformer3DModel,
)

from .cic_transport_transformer import (
    TRANSPORT_CONFIG_NAME,
    CICTransportAdapter,
    CICTransportGigaWorld0Transformer3DModel,
)


def tiny_model_kwargs() -> dict:
    return {
        "max_img_h": 8,
        "max_img_w": 8,
        "max_frames": 4,
        "in_channels": 2,
        "out_channels": 2,
        "patch_spatial": 1,
        "patch_temporal": 1,
        "concat_padding_mask": False,
        "model_channels": 8,
        "num_blocks": 2,
        "num_heads": 2,
        "mlp_ratio": 2,
        "crossattn_emb_channels": 8,
        "adaln_lora_dim": 4,
        "natten_parameters": None,
        "moe_parameters": None,
    }


def model_inputs() -> dict[str, torch.Tensor | int | None]:
    return {
        "x": torch.randn(1, 2, 3, 4, 4),
        "timesteps": torch.rand(1, 1, 3, 1, 1),
        "crossattn_emb": torch.randn(1, 5, 8),
        "fps": 16,
        "padding_mask": None,
    }


class CICTransportAdapterTest(unittest.TestCase):
    def test_zero_init_is_exact_identity_and_first_frame_stays_fixed(self):
        adapter = CICTransportAdapter(8, rank=4, window_radius=1)
        x = torch.randn(2, 3, 4, 5, 8, requires_grad=True)
        y = adapter(x)
        self.assertEqual(y.shape, x.shape)
        torch.testing.assert_close(y, x, rtol=0, atol=0)

        with torch.no_grad():
            adapter.output_proj.weight.fill_(0.125)
        changed = adapter(x)
        torch.testing.assert_close(changed[:, 0], x[:, 0], rtol=0, atol=0)
        self.assertGreater((changed[:, 1:] - x[:, 1:]).abs().max().item(), 0)

    def test_local_soft_splat_tracks_one_cell_motion_and_masks_borders(self):
        # Each cell has a unique direction. Frame 1 shifts those directions one
        # cell right and doubles their magnitude, so cosine matching determines
        # the destination while the transported magnitude remains observable.
        height, width, channels = 2, 3, 6
        prev = torch.eye(channels).reshape(height, width, channels)
        curr = torch.zeros_like(prev)
        curr[:, 1:] = 2 * prev[:, :-1]
        curr[:, 0] = 2 * prev[:, 0]
        x = torch.stack((prev, curr)).unsqueeze(0)

        adapter = CICTransportAdapter(
            channels,
            rank=channels,
            window_radius=1,
            temperature=1e-3,
            residual_scale=0.5,
        )
        with torch.no_grad():
            adapter.input_proj.weight.copy_(torch.eye(channels))
            adapter.output_proj.weight.copy_(torch.eye(channels))
        y = adapter(x)

        # Deliberately slow per-cell reference. It checks that unfold/fold uses
        # source-to-destination orientation and excludes padded border cells.
        accumulation = torch.zeros_like(curr)
        mass = torch.zeros(height, width)
        for source_y in range(height):
            for source_x in range(width):
                candidates = []
                logits = []
                query = torch.nn.functional.normalize(
                    prev[source_y, source_x], dim=0, eps=1e-6
                )
                for target_y in range(max(0, source_y - 1), min(height, source_y + 2)):
                    for target_x in range(max(0, source_x - 1), min(width, source_x + 2)):
                        key = torch.nn.functional.normalize(
                            curr[target_y, target_x], dim=0, eps=1e-6
                        )
                        candidates.append((target_y, target_x))
                        logits.append(torch.dot(query, key) / 1e-3)
                weights = torch.softmax(torch.stack(logits), dim=0)
                for (target_y, target_x), weight in zip(candidates, weights):
                    accumulation[target_y, target_x] += (
                        weight * prev[source_y, source_x]
                    )
                    mass[target_y, target_x] += weight
        transported = accumulation / mass.clamp_min(1e-6).unsqueeze(-1)
        expected = curr + 0.5 * torch.tanh(transported - curr)
        torch.testing.assert_close(y[0, 1], expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(y).all())
        stats = adapter.sentinel_stats()
        self.assertGreater(stats["match_peak"], 0.5)

    def test_gradient_staging(self):
        adapter = CICTransportAdapter(8, rank=4, window_radius=1)
        x = torch.randn(1, 3, 3, 4, 8, requires_grad=True)
        adapter(x).square().mean().backward()
        self.assertGreater(adapter.output_proj.weight.grad.norm().item(), 0)
        self.assertEqual(adapter.input_proj.weight.grad.norm().item(), 0)

        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.output_proj.weight.fill_(1e-2)
        adapter(x.detach()).square().mean().backward()
        self.assertGreater(adapter.input_proj.weight.grad.norm().item(), 0)


class CICTransportModelTest(unittest.TestCase):
    def _write_base(self, path: Path) -> GigaWorld0Transformer3DModel:
        base = GigaWorld0Transformer3DModel(**tiny_model_kwargs())
        base.save_pretrained(path, safe_serialization=False)
        return base

    def test_rng_preservation_zero_equivalence_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "base"
            trained_dir = Path(tmp) / "trained"
            base = self._write_base(base_dir)

            torch.manual_seed(1234)
            rng_before = torch.get_rng_state().clone()
            model = CICTransportGigaWorld0Transformer3DModel.from_pretrained_base(
                base_dir,
                after_block="block0",
                rank=4,
                window_radius=1,
            )
            torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)

            torch.manual_seed(99)
            inputs = model_inputs()
            base.eval()
            model.eval()
            with torch.no_grad():
                baseline = base(**inputs)
                adapted = model(**inputs)
            torch.testing.assert_close(adapted, baseline, rtol=0, atol=0)

            with torch.no_grad():
                model.cic_transport.output_proj.weight.fill_(0.0125)
            model.save_pretrained(trained_dir, safe_serialization=False)
            self.assertTrue((trained_dir / TRANSPORT_CONFIG_NAME).is_file())
            sidecar = json.loads(
                (trained_dir / TRANSPORT_CONFIG_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["after_block"], "block0")

            reloaded = (
                CICTransportGigaWorld0Transformer3DModel.from_pretrained_transport(
                    trained_dir
                )
            )
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    reloaded.state_dict()[key], value, rtol=0, atol=0
                )

    def test_activation_checkpoint_wrapper_and_outer_cic_hook(self):
        model = CICTransportGigaWorld0Transformer3DModel(
            **tiny_model_kwargs()
        ).enable_cic_transport(
            after_block="block0", rank=4, window_radius=1
        )
        with torch.no_grad():
            model.cic_transport.output_proj.weight.fill_(0.01)

        # Mirrors training: transport is registered first on the block; then
        # activation checkpointing wraps it; finally CIC registers on wrapper.
        model.blocks["block0"] = checkpoint_wrapper(model.blocks["block0"])
        captured: list[torch.Tensor] = []
        model.blocks["block0"].register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        output = model(**model_inputs())
        output.square().mean().backward()

        self.assertTrue(captured)
        self.assertTrue(captured[0].requires_grad)
        self.assertGreater(model.cic_transport.output_proj.weight.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
