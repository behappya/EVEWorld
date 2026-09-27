"""TIA moves instance appearance along the backbone's own patch correspondence.

The local window is the transport support, the weights are a softmax over that
window, and the adapter starts as an exact identity so that attaching it to a
frozen backbone changes nothing until training starts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eveworld.methods.tia.adapter import TIAConfig, build_adapter
from eveworld.methods.tia.loss import contrastive_loss, lambda_schedule, noise_gate
from eveworld.methods.tia.matcher import (
    correlation_matrix,
    local_window_indices,
    normalize_features,
    select_layer,
)
from eveworld.methods.tia.transport import transport, transport_weights


def test_local_window_is_square_and_centred():
    windows = local_window_indices(5, 5, radius=3)
    assert windows.shape == (25, 49)
    assert windows.dtype == np.int64
    centre = 3 * 7 + 3
    for index in range(25):
        assert windows[index, centre] == index
        assert windows[index].min() >= 0
        assert windows[index].max() < 25


def test_normalise_features_returns_unit_vectors():
    features = torch.randn(4, 10, 8)
    normed = normalize_features(features, dim=-1)
    assert torch.allclose(normed.norm(dim=-1), torch.ones(4, 10), atol=1e-5)


def test_correlation_matrix_is_square_per_batch_element():
    source = torch.randn(2, 6, 8)
    target = torch.randn(2, 6, 8)
    correlation = correlation_matrix(source, target)
    assert correlation.shape == (2, 6, 6)


def test_transport_weights_are_row_stochastic():
    source = torch.randn(1, 16, 8)
    target = torch.randn(1, 16, 8)
    weights = transport_weights(source, target, window=7, temperature=0.07)
    assert weights.shape == (1, 16, 16)
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 16), atol=1e-5)


def test_transport_splats_back_onto_the_target_tokens():
    source = torch.randn(1, 16, 8)
    target = torch.randn(1, 16, 8)
    out = transport(source, target, window=7, temperature=0.07)
    assert out.shape == target.shape
    assert torch.isfinite(out).all()


def test_transport_without_mass_normalisation_is_still_finite():
    source = torch.randn(1, 16, 8)
    target = torch.randn(1, 16, 8)
    out = transport(source, target, window=7, temperature=0.07, normalize_mass=False)
    assert out.shape == target.shape
    assert torch.isfinite(out).all()


def test_the_paper_defaults_are_the_dataclass_defaults():
    config = TIAConfig()
    assert config.dim == 64
    assert config.window == 7
    assert config.temperature == pytest.approx(0.07)
    assert config.gamma == pytest.approx(0.1)
    assert config.layer == 23
    assert config.first_frame_frozen


def test_the_adapter_starts_as_an_exact_identity():
    adapter = build_adapter(32, layer=5)
    hidden = torch.randn(2, 16, 32)
    assert torch.allclose(adapter(hidden, frame_index=3), hidden, atol=1e-6)


def test_the_first_frame_stays_frozen_once_the_adapter_moves():
    adapter = build_adapter(32, TIAConfig(layer=5))
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    hidden = torch.randn(2, 16, 32)
    assert torch.allclose(adapter(hidden, frame_index=0), hidden, atol=1e-6)


def test_contrastive_loss_is_small_when_the_alignment_is_obvious():
    scores = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    loss = contrastive_loss(scores, torch.tensor([0, 1]), temperature=0.07)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_contrastive_loss_punishes_a_swapped_alignment():
    scores = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    right = contrastive_loss(scores, torch.tensor([0, 1]), temperature=0.07)
    wrong = contrastive_loss(scores, torch.tensor([1, 0]), temperature=0.07)
    assert wrong.item() > right.item()


def test_lambda_schedule_warms_up_over_twenty_steps():
    assert lambda_schedule(0) == 0.0
    assert lambda_schedule(20) == pytest.approx(0.5)
    assert lambda_schedule(200) == pytest.approx(0.5)
    assert 0.0 < lambda_schedule(5) < lambda_schedule(10) < 0.5


def test_noise_gate_covers_the_configured_sigma_band():
    assert noise_gate(0.2)
    assert noise_gate(0.35)
    assert noise_gate(0.5)
    assert not noise_gate(0.19)
    assert not noise_gate(0.51)


def test_select_layer_picks_the_lowest_epe():
    result = select_layer([8, 12, 23], [1.42, 0.88, 1.05])
    assert result.layer == 12
    assert result.epe == pytest.approx(0.88)
