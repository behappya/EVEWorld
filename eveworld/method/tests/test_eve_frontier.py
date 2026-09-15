from __future__ import annotations

import unittest

import torch
from torch import nn

from eveworld.method.eve_frontier_loss import (
    CommittedHistory,
    FrontierConfig,
    boundary_commit_guard,
    frontier_loss_from_prediction,
    frontier_slices,
    frontier_score_from_prediction,
    fixed_sigma_frontier_batches,
    fixed_sigma_frontier_report,
    future_candidate_target,
    latent_frames_from_pixels,
    masked_edm_loss,
    pixel_frames_from_latents,
    prepare_frontier_batch,
    prepare_frontier_candidate_batch,
    prepare_frontier_sampling_inputs,
    reference_centered_skip_loss,
    sample_frontier,
    terminal_candidate_target,
)


class TemporalToyModel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class EveFrontierMechanismTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = FrontierConfig(block_size=4, condition_latents=1)
        self.latents = torch.randn(2, 3, 24, 2, 2)

    def test_future_perturbation_does_not_change_prefix_or_active_output(self) -> None:
        frontier = frontier_slices(24, 4, 1)[2]
        changed = self.latents.clone()
        changed[:, :, frontier.end :, :, :] = torch.randn_like(changed[:, :, frontier.end :, :, :]) * 100
        noise = torch.randn_like(self.latents[:, :, frontier.active, :, :])

        original_batch = prepare_frontier_batch(self.latents, frontier, sigma=torch.tensor([0.3, 0.7]), noise=noise, config=self.config)
        changed_batch = prepare_frontier_batch(changed, frontier, sigma=torch.tensor([0.3, 0.7]), noise=noise, config=self.config)
        self.assertTrue(torch.equal(original_batch.model_input, changed_batch.model_input))
        self.assertEqual(original_batch.input_length, frontier.end)

        model = TemporalToyModel(channels=3)
        original_output = model(original_batch.model_input)
        changed_output = model(changed_batch.model_input)
        self.assertTrue(torch.equal(original_output, changed_output))

    def test_committed_history_is_immutable_across_steps(self) -> None:
        state = CommittedHistory.from_first_latent(self.latents, self.config)
        first_condition = state.committed.clone()
        previous = state.committed

        while not state.complete:
            frontier = state.next_frontier
            self.assertIsNotNone(frontier)
            active = self.latents[:, :, frontier.active, :, :]
            new_state = state.commit(active)
            self.assertTrue(torch.equal(new_state.committed[:, :, : previous.shape[2]], previous))
            self.assertTrue(torch.equal(state.committed, previous))
            previous = new_state.committed
            state = new_state

        exposed_copy = state.committed
        exposed_copy.zero_()
        self.assertFalse(torch.equal(exposed_copy, state.committed))
        self.assertTrue(torch.equal(state.committed[:, :, :1], first_condition))

    def test_boundary_guard_aligns_velocity_anchor_and_decays(self) -> None:
        committed = self.latents[:, :, :8].clone()
        active = torch.randn(2, 3, 4, 2, 2)
        history_snapshot = committed.clone()
        unchanged = boundary_commit_guard(committed, active, strength=0.0)
        self.assertTrue(torch.equal(unchanged, active))

        guarded = boundary_commit_guard(
            committed,
            active,
            strength=1.0,
            velocity_scale=0.5,
            decay=1.0,
        )
        anchor = committed[:, :, -1] + 0.5 * (committed[:, :, -1] - committed[:, :, -2])
        self.assertTrue(torch.allclose(guarded[:, :, 0], anchor))
        correction = (guarded - active).abs().mean(dim=(0, 1, 3, 4))
        self.assertTrue(torch.all(correction[:-1] > correction[1:]))
        self.assertTrue(torch.equal(committed, history_snapshot))

    def test_loss_and_gradient_are_active_only(self) -> None:
        prediction = torch.randn_like(self.latents, requires_grad=True)
        target = torch.zeros_like(self.latents)
        mask = torch.zeros(2, 1, 24, 1, 1)
        mask[:, :, 8:12] = 1
        masked_edm_loss(prediction, target, mask, sigma=torch.tensor([0.5, 1.0])).mean().backward()

        self.assertEqual(torch.count_nonzero(prediction.grad[:, :, :8]).item(), 0)
        self.assertGreater(torch.count_nonzero(prediction.grad[:, :, 8:12]).item(), 0)
        self.assertEqual(torch.count_nonzero(prediction.grad[:, :, 12:]).item(), 0)

        frontier = frontier_slices(24, 4, 1)[3]
        batch = prepare_frontier_batch(self.latents, frontier, sigma=0.8, config=self.config)
        frontier_prediction = torch.randn_like(batch.model_input, requires_grad=True)
        frontier_loss_from_prediction(batch, frontier_prediction, self.config).backward()
        self.assertEqual(torch.count_nonzero(frontier_prediction.grad[:, :, : frontier.history_length]).item(), 0)
        self.assertGreater(torch.count_nonzero(frontier_prediction.grad[:, :, frontier.history_length :]).item(), 0)

    def test_every_frontier_is_sampled_and_receives_gradient(self) -> None:
        candidates = frontier_slices(24, 4, 1)
        self.assertEqual([(item.active.start, item.active.stop) for item in candidates], [(1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)])

        generator = torch.Generator().manual_seed(123)
        sampled = {sample_frontier(24, 4, 1, generator).index for _ in range(256)}
        self.assertEqual(sampled, {item.index for item in candidates})

        for frontier in candidates:
            batch = prepare_frontier_batch(self.latents, frontier, sigma=0.6, config=self.config)
            scale = torch.tensor(0.25, requires_grad=True)
            prediction = batch.model_input * scale
            frontier_loss_from_prediction(batch, prediction, self.config).backward()
            self.assertIsNotNone(scale.grad)
            self.assertGreater(abs(scale.grad.item()), 0.0, msg=f"frontier {frontier.index} received zero gradient")

    def test_temporal_length_and_first_frame_condition_do_not_drift(self) -> None:
        self.assertEqual(latent_frames_from_pixels(93), 24)
        self.assertEqual(pixel_frames_from_latents(24), 93)
        with self.assertRaises(ValueError):
            latent_frames_from_pixels(92)

        state = CommittedHistory.from_first_latent(self.latents, self.config)
        first_condition = state.committed
        while not state.complete:
            frontier = state.next_frontier
            state = state.commit(self.latents[:, :, frontier.active, :, :])
        self.assertEqual(state.committed.shape[2], 24)
        self.assertEqual(pixel_frames_from_latents(state.committed.shape[2]), 93)
        self.assertTrue(torch.equal(state.committed[:, :, :1], first_condition))

    def test_timestep_and_masks_match_transformer_contract(self) -> None:
        frontier = frontier_slices(24, 4, 1)[4]
        batch = prepare_frontier_batch(self.latents, frontier, sigma=torch.tensor([0.4, 1.2]), config=self.config)
        self.assertEqual(batch.timesteps.shape, (2, 1, frontier.end, 1, 1))
        self.assertEqual(batch.condition_mask.shape, (2, 1, frontier.end, 1, 1))
        self.assertEqual(batch.active_mask.shape, (2, 1, frontier.end, 1, 1))
        self.assertTrue(torch.all(batch.condition_mask[:, :, : frontier.history_length] == 1))
        self.assertTrue(torch.all(batch.active_mask[:, :, : frontier.history_length] == 0))
        self.assertTrue(torch.all(batch.active_mask[:, :, frontier.history_length :] == 1))

    def test_sampling_inputs_include_only_committed_and_active_tokens(self) -> None:
        committed = self.latents[:, :, :12].clone()
        active = torch.randn(2, 3, 4, 2, 2)
        committed_snapshot = committed.clone()
        model_input, timesteps, condition_mask = prepare_frontier_sampling_inputs(
            committed,
            active,
            active_timestep=torch.tensor([0.25, 0.75]),
            config=self.config,
        )

        self.assertEqual(model_input.shape, (2, 4, 16, 2, 2))
        self.assertEqual(timesteps.shape, (2, 1, 16, 1, 1))
        self.assertEqual(condition_mask.shape, (2, 1, 16, 1, 1))
        self.assertTrue(torch.equal(model_input[:, :3, :12], committed))
        self.assertTrue(torch.equal(model_input[:, :3, 12:], active))
        self.assertTrue(torch.all(model_input[:, 3:4, :12] == 1))
        self.assertTrue(torch.all(model_input[:, 3:4, 12:] == 0))
        self.assertTrue(torch.equal(committed, committed_snapshot))

    def test_skip_candidates_share_history_sigma_and_noise(self) -> None:
        frontier = frontier_slices(24, 4, 1)[1]
        future = future_candidate_target(self.latents, frontier, self.config, offset_blocks=1)
        self.assertIsNotNone(future)
        noise = torch.randn(2, 3, frontier.active_length, 2, 2)
        sigma = torch.tensor([0.4, 0.8])
        next_batch = prepare_frontier_batch(self.latents, frontier, sigma=sigma, noise=noise, config=self.config)
        future_batch = prepare_frontier_candidate_batch(
            self.latents,
            frontier,
            active_target=future,
            sigma=sigma,
            noise=noise,
            config=self.config,
        )
        self.assertTrue(torch.equal(next_batch.model_input[:, :, : frontier.history_length], future_batch.model_input[:, :, : frontier.history_length]))
        self.assertTrue(torch.equal(next_batch.sigma, future_batch.sigma))
        self.assertTrue(torch.allclose(next_batch.raw_noisy_active - next_batch.active_target(), future_batch.raw_noisy_active - future_batch.active_target()))
        self.assertFalse(torch.equal(next_batch.active_target(), future_batch.active_target()))

    def test_terminal_negative_is_final_state_and_never_overlaps_active(self) -> None:
        frontiers = frontier_slices(24, 4, 1)
        for frontier in frontiers[:-1]:
            terminal = terminal_candidate_target(self.latents, frontier, self.config)
            self.assertIsNotNone(terminal)
            self.assertEqual(terminal.shape[2], frontier.active_length)
            self.assertTrue(torch.equal(terminal, self.latents[:, :, -frontier.active_length :]))
        self.assertIsNone(terminal_candidate_target(self.latents, frontiers[-1], self.config))

    def test_reference_centered_skip_loss_only_penalizes_relative_regression(self) -> None:
        # Both models prefer ``next`` in absolute terms, but theta improves the
        # preference over base, so the centered hinge is inactive.
        good = reference_centered_skip_loss(
            torch.tensor([1.0, 1.2], requires_grad=True),
            torch.tensor([2.0, 2.2], requires_grad=True),
            torch.tensor([0.5, 0.7]),
            torch.tensor([1.0, 1.1]),
            margin=0.1,
            reduction="none",
        )
        self.assertTrue(torch.equal(good, torch.zeros_like(good)))

        model_next = torch.tensor([2.0, 1.8], requires_grad=True)
        model_future = torch.tensor([1.0, 1.2], requires_grad=True)
        base_next = torch.tensor([0.5, 0.7], requires_grad=True)
        base_future = torch.tensor([1.0, 1.1], requires_grad=True)
        loss = reference_centered_skip_loss(model_next, model_future, base_next, base_future, margin=0.1)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertGreater(torch.count_nonzero(model_next.grad).item(), 0)
        self.assertGreater(torch.count_nonzero(model_future.grad).item(), 0)
        self.assertIsNone(base_next.grad)
        self.assertIsNone(base_future.grad)

    def test_one_sided_skip_loss_never_degrades_future_candidate(self) -> None:
        model_next = torch.tensor([2.0, 1.8], requires_grad=True)
        model_future = torch.tensor([1.0, 1.2], requires_grad=True)
        loss = reference_centered_skip_loss(
            model_next,
            model_future,
            torch.tensor([0.5, 0.7]),
            torch.tensor([1.0, 1.1]),
            margin=0.1,
            detach_model_future=True,
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertGreater(torch.count_nonzero(model_next.grad).item(), 0)
        self.assertIsNone(model_future.grad)

    def test_frontier_score_is_per_sample_and_gradient_bearing(self) -> None:
        frontier = frontier_slices(24, 4, 1)[2]
        batch = prepare_frontier_batch(self.latents, frontier, sigma=0.6, config=self.config)
        prediction = torch.zeros_like(batch.model_input, requires_grad=True)
        score = frontier_score_from_prediction(batch, prediction, self.config)
        self.assertEqual(score.shape, (self.latents.shape[0],))
        score.mean().backward()
        self.assertGreater(torch.count_nonzero(prediction.grad[:, :, frontier.history_length :]).item(), 0)

    def test_fixed_sigma_diagnostic_covers_all_frontiers_deterministically(self) -> None:
        first = fixed_sigma_frontier_batches(self.latents, sigma=0.35, config=self.config, seed=19)
        second = fixed_sigma_frontier_batches(self.latents, sigma=0.35, config=self.config, seed=19)
        self.assertEqual(len(first), 6)
        self.assertEqual(len(second), len(first))
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left.model_input, right.model_input))
            self.assertTrue(torch.equal(left.raw_noisy_active, right.raw_noisy_active))

        report = fixed_sigma_frontier_report(
            self.latents,
            lambda batch: torch.zeros_like(batch.model_input),
            sigma=0.35,
            config=self.config,
            seed=19,
        )
        self.assertEqual([row["frontier"] for row in report], list(range(6)))
        self.assertTrue(all(row["sigma"] == 0.35 for row in report))
        self.assertTrue(all(row["score_mean"] >= 0.0 for row in report))


if __name__ == "__main__":
    unittest.main()
