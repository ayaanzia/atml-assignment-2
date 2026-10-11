"""CPU checks for GRPO math, fixed-budget grouping, and resumability."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from task3_grpo.analyze_group_size import regroup_equal_generation_budget
from task3_grpo.continue_train import _optimize_group
from task3_grpo.grpo import (
    group_relative_advantages, grpo_policy_loss, mask_truncated_sequences,
)
from task3_grpo.runtime import (
    length_bin, load_training_state, save_training_state,
)


class GRPOTests(unittest.TestCase):
    def test_advantages_are_normalized_within_each_prompt(self):
        rewards = torch.tensor([1.0, 3.0, 100.0, 100.0, 102.0, 98.0])
        groups = torch.tensor([10, 10, 20, 30, 30, 30])
        actual = group_relative_advantages(rewards, groups)
        self.assertTrue(torch.allclose(actual[:2], torch.tensor([-1.0, 1.0])))
        self.assertEqual(actual[2].item(), 0.0)
        self.assertAlmostEqual(actual[3:].mean().item(), 0.0, places=6)
        self.assertAlmostEqual(actual[3:].std(unbiased=False).item(), 1.0, places=6)
        tiny = group_relative_advantages(torch.tensor([1.0, 1.0 + 1e-7]),
                                         torch.tensor([0, 0]))
        self.assertTrue(torch.equal(tiny, torch.zeros_like(tiny)))

    def test_masking_and_normalization_change_sequence_weight(self):
        zeros = torch.zeros(2, 4)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        canonical, _ = grpo_policy_loss(zeros, zeros, advantages, mask, zeros,
                                        eps=0.2, beta=0.0, loss_type="grpo")
        dr, _ = grpo_policy_loss(zeros, zeros, advantages, mask, zeros,
                                 eps=0.2, beta=0.0, loss_type="dr_grpo",
                                 max_completion_length=4)
        self.assertAlmostEqual(canonical.item(), 0.0, places=6)
        self.assertAlmostEqual(dr.item(), 0.25, places=6)
        masked = mask_truncated_sequences(mask, [False, True])
        self.assertEqual(masked[1].sum().item(), 0.0)

    def test_clipped_surrogate_uses_minimum_and_kl_penalty_is_nonnegative(self):
        new = torch.log(torch.tensor([[1.3], [0.7]]))
        old = torch.zeros_like(new)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones_like(new)
        loss, details = grpo_policy_loss(new, old, advantages, mask, old,
                                         eps=0.2, beta=0.0)
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        self.assertAlmostEqual(details["clip_fraction"].item(), 1.0, places=6)
        self.assertGreaterEqual(details["kl_penalty"].item(), 0.0)

    def test_microbatch_update_matches_full_group_gradient(self):
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

        features = [1.0, 2.0, 4.0, 8.0]
        advantages = [1.0, -1.0, 1.0, -1.0]
        policy = TinyPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
        entries = []
        masks = [torch.ones(1, 2), torch.ones(1, 2),
                 torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2)]
        for feature, advantage, mask in zip(features, advantages, masks):
            entries.append({
                "generated": {"feature": feature},
                "old_logp": torch.zeros(1, 2),
                "ref_logp": torch.full((1, 2), -0.1),
                "advantage": torch.tensor([advantage]),
                "train_mask": mask,
            })
        expected_policy = TinyPolicy()
        expected_new = expected_policy.weight * torch.tensor(features).view(4, 1).expand(4, 2)
        expected, _ = grpo_policy_loss(
            expected_new, torch.zeros(4, 2), torch.tensor(advantages),
            torch.cat(masks), torch.full((4, 2), -0.1),
            eps=0.2, beta=0.1, loss_type="grpo",
        )
        expected.backward()
        expected_gradient = expected_policy.weight.grad.item()
        step_counts = {"optimizer": 0}
        hook = optimizer.register_step_post_hook(
            lambda *_: step_counts.__setitem__("optimizer", step_counts["optimizer"] + 1))
        try:
            with patch("task3_grpo.continue_train.response_logps",
                       side_effect=lambda model, generated: (
                           model.weight * torch.full((1, 2), generated["feature"]), None)):
                details = _optimize_group(
                    {"policy": policy, "optimizer": optimizer, "scaler": scaler},
                    {"entries": entries, "usable_tokens": 5},
                    {"kl_beta": 0.1, "clip_epsilon": 0.2,
                     "max_completion_length": 512, "max_grad_norm": 1000000.0},
                    "grpo", 0, step_counts,
                )
        finally:
            hook.remove()
        self.assertEqual(details["backward_passes"], 3)
        self.assertEqual(step_counts["optimizer"], 1)
        self.assertAlmostEqual(policy.weight.item(), -0.1 * expected_gradient, places=5)
        self.assertAlmostEqual(details["total_loss"], expected.item(), places=5)

    def test_group_cache_partition_keeps_each_generation_once(self):
        source = {
            str(prompt): [{"prompt_id": f"p{prompt}", "generation_index": i,
                           "reward": float(prompt + i)} for i in range(8)]
            for prompt in range(3)
        }
        for k, expected_count in ((2, 12), (4, 6), (8, 3)):
            groups = regroup_equal_generation_budget(source, k)
            self.assertEqual(len(groups), expected_count)
            self.assertEqual(sum(len(g["completion_rows"]) for g in groups), 24)
            for prompt in range(3):
                indices = [row["generation_index"] for group in groups
                           if group["source_index"] == str(prompt)
                           for row in group["completion_rows"]]
                self.assertEqual(indices, list(range(8)))

    def test_length_bins_and_checkpoint_round_trip(self):
        self.assertEqual(length_bin(127), "1-127")
        self.assertEqual(length_bin(128), "128-255")
        self.assertEqual(length_bin(512), "384-512")
        policy = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
        before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "latest_state.pt"
            save_training_state(path, "signature", 3, {"completed_updates": 3},
                                policy, optimizer, scaler)
            with torch.no_grad():
                for parameter in policy.parameters():
                    parameter.add_(7.0)
            state = load_training_state(path, "signature", policy, optimizer, scaler)
            self.assertEqual(state["next_update"], 3)
            for name, parameter in policy.named_parameters():
                self.assertTrue(torch.equal(parameter.detach(), before[name]))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_training_state(path, "different", policy, optimizer, scaler)


if __name__ == "__main__":
    unittest.main()
