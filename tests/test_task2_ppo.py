"""CPU checks for PPO mathematics, fixed schedules, and recoverability."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from task2_ppo.analyze_clipping import load_cached_rollouts
from task2_ppo.continue_train import _optimize_pass
from task2_ppo.ppo import (
    affected_token_fraction, compute_gae, ppo_policy_loss, shaped_rewards,
)
from task2_ppo.runtime import (
    generate_one, load_training_state, response_values, save_training_state, scheduled_rows,
)


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "test"

    def __call__(self, text, truncation=False):
        return {"input_ids": [1, 2]}


class FakeScaler:
    value = 4

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class PPOChecks(unittest.TestCase):
    def test_clipped_surrogate_uses_minimum_and_affected_fraction(self):
        old = torch.zeros(3)
        new = torch.log(torch.tensor([1.5, 0.5, 1.5]))
        advantage = torch.tensor([1.0, -1.0, -1.0])
        mask = torch.ones(3)
        loss, ratio, clip_fraction = ppo_policy_loss(new, old, advantage, mask, 0.2)
        self.assertAlmostEqual(float(loss), -(1.2 - 0.8 - 1.5) / 3, places=6)
        self.assertAlmostEqual(float(clip_fraction), 1.0)
        self.assertAlmostEqual(float(affected_token_fraction(ratio, advantage, mask, 0.2)), 2 / 3)

    def test_gae_mask_and_terminal_reward(self):
        reward = torch.tensor([[0.0, 2.0, 0.0]])
        values = torch.tensor([[0.5, 0.7, 99.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        advantage, returns = compute_gae(reward, values, mask, 1.0, 0.95)
        self.assertAlmostEqual(float(advantage[0, 1]), 1.3, places=6)
        self.assertAlmostEqual(float(advantage[0, 0]), 1.435, places=6)
        self.assertEqual(float(advantage[0, 2]), 0.0)
        self.assertAlmostEqual(float(returns[0, 1]), 2.0, places=6)
        shaped = shaped_rewards(
            torch.tensor([3.0]), torch.tensor([[0.1, 0.2]]),
            torch.tensor([[0.0, 0.0]]), torch.ones(1, 2), 0.1,
        )
        self.assertTrue(torch.allclose(shaped, torch.tensor([[-0.01, 2.98]])))

    def test_short_forks_share_standard_schedule_prefix(self):
        rows = [{"prompt_id": str(i)} for i in range(20)]
        full = scheduled_rows(rows, 6304, 20)
        short = scheduled_rows(rows, 6304, 8)
        self.assertEqual([row["prompt_id"] for row in short],
                         [row["prompt_id"] for row in full[:8]])

    def test_generation_clones_inference_tensors(self):
        with torch.inference_mode():
            sequences = torch.tensor([[1, 2, 3]])
            generated = {
                "sequences": sequences,
                "attention_mask": torch.ones_like(sequences),
                "prompt_width": 2,
                "response_ids": sequences[:, 2:],
                "response_mask": torch.ones(1, 1),
                "responses": ["answer"], "response_lengths": [1],
                "terminated_with_eos": [True], "truncated": [False],
            }
        with patch("task2_ppo.runtime.batch_generate", return_value=generated):
            safe = generate_one(None, TinyTokenizer(),
                                {"messages": [{"role": "user", "content": "hi"}]},
                                {"max_prompt_length": 2, "generation": {}}, 123, 2)
        self.assertFalse(safe["response_ids"].is_inference())
        self.assertFalse(safe["sequences"].is_inference())

    def test_two_model_checkpoint_round_trip(self):
        policy, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
        policy_optimizer = torch.optim.SGD(policy.parameters(), lr=0.1, momentum=0.9)
        critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.2, momentum=0.9)
        scaler = FakeScaler()
        for model, optimizer in ((policy, policy_optimizer), (critic, critic_optimizer)):
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        policy_before = policy.weight.detach().clone()
        critic_before = critic.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo.pt"
            save_training_state(path, "fixed-schedule", 3, {"completed_updates": 3},
                                policy, critic, policy_optimizer, critic_optimizer, scaler)
            with torch.no_grad():
                policy.weight.add_(10)
                critic.weight.add_(10)
            scaler.value = 20
            state = load_training_state(path, "fixed-schedule", policy, critic,
                                        policy_optimizer, critic_optimizer, scaler)
        self.assertEqual(state["next_update"], 3)
        self.assertEqual(state["counters"]["completed_updates"], 3)
        self.assertTrue(torch.equal(policy.weight, policy_before))
        self.assertTrue(torch.equal(critic.weight, critic_before))
        self.assertEqual(scaler.value, 4)

    def test_critic_helper_aligns_response_values_and_backpropagates(self):
        from peft import get_peft_model
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        from common.models import make_value_lora_config

        config = Qwen2Config(vocab_size=40, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, num_labels=1, pad_token_id=0)
        critic = get_peft_model(
            Qwen2ForSequenceClassification(config),
            make_value_lora_config({"lora": {"r": 2, "alpha": 4,
                                            "dropout": 0.0,
                                            "target_modules": ["q_proj", "v_proj"]}}),
        )
        generated = {
            "sequences": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "prompt_width": 2, "response_ids": torch.tensor([[3, 4]]),
        }
        values = response_values(critic, generated)
        self.assertEqual(tuple(values.shape), (1, 2))
        values.sum().backward()
        self.assertTrue(any(p.grad is not None for p in critic.parameters() if p.requires_grad))

    def test_supplied_cache_lengths_and_penalty(self):
        rows = load_cached_rollouts("cached/ppo_rollout.pt")
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(row["old_logprobs"].numel() == row["response_tokens"] for row in rows))
        self.assertTrue(all(math.isclose(
            row["effective_terminal_reward"] - row["raw_terminal_reward"],
            0.0 if row["terminated_with_eos"] else -1.0,
        ) for row in rows))

    def test_two_ppo_passes_step_both_optimizers_on_one_fixed_rollout(self):
        class ScalarModel(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(float(value)))

        policy, critic = ScalarModel(0.0), ScalarModel(0.5)
        policy_optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
        critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
        counts = {"policy": 0, "critic": 0}
        policy_hook = policy_optimizer.register_step_post_hook(
            lambda *_: counts.__setitem__("policy", counts["policy"] + 1))
        critic_hook = critic_optimizer.register_step_post_hook(
            lambda *_: counts.__setitem__("critic", counts["critic"] + 1))
        bundle = {"policy": policy, "critic": critic,
                  "policy_optimizer": policy_optimizer, "critic_optimizer": critic_optimizer,
                  "scaler": torch.amp.GradScaler("cpu", init_scale=8.0)}
        generated = {"response_ids": torch.tensor([[3, 4]])}
        rollout = {"generated": generated, "mask": torch.ones(1, 2),
                   "old_logp": torch.zeros(1, 2), "advantages": torch.ones(1, 2),
                   "returns": torch.tensor([[1.0, 2.0]])}
        cfg = {"clip_epsilon": 0.2, "value_coef": 0.5, "max_grad_norm": 1.0}

        def fake_logps(model, _generated):
            return model.weight * torch.tensor([[1.0, 2.0]]), None

        def fake_values(model, _generated):
            return model.weight * torch.tensor([[1.0, 2.0]])

        try:
            with patch("task2_ppo.continue_train._sync", return_value=0.0), \
                    patch("task2_ppo.continue_train.response_logps", side_effect=fake_logps), \
                    patch("task2_ppo.continue_train.response_values", side_effect=fake_values):
                first = _optimize_pass(bundle, rollout, cfg, 0, counts)
                second = _optimize_pass(bundle, rollout, cfg, 1, counts)
        finally:
            policy_hook.remove()
            critic_hook.remove()
        self.assertEqual(counts, {"policy": 2, "critic": 2})
        self.assertEqual((first["ppo_epoch"], second["ppo_epoch"]), (1, 2))
        self.assertEqual(float(rollout["old_logp"].sum()), 0.0)
        self.assertNotEqual(float(policy.weight.detach()), 0.0)
        self.assertNotEqual(float(critic.weight.detach()), 0.5)


if __name__ == "__main__":
    unittest.main()
