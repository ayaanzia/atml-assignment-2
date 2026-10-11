"""Small CPU regression checks for the Task 1 objective and data protocol."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from common.data import encode_prompt_response, read_jsonl
from common.metrics import parse_word_limit
from task1_dpo.data import prepare_preference_examples, seed_for_prompt, selection_summary
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import _epoch_loader, _load_state, _save_state


class TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = [10] + [ord(char) % 50 + 20 for char in messages[0]["content"]] + [11]
        return {"input_ids": ids} if tokenize else "rendered"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 50 + 20 for char in text]}


def row(number: int, prompt: str = "p") -> dict:
    return {
        "prompt_id": f"p{number}", "source_index": number,
        "chosen": [{"role": "user", "content": prompt},
                   {"role": "assistant", "content": "yes"}],
        "rejected": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": "no"}],
    }


class Task1Checks(unittest.TestCase):
    def test_reference_margin_is_subtracted(self):
        pc = torch.tensor([3.0], requires_grad=True)
        pr = torch.tensor([1.0])
        rc = torch.tensor([5.0])
        rr = torch.tensor([1.0])
        loss, diag = dpo_loss(pc, pr, rc, rr, 0.1)
        self.assertAlmostEqual(float(loss.detach()), math.log1p(math.exp(0.2)), places=6)
        self.assertEqual(float(diag["preference_accuracy"]), 0.0)
        loss.backward()
        self.assertLess(float(pc.grad), 0.0)

    def test_zero_adjusted_margin_has_log_two_loss(self):
        loss, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([0.0]),
                           torch.tensor([2.0]), torch.tensor([1.0]), 0.3)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)

    def test_encoding_preserves_prompt_and_masks_response(self):
        tok = TinyTokenizer()
        ids, mask = encode_prompt_response(tok, [{"role": "user", "content": "abc"}], "long answer", 8)
        self.assertEqual(ids[:5], [10, 20 + ord("a") % 50, 20 + ord("b") % 50,
                                   20 + ord("c") % 50, 11])
        self.assertEqual(mask, [0] * 5 + [1] * 3)
        self.assertEqual(ids[-1], tok.eos_token_id)

    def test_beta_selection_after_prompt_exclusion_and_deterministic_order(self):
        tok = TinyTokenizer()
        rows = [row(0, "x" * 20)] + [row(i) for i in range(1, 6)]
        selected, excluded = prepare_preference_examples(rows, tok, 12, max_examples=3)
        self.assertEqual([item["prompt_id"] for item in selected], ["p1", "p2", "p3"])
        self.assertEqual(excluded[0]["prompt_id"], "p0")
        self.assertEqual(excluded[0]["reason"], "prompt_exceeds_sequence_cap")
        same, same_excluded = prepare_preference_examples(rows, tok, 12, max_examples=3)
        self.assertEqual(selection_summary(selected, excluded)["selected_ids_sha256"],
                         selection_summary(same, same_excluded)["selected_ids_sha256"])

    def test_epoch_order_and_batch_cursor(self):
        tok = TinyTokenizer()
        examples, _ = prepare_preference_examples([row(i) for i in range(7)], tok, 12)
        full, total = _epoch_loader(examples, tok, 2, 6304, 0, 0, 0)
        batches = [ids for _, _, ids in full]
        resumed, resumed_total = _epoch_loader(examples, tok, 2, 6304, 0, 2, 0)
        self.assertEqual(total, 4)
        self.assertEqual(resumed_total, total)
        self.assertEqual([ids for _, _, ids in resumed], batches[2:])
        self.assertEqual(sum(len(ids) for ids in batches), 7)
        self.assertEqual(seed_for_prompt(6304, "p1"), seed_for_prompt(6304, "p1"))
        self.assertNotEqual(seed_for_prompt(6304, "p1"), seed_for_prompt(6304, "p2"))

    def test_fixed_word_limit_prompts_are_parseable(self):
        rows = read_jsonl("data/word_limit_prompts.jsonl")
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(parse_word_limit(row["messages"][0]["content"]) is not None
                            for row in rows))

    def test_checkpoint_restores_weights_optimizer_and_cursor(self):
        class Scaler:
            value = 3

            def state_dict(self):
                return {"value": self.value}

            def load_state_dict(self, state):
                self.value = state["value"]

        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
        scaler = Scaler()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        expected_weight = model.weight.detach().clone()
        expected_momentum = optimizer.state[model.weight]["momentum_buffer"].clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            _save_state(checkpoint, model, optimizer, scaler, "same-run",
                        {"optimizer_updates": 1}, 0, 3)
            with torch.no_grad():
                model.weight.add_(9)
            optimizer.state[model.weight]["momentum_buffer"].zero_()
            scaler.value = 99
            restored = _load_state(checkpoint, model, optimizer, scaler, "same-run")
        self.assertEqual((restored["epoch"], restored["next_batch"]), (0, 3))
        self.assertEqual(restored["counters"]["optimizer_updates"], 1)
        self.assertTrue(torch.equal(model.weight, expected_weight))
        self.assertTrue(torch.equal(optimizer.state[model.weight]["momentum_buffer"], expected_momentum))
        self.assertEqual(scaler.value, 3)


if __name__ == "__main__":
    unittest.main()
