"""CPU checks for Task 5 fixed inputs, aggregation, and result provenance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common.data import load_yaml, write_jsonl
from common.logging_utils import save_json
from task5_feedback.compare import compare_feedback
from task5_feedback.diagnostics import (
    _judge_protocol, diagnostic_dir, fixed_diagnostic_groups, score_exact,
    aggregate_diagnostics,
)
from task5_feedback.generate import generation_protocol, result_dir
from task5_feedback.evaluate_math import run_math_evaluation
from task5_feedback.judge import judge_protocol
from task5_feedback.metrics import aggregate_dataset
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.score_perturbations import run_diagnostics
from task5_feedback.runtime import (
    COMPETITORS, POLICIES, adapter_digest, compare_binary_scores,
    file_digest, load_fixed_rows, policy_paths, preference_rates, signature,
)


class Task5FeedbackTests(unittest.TestCase):
    def test_stage_failures_do_not_erase_independent_work(self):
        cfg = load_yaml("configs/feedback.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            cfg["results_dir"] = temporary
            with patch("task5_feedback.evaluate_math.load_yaml", return_value=cfg), \
                    patch("task5_feedback.generate.torch.cuda.is_available", return_value=False):
                outcome = run_math_evaluation("unused", "gsm")
            self.assertEqual(outcome["status"], "INCOMPLETE")
            self.assertEqual(len(outcome["generation"]), 3)
            self.assertTrue(all(row["status"] == "FAIL" for row in outcome["generation"]))
            self.assertTrue(all(row["status"] == "SKIPPED" for row in outcome["judging"]))
            self.assertEqual(outcome["aggregation"], "INCOMPLETE")
            with patch("task5_feedback.score_perturbations.load_yaml", return_value=cfg), \
                    patch("task5_feedback.diagnostics.torch.cuda.is_available", return_value=False):
                diagnostic = run_diagnostics("unused")
            states = {row["stage"]: row["status"] for row in diagnostic["stages"]}
            self.assertEqual(states, {"exact": "PASS", "judge": "FAIL",
                                      "aggregate": "INCOMPLETE"})

    def test_fixed_inputs_and_tie_rule(self):
        cfg = load_yaml("configs/feedback.yaml")
        self.assertEqual(len(load_fixed_rows(cfg, "gsm")), 300)
        self.assertEqual(len(load_fixed_rows(cfg, "transfer")), 100)
        variants, pairs = fixed_diagnostic_groups(cfg)
        self.assertEqual((len(variants), len(pairs)), (100, 80))
        self.assertTrue(all(exact_reward(row["response"], row["gold_final"])
                            == float(row["expected_exact_reward"]) for row in variants))
        self.assertEqual(preference_rates(["A", "TIE", "B", "A"], "A")
                         ["tie_adjusted_better_rate"], 0.625)

    def test_complete_aggregation_and_changed_record_detection(self):
        cfg = load_yaml("configs/feedback.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            cfg["results_dir"] = temporary
            for dataset in ("gsm", "transfer"):
                sources = load_fixed_rows(cfg, dataset)
                root = result_dir(cfg, dataset)
                root.mkdir(parents=True)
                generated = {}
                for policy in POLICIES:
                    records = []
                    for position, source in enumerate(sources):
                        correct = (position + POLICIES.index(policy)) % 3 == 0
                        response = (f"Work. #### {source['gold_final']}" if correct
                                    else "Work. #### 99999999")
                        final = extract_designated_final(response)
                        records.append({
                            "position": position, "dataset": dataset, "policy": policy,
                            "source_id": str(source["source_index"]),
                            "question": source["question"], "gold_final": source["gold_final"],
                            "response": response, "response_tokens": 8,
                            "terminated_with_eos": True,
                            "truncated_at_generation_cap": False,
                            "designated_final": final, "format_compliant": final is not None,
                            "exact_reward": exact_reward(response, source["gold_final"]),
                        })
                    generated[policy] = records
                    path = root / f"generated_{policy}.jsonl"
                    write_jsonl(path, records)
                    identity = signature(generation_protocol(
                        cfg, dataset, policy, sources,
                        adapter_digest(policy_paths(cfg)[policy])))
                    save_json(root / f"generation_manifest_{policy}.json",
                              {"signature": identity})
                    save_json(root / f"generation_summary_{policy}.json",
                              {"status": "PASS", "signature": identity,
                               "generated_sha256": file_digest(path)})
                for competitor in COMPETITORS:
                    records = []
                    for position, source in enumerate(sources):
                        baseline = generated["sft"][position]
                        candidate = generated[competitor][position]
                        ai = ("B", "TIE", "A")[position % 3]
                        exact = compare_binary_scores(baseline["exact_reward"],
                                                      candidate["exact_reward"])
                        records.append({
                            "position": position, "dataset": dataset,
                            "source_id": str(source["source_index"]),
                            "competitor": competitor, "ai_preference": ai,
                            "exact_preference": exact, "judge_cache_hit": False,
                        })
                    path = root / f"judged_{competitor}.jsonl"
                    write_jsonl(path, records)
                    hashes = {name: file_digest(root / f"generated_{name}.jsonl")
                              for name in ("sft", competitor)}
                    identity = signature(judge_protocol(cfg, dataset, competitor, hashes))
                    save_json(root / f"judge_manifest_{competitor}.json",
                              {"signature": identity})
                    save_json(root / f"judge_summary_{competitor}.json",
                              {"status": "PASS", "signature": identity,
                               "judged_sha256": file_digest(path)})
                domain = aggregate_dataset(cfg, dataset)
                self.assertEqual(domain["status"], "COMPLETE")
                expected_win_rate = 0.5 if dataset == "gsm" else 0.505
                self.assertAlmostEqual(domain["comparisons"][0]["metrics"]
                                       ["ai_win_rate_vs_sft"], expected_win_rate)
                self.assertTrue((root / "policy_metrics.csv").exists())
                self.assertTrue((root / "comparison.png").exists())

            exact = score_exact(cfg)
            self.assertEqual((exact["variants"], exact["pairs"]), (100, 80))
            root = diagnostic_dir(cfg)
            exact_pairs = [json.loads(line) for line in
                           (root / "exact_pairs.jsonl").read_text().splitlines()]
            judged = [{"position": position, "problem_id": row["problem_id"],
                       "variant_type": row["variant_type"], "ai_preference": "TIE",
                       "judge_cache_hit": False}
                      for position, row in enumerate(exact_pairs)]
            path = root / "judged_pairs.jsonl"
            write_jsonl(path, judged)
            identity = signature(_judge_protocol(cfg, exact["exact_pairs_sha256"]))
            save_json(root / "judge_manifest.json", {"signature": identity})
            save_json(root / "judge_summary.json",
                      {"status": "PASS", "signature": identity,
                       "judged_pairs_sha256": file_digest(path)})
            diagnostics = aggregate_diagnostics(cfg)
            self.assertEqual(diagnostics["status"], "COMPLETE")
            self.assertEqual(len(diagnostics["contrasts"]), 8)
            comparison = compare_feedback(cfg)
            self.assertEqual(comparison["status"], "COMPLETE")
            self.assertEqual(len(comparison["transfer_comparison"]), 3)
            self.assertTrue((Path(temporary) / "task5_feedback" /
                             "transfer_comparison.csv").exists())

            tampered = result_dir(cfg, "gsm") / "generated_sft.jsonl"
            tampered.write_text(tampered.read_text() + "\n", encoding="utf-8")
            invalid = compare_feedback(cfg)
            self.assertEqual(invalid["status"], "INCOMPLETE")
            self.assertEqual(next(row for row in invalid["stages"]
                                  if row["stage"] == "gsm")["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
