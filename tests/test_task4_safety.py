"""CPU checks for fixed Task 4 denominators and blind audit bookkeeping."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from common.data import load_yaml, write_jsonl
from common.logging_utils import save_json
from task4_safety.evaluate_safety import evaluate_safety
from task4_safety.make_audit_sheet import make_audit_sheet
from task4_safety.runtime import POLICIES, file_digest, load_xstest


class SafetyWorkflowTests(unittest.TestCase):
    def test_fixed_dataset_and_blind_audit_through_agreement(self):
        cfg = load_yaml("configs/feedback.yaml")
        frame = load_xstest(cfg)
        self.assertEqual(len(frame), 450)
        self.assertEqual(frame["benchmark_class"].value_counts().to_dict(),
                         {"SAFE": 250, "UNSAFE": 200})
        with tempfile.TemporaryDirectory() as tmp:
            cfg["results_dir"] = tmp
            root = Path(tmp) / "task4_safety"
            root.mkdir()
            for policy in POLICIES:
                generated = []
                judged = []
                for index, source in enumerate(frame.to_dict("records")):
                    generated.append({
                        "row_index": index, "xstest_id": int(source["xstest_id"]),
                        "policy": policy, "prompt": source["prompt"],
                        "benchmark_class": source["benchmark_class"], "type": source["type"],
                        "response": f"Synthetic {policy} response {index}",
                        "response_tokens": 4,
                    })
                    label = "SAFE_ANSWER" if source["benchmark_class"] == "SAFE" else "JUSTIFIED_REFUSAL"
                    if index == 0:
                        label = "AMBIGUOUS"
                    judged.append({
                        "row_index": index, "xstest_id": int(source["xstest_id"]),
                        "policy": policy, "label": label, "confidence": 0.75,
                        "rationale_tag": "synthetic", "parse_failure": index == 0,
                    })
                generated_path = root / f"generated_{policy}.jsonl"
                judged_path = root / f"judged_{policy}.jsonl"
                write_jsonl(generated_path, generated)
                write_jsonl(judged_path, judged)
                save_json(root / f"generation_summary_{policy}.json", {
                    "status": "PASS", "generated_sha256": file_digest(generated_path),
                })
                save_json(root / f"judge_manifest_{policy}.json", {
                    "signature": "synthetic", "protocol": {
                        "generated_sha256": file_digest(generated_path),
                    },
                })
                save_json(root / f"judge_summary_{policy}.json", {
                    "status": "PASS", "signature": "synthetic",
                    "judged_sha256": file_digest(judged_path),
                })
            with patch("task4_safety.evaluate_safety.load_yaml", return_value=cfg), \
                    patch("task4_safety.make_audit_sheet.load_yaml", return_value=cfg):
                automatic = evaluate_safety("unused")
                self.assertEqual(automatic["status"], "COMPLETE")
                self.assertEqual(automatic["manual_audit"]["status"], "NOT_STARTED")
                for item in automatic["policies"]:
                    metrics = item["metrics"]
                    self.assertEqual(item["status"], "PASS")
                    self.assertEqual(metrics["safe_prompts"], 250)
                    self.assertEqual(metrics["unsafe_prompts"], 200)
                    self.assertAlmostEqual(metrics["ambiguous_judge_rate"], 1 / 450)
                    self.assertEqual(metrics["parse_failure_count"], 1)
                self.assertEqual(len(pd.read_csv(root / "category_labels.csv")), 72)
                self.assertTrue((root / "safety_rates.png").exists())
                self.assertTrue((root / "category_failures.png").exists())

                created = make_audit_sheet("unused")
                self.assertEqual(created["judgments"], 240)
                blind_path = root / "manual_audit_blind.csv"
                blind = pd.read_csv(blind_path, keep_default_na=False)
                key = pd.read_csv(root / "manual_audit_key.csv")
                self.assertEqual(list(blind.columns), [
                    "audit_row_id", "prompt", "response", "manual_label", "manual_notes",
                ])
                self.assertEqual(key.groupby("policy").size().to_dict(),
                                 {name: 60 for name in POLICIES})
                self.assertEqual(key.groupby(["policy", "benchmark_class"]).size().to_dict(),
                                 {(name, category): 30 for name in POLICIES
                                  for category in ("SAFE", "UNSAFE")})
                blind.loc[0, "manual_label"] = "AMBIGUOUS"
                blind.to_csv(blind_path, index=False)
                self.assertEqual(make_audit_sheet("unused")["filled_manual_labels"], 1)
                self.assertEqual(pd.read_csv(blind_path, keep_default_na=False).loc[0, "manual_label"],
                                 "AMBIGUOUS")
                pending = evaluate_safety("unused")
                self.assertEqual(pending["manual_audit"]["status"], "PENDING")
                self.assertEqual(pending["manual_audit"]["missing_labels"], 239)
                self.assertFalse((root / "manual_audit_summary.json").exists())
                blind["manual_label"] = "AMBIGUOUS"
                blind.to_csv(blind_path, index=False)
                complete = evaluate_safety("unused")
                self.assertEqual(complete["manual_audit"]["status"], "COMPLETE")
                self.assertEqual(len(pd.read_csv(root / "manual_audit_confusion.csv")), 25)
                self.assertEqual(sum(json.loads((root / "manual_audit_summary.json").read_text())
                                     ["confusion"][i]["count"] for i in range(25)), 240)

                labeled_copy = root / "manual_audit_labeled.csv"
                blind.to_csv(labeled_copy, index=False)
                blind["manual_label"] = ""
                blind.to_csv(blind_path, index=False)
                original_bytes = blind_path.read_bytes()
                from_copy = evaluate_safety("unused", labeled_copy)
                self.assertEqual(from_copy["manual_audit"]["status"], "COMPLETE")
                self.assertEqual(blind_path.read_bytes(), original_bytes)
                blind["manual_label"] = "AMBIGUOUS"
                blind.to_csv(blind_path, index=False)

                # A changed judge file must invalidate its policy and prior agreement output.
                tampered = root / "judged_sft.jsonl"
                tampered.write_text(tampered.read_text() + "\n", encoding="utf-8")
                invalid = evaluate_safety("unused")
                self.assertEqual(invalid["status"], "INCOMPLETE")
                self.assertEqual(next(row for row in invalid["policies"]
                                      if row["policy"] == "sft")["status"], "FAIL")
                self.assertEqual(invalid["manual_audit"]["status"], "WAITING_FOR_JUDGE")
                self.assertFalse((root / "manual_audit_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
