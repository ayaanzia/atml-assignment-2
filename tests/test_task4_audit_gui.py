"""The blind labeling tool saves a separate, resumable CSV."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from task4_safety.manual_audit_gui import (
    COLUMNS, load_blind_work, read_rows, save_copy,
)


class AuditGuiFileTests(unittest.TestCase):
    def test_separate_copy_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "manual_audit_blind.csv"
            output = Path(directory) / "manual_audit_labeled.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                writer.writeheader()
                for index in range(240):
                    writer.writerow({"audit_row_id": f"audit_{index + 1:04d}",
                                     "prompt": f"Prompt {index}",
                                     "response": f"Response {index}",
                                     "manual_label": "", "manual_notes": ""})
            original = source.read_bytes()
            rows = load_blind_work(source, output)
            rows[0]["manual_label"] = "SAFE_ANSWER"
            save_copy(output, rows)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(read_rows(output)[0]["manual_label"], "SAFE_ANSWER")
            self.assertEqual(load_blind_work(source, output)[0]["manual_label"],
                             "SAFE_ANSWER")
            with self.assertRaises(ValueError):
                load_blind_work(source, source)
            changed = read_rows(output)
            changed[0]["prompt"] = "Different prompt"
            save_copy(output, changed)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_blind_work(source, output)


if __name__ == "__main__":
    unittest.main()
