"""Prepare a fixed, blind human audit sheet; never assign manual labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import print_gpu_info, save_json
from task4_safety.runtime import (
    LABEL_ORDER, POLICIES, check_policy_rows, file_digest, load_xstest,
)


def fixed_audit_ids(base_rows: list[dict], per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ("SAFE", "UNSAFE"):
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(value) for value in ids)


def make_audit_sheet(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    frame = load_xstest(cfg)
    root = repo_path(cfg["results_dir"]) / "task4_safety"
    root.mkdir(parents=True, exist_ok=True)
    generated = {}
    digests = {}
    for policy in POLICIES:
        path = root / f"generated_{policy}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Complete {policy} generation is needed before the audit sheet")
        rows = read_jsonl(path)
        check_policy_rows(rows, frame, policy)
        summary_path = root / f"generation_summary_{policy}.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Complete {policy} generation summary is needed before the audit sheet")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS" or summary.get("generated_sha256") != file_digest(path):
            raise ValueError(f"{policy} generated responses changed or generation did not complete")
        generated[policy] = {int(row["xstest_id"]): row for row in rows}
        digests[policy] = file_digest(path)
    ids = fixed_audit_ids(read_jsonl(root / "generated_sft.jsonl"),
                          int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    if len(ids) != 60 or len({value for value in ids}) != 60:
        raise ValueError("Task 4 manual audit requires 60 unique fixed prompt IDs")
    by_id = {int(row["xstest_id"]): row for row in frame.to_dict("records")}
    if sum(by_id[value]["benchmark_class"] == "SAFE" for value in ids) != 30:
        raise ValueError("Task 4 manual audit selection is not balanced 30 SAFE / 30 UNSAFE")
    blind_path = root / "manual_audit_blind.csv"
    key_path = root / "manual_audit_key.csv"
    ids_path = root / "manual_audit_ids.csv"
    manifest_path = root / "manual_audit_manifest.json"
    present = [path.exists() for path in (blind_path, key_path, ids_path, manifest_path)]
    if any(present):
        if not all(present):
            raise FileExistsError("Partial manual-audit artifacts exist; preserve human labels and inspect them")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generated_sha256") != digests or manifest.get("selected_ids") != ids or \
                manifest.get("key_sha256") != file_digest(key_path):
            raise ValueError("Existing blind audit sheet does not match fixed responses or mapping")
        blind = pd.read_csv(blind_path, keep_default_na=False)
        key = pd.read_csv(key_path, keep_default_na=False)
        blind_ids = blind["audit_row_id"].astype(str).tolist()
        key_ids = key["audit_row_id"].astype(str).tolist()
        if len(blind) != 240 or len(key) != 240 or \
                len(set(blind_ids)) != 240 or set(blind_ids) != set(key_ids):
            raise ValueError("Existing manual-audit sheet/key has changed row identity")
        return {"status": "EXISTS_UNCHANGED", "blind_sheet": str(blind_path),
                "mapping": str(key_path), "prompts": len(ids), "judgments": len(blind),
                "filled_manual_labels": sum(bool(value.strip()) for value in blind["manual_label"].astype(str))}
    entries = [(value, policy) for value in ids for policy in POLICIES]
    permutation = np.random.default_rng(int(cfg["seed"])).permutation(len(entries))
    blind_rows = []
    key_rows = []
    for position, source_position in enumerate(permutation):
        xstest_id, policy = entries[int(source_position)]
        item = generated[policy][xstest_id]
        row_id = f"audit_{position + 1:04d}"
        blind_rows.append({
            "audit_row_id": row_id, "prompt": item["prompt"],
            "response": item["response"], "manual_label": "", "manual_notes": "",
        })
        key_rows.append({
            "audit_row_id": row_id, "xstest_id": xstest_id, "policy": policy,
            "benchmark_class": item["benchmark_class"], "type": item["type"],
        })
    pd.DataFrame(blind_rows).to_csv(blind_path, index=False)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    pd.DataFrame({"xstest_id": ids}).to_csv(ids_path, index=False)
    manifest = {
        "selection_seed": int(cfg["seed"]),
        "per_class": int(cfg["manual_audit_per_class"]),
        "selected_ids": ids, "selected_prompts": len(ids),
        "human_judgments_required": len(blind_rows),
        "generated_sha256": digests,
        "key_sha256": file_digest(key_path),
        "manual_labels": list(LABEL_ORDER),
        "instruction": "Label prompt and response without opening manual_audit_key.csv or AI judge files; do not alter audit_row_id",
    }
    save_json(manifest_path, manifest)
    print(f"Wrote blind Task 4 audit sheet: {blind_path} ({len(ids)} prompts, {len(blind_rows)} responses)", flush=True)
    return {"status": "CREATED", "blind_sheet": str(blind_path),
            "mapping": str(key_path), "prompts": len(ids), "judgments": len(blind_rows),
            "filled_manual_labels": 0}


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feedback.yaml")
    args = parser.parse_args()
    make_audit_sheet(args.config)


if __name__ == "__main__":
    main()
