"""Fixed Task 4 identities, data checks, and small runtime utilities."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from common.data import repo_path


POLICIES = ("sft", "dpo", "ppo", "grpo")
LABEL_ORDER = ("SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE",
               "OVER_REFUSAL", "AMBIGUOUS")
EXPECTED_SAFE = 250
EXPECTED_UNSAFE = 200
EXPECTED_TOTAL = EXPECTED_SAFE + EXPECTED_UNSAFE


def load_xstest(cfg: dict) -> pd.DataFrame:
    frame = pd.read_csv(repo_path(cfg["paths"]["xstest"]), keep_default_na=False)
    required = {"xstest_id", "prompt", "type", "benchmark_class"}
    if not required.issubset(frame.columns):
        raise ValueError(f"XSTest CSV lacks {sorted(required - set(frame.columns))}")
    if len(frame) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} fixed XSTest prompts, got {len(frame)}")
    frame["xstest_id"] = frame["xstest_id"].astype(int)
    if frame["xstest_id"].duplicated().any() or (frame["prompt"].astype(str).str.len() == 0).any():
        raise ValueError("XSTest IDs must be unique and prompts nonempty")
    counts = frame["benchmark_class"].value_counts().to_dict()
    if counts != {"SAFE": EXPECTED_SAFE, "UNSAFE": EXPECTED_UNSAFE}:
        raise ValueError(f"Unexpected XSTest SAFE/UNSAFE counts: {counts}")
    if frame["type"].nunique() != 18:
        raise ValueError("Expected 18 XSTest type categories")
    return frame


def dataset_digest(frame: pd.DataFrame) -> str:
    records = frame[["xstest_id", "prompt", "type", "benchmark_class"]].to_dict("records")
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def adapter_digest(path: str | None) -> str | None:
    if path is None:
        return None
    directory = repo_path(path)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = directory / name
        if candidate.exists():
            return file_digest(candidate)
    raise FileNotFoundError(f"No policy adapter weights in {directory}")


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def model_audit(model, name: str) -> dict:
    params = list(model.parameters())
    return {
        "name": name,
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in params)),
        "devices": dict(Counter(str(p.device) for p in params)),
        "loaded_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "loaded_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
    }


def gpu_memory() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
    }


def check_policy_rows(rows: list[dict], frame: pd.DataFrame, policy: str) -> None:
    expected = frame.to_dict("records")
    if len(rows) != len(expected):
        raise ValueError(f"{policy}: expected {len(expected)} records, got {len(rows)}")
    for index, (record, source) in enumerate(zip(rows, expected)):
        if record.get("policy") != policy or int(record.get("xstest_id", -1)) != int(source["xstest_id"]) or \
                record.get("prompt") != source["prompt"] or \
                record.get("benchmark_class") != source["benchmark_class"] or \
                record.get("type") != source["type"]:
            raise ValueError(f"{policy}: generated row {index} differs from the fixed XSTest source")
