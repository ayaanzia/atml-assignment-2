"""Fixed Task 5 identities, validation, and small evaluation utilities."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from common.data import read_jsonl, repo_path


POLICIES = ("sft", "rlvr", "rlaif")
COMPETITORS = ("rlvr", "rlaif")
DATASETS = {"gsm": ("gsm_eval", 300), "transfer": ("math_transfer_eval", 100)}
VARIANTS = (
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
)
CONTRASTS = VARIANTS[1:]


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repo_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def gpu_memory() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0}
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
    }


def model_audit(model, name: str) -> dict:
    params = list(model.parameters())
    return {
        "name": name,
        "parameter_dtypes": dict(Counter(str(param.dtype) for param in params)),
        "devices": dict(Counter(str(param.device) for param in params)),
        "loaded_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "loaded_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "model_commit_hash": getattr(getattr(model, "config", None), "_commit_hash", None),
    }


def policy_paths(cfg: dict) -> dict[str, str | None]:
    paths = {"sft": None, "rlvr": cfg["policies"]["rlvr"],
             "rlaif": cfg["policies"]["rlaif"]}
    fixed = {"sft": None, "rlvr": "checkpoints/rlvr_policy",
             "rlaif": "checkpoints/rlaif_policy"}
    if paths != fixed or cfg["policies"].get("sft") is not None:
        raise ValueError("Task 5 requires the untouched SFT base and the two fixed course adapters")
    return paths


def check_released_precision(cfg: dict) -> None:
    if str(cfg.get("dtype", "")).lower() not in {"float16", "fp16", "half"} or \
            cfg.get("quantize_frozen_models") is not True:
        raise ValueError("Task 5 uses the released FP16 policies and 4-bit frozen AI judge")


def check_released_generation(cfg: dict) -> None:
    expected = {"temperature": 0.7, "top_p": 0.9, "do_sample": True}
    if cfg.get("generation") != expected or int(cfg.get("math_max_new_tokens", 0)) != 512:
        raise ValueError("Task 5 uses released sampled decoding and the 512-token math cap")


def adapter_digest(path: str | None) -> str | None:
    if path is None:
        return None
    directory = repo_path(path)
    config = directory / "adapter_config.json"
    if not config.exists():
        raise FileNotFoundError(f"Missing frozen adapter configuration in {directory}")
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = directory / name
        if candidate.exists():
            return signature({"config_sha256": file_digest(config),
                              "weights_sha256": file_digest(candidate)})
    raise FileNotFoundError(f"Missing frozen adapter weights in {directory}")


def source_id(row: dict) -> str:
    return str(row["source_index"])


def load_fixed_rows(cfg: dict, dataset: str) -> list[dict]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown Task 5 dataset: {dataset}")
    path_key, count = DATASETS[dataset]
    rows = read_jsonl(cfg["paths"][path_key])
    if len(rows) != count:
        raise ValueError(f"{dataset} requires {count} fixed rows, found {len(rows)}")
    needed = {"source_index", "question", "messages", "gold_final"}
    if any(not needed.issubset(row) or not isinstance(row["messages"], list)
           or not row["messages"] or not str(row["question"]).strip()
           for row in rows):
        raise ValueError(f"{dataset} has a missing question, messages, source ID, or gold answer")
    ids = [source_id(row) for row in rows]
    if len(set(ids)) != count:
        raise ValueError(f"{dataset} source IDs are not unique")
    return rows


def dataset_digest(rows: list[dict]) -> str:
    return signature({"rows": rows})


def generation_seed(seed: int, dataset: str, row_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:task5:{dataset}:{row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def check_generated_rows(records: list[dict], sources: list[dict],
                         dataset: str, policy: str) -> None:
    if len(records) != len(sources):
        raise ValueError(f"{dataset}/{policy}: need {len(sources)} responses, found {len(records)}")
    for position, (record, source) in enumerate(zip(records, sources)):
        if record.get("position") != position or record.get("dataset") != dataset or \
                record.get("policy") != policy or record.get("source_id") != source_id(source) or \
                record.get("question") != source["question"] or \
                str(record.get("gold_final")) != str(source["gold_final"]):
            raise ValueError(f"{dataset}/{policy}: generated row {position} differs from the fixed source")


def compare_binary_scores(score_a: float, score_b: float) -> str:
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return "TIE"


def preference_rates(verdicts: list[str], better: str) -> dict:
    if better not in {"A", "B"}:
        raise ValueError("better must be A or B")
    if not verdicts or any(value not in {"A", "B", "TIE"} for value in verdicts):
        raise ValueError("Need nonempty valid pairwise verdicts")
    n = len(verdicts)
    wins = sum(value == better for value in verdicts)
    ties = sum(value == "TIE" for value in verdicts)
    wrong = n - wins - ties
    return {
        "pairs": n, "better_count": wins, "tie_count": ties, "wrong_count": wrong,
        "better_rate": wins / n, "tie_rate": ties / n,
        "wrong_rate": wrong / n, "tie_adjusted_better_rate": (wins + 0.5 * ties) / n,
    }
