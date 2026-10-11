"""Task 3 runtime helpers. SSH and Vast setup are deliberately outside this module."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

# These are data/model-neutral helpers already used by Task 2. Reusing them also
# keeps long-prompt handling identical for the shared UltraFeedback prompt pool.
from task2_ppo.runtime import (
    changed_parameter_count,
    generate_one,
    parameter_audit,
    promote_trainables,
    response_logps,
    restore_rng,
    reward_input_length,
    rng_state,
    scheduled_rows,
    single_prompt_collate,
    stable_seed,
    trainable_snapshot,
)


LENGTH_BINS = ("1-127", "128-255", "256-383", "384-512")


def length_bin(tokens: int) -> str:
    if tokens < 1:
        raise ValueError("A completion must have at least one generated token")
    if tokens < 128:
        return LENGTH_BINS[0]
    if tokens < 256:
        return LENGTH_BINS[1]
    if tokens < 384:
        return LENGTH_BINS[2]
    if tokens <= 512:
        return LENGTH_BINS[3]
    return ">512"


def training_seed(seed: int, update_index: int, prompt_id: str, generation_index: int) -> int:
    return stable_seed(seed, f"grpo-train-{update_index}-{prompt_id}-{generation_index}")


def evaluation_seed(seed: int, prompt_id: str) -> int:
    return stable_seed(seed, f"grpo-eval-{prompt_id}")


def signature_for(cfg: dict, rows: list[dict], loss_type: str) -> str:
    payload = {"config": cfg, "prompt_ids": [row["prompt_id"] for row in rows],
               "loss_type": loss_type}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def save_training_state(path: Path, signature: str, next_update: int, counters: dict,
                        policy, optimizer, scaler) -> None:
    state = {
        "schema": 1, "signature": signature, "next_update": next_update,
        "counters": dict(counters), "policy_trainable": trainable_snapshot(policy),
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "rng": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_training_state(path: Path, signature: str, policy, optimizer, scaler) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != 1 or state.get("signature") != signature:
        raise ValueError("Task 3 checkpoint configuration, loss type, or prompt schedule mismatch")
    current = {name: p for name, p in policy.named_parameters() if p.requires_grad}
    if set(current) != set(state["policy_trainable"]):
        raise ValueError("Task 3 checkpoint trainable parameter names differ")
    with torch.no_grad():
        for name, parameter in current.items():
            saved = state["policy_trainable"][name]
            if saved.shape != parameter.shape:
                raise ValueError(f"Checkpoint parameter shape mismatch: {name}")
            parameter.copy_(saved.to(parameter.device, parameter.dtype))
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state
