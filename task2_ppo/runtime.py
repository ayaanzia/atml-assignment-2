"""Shared, explicit Task 2 runtime helpers; no cloud connection code."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common.data import prompt_messages
from common.generation import batch_generate, response_token_logprobs


def stable_seed(base_seed: int, identity: str) -> int:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def scheduled_rows(rows: list[dict], seed: int, updates: int) -> list[dict]:
    if updates < 1 or updates > len(rows):
        raise ValueError(f"Need 1..{len(rows)} updates, got {updates}")
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    return [rows[index] for index in order[:updates]]


def single_prompt_collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("The released PPO setting uses one prompt per update")
    return batch[0]


def prompt_token_lengths(tokenizer, messages: list[dict], cap: int) -> tuple[int, int, bool]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = len(tokenizer(rendered, truncation=False)["input_ids"])
    return full, min(full, cap), full > cap


def conditioning_messages(tokenizer, messages: list[dict], cap: int) -> tuple[list[dict], int, int, bool]:
    """Keep the chat wrapper and assistant suffix; trim older user content only."""
    full, _, truncated = prompt_token_lengths(tokenizer, messages, cap)
    if not truncated:
        return messages, full, full, False
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Task 2 prompt trimming expects the released single-user prompt schema")
    original_ids = tokenizer(str(messages[0]["content"]), add_special_tokens=False)["input_ids"]
    best = None
    low, high = 0, len(original_ids)
    while low <= high:
        keep = (low + high) // 2
        suffix = original_ids[-keep:] if keep else []
        content = tokenizer.decode(suffix, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        candidate = [{**messages[0], "content": content}]
        rendered = tokenizer.apply_chat_template(candidate, tokenize=False,
                                                 add_generation_prompt=True)
        length = len(tokenizer(rendered, truncation=False)["input_ids"])
        if length <= cap:
            best = (candidate, length)
            low = keep + 1
        else:
            high = keep - 1
    if best is None:
        raise ValueError("Chat wrapper alone does not fit the configured Task 2 prompt cap")
    return best[0], full, best[1], True


def generate_one(model, tokenizer, row: dict, cfg: dict, seed: int,
                 response_cap: int) -> dict:
    """The helper generates in inference mode; clone outputs for later backward."""
    messages = prompt_messages(row)
    conditioned, prompt_full, prompt_used, prompt_truncated = conditioning_messages(
        tokenizer, messages, int(cfg["max_prompt_length"])
    )
    cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        raw = batch_generate(
            model, tokenizer, [conditioned], int(cfg["max_prompt_length"]), response_cap,
            **cfg["generation"],
        )
    result = {key: value.clone() if isinstance(value, torch.Tensor) else value
              for key, value in raw.items()}
    del raw
    if int(result["prompt_width"]) != prompt_used:
        raise ValueError("Processed prompt length does not match the generation helper")
    if any(value.is_inference() for value in result.values() if isinstance(value, torch.Tensor)):
        raise RuntimeError("Generated tensors remain inference tensors after cloning")
    result.update({
        "messages": messages, "conditioning_messages": conditioned,
        "seed": seed, "prompt_tokens_full": prompt_full,
        "prompt_tokens_used": prompt_used, "prompt_truncated": prompt_truncated,
    })
    return result


def response_logps(model, generated: dict, entropy: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    result = response_token_logprobs(
        model, generated["sequences"], generated["attention_mask"],
        generated["prompt_width"], generated["response_ids"],
    )
    selected = result[0].float()
    entropy_values = token_entropy_chunked(result[1]) if entropy else None
    del result
    return selected, entropy_values


def token_entropy_chunked(logits: torch.Tensor, chunk_tokens: int = 32) -> torch.Tensor:
    """Exact categorical entropy in response-position chunks to limit FP32 storage."""
    pieces = []
    for start in range(0, logits.shape[1], chunk_tokens):
        x = logits[:, start:start + chunk_tokens, :].float()
        log_z = torch.logsumexp(x, dim=-1)
        probabilities = torch.softmax(x, dim=-1)
        pieces.append(log_z - (probabilities * x).sum(dim=-1))
        del x, log_z, probabilities
    return torch.cat(pieces, dim=1)


def response_values(value_model, generated: dict) -> torch.Tensor:
    from common.models import token_values

    values = token_values(value_model, generated["sequences"], generated["attention_mask"])
    start = int(generated["prompt_width"]) - 1
    response_steps = generated["response_ids"].shape[1]
    selected = values[:, start:start + response_steps].float()
    if selected.shape != generated["response_ids"].shape:
        raise ValueError("Critic response-value alignment failed")
    return selected


def reward_input_length(tokenizer, messages: list[dict], response: str) -> int:
    rendered = tokenizer.apply_chat_template(
        list(messages) + [{"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False,
    )
    return len(tokenizer(rendered, truncation=False)["input_ids"])


def parameter_audit(model, name: str) -> dict:
    parameters = list(model.parameters())
    return {
        "name": name,
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in parameters)),
        "trainable_dtypes": dict(Counter(str(p.dtype) for p in parameters if p.requires_grad)),
        "devices": dict(Counter(str(p.device) for p in parameters)),
        "loaded_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "loaded_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
    }


def promote_trainables(model) -> list[str]:
    """GradScaler cannot unscale FP16 parameter gradients; retain FP16 bases."""
    promoted = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.float()
            promoted.append(name)
    return promoted


def trainable_snapshot(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def changed_parameter_count(model, before: dict[str, torch.Tensor]) -> int:
    return sum(not torch.equal(p.detach().cpu(), before[name])
               for name, p in model.named_parameters() if p.requires_grad)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(path: Path, signature: str, next_update: int, counters: dict,
                        policy, critic, policy_optimizer, critic_optimizer, scaler) -> None:
    state = {
        "schema": 1, "signature": signature, "next_update": next_update,
        "counters": counters.copy(),
        "policy_trainable": trainable_snapshot(policy),
        "critic_trainable": trainable_snapshot(critic),
        "policy_optimizer": policy_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "scaler": scaler.state_dict(), "rng": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_training_state(path: Path, signature: str, policy, critic,
                        policy_optimizer, critic_optimizer, scaler) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != 1 or state.get("signature") != signature:
        raise ValueError("Task 2 checkpoint configuration or prompt schedule mismatch")
    for model, key in ((policy, "policy_trainable"), (critic, "critic_trainable")):
        current = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if set(current) != set(state[key]):
            raise ValueError(f"Checkpoint {key} names do not match loaded model")
        with torch.no_grad():
            for name, param in current.items():
                stored = state[key][name]
                if stored.shape != param.shape:
                    raise ValueError(f"Checkpoint shape mismatch: {name}")
                param.copy_(stored.to(param.device, param.dtype))
    policy_optimizer.load_state_dict(state["policy_optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state


def signature_for(config: dict, scheduled: list[dict]) -> str:
    payload = {"config": config, "prompt_ids": [row["prompt_id"] for row in scheduled]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
