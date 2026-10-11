"""Deterministic Task 1 data preparation and selection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from common.data import (
    chat_template_token_ids,
    encode_prompt_response,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
)


def example_identity(row: dict, position: int) -> dict:
    return {
        "file_position": position,
        "source_index": row.get("source_index"),
        "prompt_id": row.get("prompt_id"),
        "length_stratum": row.get("length_stratum"),
    }


def prepare_preference_examples(
    rows: list[dict], tokenizer, max_length: int, max_examples: int | None = None
) -> tuple[list[dict], list[dict]]:
    """Skip prompt-only overflows, then take the first N eligible rows in file order."""
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    eligible, excluded = [], []
    for position, row in enumerate(rows):
        identity = example_identity(row, position)
        prompt = prompt_messages_from_preference(row)
        prompt_ids = chat_template_token_ids(tokenizer, prompt)
        if len(prompt_ids) >= max_length:
            excluded.append({**identity, "reason": "prompt_exceeds_sequence_cap", "prompt_tokens": len(prompt_ids)})
            continue
        chosen_text, rejected_text = preference_responses(row)
        chosen = encode_prompt_response(tokenizer, prompt, chosen_text, max_length)
        rejected = encode_prompt_response(tokenizer, prompt, rejected_text, max_length)
        eos_tokens = int(tokenizer.eos_token_id is not None)
        chosen_raw = len(tokenizer(chosen_text, add_special_tokens=False)["input_ids"]) + eos_tokens
        rejected_raw = len(tokenizer(rejected_text, add_special_tokens=False)["input_ids"]) + eos_tokens
        eligible.append({
            **identity,
            "prompt_tokens": len(prompt_ids),
            "chosen_tokens_raw": chosen_raw,
            "rejected_tokens_raw": rejected_raw,
            "chosen_tokens_used": sum(chosen[1]),
            "rejected_tokens_used": sum(rejected[1]),
            "chosen_truncated": chosen_raw > sum(chosen[1]),
            "rejected_truncated": rejected_raw > sum(rejected[1]),
            "chosen": chosen,
            "rejected": rejected,
        })
        if max_examples is not None and len(eligible) == max_examples:
            break
    if max_examples is not None and len(eligible) < max_examples:
        raise ValueError(f"Only {len(eligible)} eligible pairs; need {max_examples}")
    if not eligible:
        raise ValueError("No eligible preference pairs")
    return eligible, excluded


def selection_summary(examples: list[dict], excluded: list[dict]) -> dict:
    ids = [
        {k: row[k] for k in ("file_position", "source_index", "prompt_id", "length_stratum")}
        for row in examples
    ]
    digest = hashlib.sha256(json.dumps(ids, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "eligible_pairs": len(examples),
        "excluded_prompt_pairs": len(excluded),
        "selected_ids_sha256": digest,
        "strata": dict(Counter(str(row["length_stratum"]) for row in examples)),
        "chosen_truncated_pairs": sum(bool(row["chosen_truncated"]) for row in examples),
        "rejected_truncated_pairs": sum(bool(row["rejected_truncated"]) for row in examples),
        "max_processed_sequence_tokens": max(
            max(len(row["chosen"][0]), len(row["rejected"][0])) for row in examples
        ),
    }


class PreferenceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict]):
        return (
            pad_batch(self.tokenizer, [row["chosen"] for row in rows]),
            pad_batch(self.tokenizer, [row["rejected"] for row in rows]),
            [example_identity(row, row["file_position"]) for row in rows],
        )


def seed_for_prompt(base_seed: int, prompt_id: str) -> int:
    digest = hashlib.sha256(str(prompt_id).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)
