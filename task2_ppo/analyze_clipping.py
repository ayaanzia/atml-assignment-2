"""Fixed cached PPO clipping geometry and matched short continuation forks."""

from __future__ import annotations

import argparse
import csv
import gc
import statistics

import torch
from tqdm.auto import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import print_gpu_info, save_json, set_seed
from common.models import load_policy, load_tokenizer
from task2_ppo.ppo import (
    affected_token_fraction, compute_gae, ppo_policy_loss, shaped_rewards,
)
from task2_ppo.runtime import response_logps
from task2_ppo.workflow import run_condition, verify_matched_conditions


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)
    required = {"source_index", "prompt_id", "response", "response_tokens", "old_logprobs",
                "ref_logprobs", "values", "effective_terminal_reward", "terminated_with_eos"}
    if any(not required.issubset(row) for row in normalized):
        raise ValueError(f"Unexpected PPO cache schema; need {sorted(required)} in every row")
    for row in normalized:
        count = int(row["response_tokens"])
        if count < 1 or any(int(row[key].numel()) != count for key in
                               ("old_logprobs", "ref_logprobs", "values")):
            raise ValueError(f"Cache vector length mismatch for source index {row['source_index']}")
    return normalized


def _reconstruct(tokenizer, prompt: dict, cache_row: dict, cap: int,
                 device: str = "cuda") -> dict:
    # Staff cached rollouts were created with the released helper's original
    # right-side truncation. New Task 2 rollouts shorten user content while
    # preserving the complete chat wrapper and assistant suffix.
    rendered = tokenizer.apply_chat_template(
        prompt_messages(prompt), tokenize=False, add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, truncation=True, max_length=cap)["input_ids"]
    response = tokenizer(cache_row["response"], add_special_tokens=False)["input_ids"]
    if cache_row["terminated_with_eos"]:
        response.append(tokenizer.eos_token_id)
    if len(response) != int(cache_row["response_tokens"]):
        raise ValueError(f"Cached response {cache_row['source_index']} cannot be tokenized to its recorded length")
    all_ids = torch.tensor([encoded + response], dtype=torch.long, device=device)
    return {
        "sequences": all_ids, "attention_mask": torch.ones_like(all_ids),
        "prompt_width": len(encoded),
        "response_ids": torch.tensor([response], dtype=torch.long, device=device),
        "response_mask": torch.ones((1, len(response)), dtype=torch.float32, device=device),
        "full_prompt_tokens": len(tokenizer(rendered, truncation=False)["input_ids"]),
    }


def analyze_cached_batch(config_path: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Cached PPO geometry requires a CUDA GPU to score the midpoint")
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    cache = load_cached_rollouts(cfg["cached_rollouts"])
    heldout = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    by_index = {row["source_index"]: row for row in heldout}
    if len(by_index) != len(heldout):
        raise ValueError("Held-out prompt source indices are not unique")
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "right"
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    policy.eval()
    rows, advantage_parts, new_parts, old_parts = [], [], [], []
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            for cache_row in tqdm(cache, desc="PPO cached clipping geometry", unit="rollout"):
                prompt = by_index.get(cache_row["source_index"])
                if prompt is None or prompt["prompt_id"] != cache_row["prompt_id"]:
                    raise ValueError("Cached rollout does not match the fixed held-out prompt")
                generated = _reconstruct(tokenizer, prompt, cache_row, int(cfg["max_prompt_length"]))
                new = response_logps(policy, generated)[0].float()
                old = cache_row["old_logprobs"].reshape(1, -1).float().cuda()
                ref = cache_row["ref_logprobs"].reshape(1, -1).float().cuda()
                values = cache_row["values"].reshape(1, -1).float().cuda()
                mask = generated["response_mask"]
                terminal = torch.tensor([cache_row["effective_terminal_reward"]],
                                        dtype=torch.float32, device="cuda")
                rewards = shaped_rewards(terminal, old, ref, mask, float(cfg["kl_beta"]))
                advantage, returns = compute_gae(
                    rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"])
                )
                if not all(bool(torch.isfinite(x).all().item()) for x in (new, old, ref, values, advantage, returns)):
                    raise FloatingPointError(f"Nonfinite cached rollout values for {cache_row['source_index']}")
                discrepancy = (new - old).abs()
                rows.append({
                    "source_index": cache_row["source_index"], "prompt_id": cache_row["prompt_id"],
                    "response_tokens": int(cache_row["response_tokens"]),
                    "prompt_tokens_full": generated["full_prompt_tokens"],
                    "prompt_tokens_used": int(generated["prompt_width"]),
                    "prompt_truncated": generated["full_prompt_tokens"] > int(cfg["max_prompt_length"]),
                    "mean_abs_midpoint_vs_cached_old_logp": float(discrepancy.mean().item()),
                    "max_abs_midpoint_vs_cached_old_logp": float(discrepancy.max().item()),
                    "mean_return": float(returns.mean().item()),
                })
                advantage_parts.append(advantage.flatten().cpu())
                new_parts.append(new.flatten().cpu())
                old_parts.append(old.flatten().cpu())
                del generated, new, old, ref, values, mask, rewards, advantage, returns
        advantage = torch.cat(advantage_parts).float()
        new = torch.cat(new_parts).float()
        old = torch.cat(old_parts).float()
        std = advantage.std(unbiased=False)
        normalized = (advantage - advantage.mean()) / std.clamp_min(1e-6)
        mask = torch.ones_like(old)
        table = []
        for epsilon in cfg["clip_values"]:
            epsilon = float(epsilon)
            loss, ratio, clip_fraction = ppo_policy_loss(new, old, normalized, mask, epsilon)
            affected = affected_token_fraction(ratio, normalized, mask, epsilon)
            table.append({
                "clip_epsilon": epsilon, "cached_rollouts": len(rows),
                "valid_tokens": int(mask.sum().item()),
                "clipped_surrogate_loss": float(loss.item()),
                "out_of_range_ratio_fraction": float(clip_fraction.item()),
                "affected_token_fraction": float(affected.item()),
            })
        summary = {
            "cache_path": str(repo_path(cfg["cached_rollouts"])),
            "prompt_reconstruction": "original helper right truncation; original token IDs absent",
            "advantage_normalization": "all valid tokens in the fixed 32-rollout batch",
            "mean_abs_midpoint_vs_cached_old_logp": statistics.fmean(
                row["mean_abs_midpoint_vs_cached_old_logp"] for row in rows),
            "prompt_truncated_count": sum(row["prompt_truncated"] for row in rows),
            "rows": rows, "clip_geometry": table,
        }
        root = repo_path(cfg["results_dir"])
        root.mkdir(parents=True, exist_ok=True)
        save_json(root / "cached_clipping.json", summary)
        with (root / "cached_clipping.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        return summary
    finally:
        del policy
        gc.collect()
        torch.cuda.empty_cache()


def run_clipping_study(config_path: str, num_workers: int = 0,
                       resume: bool = False, cache_only: bool = False) -> dict:
    cfg = load_yaml(config_path)
    cached = analyze_cached_batch(config_path)
    if cache_only:
        return cached
    conditions = []
    for epsilon in cfg["clip_values"]:
        epsilon = float(epsilon)
        name = f"clip_{epsilon:.2f}".replace(".", "p")
        manifest, train, evaluation = run_condition(
            config_path, cfg, name, int(cfg["fork_updates"]), epsilon,
            float(cfg["kl_beta"]), num_workers, resume,
        )
        logs = read_jsonl(repo_path(cfg["results_dir"]) / name / "train_updates.jsonl")
        losses = [statistics.fmean(p["policy_loss"] for p in row["ppo_passes"]) for row in logs]
        gradient_norms = [statistics.fmean(p["policy_gradient_norm_preclip"] for p in row["ppo_passes"])
                          for row in logs]
        metrics = evaluation["metrics"]
        conditions.append({
            "run_name": name, "clip_epsilon": epsilon,
            "training_prompt_ids": manifest["prompt_ids"],
            "completed_updates": train["counters"]["completed_updates"],
            "actual_generated_response_tokens": train["counters"]["generated_response_tokens"],
            "policy_loss_std_across_updates": statistics.pstdev(losses),
            "policy_gradient_norm_std_across_updates": statistics.pstdev(gradient_norms),
            "optimizer_steps_skipped": sum(p["optimizer_step_skipped"] for row in logs for p in row["ppo_passes"]),
            "heldout_reward": metrics["mean_raw_reward"],
            "heldout_sampled_kl_token": metrics["sampled_kl_token_mean"],
            "heldout_response_tokens": metrics["mean_response_tokens"],
            "heldout_entropy": metrics["token_entropy_mean"],
        })
    matched = verify_matched_conditions(cfg, [item["run_name"] for item in conditions])
    result = {"study": "Task 2 PPO clipping", "cached_geometry": cached["clip_geometry"],
              "conditions": conditions, "matched": matched,
              "stability_statistic": "population SD of mean two-pass policy loss across eight updates"}
    root = repo_path(cfg["results_dir"])
    save_json(root / "clipping_summary.json", result)
    fields = [key for key in conditions[0] if key != "training_prompt_ids"]
    with (root / "clipping_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in conditions])
    from task2_ppo.plots import plot_condition_comparison
    plot_condition_comparison(conditions, "clip_epsilon", root / "clipping_comparison.png")
    print(f"PPO clipping comparison: {root / 'clipping_summary.json'}", flush=True)
    return result


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    run_clipping_study(args.config, args.num_workers, args.resume, args.cache_only)


if __name__ == "__main__":
    main()
