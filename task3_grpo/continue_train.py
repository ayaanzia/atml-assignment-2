"""GRPO continuation from the supplied midpoint; no cloud connection code."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import traceback
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, print_gpu_info, save_json, set_seed
from common.models import (
    load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters,
)
from task3_grpo.grpo import (
    group_relative_advantages, grpo_policy_loss, mask_truncated_sequences,
)
from task3_grpo.runtime import (
    changed_parameter_count, generate_one, length_bin, load_training_state,
    parameter_audit, promote_trainables, response_logps, reward_input_length,
    save_training_state, scheduled_rows, signature_for, single_prompt_collate,
    sync_time, trainable_snapshot, training_seed,
)


def prepare_grpo_continuation(config_path: str, cfg: dict | None = None) -> dict:
    cfg = load_yaml(config_path) if cfg is None else cfg
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"  # Guard; the prepared chat prompt already fits.
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    promoted = promote_trainables(policy)
    parameters = trainable_parameters(policy)
    if not parameters:
        raise RuntimeError("GRPO midpoint has no trainable policy adapter parameters")
    optimizer = AdamW(parameters, lr=float(cfg["learning_rate"]))
    reward_model, reward_tokenizer = load_reward_model(cfg)
    reward_tokenizer.truncation_side = "left"  # Keep the generated response visible.
    # Evaluation mode disables LoRA dropout but still permits gradients.
    policy.eval()
    return {
        "cfg": cfg, "tokenizer": tokenizer, "policy": policy,
        "reward_model": reward_model, "reward_tokenizer": reward_tokenizer,
        "optimizer": optimizer, "promoted_trainables": promoted,
        "model_audit": [parameter_audit(policy, "GRPO policy"),
                        parameter_audit(reward_model, "frozen reward model")],
    }


def _output_dir(cfg: dict, run_name: str, output: str | None) -> Path:
    if output:
        return repo_path(output)
    return repo_path(cfg["output"] if run_name == "standard"
                     else f"outputs/task3_grpo/{run_name}")


def _truncate_update_log(path: Path, completed: int) -> None:
    if not path.exists():
        return
    rows = read_jsonl(path)
    kept = [row for row in rows if int(row["update"]) <= completed]
    if len(kept) != len(rows):
        write_jsonl(path, kept)
        print(f"Removed {len(rows) - len(kept)} Task 3 logs after the last checkpoint", flush=True)


def _collect_rollout(bundle: dict, row: dict, cfg: dict, update_index: int) -> dict:
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    k = int(cfg["num_generations"])
    if k < 2:
        raise ValueError("GRPO requires at least two completions per prompt")
    entries = []
    generation_seconds = 0.0
    scoring_seconds = 0.0
    for generation_index in tqdm(range(k), total=k,
                                 desc=f"GRPO update {update_index + 1} generation",
                                 unit="completion", leave=False):
        seed = training_seed(int(cfg["seed"]), update_index, row["prompt_id"], generation_index)
        start = sync_time()
        generated = generate_one(policy, tokenizer, row, cfg, seed,
                                 int(cfg["max_completion_length"]))
        generation_seconds += sync_time() - start
        full_mask = generated["response_mask"].float()
        if int(full_mask.sum().item()) < 1:
            raise ValueError(f"Empty GRPO response for {row['prompt_id']} generation {generation_index}")
        start = sync_time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            old_logp, entropy_tokens = response_logps(policy, generated, entropy=True)
            with reference_mode(policy):
                ref_logp = response_logps(policy, generated)[0]
            reward = score_reward_pairs(
                bundle["reward_model"], bundle["reward_tokenizer"],
                [generated["messages"]], generated["responses"],
                max_length=int(cfg["reward_max_length"]),
            ).float()
        reward_length = reward_input_length(
            bundle["reward_tokenizer"], generated["messages"], generated["responses"][0]
        )
        scoring_seconds += sync_time() - start
        if not all(bool(torch.isfinite(value).all().item())
                   for value in (old_logp, ref_logp, entropy_tokens, reward)):
            raise FloatingPointError("Nonfinite GRPO rollout probability, entropy, or reward")
        train_mask = mask_truncated_sequences(full_mask, generated["truncated"])
        entries.append({
            "generation_index": generation_index, "seed": seed,
            "generated": generated, "full_mask": full_mask, "train_mask": train_mask,
            "old_logp": old_logp.detach(), "ref_logp": ref_logp.detach(),
            "entropy_tokens": entropy_tokens.detach(),
            "reward": float(reward[0].item()),
            "reward_input_tokens_full": reward_length,
            "reward_input_truncated": reward_length > int(cfg["reward_max_length"]),
        })
        del old_logp, ref_logp, entropy_tokens, reward
    rewards = torch.tensor([entry["reward"] for entry in entries],
                           dtype=torch.float32, device=entries[0]["old_logp"].device)
    advantages = group_relative_advantages(rewards, torch.zeros(k, dtype=torch.long,
                                                                  device=rewards.device))
    reward_std = float(rewards.std(unbiased=False).item())
    for index, entry in enumerate(entries):
        entry["advantage"] = advantages[index:index + 1].detach()
    usable = sum(int(entry["train_mask"].sum().item()) for entry in entries)
    if usable == 0:
        raise ValueError(f"Every completion was masked at the generation cap for {row['prompt_id']}")
    sampled_kl_sum = sum(float(((entry["old_logp"] - entry["ref_logp"])
                                * entry["train_mask"]).sum().item()) for entry in entries)
    entropy_sum = sum(float((entry["entropy_tokens"] * entry["train_mask"]).sum().item())
                      for entry in entries)
    return {
        "entries": entries, "reward_std": reward_std,
        "informative_group": reward_std > 1e-6,
        "usable_tokens": usable,
        "sampled_kl_token_mean": sampled_kl_sum / usable,
        "token_entropy_mean": entropy_sum / usable,
        "generation_seconds": generation_seconds,
        "scoring_seconds": scoring_seconds,
    }


def _optimize_group(bundle: dict, rollout: dict, cfg: dict, loss_type: str,
                    update_index: int, step_counts: dict[str, int]) -> dict:
    policy, optimizer, scaler = bundle["policy"], bundle["optimizer"], bundle["scaler"]
    entries = rollout["entries"]
    k = len(entries)
    total_tokens = rollout["usable_tokens"]
    optimizer.zero_grad(set_to_none=True)
    start = sync_time()
    policy_sum = kl_sum = clip_sum = ratio_sum = 0.0
    backward_passes = 0
    contributions = []
    for entry in tqdm(entries, total=k,
                      desc=f"GRPO update {update_index + 1} epoch 1/1",
                      unit="microbatch", leave=False):
        mask = entry["train_mask"]
        token_count = int(mask.sum().item())
        if token_count == 0:
            contributions.append(0.0)
            continue
        with torch.amp.autocast("cuda", dtype=torch.float16):
            new_logp = response_logps(policy, entry["generated"])[0]
            # Generation and old scoring both disabled dropout. The policy has
            # not changed since rollout collection, so these must agree.
            if not torch.allclose(new_logp.detach() * mask, entry["old_logp"] * mask,
                                  atol=1e-2, rtol=1e-3):
                difference = float(((new_logp.detach() - entry["old_logp"]).abs()
                                    * mask).max().item())
                raise RuntimeError(f"Unchanged GRPO old/new log probabilities disagree: {difference:.5g}")
            # The helper averages its policy term over its local batch and KL
            # over local tokens. This coefficient makes four microbatches
            # algebraically equal to one K-completion batch with global KL/token.
            scaled_beta = float(cfg["kl_beta"]) * k * token_count / total_tokens
            local_loss, details = grpo_policy_loss(
                new_logp.float(), entry["old_logp"], entry["advantage"], mask,
                entry["ref_logp"], float(cfg["clip_epsilon"]), scaled_beta,
                loss_type=loss_type,
                max_completion_length=int(cfg["max_completion_length"]),
            )
            loss = local_loss.float() / k
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Nonfinite GRPO loss")
        scaler.scale(loss).backward()
        backward_passes += 1
        contribution = float(details["policy_term"].item()) / k
        contributions.append(abs(contribution))
        policy_sum += contribution
        kl_sum += float(details["kl_penalty"].item()) * token_count / total_tokens
        clip_sum += float(details["clip_fraction"].item()) * token_count
        ratio_sum += float(details["ratio_mean"].item()) * token_count
        del new_logp, local_loss, loss, details
    scaler.unscale_(optimizer)
    parameters = trainable_parameters(policy)
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads or not all(bool(torch.isfinite(grad).all().item()) for grad in grads):
        raise FloatingPointError("Missing or nonfinite GRPO gradients; optimizer was not stepped")
    norm = float(torch.nn.utils.clip_grad_norm_(
        parameters, float(cfg["max_grad_norm"]), error_if_nonfinite=True).item())
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("GRPO preclip gradient norm is zero or nonfinite")
    before_steps = step_counts["optimizer"]
    scaler.step(optimizer)
    scaler.update()
    if step_counts["optimizer"] != before_steps + 1:
        raise RuntimeError("GradScaler skipped the GRPO optimizer step")
    optimizer.zero_grad(set_to_none=True)
    return {
        "policy_loss": policy_sum, "kl_penalty_token_mean": kl_sum,
        "total_loss": policy_sum + float(cfg["kl_beta"]) * kl_sum,
        "clip_fraction": clip_sum / total_tokens,
        "ratio_mean": ratio_sum / total_tokens,
        "policy_gradient_norm_preclip": norm,
        "optimizer_step_skipped": False,
        "backward_passes": backward_passes,
        "policy_objective_contribution_abs": contributions,
        "optimization_seconds": sync_time() - start,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None,
             loss_type: str = "grpo", run_name: str = "standard",
             num_workers: int = 0, resume: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Task 3 continuation requires a CUDA GPU")
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("run_name must be one simple directory name")
    if num_workers < 0 or loss_type not in {"grpo", "dr_grpo"}:
        raise ValueError("Invalid worker count or GRPO normalization")
    cfg = load_yaml(config_path)
    if updates is not None:
        cfg["updates"] = int(updates)
    k = int(cfg["num_generations"])
    if int(cfg["policy_epochs"]) != 1 or k != 4:
        raise ValueError("Released Task 3 continuation requires one policy epoch and K=4")
    if not bool(cfg["mask_truncated_completions"]):
        raise ValueError("Released Task 3 continuation masks max-length completions")
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    if len(rows) != 1200 or len({row["prompt_id"] for row in rows}) != len(rows):
        raise ValueError("Task 3 expects the fixed 1,200-row training prompt pool with unique IDs")
    out = _output_dir(cfg, run_name, output)
    results = repo_path(cfg["results_dir"]) / run_name
    manifest_path = results / "run_config.json"
    summary_path = results / "train_summary.json"
    log_path = results / "train_updates.jsonl"
    state_path = out / "latest_state.pt"
    if resume and manifest_path.exists():
        resume_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows_by_id = {row["prompt_id"]: row for row in rows}
        prompt_ids = resume_manifest.get("prompt_ids", [])
        if len(prompt_ids) != int(cfg["updates"]) or any(pid not in rows_by_id for pid in prompt_ids):
            raise ValueError("GRPO resume manifest has an invalid prompt schedule")
        selected = [rows_by_id[pid] for pid in prompt_ids]
    else:
        selected = scheduled_rows(rows, int(cfg["seed"]), int(cfg["updates"]))
    signature = signature_for(cfg, selected, loss_type)
    if summary_path.exists():
        if resume:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("signature") != signature:
                raise ValueError("Completed GRPO run has different settings")
            if not (out / "adapter_config.json").exists():
                raise FileNotFoundError(f"Completed GRPO adapter is missing from {out}")
            return summary
        raise FileExistsError(f"Completed GRPO run {run_name} exists; use --resume")
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"No GRPO run manifest to resume at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") != signature:
            raise ValueError("GRPO resume configuration, prompts, or loss type differs")
    else:
        if manifest_path.exists() or state_path.exists() or log_path.exists() or out.exists():
            raise FileExistsError(f"GRPO run {run_name} already has output; use --resume or a new name")
        results.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "signature": signature, "config": cfg, "loss_type": loss_type,
            "prompt_ids": [row["prompt_id"] for row in selected],
            "training_seeds": [[training_seed(int(cfg["seed"]), i, row["prompt_id"], j)
                                for j in range(k)] for i, row in enumerate(selected)],
            "prompt_truncation": "trim older user content; keep chat wrapper and assistant suffix",
            "reward_input_truncation": "left", "reward_input_cap": cfg["reward_max_length"],
            "planned_completion_attempts": len(selected) * k,
        }
        save_json(manifest_path, manifest)
    print(f"GRPO {run_name}: {len(selected)} updates, K={k}, workers={num_workers}, "
          f"loss_type={loss_type}, FP16 autocast", flush=True)
    torch.cuda.reset_peak_memory_stats()
    load_start = sync_time()
    bundle = prepare_grpo_continuation(config_path, cfg)
    load_seconds = sync_time() - load_start
    bundle["scaler"] = torch.amp.GradScaler("cuda", init_scale=1024.0)
    policy = bundle["policy"]
    initial_policy = trainable_snapshot(policy)
    if resume:
        if manifest.get("model_audit") != bundle["model_audit"] or \
                manifest.get("promoted_trainables") != bundle["promoted_trainables"]:
            raise ValueError("Resumed GRPO model precision, placement, or trainable promotion differs")
    else:
        manifest["model_audit"] = bundle["model_audit"]
        manifest["promoted_trainables"] = bundle["promoted_trainables"]
        save_json(manifest_path, manifest)
    counters = {
        "completed_updates": 0, "optimizer_steps": 0, "backward_passes": 0,
        "generated_completions": 0, "generated_response_tokens": 0,
        "usable_response_tokens": 0, "masked_completions": 0,
        "uninformative_groups": 0, "truncated_prompts": 0,
        "truncated_reward_inputs": 0,
    }
    cursor = 0
    if resume and state_path.exists():
        state = load_training_state(state_path, signature, policy, bundle["optimizer"], bundle["scaler"])
        counters = state["counters"]
        cursor = int(state["next_update"])
        if cursor < 0 or cursor > len(selected) or counters["completed_updates"] != cursor:
            raise ValueError("GRPO checkpoint cursor and completed-update count disagree")
        _truncate_update_log(log_path, cursor)
        print(f"Resuming {run_name} at update {cursor + 1}", flush=True)
    elif resume:
        _truncate_update_log(log_path, 0)
        print(f"No saved update yet; restarting {run_name} from midpoint", flush=True)
    step_counts = {"optimizer": 0}
    step_hook = bundle["optimizer"].register_step_post_hook(
        lambda *_: step_counts.__setitem__("optimizer", step_counts["optimizer"] + 1)
    )
    train_start = sync_time()
    try:
        loader = DataLoader(selected[cursor:], batch_size=1, shuffle=False,
                            num_workers=num_workers, collate_fn=single_prompt_collate,
                            pin_memory=True)
        outer = tqdm(loader, total=len(selected), initial=cursor,
                     desc=f"GRPO {run_name} continuation", unit="update")
        for update_index, row in enumerate(outer, start=cursor):
            rollout = _collect_rollout(bundle, row, cfg, update_index)
            before = trainable_snapshot(policy)
            optimization = _optimize_group(bundle, rollout, cfg, loss_type, update_index, step_counts)
            changed = changed_parameter_count(policy, before)
            if changed == 0:
                raise RuntimeError("GRPO optimizer step did not change any trainable policy parameter")
            completion_rows = []
            for entry, contribution in zip(rollout["entries"],
                                           optimization["policy_objective_contribution_abs"]):
                generated = entry["generated"]
                length = int(generated["response_lengths"][0])
                completion_rows.append({
                    "generation_index": entry["generation_index"], "seed": entry["seed"],
                    "response": generated["responses"][0], "response_tokens": length,
                    "length_bin": length_bin(length),
                    "usable_response_tokens": int(entry["train_mask"].sum().item()),
                    "terminated_with_eos": bool(generated["terminated_with_eos"][0]),
                    "truncated_at_generation_cap": bool(generated["truncated"][0]),
                    "reward": entry["reward"], "advantage": float(entry["advantage"].item()),
                    "policy_objective_contribution_abs": contribution,
                    "reward_input_tokens_full": entry["reward_input_tokens_full"],
                    "reward_input_truncated": entry["reward_input_truncated"],
                })
            first = rollout["entries"][0]["generated"]
            record = {
                "update": update_index + 1, "prompt_id": row["prompt_id"],
                "source_index": row["source_index"],
                "prompt_tokens_full": first["prompt_tokens_full"],
                "prompt_tokens_used": first["prompt_tokens_used"],
                "prompt_truncated": first["prompt_truncated"],
                "loss_type": loss_type, "num_generations": k,
                "completions": completion_rows,
                "mean_raw_reward": statistics.fmean(item["reward"] for item in completion_rows),
                "within_group_reward_std": rollout["reward_std"],
                "informative_group": rollout["informative_group"],
                "mean_response_tokens": statistics.fmean(item["response_tokens"] for item in completion_rows),
                "std_response_tokens": statistics.pstdev(item["response_tokens"] for item in completion_rows),
                "usable_response_tokens": rollout["usable_tokens"],
                "sampled_kl_token_mean": rollout["sampled_kl_token_mean"],
                "token_entropy_mean": rollout["token_entropy_mean"],
                "trainable_parameters_changed": changed,
                "generation_seconds": round(rollout["generation_seconds"], 3),
                "scoring_seconds": round(rollout["scoring_seconds"], 3),
                "peak_cuda_allocated_gib_so_far": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                **{key: value for key, value in optimization.items()
                   if key != "policy_objective_contribution_abs"},
            }
            if not all(math.isfinite(record[key]) for key in
                       ("mean_raw_reward", "within_group_reward_std", "sampled_kl_token_mean",
                        "token_entropy_mean", "policy_loss", "total_loss",
                        "policy_gradient_norm_preclip")):
                raise FloatingPointError("Nonfinite GRPO update metric")
            append_jsonl(log_path, record)
            counters["completed_updates"] += 1
            counters["optimizer_steps"] += 1
            counters["backward_passes"] += optimization["backward_passes"]
            counters["generated_completions"] += k
            counters["generated_response_tokens"] += sum(item["response_tokens"] for item in completion_rows)
            counters["usable_response_tokens"] += rollout["usable_tokens"]
            counters["masked_completions"] += sum(item["truncated_at_generation_cap"] for item in completion_rows)
            counters["uninformative_groups"] += int(not rollout["informative_group"])
            counters["truncated_prompts"] += int(first["prompt_truncated"])
            counters["truncated_reward_inputs"] += sum(item["reward_input_truncated"] for item in completion_rows)
            save_training_state(state_path, signature, update_index + 1, counters,
                                policy, bundle["optimizer"], bundle["scaler"])
            outer.set_postfix(reward=f"{record['mean_raw_reward']:.3f}",
                              informative=int(rollout["informative_group"]))
            # Loop variables otherwise keep the first and final completion's
            # GPU tensors alive during the next rollout.
            del rollout, before, entry, generated, first, completion_rows
        if counters["completed_updates"] != len(selected):
            raise RuntimeError("GRPO stopped before all scheduled updates")
        if counters["generated_completions"] != len(selected) * k or \
                counters["optimizer_steps"] != len(selected):
            raise RuntimeError("GRPO completion or optimizer-step count differs from the released update budget")
        if changed_parameter_count(policy, initial_policy) == 0:
            raise RuntimeError("Final GRPO adapter matches its starting trainable weights")
        policy.save_pretrained(out)
        logs = read_jsonl(log_path)
        summary = {
            "run_name": run_name, "loss_type": loss_type,
            "signature": signature, "configuration": cfg,
            "prompt_ids": [row["prompt_id"] for row in selected],
            "counters": counters,
            "mean_raw_reward": statistics.fmean(row["mean_raw_reward"] for row in logs),
            "mean_within_group_reward_std": statistics.fmean(row["within_group_reward_std"] for row in logs),
            "uninformative_group_fraction": counters["uninformative_groups"] / len(logs),
            "relative_reward_signal_inconclusive": counters["uninformative_groups"] == len(logs),
            "mean_sampled_kl_token": sum(row["sampled_kl_token_mean"] * row["usable_response_tokens"]
                                         for row in logs) / counters["usable_response_tokens"],
            "mean_token_entropy": sum(row["token_entropy_mean"] * row["usable_response_tokens"]
                                      for row in logs) / counters["usable_response_tokens"],
            "mean_response_tokens": counters["generated_response_tokens"] / counters["generated_completions"],
            "logged_generation_seconds": round(sum(row["generation_seconds"] for row in logs), 3),
            "logged_scoring_seconds": round(sum(row["scoring_seconds"] for row in logs), 3),
            "logged_optimization_seconds": round(sum(row["optimization_seconds"] for row in logs), 3),
            "model_loading_seconds": round(load_seconds, 3),
            "continuation_seconds": round(sync_time() - train_start, 3),
            "timing_note": "loading/continuation cover this invocation; logged phase seconds include prior resumed updates",
            "model_audit": bundle["model_audit"],
            "autocast_dtype": "torch.float16",
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "policy_adapter": str(out), "rolling_state": str(state_path),
        }
        from task3_grpo.plots import plot_training_trajectories
        plot_training_trajectories(cfg, [run_name], results / "training_trajectories.png")
        save_json(summary_path, summary)
        (results / "last_failure.json").unlink(missing_ok=True)
        print(f"Completed GRPO {run_name}: {counters['completed_updates']} updates, "
              f"{counters['optimizer_steps']} optimizer steps", flush=True)
        return summary
    except Exception as exc:
        save_json(results / "last_failure.json", {
            "run_name": run_name, "error_type": type(exc).__name__,
            "error_message": str(exc), "traceback": traceback.format_exc(),
            "last_completed_update": counters["completed_updates"],
            "resumable_state": str(state_path) if state_path.exists() else None,
        })
        raise
    finally:
        step_hook.remove()
        del bundle, policy
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--output")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--loss-type", choices=("grpo", "dr_grpo"), default="grpo")
    parser.add_argument("--run-name", default="standard")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type,
             args.run_name, args.num_workers, args.resume)


if __name__ == "__main__":
    main()
