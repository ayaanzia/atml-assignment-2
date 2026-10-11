"""Continue PPO from the supplied midpoint with fixed two-pass rollouts."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, print_gpu_info, save_json, set_seed
from common.metrics import masked_mean, sampled_kl
from common.models import (
    load_policy, load_reward_model, load_tokenizer, load_value_model,
    reference_mode, trainable_parameters, value_parameter_groups,
)
from task2_ppo.ppo import (
    affected_token_fraction, compute_gae, normalize_advantages, ppo_policy_loss,
    shaped_rewards, value_mse_loss,
)
from task2_ppo.runtime import (
    changed_parameter_count, generate_one, load_training_state, parameter_audit,
    promote_trainables, response_logps, response_values, save_training_state,
    scheduled_rows, signature_for, single_prompt_collate, stable_seed,
    trainable_snapshot, reward_input_length,
)


def prepare_ppo_continuation(config_path: str, cfg: dict | None = None) -> dict:
    cfg = load_yaml(config_path) if cfg is None else cfg
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"  # Guard against accidental loss of the assistant suffix.
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    critic = load_value_model(
        cfg, cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "lora_head"),
    )
    policy_promoted = promote_trainables(policy)
    critic_promoted = promote_trainables(critic)
    policy_params = trainable_parameters(policy)
    critic_groups = value_parameter_groups(
        critic, float(cfg["value_lora_learning_rate"]), float(cfg["value_head_learning_rate"])
    )
    if not policy_params:
        raise RuntimeError("Midpoint policy has no trainable adapter parameters")
    policy_optimizer = AdamW(policy_params, lr=float(cfg["policy_learning_rate"]))
    critic_optimizer = AdamW(critic_groups, weight_decay=0.0)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    reward_tokenizer.truncation_side = "left"  # Keep the response within the fixed reward cap.
    # Evaluation mode disables dropout, while autograd remains enabled for PPO updates.
    policy.eval()
    critic.eval()
    return {
        "cfg": cfg, "tokenizer": tokenizer, "policy": policy, "critic": critic,
        "reward_model": reward_model, "reward_tokenizer": reward_tokenizer,
        "policy_optimizer": policy_optimizer, "critic_optimizer": critic_optimizer,
        "promoted_trainables": {"policy": policy_promoted, "critic": critic_promoted},
        "model_audit": [parameter_audit(policy, "policy"), parameter_audit(critic, "critic"),
                        parameter_audit(reward_model, "reward_model")],
    }


def _output_dir(cfg: dict, run_name: str, output: str | None) -> Path:
    if output:
        return repo_path(output)
    return repo_path(cfg["output"] if run_name == "standard" else f"outputs/task2_ppo/{run_name}")


def _truncate_update_log(path: Path, completed: int) -> None:
    if not path.exists():
        return
    rows = read_jsonl(path)
    kept = [row for row in rows if int(row["update"]) <= completed]
    if len(kept) != len(rows):
        write_jsonl(path, kept)
        print(f"Removed {len(rows) - len(kept)} logs after the last saved PPO update", flush=True)


def _sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _rollout(bundle: dict, row: dict, cfg: dict, update_index: int) -> dict:
    policy, critic = bundle["policy"], bundle["critic"]
    tokenizer = bundle["tokenizer"]
    seed = stable_seed(int(cfg["seed"]), f"ppo-train-{update_index}-{row['prompt_id']}")
    generation_start = _sync()
    generated = generate_one(policy, tokenizer, row, cfg, seed, int(cfg["max_response_length"]))
    generation_seconds = _sync() - generation_start
    mask = generated["response_mask"].float()
    usable = int(mask.sum().item())
    if usable < 1:
        raise ValueError(f"Rollout {row['prompt_id']} has no usable response tokens")
    score_start = _sync()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        old_logp, entropy_tokens = response_logps(policy, generated, entropy=True)
        with reference_mode(policy):
            ref_logp = response_logps(policy, generated)[0]
        old_values = response_values(critic, generated)
        raw_reward = score_reward_pairs(
            bundle["reward_model"], bundle["reward_tokenizer"],
            [generated["messages"]], generated["responses"],
            max_length=int(cfg["reward_max_length"]),
        ).float()
    effective_reward = raw_reward - (
        0.0 if generated["terminated_with_eos"][0] else float(cfg["missing_eos_penalty"])
    )
    reward_tokens = reward_input_length(
        bundle["reward_tokenizer"], generated["messages"], generated["responses"][0]
    )
    rewards = shaped_rewards(effective_reward, old_logp, ref_logp, mask, float(cfg["kl_beta"]))
    advantages, returns = compute_gae(
        rewards, old_values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"])
    )
    normalized = normalize_advantages(advantages, mask)
    if not all(bool(torch.isfinite(value).all().item()) for value in
               (old_logp, ref_logp, old_values, raw_reward, rewards, advantages, returns, normalized, entropy_tokens)):
        raise FloatingPointError(f"Nonfinite PPO rollout statistics for {row['prompt_id']}")
    scoring_seconds = _sync() - score_start
    return {
        "generated": generated, "mask": mask,
        "old_logp": old_logp.detach(), "ref_logp": ref_logp.detach(),
        "old_values": old_values.detach(), "advantages": normalized.detach(),
        "returns": returns.detach(), "raw_reward": raw_reward.detach(),
        "effective_reward": effective_reward.detach(),
        "sampled_kl": float(sampled_kl(old_logp, ref_logp, mask).item()),
        "entropy": float(masked_mean(entropy_tokens, mask).item()),
        "usable_tokens": usable, "generation_seconds": generation_seconds,
        "scoring_seconds": scoring_seconds,
        "reward_input_tokens_full": reward_tokens,
        "reward_input_truncated": reward_tokens > int(cfg["reward_max_length"]),
    }


def _optimize_pass(bundle: dict, rollout: dict, cfg: dict, epoch: int,
                   step_counts: dict[str, int]) -> dict:
    policy, critic = bundle["policy"], bundle["critic"]
    policy_optimizer, critic_optimizer = bundle["policy_optimizer"], bundle["critic_optimizer"]
    scaler = bundle["scaler"]
    generated, mask = rollout["generated"], rollout["mask"]
    policy_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    start = _sync()
    policy_params = trainable_parameters(policy)
    critic_params = trainable_parameters(critic)
    max_overflow_retries = 10
    for retry in range(max_overflow_retries + 1):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            new_logp = response_logps(policy, generated)[0]
            if epoch == 0 and not torch.allclose(
                new_logp.detach() * mask, rollout["old_logp"] * mask,
                atol=1e-2, rtol=1e-3,
            ):
                max_diff = float(((new_logp.detach() - rollout["old_logp"]).abs() * mask).max().item())
                raise RuntimeError(
                    f"Unchanged PPO policy old/new log probabilities disagree: max {max_diff:.5g}"
                )
            policy_loss, ratio, clip_fraction = ppo_policy_loss(
                new_logp.float(), rollout["old_logp"], rollout["advantages"], mask,
                float(cfg["clip_epsilon"]),
            )
            critic_values = response_values(critic, generated)
            value_loss = value_mse_loss(critic_values.float(), rollout["returns"], mask)
            total_loss = policy_loss.float() + float(cfg["value_coef"]) * value_loss.float()
        if not bool(torch.isfinite(total_loss).item()):
            raise FloatingPointError("Nonfinite PPO policy/value loss")
        affected = affected_token_fraction(ratio, rollout["advantages"], mask, float(cfg["clip_epsilon"]))
        scaler.scale(total_loss).backward()
        scaler.unscale_(policy_optimizer)
        scaler.unscale_(critic_optimizer)
        policy_grads = [p.grad for p in policy_params if p.grad is not None]
        critic_grads = [p.grad for p in critic_params if p.grad is not None]
        if not policy_grads or not critic_grads:
            raise RuntimeError("PPO backward did not produce both policy and critic gradients")
        bad_policy = [
            name for name, param in policy.named_parameters()
            if param.requires_grad and param.grad is not None
            and not bool(torch.isfinite(param.grad).all().item())
        ]
        bad_critic = [
            name for name, param in critic.named_parameters()
            if param.requires_grad and param.grad is not None
            and not bool(torch.isfinite(param.grad).all().item())
        ]
        if not bad_policy and not bad_critic:
            break

        scale = float(scaler.get_scale())
        if retry == max_overflow_retries or scale <= 1.0:
            raise FloatingPointError(
                "Nonfinite PPO gradients after loss-scale retries; "
                f"scale={scale:g}, policy={bad_policy[:3]}, critic={bad_critic[:3]}, "
                f"policy_loss={policy_loss.detach().item():.6g}, "
                f"value_loss={value_loss.detach().item():.6g}"
            )
        next_scale = max(1.0, scale / 2.0)
        print(
            f"Nonfinite PPO gradients; retrying this same pass at loss scale {next_scale:g} "
            f"(policy={bad_policy[:2]}, critic={bad_critic[:2]})",
            flush=True,
        )
        policy_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        scaler.update(new_scale=next_scale)
        del (new_logp, policy_loss, ratio, clip_fraction, critic_values, value_loss,
             total_loss, affected, policy_grads, critic_grads, bad_policy, bad_critic)
    else:
        raise FloatingPointError("PPO gradient retry loop ended without a valid pass")
    policy_norm = float(torch.nn.utils.clip_grad_norm_(
        policy_params, float(cfg["max_grad_norm"]), error_if_nonfinite=True).item())
    critic_norm = float(torch.nn.utils.clip_grad_norm_(
        critic_params, float(cfg["max_grad_norm"]), error_if_nonfinite=True).item())
    if policy_norm == 0 or critic_norm == 0:
        raise RuntimeError("Zero policy or critic gradient norm")
    before = step_counts.copy()
    scaler.step(policy_optimizer)
    scaler.step(critic_optimizer)
    scaler.update()
    if step_counts["policy"] != before["policy"] + 1 or step_counts["critic"] != before["critic"] + 1:
        raise RuntimeError("GradScaler skipped a policy or critic optimizer step")
    policy_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    seconds = _sync() - start
    return {
        "ppo_epoch": epoch + 1, "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "clip_fraction": float(clip_fraction.item()),
        "affected_token_fraction": float(affected.item()),
        "ratio_mean": float(masked_mean(ratio, mask).item()),
        "policy_gradient_norm_preclip": policy_norm,
        "critic_gradient_norm_preclip": critic_norm,
        "grad_scaler_scale": float(scaler.get_scale()),
        "grad_scaler_retries": retry,
        "optimizer_step_skipped": False, "seconds": round(seconds, 3),
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None,
            clip_epsilon: float | None = None, kl_beta: float | None = None,
            run_name: str = "standard", num_workers: int = 0, resume: bool = False) -> dict:
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("run_name must be a simple directory name")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 2 PPO continuation requires a CUDA GPU")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    cfg = load_yaml(config_path)
    cfg["updates"] = int(cfg["updates"] if updates is None else updates)
    cfg["clip_epsilon"] = float(cfg["clip_epsilon"] if clip_epsilon is None else clip_epsilon)
    cfg["kl_beta"] = float(cfg["kl_beta"] if kl_beta is None else kl_beta)
    if int(cfg["prompts_per_update"]) != 1 or int(cfg["ppo_epochs"]) != 2:
        raise ValueError("Task 2 implementation requires released one-prompt, two-epoch PPO setting")
    if str(cfg["dtype"]).lower() not in {"float16", "fp16", "half"}:
        raise ValueError("Task 2 main run requires released FP16 model storage")
    set_seed(int(cfg["seed"]))
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    selected = scheduled_rows(rows, int(cfg["seed"]), int(cfg["updates"]))
    signature = signature_for(cfg, selected)
    out = _output_dir(cfg, run_name, output)
    results = repo_path(cfg["results_dir"]) / run_name
    state_path = out / "latest_state.pt"
    manifest_path = results / "run_config.json"
    log_path = results / "train_updates.jsonl"
    final_adapter = out / "adapter_config.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"No prior run manifest at {manifest_path}")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Resume configuration or prompt schedule differs from original run")
    elif state_path.exists() or manifest_path.exists() or final_adapter.exists() or log_path.exists():
        raise FileExistsError(f"Task 2 run {run_name} already has output; use --resume")
    out.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if not resume:
        save_json(manifest_path, {
            "run_name": run_name, "signature": signature, "config": cfg,
            "num_workers": num_workers, "prompt_ids": [r["prompt_id"] for r in selected],
            "source_indices": [r["source_index"] for r in selected],
            "prompt_truncation": "trim older user content; preserve chat wrapper and assistant suffix",
            "reward_input_truncation_side": "left: retain response",
            "precision": "FP16 policy/critic/autocast; trainable dtype audited after loading",
            "two_passes_same_rollout": True,
        })
    print(f"PPO {run_name}: {len(selected)} scheduled updates, 2 passes per rollout, "
          f"workers={num_workers}, epsilon={cfg['clip_epsilon']}, KL beta={cfg['kl_beta']}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    load_start = _sync()
    bundle = prepare_ppo_continuation(config_path, cfg)
    load_seconds = _sync() - load_start
    bundle["scaler"] = torch.amp.GradScaler("cuda", init_scale=1024.0)
    policy, critic = bundle["policy"], bundle["critic"]
    initial_policy = trainable_snapshot(policy)
    initial_critic = trainable_snapshot(critic)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if resume:
        if manifest.get("model_audit") != bundle["model_audit"] or \
                manifest.get("promoted_trainables") != bundle["promoted_trainables"]:
            raise ValueError("Resumed model dtypes, devices, or trainable promotions differ")
    else:
        manifest["model_audit"] = bundle["model_audit"]
        manifest["promoted_trainables"] = bundle["promoted_trainables"]
        save_json(manifest_path, manifest)
    counters = {"completed_updates": 0, "policy_optimizer_steps": 0,
                "critic_optimizer_steps": 0, "backward_passes": 0,
                "generated_response_tokens": 0, "truncated_prompts": 0,
                "truncated_reward_inputs": 0}
    cursor = 0
    if resume and state_path.exists():
        state = load_training_state(
            state_path, signature, policy, critic,
            bundle["policy_optimizer"], bundle["critic_optimizer"], bundle["scaler"]
        )
        counters = state["counters"]
        cursor = int(state["next_update"])
        _truncate_update_log(log_path, cursor)
        print(f"Resuming {run_name} at update {cursor + 1}", flush=True)
    elif resume:
        _truncate_update_log(log_path, 0)
        print(f"No saved update yet; restarting {run_name} from midpoint", flush=True)
    step_counts = {"policy": 0, "critic": 0}
    policy_hook = bundle["policy_optimizer"].register_step_post_hook(
        lambda *_: step_counts.__setitem__("policy", step_counts["policy"] + 1))
    critic_hook = bundle["critic_optimizer"].register_step_post_hook(
        lambda *_: step_counts.__setitem__("critic", step_counts["critic"] + 1))
    train_start = _sync()
    try:
        loader = DataLoader(
            selected[cursor:], batch_size=1, shuffle=False, num_workers=num_workers,
            collate_fn=single_prompt_collate, pin_memory=True,
        )
        outer = tqdm(loader, total=len(selected), initial=cursor,
                     desc=f"PPO {run_name} continuation", unit="update")
        for update_index, row in enumerate(outer, start=cursor):
            rollout = _rollout(bundle, row, cfg, update_index)
            policy_before = trainable_snapshot(policy)
            critic_before = trainable_snapshot(critic)
            passes = []
            for epoch in range(int(cfg["ppo_epochs"])):
                inner = tqdm(range(1), total=1,
                             desc=f"PPO update {update_index + 1} epoch {epoch + 1}/2",
                             unit="batch", leave=False)
                for _batch_iteration in inner:
                    passes.append(_optimize_pass(bundle, rollout, cfg, epoch, step_counts))
            policy_changed = changed_parameter_count(policy, policy_before)
            critic_changed = changed_parameter_count(critic, critic_before)
            if policy_changed == 0 or critic_changed == 0:
                raise RuntimeError("PPO update did not change both policy and critic trainable weights")
            generated = rollout["generated"]
            record = {
                "update": update_index + 1, "prompt_id": row["prompt_id"],
                "source_index": row["source_index"], "seed": generated["seed"],
                "prompt_tokens_full": generated["prompt_tokens_full"],
                "prompt_tokens_used": generated["prompt_tokens_used"],
                "prompt_truncated": generated["prompt_truncated"],
                "response": generated["responses"][0],
                "response_tokens": int(generated["response_lengths"][0]),
                "usable_response_tokens": rollout["usable_tokens"],
                "terminated_with_eos": bool(generated["terminated_with_eos"][0]),
                "truncated_at_generation_cap": bool(generated["truncated"][0]),
                "raw_reward": float(rollout["raw_reward"][0].item()),
                "effective_reward": float(rollout["effective_reward"][0].item()),
                "reward_input_tokens_full": rollout["reward_input_tokens_full"],
                "reward_input_truncated": rollout["reward_input_truncated"],
                "reward_input_truncation_side": "left",
                "sampled_kl_token_mean": rollout["sampled_kl"],
                "token_entropy": rollout["entropy"],
                "mean_old_critic_value": float(masked_mean(rollout["old_values"], rollout["mask"]).item()),
                "mean_fixed_return": float(masked_mean(rollout["returns"], rollout["mask"]).item()),
                "critic_value_minus_return": float(masked_mean(
                    rollout["old_values"] - rollout["returns"], rollout["mask"]).item()),
                "policy_trainable_parameters_changed": policy_changed,
                "critic_trainable_parameters_changed": critic_changed,
                "ppo_passes": passes, "backward_passes": len(passes),
                "policy_optimizer_steps": len(passes), "critic_optimizer_steps": len(passes),
                "generation_seconds": round(rollout["generation_seconds"], 3),
                "scoring_seconds": round(rollout["scoring_seconds"], 3),
                "optimization_seconds": round(sum(item["seconds"] for item in passes), 3),
                "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            }
            if not all(math.isfinite(value) for value in (
                record["raw_reward"], record["effective_reward"],
                record["sampled_kl_token_mean"], record["token_entropy"]
            )):
                raise FloatingPointError("Nonfinite PPO update metric")
            append_jsonl(log_path, record)
            counters["completed_updates"] += 1
            counters["policy_optimizer_steps"] += len(passes)
            counters["critic_optimizer_steps"] += len(passes)
            counters["backward_passes"] += len(passes)
            counters["generated_response_tokens"] += record["response_tokens"]
            counters["truncated_prompts"] += int(record["prompt_truncated"])
            counters["truncated_reward_inputs"] += int(record["reward_input_truncated"])
            save_training_state(
                state_path, signature, update_index + 1, counters, policy, critic,
                bundle["policy_optimizer"], bundle["critic_optimizer"], bundle["scaler"]
            )
            outer.set_postfix(reward=f"{record['effective_reward']:.3f}",
                              kl=f"{record['sampled_kl_token_mean']:.3f}")
            del rollout, generated, policy_before, critic_before
        if counters["completed_updates"] != len(selected):
            raise RuntimeError("PPO stopped before all scheduled updates")
        if changed_parameter_count(policy, initial_policy) == 0 or \
                changed_parameter_count(critic, initial_critic) == 0:
            raise RuntimeError("Final PPO policy or critic matches its starting trainable weights")
        policy.save_pretrained(out)
        critic.save_pretrained(out / "critic_adapter")
        torch.cuda.synchronize()
        logs = read_jsonl(log_path)
        summary = {
            "run_name": run_name, "configuration": cfg, "signature": signature,
            "prompt_ids": [r["prompt_id"] for r in selected], "counters": counters,
            "mean_raw_reward": sum(r["raw_reward"] for r in logs) / len(logs),
            "mean_effective_reward": sum(r["effective_reward"] for r in logs) / len(logs),
            "mean_sampled_kl_token": sum(
                r["sampled_kl_token_mean"] * r["usable_response_tokens"] for r in logs
            ) / sum(r["usable_response_tokens"] for r in logs),
            "mean_token_entropy": sum(
                r["token_entropy"] * r["usable_response_tokens"] for r in logs
            ) / sum(r["usable_response_tokens"] for r in logs),
            "sampled_kl_entropy_averaging": "all usable response tokens across updates",
            "mean_response_tokens": sum(r["response_tokens"] for r in logs) / len(logs),
            "logged_generation_seconds": round(sum(r["generation_seconds"] for r in logs), 3),
            "logged_scoring_seconds": round(sum(r["scoring_seconds"] for r in logs), 3),
            "logged_optimization_seconds": round(sum(r["optimization_seconds"] for r in logs), 3),
            "model_loading_seconds": round(load_seconds, 3),
            "continuation_seconds": round(_sync() - train_start, 3),
            "invocation_timing_note": "model loading and continuation wall time cover this invocation; logged stage seconds sum all completed updates, including before resume",
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "policy_adapter": str(out), "critic_adapter": str(out / "critic_adapter"),
            "rolling_state": str(state_path),
        }
        from task2_ppo.plots import plot_training_trajectories
        plot_training_trajectories(cfg, [run_name], results / "training_trajectories.png")
        save_json(results / "train_summary.json", summary)
        (results / "last_failure.json").unlink(missing_ok=True)
        print(f"Completed PPO {run_name}: {counters['completed_updates']} updates, "
              f"{counters['policy_optimizer_steps']} policy and "
              f"{counters['critic_optimizer_steps']} critic steps", flush=True)
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
        policy_hook.remove()
        critic_hook.remove()
        del bundle, policy, critic
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--output")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--kl-beta", type=float)
    parser.add_argument("--run-name", default="standard")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon,
            args.kl_beta, args.run_name, args.num_workers, args.resume)


if __name__ == "__main__":
    main()
