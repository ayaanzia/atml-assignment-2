"""Task 1 DPO training, with deterministic selection and resumable updates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, print_gpu_info, save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.data import PreferenceCollator, prepare_preference_examples, selection_summary
from task1_dpo.dpo import dpo_loss


def sequence_logps(model, batch: dict) -> torch.Tensor:
    result = response_sequence_logprobs(model, batch)
    sequence = result[0].float()
    del result
    return sequence


def _to_device(batch: dict) -> dict:
    return {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}


def _signature(cfg: dict, dataset: str, beta: float, ids_hash: str) -> str:
    payload = {"config": cfg, "dataset": str(repo_path(dataset)), "beta": beta, "ids_hash": ids_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _save_state(path: Path, model, optimizer, scaler, signature: str,
                counters: dict, epoch: int, next_batch: int) -> None:
    state = {
        "schema": 1,
        "signature": signature,
        "epoch": epoch,
        "next_batch": next_batch,
        "counters": counters.copy(),
        "trainable": {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _rng_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _load_state(path: Path, model, optimizer, scaler, signature: str) -> dict:
    # This state is produced by this script in the user's own output directory.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != 1 or state.get("signature") != signature:
        raise ValueError("Checkpoint schema, configuration, or selected IDs do not match this run")
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != set(state["trainable"]):
        raise ValueError("Checkpoint trainable parameter names do not match the policy")
    with torch.no_grad():
        for name, param in trainable.items():
            saved = state["trainable"][name]
            if saved.shape != param.shape:
                raise ValueError(f"Checkpoint shape mismatch for {name}")
            param.copy_(saved.to(device=param.device, dtype=param.dtype))
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    _restore_rng(state["rng"])
    return state


def _epoch_loader(examples: list[dict], tokenizer, batch_size: int, seed: int, epoch: int,
                  start_batch: int, num_workers: int) -> tuple[DataLoader, int]:
    order_rng = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(examples), generator=order_rng).tolist()
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if start_batch > len(batches):
        raise ValueError("Checkpoint batch cursor exceeds the number of batches")
    worker_rng = torch.Generator().manual_seed(seed + 100_000 + epoch)
    loader = DataLoader(
        examples, batch_sampler=batches[start_batch:],
        collate_fn=PreferenceCollator(tokenizer), num_workers=num_workers,
        pin_memory=True, generator=worker_rng,
    )
    return loader, len(batches)


def _truncate_step_log(path: Path, backward_passes: int) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    kept = [row for row in rows if row["backward_passes"] <= backward_passes]
    if len(kept) != len(rows):
        write_jsonl(path, kept)
        print(f"Discarded {len(rows) - len(kept)} step logs after the last saved checkpoint", flush=True)


def _output_dir(cfg: dict, run_name: str, output_path: str | None) -> Path:
    if output_path:
        return repo_path(output_path)
    if run_name == "standard":
        return repo_path(cfg["standard_output"])
    if run_name == "length_balanced":
        return repo_path(cfg["length_output"])
    return repo_path(f"outputs/task1_dpo/{run_name}")


def run_training(config_path: str, run_name: str, dataset_path: str | None = None,
                 output_path: str | None = None, beta: float | None = None,
                 max_examples: int | None = None, num_workers: int = 0,
                 checkpoint_every: int = 25, resume: bool = False) -> dict:
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("run_name must be a simple directory name")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 1 training requires a CUDA GPU")
    if num_workers < 0 or checkpoint_every < 1:
        raise ValueError("num_workers must be >= 0 and checkpoint_every must be >= 1")
    cfg = load_yaml(config_path)
    if str(cfg["dtype"]).lower() not in {"float16", "fp16", "half"}:
        raise ValueError("Task 1 requires the released FP16 policy configuration")
    seed = int(cfg["seed"])
    set_seed(seed)
    dataset = dataset_path or cfg["paths"]["dpo_standard_train"]
    chosen_beta = float(cfg["beta"] if beta is None else beta)
    output = _output_dir(cfg, run_name, output_path)
    results = repo_path(cfg["results_dir"]) / run_name
    state_path = output / "latest_state.pt"
    steps_path = results / "train_steps.jsonl"
    manifest_path = results / "run_config.json"
    final_adapter = output / "adapter_config.json"
    if resume:
        if not state_path.exists() and not manifest_path.exists():
            raise FileNotFoundError(f"No run manifest or resumable state for {run_name}")
    elif state_path.exists() or final_adapter.exists() or steps_path.exists():
        raise FileExistsError(f"Run {run_name!r} already has output; use --resume or a new run name")

    prep_start = time.perf_counter()
    tokenizer = load_tokenizer(cfg["base_model"])
    examples, excluded = prepare_preference_examples(
        read_jsonl(dataset), tokenizer, int(cfg["max_sequence_length"]), max_examples
    )
    selection = selection_summary(examples, excluded)
    signature = _signature(cfg, dataset, chosen_beta, selection["selected_ids_sha256"])
    prep_seconds = time.perf_counter() - prep_start
    output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_name": run_name, "config_path": config_path, "configuration": cfg,
        "dataset_path": str(repo_path(dataset)), "beta": chosen_beta,
        "requested_examples": max_examples, "num_workers": num_workers,
        "checkpoint_every_updates": checkpoint_every, "selection": selection,
        "run_signature": signature, "precision": "FP16 base/autocast; trainable dtype audited below",
    }
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing run manifest at {manifest_path}")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("run_signature") != signature:
            raise ValueError("Saved manifest does not match configuration or selected examples")
    else:
        save_json(manifest_path, manifest)
        write_jsonl(results / "selected_examples.jsonl", [
            {key: value for key, value in row.items() if key not in {"chosen", "rejected"}}
            for row in examples
        ])
        write_jsonl(results / "excluded_examples.jsonl", excluded)
    print(f"DPO {run_name}: {len(examples)} eligible pairs, {len(excluded)} prompt exclusions, "
          f"workers={num_workers}, beta={chosen_beta}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    promoted = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.float()
            promoted.append(name)
    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("Policy has no trainable LoRA parameters")
    initial_trainable = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters() if param.requires_grad
    }
    manifest["trainable_parameter_dtypes"] = dict(Counter(str(p.dtype) for p in params))
    manifest["promoted_trainable_fp16_parameters"] = promoted
    if resume:
        if old.get("trainable_parameter_dtypes") != manifest["trainable_parameter_dtypes"] or \
                old.get("promoted_trainable_fp16_parameters") != promoted:
            raise ValueError("Resumed model trainable dtypes do not match the original run")
    else:
        save_json(manifest_path, manifest)
    optimizer = AdamW(params, lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    load_seconds = time.perf_counter() - load_start

    counters = {"optimizer_updates": 0, "skipped_optimizer_steps": 0,
                "backward_passes": 0, "processed_pairs": 0,
                "loss_sum": 0.0, "consecutive_skipped_steps": 0}
    start_epoch, start_batch = 0, 0
    if resume:
        if state_path.exists():
            saved = _load_state(state_path, model, optimizer, scaler, signature)
            counters = saved["counters"]
            start_epoch, start_batch = int(saved["epoch"]), int(saved["next_batch"])
            _truncate_step_log(steps_path, counters["backward_passes"])
            print(f"Resuming {run_name} at epoch {start_epoch + 1}, batch {start_batch + 1}", flush=True)
        else:
            _truncate_step_log(steps_path, 0)
            print(f"No checkpoint yet; restarting {run_name} from its verified manifest", flush=True)
    step_count = {"value": 0}
    hook = optimizer.register_step_post_hook(lambda *_: step_count.__setitem__("value", step_count["value"] + 1))
    train_start = time.perf_counter()
    try:
        model.train()
        for epoch in range(start_epoch, int(cfg["epochs"])):
            first_batch = start_batch if epoch == start_epoch else 0
            loader, total_batches = _epoch_loader(
                examples, tokenizer, int(cfg["batch_size"]), seed, epoch, first_batch, num_workers
            )
            optimizer.zero_grad(set_to_none=True)
            group_losses, group_ids = [], []
            bar = tqdm(loader, total=total_batches, initial=first_batch,
                       desc=f"DPO {run_name} epoch {epoch + 1}/{cfg['epochs']}", unit="batch")
            for batch_index, (chosen_cpu, rejected_cpu, identities) in enumerate(bar, start=first_batch):
                chosen, rejected = _to_device(chosen_cpu), _to_device(rejected_cpu)
                with torch.no_grad(), reference_mode(model), torch.amp.autocast("cuda", dtype=torch.float16):
                    ref_chosen = sequence_logps(model, chosen)
                    ref_rejected = sequence_logps(model, rejected)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    policy_chosen = sequence_logps(model, chosen)
                    policy_rejected = sequence_logps(model, rejected)
                    loss, diagnostics = dpo_loss(
                        policy_chosen, policy_rejected, ref_chosen, ref_rejected, chosen_beta
                    )
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Nonfinite DPO loss at epoch {epoch + 1}, batch {batch_index + 1}")
                accumulation = int(cfg["grad_accum_steps"])
                batch_size = int(cfg["batch_size"])
                group_start = (batch_index // accumulation) * accumulation
                group_end = min(group_start + accumulation, total_batches)
                group_pairs = min((group_end - group_start) * batch_size,
                                  len(examples) - group_start * batch_size)
                scaler.scale(loss * (len(identities) / group_pairs)).backward()
                counters["backward_passes"] += 1
                counters["processed_pairs"] += len(identities)
                counters["loss_sum"] += float(loss.detach()) * len(identities)
                group_losses.append(float(loss.detach()))
                group_ids.extend(identities)
                bar.set_postfix(loss=f"{float(loss.detach()):.4f}", updates=counters["optimizer_updates"])
                del chosen, rejected, ref_chosen, ref_rejected, policy_chosen, policy_rejected, loss, diagnostics
                if batch_index + 1 != group_end:
                    continue

                scaler.unscale_(optimizer)
                gradients = [p.grad for p in params if p.grad is not None]
                if not gradients:
                    raise RuntimeError("No LoRA gradients were produced")
                finite_gradients = all(bool(torch.isfinite(g).all().item()) for g in gradients)
                preclip_norm = torch.nn.utils.clip_grad_norm_(
                    params, float(cfg["max_grad_norm"]), error_if_nonfinite=False
                )
                norm_value = float(preclip_norm.detach().cpu())
                if finite_gradients and not math.isfinite(norm_value):
                    raise FloatingPointError("Nonfinite preclip gradient norm despite finite gradient tensors")
                if finite_gradients and norm_value == 0:
                    raise RuntimeError("Zero DPO gradient norm; no learning signal")
                previous_steps = step_count["value"]
                scaler.step(optimizer)
                scaler.update()
                stepped = step_count["value"] > previous_steps
                counters["optimizer_updates"] += int(stepped)
                counters["skipped_optimizer_steps"] += int(not stepped)
                counters["consecutive_skipped_steps"] = 0 if stepped else counters["consecutive_skipped_steps"] + 1
                optimizer.zero_grad(set_to_none=True)
                append_jsonl(steps_path, {
                    "run_name": run_name, "epoch": epoch + 1,
                    "last_batch_iteration": batch_index + 1, "iterations_this_epoch": total_batches,
                    "optimizer_updates": counters["optimizer_updates"],
                    "optimizer_step_skipped": not stepped, "backward_passes": counters["backward_passes"],
                    "pairs_in_accumulation": len(group_ids),
                    "mean_microbatch_loss": sum(group_losses) / len(group_losses),
                    "finite_gradients": finite_gradients,
                    "preclip_gradient_norm": norm_value if math.isfinite(norm_value) else None,
                    "grad_scaler_scale": float(scaler.get_scale()),
                    "source_indices": [row["source_index"] for row in group_ids],
                    "prompt_ids": [row["prompt_id"] for row in group_ids],
                })
                group_losses, group_ids = [], []
                next_epoch = epoch + int(batch_index + 1 == total_batches)
                next_batch = 0 if next_epoch > epoch else batch_index + 1
                if stepped and counters["optimizer_updates"] % checkpoint_every == 0:
                    _save_state(state_path, model, optimizer, scaler, signature, counters, next_epoch, next_batch)
                if counters["consecutive_skipped_steps"] >= 3:
                    raise FloatingPointError("Three consecutive optimizer steps skipped by GradScaler")
            start_batch = 0
        _save_state(state_path, model, optimizer, scaler, signature, counters, int(cfg["epochs"]), 0)
        changed = [name for name, param in model.named_parameters()
                   if param.requires_grad and not torch.equal(param.detach().cpu(), initial_trainable[name])]
        if counters["optimizer_updates"] == 0 or not changed:
            raise RuntimeError("Training completed without a verified change to trainable LoRA weights")
        model.save_pretrained(output)
        torch.cuda.synchronize()
        summary = {
            "run_name": run_name, "beta": chosen_beta, "dataset_path": str(repo_path(dataset)),
            "selection": selection, "epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]), "grad_accum_steps": int(cfg["grad_accum_steps"]),
            "backward_passes": counters["backward_passes"],
            "optimizer_updates": counters["optimizer_updates"],
            "trainable_parameters_changed": len(changed),
            "trainable_parameter_dtype_counts": manifest["trainable_parameter_dtypes"],
            "skipped_optimizer_steps": counters["skipped_optimizer_steps"],
            "processed_pairs": counters["processed_pairs"],
            "mean_train_loss": counters["loss_sum"] / max(1, counters["processed_pairs"]),
            "preparation_seconds": round(prep_seconds, 3),
            "model_loading_seconds": round(load_seconds, 3),
            "optimization_seconds": round(time.perf_counter() - train_start, 3),
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "adapter_path": str(output), "latest_state_path": str(state_path),
            "run_signature": signature,
        }
        save_json(results / "train_summary.json", summary)
        print(f"Completed DPO {run_name}: {counters['optimizer_updates']} updates, "
              f"{counters['skipped_optimizer_steps']} skipped, adapter={output}", flush=True)
        return summary
    finally:
        hook.remove()
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    print_gpu_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--run-name", default="standard")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--beta", type=float)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta,
                 args.max_examples, args.num_workers, args.checkpoint_every, args.resume)


if __name__ == "__main__":
    main()
