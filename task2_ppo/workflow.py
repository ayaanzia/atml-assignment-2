"""Separate-process orchestration for Task 2 short forks."""

from __future__ import annotations

import json
import subprocess
import sys

from common.data import read_jsonl, repo_path


def invoke(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def evaluate_midpoint(config_path: str, cfg: dict, resume: bool) -> dict:
    result = repo_path(cfg["results_dir"]) / "midpoint" / "heldout_eval"
    summary_path = result / "summary.json"
    if not summary_path.exists():
        partial = (result / "evaluation_manifest.json").exists()
        if partial and not resume:
            raise FileExistsError("Partial midpoint evaluation exists; rerun with --resume")
        args = ["--config", config_path, "--adapter",
                cfg["paths"]["ppo_midpoint_policy"], "--name", "midpoint"]
        if partial:
            args.append("--resume")
        invoke("task2_ppo.evaluate", *args)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_condition(config_path: str, cfg: dict, name: str, updates: int,
                  epsilon: float, beta: float, num_workers: int,
                  resume: bool) -> tuple[dict, dict, dict]:
    output = repo_path(f"outputs/task2_ppo/{name}")
    result = repo_path(cfg["results_dir"]) / name
    train_summary = result / "train_summary.json"
    if not train_summary.exists():
        partial = (result / "run_config.json").exists() or (output / "latest_state.pt").exists()
        if partial and not resume:
            raise FileExistsError(f"Partial {name} training exists; rerun with --resume")
        args = ["--config", config_path, "--run-name", name, "--updates", str(updates),
                "--clip-epsilon", str(epsilon), "--kl-beta", str(beta),
                "--num-workers", str(num_workers)]
        if partial:
            args.append("--resume")
        invoke("task2_ppo.continue_train", *args)
    manifest = json.loads((result / "run_config.json").read_text(encoding="utf-8"))
    train = json.loads(train_summary.read_text(encoding="utf-8"))
    effective = manifest["config"]
    expected = dict(cfg)
    expected.update({"updates": updates, "clip_epsilon": epsilon, "kl_beta": beta})
    if (effective != expected or
            train["counters"]["completed_updates"] != updates):
        raise ValueError(f"Completed {name} training does not match requested fork")
    eval_dir = result / "heldout_eval"
    eval_summary = eval_dir / "summary.json"
    if not eval_summary.exists():
        partial = (eval_dir / "evaluation_manifest.json").exists()
        if partial and not resume:
            raise FileExistsError(f"Partial {name} evaluation exists; rerun with --resume")
        args = ["--config", config_path, "--adapter", str(output), "--name", name]
        if partial:
            args.append("--resume")
        invoke("task2_ppo.evaluate", *args)
    evaluation = json.loads(eval_summary.read_text(encoding="utf-8"))
    return manifest, train, evaluation


def verify_matched_conditions(cfg: dict, names: list[str]) -> dict:
    root = repo_path(cfg["results_dir"])
    training_ids, training_keys, evaluation_ids, generation_keys = [], [], [], []
    for name in names:
        run_manifest = json.loads((root / name / "run_config.json").read_text(encoding="utf-8"))
        eval_manifest = json.loads((root / name / "heldout_eval" / "evaluation_manifest.json").read_text(encoding="utf-8"))
        training_ids.append(run_manifest["prompt_ids"])
        train_rows = read_jsonl(root / name / "train_updates.jsonl")
        training_keys.append([(row["update"], row["prompt_id"], row["seed"])
                              for row in train_rows])
        evaluation_ids.append(eval_manifest["prompt_ids"])
        if eval_manifest["generation"] != cfg["generation"] or \
                int(eval_manifest["response_cap"]) != int(cfg["eval_max_response_length"]):
            raise ValueError(f"Held-out generation settings differ for {name}")
        rows = read_jsonl(root / name / "heldout_eval" / "generations.jsonl")
        generation_keys.append(sorted((row["file_position"], row["prompt_id"], row["seed"])
                                      for row in rows))
    if any(ids != training_ids[0] for ids in training_ids[1:]):
        raise ValueError("Short PPO forks used different training prompt IDs")
    if any(keys != training_keys[0] for keys in training_keys[1:]):
        raise ValueError("Short PPO forks used different training sampling seeds")
    if any(ids != evaluation_ids[0] for ids in evaluation_ids[1:]):
        raise ValueError("Short PPO forks used different held-out prompt IDs")
    if any(keys != generation_keys[0] for keys in generation_keys[1:]):
        raise ValueError("Short PPO forks used different held-out sampling seeds")
    return {"training_prompt_ids": training_ids[0],
            "heldout_prompt_ids": evaluation_ids[0]}
