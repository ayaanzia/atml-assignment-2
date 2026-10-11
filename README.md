# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/ayaanzia/atml-assignment-2.git
cd atml-assignment-2
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This working repository is based on the ATML PA2 student starter. The upstream release was intentionally incomplete: Tasks 1–3 included model/data loading, objective helpers, checkpoint restoration, and experiment entry points, while the training loops, ablation orchestration, and objective fixes were student work. Those implementations and validations are included in this working copy.

Upstream course starter: https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining

This working copy contains the completed implementations and local result
artifacts for Tasks 1–5. Task 2 was run on Kaggle and the other GPU runs used
Vast.ai; task outputs, metric histories, generated responses, figures, and
trained adapters have been copied into `results/` and `outputs/`. Task 4's
240-response manual audit is complete. Task 6 requires no new training; it is
the cross-task analysis in the report.

## 1. Clone and install

```bash
git clone https://github.com/ayaanzia/atml-assignment-2.git
cd atml-assignment-2
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

Run these commands from the repository root on the rented GPU machine, in the
order shown. The standard and length analyses require the finished standard
adapter. Each entry point prints the selected GPU and free/total VRAM at
startup. Training shows a `tqdm` bar whose iterations are batches within the
current epoch. `--num-workers` defaults to `0`; choose a higher value only after
checking the rented machine's CPU and RAM. For example, append
`--num-workers 2` to `train`, `ablate_beta`, or `analyze_length` if appropriate.

Training uses the released FP16 model, batch size, gradient accumulation, and
sequence cap from `configs/dpo.yaml`. It saves a rolling resumable state every
25 successful optimizer updates and a final LoRA adapter. To resume an
interrupted training or evaluation command, repeat it with `--resume`.
The beta driver creates independent `0.03`, `0.10`, and `0.30` runs from fresh
initialization, each using the first 600 prompt-eligible standard examples.
The length driver trains the supplied balanced set and evaluates both the
standard and balanced adapters on the fixed stratified and word-limit sets.

Adapters and rolling states are written to `outputs/task1_dpo/<run>/`;
JSONL/JSON/CSV records are written to `results/task1_dpo/`. On Vast.ai these
paths are **on the rented instance**. Copy both directories to local storage
before deleting the instance; no cloud connection or transfer code is embedded
in the assignment scripts. The planning and design notes in `rough/` are
intentionally Git-ignored. Completed Task 1 histories and evaluation records
are in `results/task1_dpo/`; trained adapters and rolling states are in
`outputs/task1_dpo/`.

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

Task 2 now has working entry points in this working copy. Run them on the
rented GPU machine from the repository root in the order shown. The standard
run continues the supplied midpoint for 20 updates. Both analysis drivers
launch independent eight-update forks from that same midpoint and evaluate
each condition on the fixed 200-prompt held-out pool. `analyze_clipping`
also computes the fixed 32-rollout cache geometry; use `--cache-only` if you
want to inspect that part before launching its forks.
The KL driver also evaluates the untouched PPO midpoint on those same held-out
prompts, giving the short-fork comparisons a fixed starting baseline.

Each script prints the selected GPU and free/total VRAM at startup. Training
uses an update progress bar and an iteration bar for each of the two PPO
epochs over the same rollout. `--num-workers` defaults to zero; set it after
checking the rented CPU/RAM. Training saves a rolling state after every
successful update. Repeat an interrupted command with `--resume`; a run that
stopped before its first checkpoint restarts from the verified midpoint.

Policy adapters and rolling policy/critic states go under
`outputs/task2_ppo/<run>/`; the final critic adapter is in its
`critic_adapter/` subdirectory. Per-update and per-prompt JSONL, comparison
CSV/JSON, and diagnostic PNG files go under `results/task2_ppo/`.
These paths are on Vast.ai when run there. Copy both directories to local
storage before deleting the instance. Planning and design decisions are in
the Git-ignored `rough/` folder. The completed Task 2 run histories,
comparisons, held-out responses, and figures are in `results/task2_ppo/`.

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

Run from the repository root on a suitable GPU. The standard continuation uses
the supplied GRPO midpoint, 20 updates, and four completions per prompt. The
group-size command is a CPU-capable analysis of the fixed 192-completion cache;
it performs no training. The normalization driver runs two independent
eight-update forks from the same midpoint, then evaluates them and the
untouched midpoint on the same fixed 200 held-out prompts. Task 3 held-out
generation uses the 512-token cap recorded in `configs/grpo.yaml`.

Each entry point prints GPU availability/name and free/total VRAM. During
training, the outer `tqdm` bar tracks prompt updates and the inner bar tracks
the four completion microbatches in the current policy epoch. `--num-workers`
defaults to 0 on `continue_train` and `compare_normalization`; select its
value after inspecting CPU and RAM on Vast. An interrupted training,
evaluation, or normalization command can be repeated with `--resume` after
checking its saved state. The group-size cache analysis is deterministic and
can simply be rerun. The completed run histories, held-out responses,
comparison tables, and figures are in `results/task3_grpo/`.

Final policy adapters and a rolling state after each successful update are
under `outputs/task3_grpo/<run>/`. Raw completions, per-update records,
evaluation records, JSON/CSV summaries, and PNG plots are under
`results/task3_grpo/`. These files live on Vast when run there and must be
copied to local or persistent storage before the instance is removed. The
implementation and results format decisions are recorded in the Git-ignored
`rough/task3_plan.md` and `rough/task3_design_choices.md`. This working copy
contains the completed standard continuation and both normalization forks.

### Task 4 - Safety calibration

Run after the Task 1–3 **standard** adapters are complete. Generation uses the untouched SFT base and only the standard DPO, PPO, and GRPO adapters. It checks the fixed 450-row XSTest file, uses batch size 4 and greedy decoding, and saves one file per policy. The fixed 4-bit judge runs separately from the policy models. Each entry point prints GPU information. No DataLoader is used here, so `num_workers` does not apply.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

If generation or judging is interrupted, rerun that command with `--resume`; saved manifests reject changed inputs or settings. `--policy sft|dpo|ppo|grpo` limits generation or judging to one policy. Missing standard adapters are reported as skipped; the four-policy comparison remains incomplete until all four have 450 generated and judged responses. Other failures are recorded in status JSON and cause a nonzero exit. Results and status files go to `results/task4_safety/`.

The audit command creates `manual_audit_blind.csv` with 240 shuffled, unlabeled responses for 60 fixed prompt IDs, plus a separate `manual_audit_key.csv`. Fill `manual_label` using one of the five labels listed in `manual_audit_manifest.json` **without opening the key or AI labels first**. The sheet is preserved on rerun. Run `evaluate_safety` again after all 240 manual labels are entered to produce agreement and confusion files. This working copy contains the completed labels at `results/task4_safety/manual_audit_labeled.csv` and the resulting agreement/confusion files. Review `policy_disagreements.jsonl` and judge errors yourself before writing the assignment interpretation.

For a local, blind keyboard workflow, copy the blind CSV to a machine with a desktop display if it was generated on a remote GPU, then use:

```bash
python -m task4_safety.manual_audit_gui
```

It reads `results/task4_safety/manual_audit_blind.csv`, shows only the prompt and response, and autosaves to a **separate** `manual_audit_labeled.csv`. Use left/up and right/down arrows to move; press 1–5 to apply the displayed label and advance; press 0 to clear a label. Reopening resumes from the first unlabeled row. It never reads the policy key or AI judge files. After reviewing all 240 labels, evaluate that copy directly without replacing the original:

```bash
python -m task4_safety.evaluate_safety --config configs/feedback.yaml --manual-audit-sheet results/task4_safety/manual_audit_labeled.csv
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied. This working copy evaluates the frozen SFT base and the fixed RLVR/RLAIF adapters, without Task 5 training. It generates 300 GSM8K and 100 SVAMP responses per policy with the released sampling settings and matched prompt seeds. It then loads the 4-bit judge separately for SFT-versus-RLVR and SFT-versus-RLAIF comparisons, and scores 20 five-variant controlled problems. Run from the repo root after validating assets and confirming the two Task 5 adapters exist:

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

`evaluate_math` supports `--stage generate|judge|aggregate|all`, `--policy sft|rlvr|rlaif` for generation only, and `--resume`. `score_perturbations` supports `--stage exact|judge|aggregate|all` and `--resume`. A repeated command with existing records needs `--resume`, which checks manifests and file hashes. Generation, judging, and aggregation can be run separately; a missing prerequisite is reported rather than counted as a completed comparison. The fixed exact diagnostic can run without a GPU; generation and AI judging require CUDA. Task 5 has no DataLoader, so `num_workers` does not apply.

Raw JSONL, status files, manifests, JSON/CSV summaries, plots, and judge caches are written under `results/task5_feedback/`. Each entry point prints the selected GPU and free/total VRAM. `comparison_summary.json` is marked COMPLETE only after both domains and the controlled diagnostic have complete validated records. The local comparison is marked COMPLETE. Judge cache hits are reported separately from new inference. The released judge maps an unparseable answer to TIE; its TIE counts cannot distinguish that case. Preserve the result directory before a temporary GPU instance ends.

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
