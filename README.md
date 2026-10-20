# ATML PA2: LLM post-training

Code, configs, logs and results for EE-5102/CS-6304 Programming Assignment 2 (Fall 2026): DPO, PPO and GRPO on Qwen2.5-1.5B-Instruct, safety calibration on XSTest, and RLVR vs RLAIF feedback on GSM8K/SVAMP. The analysis is in the separate PDF report; this repository is everything needed to reproduce its numbers.

Built on the course starter release ([AbDu11aHHH/ATML-PA2-LLM-PostTraining](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining)). See [Changes to the supplied code](#6-changes-to-the-supplied-code) and [Attribution](#7-attribution).

## Repository layout

```text
configs/            release configs (only new keys added, no release value changed)
common/             release infrastructure, plus rl_eval.py (held-out evaluation shared by Tasks 2 and 3)
scripts/            release asset download / validation
task1_dpo/ ... task5_feedback/
    *.py            the task's scripts (objective, training/continuation, ablations, evaluation)
    taskN.ipynb     thin driver: runs the scripts in order, then builds the summary tables and figures
    taskN_executed.ipynb   the executed copy from the run that produced results/
results/taskN_*/    machine-readable logs, metrics, generations, summary tables, figures
run_modal.py        the runner used for every reported run (Modal, fixed GPU type)
data/ cached/ checkpoints/ manifests/   course assets, downloaded, not committed
outputs/            trained adapters, not committed
```

## 1. Setup

```bash
git clone https://github.com/mujeeb-asad/ATML-PA2.git
cd ATML-PA2
python -m pip install -r requirements.txt
python -m pip install papermill==2.7.0 ipykernel   # to execute the task notebooks
python -m scripts.download_assets                 # course assets, pinned revision 0b350481fb03f5525a35bcdec4131bd4fe487f98
python -m scripts.validate_assets
python -m scripts.check_environment
```

`download_assets` also materialises the fixed 100-example SVAMP transfer set (`data/math_transfer_eval.jsonl`). Public models are pulled from Hugging Face at runtime: Qwen/Qwen2.5-1.5B-Instruct (policy), Qwen/Qwen2.5-0.5B-Instruct (PPO critic base), yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback (reward model), Qwen/Qwen2.5-3B-Instruct (Task 4 and Task 5 judge).

All commands run from the repository root.

## 2. How the reported runs were executed

Every reported result was produced with `run_modal.py` on [Modal](https://modal.com). It builds an image with the pinned library versions, keeps the course assets, model cache, adapters and results on a persistent volume (`atml`, under `/pa2`), and executes one task notebook top to bottom with papermill.

```bash
pip install modal && modal setup                     # once

modal run run_modal.py --task task1 --smoke          # smoke test: real models, a handful of rows/updates, results under /pa2/smoke
modal run --detach run_modal.py --task task1         # the real run (task1 ... task5)

modal volume get atml /pa2/results ./                # pull results/ back
modal volume get atml /pa2/results/task1_executed.ipynb task1_dpo/task1_executed.ipynb
```

- GPU: Tasks 2, 3 and 5 ran on an NVIDIA L4 (24 GB), the current `GPU` setting in `run_modal.py` (recorded in the Task 2/3 `train_summary_*.json`). Task 1 ran on an A100-40GB, the setting at commit `535dbc6`; set `GPU = 'A100-40GB'` to reproduce it on the same card. Task 4 reports no compute numbers.
- With `--detach`, keep the local client alive until the run finishes, or close it with Ctrl-C. Terminating the client process (e.g. with `kill` or `timeout`) cancels the run.
- The Task 4 smoke test uses the adapters left by the Task 1–3 smoke tests, so run those first.

The same notebooks run locally without Modal (GPU required for every task except the table-only steps):

```bash
papermill --cwd . task1_dpo/task1.ipynb task1_dpo/task1_executed.ipynb -p CONFIG configs/dpo.yaml
papermill --cwd . task1_dpo/task1.ipynb task1_dpo/task1_executed.ipynb -p CONFIG configs/dpo.yaml -p RUN_SCRIPTS False   # rebuild tables/figures only
```

## 3. Commands per experiment

Each notebook runs exactly the script sequence listed under it, each script as its own process, and then builds the task's summary tables and figures from the files on disk. Running the scripts by hand reproduces every logged result; the summary tables and figures come from the notebook's analysis cells (run it with `-p RUN_SCRIPTS False` afterwards).

### Task 1: DPO (`configs/dpo.yaml`, notebook `task1_dpo/task1.ipynb`)

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard --output outputs/task1_dpo/standard   # Step 1: one epoch, beta 0.1
python -m task1_dpo.evaluate --config configs/dpo.yaml --name base                                            # untrained reference row
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml       # Step 2: beta in {0.03, 0.10, 0.30}, 600 pairs each, trained + evaluated
python -m task1_dpo.analyze_length --config configs/dpo.yaml    # Step 3: length-balanced model, stratified held-out set, dataset length diagnostics
```

### Task 2: PPO (`configs/ppo.yaml`, notebook `task2_ppo/task2.ipynb`)

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard --output outputs/task2_ppo/standard   # Step 1: 20 updates from the midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --name reference
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml  # Step 2: cached-rollout study + eps in {0.05, 0.20, 0.50} forks (8 updates each)
python -m task2_ppo.ablate_kl --config configs/ppo.yaml         # Step 3: beta_KL in {0, 0.10, 0.20} forks (8 updates each)
```

`analyze_clipping --cached-only` runs the cached-rollout part alone. The eps 0.2 / beta_KL 0.1 fork is shared by Steps 2 and 3 and trained once.

### Task 3: GRPO (`configs/grpo.yaml`, notebook `task3_grpo/task3.ipynb`)

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard --output outputs/task3_grpo/standard   # Step 1: 20 updates from the midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --name reference
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml       # Step 2: K in {2, 4, 8} from the supplied completion cache, no training
python -m task3_grpo.compare_normalization --config configs/grpo.yaml    # Step 3: canonical GRPO vs Dr. GRPO forks (8 updates each) + evaluation
```

### Task 4: Safety calibration (`configs/feedback.yaml`, notebook `task4_safety/task4.ipynb`)

Needs the three standard adapters from Tasks 1–3 in `outputs/`.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # Step 1: greedy responses from SFT, DPO, PPO, GRPO to all 450 XSTest prompts
python -m task4_safety.judge_responses --config configs/feedback.yaml      # Step 2: the fixed categorical judge on every response
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml     # Step 3: the fixed 60-prompt audit set and a blind, shuffled labelling sheet
# label results/task4_safety/manual_audit_sheet.csv by hand (column manual_label), then:
python -m task4_safety.evaluate_safety --config configs/feedback.yaml      # calibration rates, category tables, judge/manual agreement
```

The audit sheet holds the 118 distinct texts among the 240 responses (60 prompts × 4 policies), with policy and benchmark class hidden. `make_audit_sheet` never overwrites a sheet that already has labels. `evaluate_safety` is table work only and runs without a GPU; the labelled sheet is in this repository, not on the Modal volume.

### Task 5: RLVR vs RLAIF (`configs/feedback.yaml`, notebook `task5_feedback/task5.ipynb`)

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm        # Step 1: SFT / RLVR / RLAIF on 300 GSM8K problems
python -m task5_feedback.score_perturbations --config configs/feedback.yaml                # Step 2: the 100-response controlled diagnostic set
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer   # Step 3: the same policies on the 100-problem SVAMP set
python -m task5_feedback.compare_feedback --config configs/feedback.yaml                   # combined in-domain / transfer / mechanism tables
```

Task 6 needs no runs; it uses the results above.

### Report diagnostics and figures (CPU only, after the runs above)

Each task has a `report_diagnostics.py`. It recomputes the report's secondary statistics from the files already in `results/`:
- per-response KL and paired SEs
- per-token strata accuracy
- the cached PPO advantage decomposition
- masking accounting
- hand-label calibration
- strict vs lenient accuracy and judge output breakdowns

It writes new CSV/JSON and figures next to the task's results. `report/make_figures.py` then builds the compact figures used in the PDF report, under `results/report/`. None of these scripts loads a model.

```bash
python -m task1_dpo.report_diagnostics --config configs/dpo.yaml
python -m task2_ppo.report_diagnostics --config configs/ppo.yaml
python -m task3_grpo.report_diagnostics --config configs/grpo.yaml
python -m task4_safety.report_diagnostics --config configs/feedback.yaml
python -m task5_feedback.report_diagnostics --config configs/feedback.yaml
python -m report.make_figures
```

KL convention in the report: the course estimator (`common.metrics.sampled_kl`, mean of log π − log π_ref over tokens) is applied to each response, then averaged over responses (the `kl_per_response*` columns). The token-pooled `kl_token_mean` and the per-response sum `kl_sequence_mean` in the `eval_*.json` files are kept for reference.

## 4. Where each result lives

| Task | Main tables | Per-example records | Figures |
|---|---|---|---|
| 1 | `results/task1_dpo/dpo_summary.csv`, `length_strata_accuracy.csv`, `length_generation_compare_with_base.csv`, `length_dataset_diagnostics.json`, `eval_*.json`, `train_summary_*.json` | `eval_*_pairs.jsonl`, `eval_*_generations.jsonl`, `eval_*_word_limit.jsonl`, `train_log_*.jsonl` | `results/task1_dpo/figures/` |
| 2 | `results/task2_ppo/standard_trajectory.csv`, `standard_critic.csv`, `standard_heldout.csv`, `cached_clip_*.csv`, `cached_clip_consistency.json`, `clipping_forks.csv`, `kl_forks.csv`, `train_summary_*.json` | `train_log_*.jsonl`, `eval_*_generations.jsonl` | `results/task2_ppo/figures/` |
| 3 | `results/task3_grpo/standard_trajectory.csv`, `standard_heldout.csv`, `group_size_study.csv`, `group_size_prompts.csv`, `group_size_groups.csv`, `normalization_comparison.csv`, `train_summary_*.json` | `train_log_*.jsonl`, `train_completions_*.jsonl`, `eval_*_generations.jsonl` | `results/task3_grpo/figures/` |
| 4 | `results/task4_safety/safety_calibration.csv`, `category_labels.csv`, `policy_disagreements.csv`, `audit_agreement.csv`, `audit_confusion.csv` | `generated_*.jsonl`, `judged_*.jsonl`, `manual_audit_*.csv`, `audit_joined.csv`, `audit_disagreements.csv` | `results/task4_safety/figures/` |
| 5 | `results/task5_feedback/summary_*.csv`, `agreement_*.csv`, `diagnostic_preference_rates.csv`, `diagnostic_sensitivity.csv`, `feedback_*.csv`, `diagnostic_verifier_check.json` | `generated_*.jsonl`, `pairwise_*.jsonl`, `diagnostic_pairs.jsonl`, `diagnostic_group_rewards.csv`, `judge_cache.json`, `judge_cache_raw.json` | `results/task5_feedback/figures/` |

Peak VRAM and wall-clock for every training run are in its `train_summary_*.json`.

## 5. Reproducibility notes

- Seed 6304 (`configs/base.yaml`) everywhere; all hyperparameters are in `configs/`.
- Exact data used: DPO training pair IDs (and the long-prompt pairs dropped) in `results/task1_dpo/train_summary_*.json`; PPO/GRPO prompt IDs per update in `train_log_*.jsonl`; `prompt_id` / `xstest_id` / `problem_id` on every generated record.
- PPO and GRPO take prompts in file order, so every fork sees the same prompts at the same update. Every fork starts from the same supplied midpoint.
- Re-running is resumable: forks whose adapter already exists, and generation/judge files that already exist, are reused instead of recomputed. Delete the matching `outputs/` and `results/` entries to recompute from scratch.
- GPU sampling and fp16 kernels are not bit-deterministic, so a re-run can differ slightly from the committed logs even with the same seed.

## 6. Changes to the supplied code

**The three deliberate defects (Tasks 1–3), fixed and commented in place:**

- `task1_dpo/dpo.py`, `dpo_loss`: the reference margin was added (`policy_margin + ref_margin`); it is subtracted.
- `task2_ppo/ppo.py`, `ppo_policy_loss`: the clipped surrogate used `torch.maximum`; it uses `torch.minimum`.
- `task3_grpo/grpo.py`, `group_relative_advantages`: mean and std were taken over the whole batch; they are taken per prompt group (`group_ids`).

**Implemented (the parts the release left as TODO):** the DPO training loop and evaluation, the PPO and GRPO continuation loops, all ablation orchestration and analysis scripts, the Task 4 judging/aggregation/audit evaluation, and the Task 5 evaluation and analysis. `common/rl_eval.py` is new.

**Other changes to supplied files:**

- `task2_ppo/continue_train.py`: the critic's trainable value head is moved to fp32 before training (the release loader left it in fp16, where AdamW is numerically unsafe).
- `task5_feedback/rlaif.py`: the pairwise judge also records its raw output (`judge_cache_raw.json`) and exposes `output_kind`, so ties, unparseable outputs and multi-label outputs can be reported separately. The judge prompt, parser and verdicts are unchanged.
- `configs/*.yaml`: new keys only (evaluation batch sizes, caps, sample counts); no release value changed.

`common/` (except the new `rl_eval.py`) and `scripts/` are as released.

## 7. Attribution

- Course starter release: [AbDu11aHHH/ATML-PA2-LLM-PostTraining](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining), including its two later updates to `common/data.py` and `task1_dpo/dpo.py` (2026-10-03). The supplied infrastructure, judges, verifier, configs and data loaders are the course's.
- Course assets: [AbDu11aHHH/ATML-PA2-assets](https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets), revision `0b350481fb03f5525a35bcdec4131bd4fe487f98` (UltraFeedback preference subsets, RL prompt pools, XSTest, GSM8K subsets, the Task 5 diagnostic set, PPO/GRPO midpoint checkpoints, RLVR/RLAIF adapters, cached rollouts). The SVAMP transfer set is built by the release installer from the official SVAMP challenge set; the Task 1 word-limit prompts ship with the release.
- Public models: Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct (Qwen team), qwen2.5-1.5b-rm-ultrafeedback (yavuz-ai).
- Libraries: PyTorch, Transformers, PEFT, bitsandbytes, Accelerate, pandas, NumPy, matplotlib, papermill, Modal (versions in `requirements.txt` and `run_modal.py`).

No other external code was reused.
