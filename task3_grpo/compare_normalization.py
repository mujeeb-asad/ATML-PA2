from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.metrics import safe_corr
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import run_evaluation

LOSS_TYPES = ["grpo", "dr_grpo"]


def fork_name(loss_type):
    return f"fork_{loss_type}" #fork_grpo, fork_dr_grpo


def length_conditioned_stats(completions):
    #what the normalization does to the gradient, measured on the completions that actually carried gradient
    #(not masked, informative group). each completion's gradient norm is divided by |A_k|, so what is left is the
    #part of its weight that comes from the normalization and not from how good or bad it was
    active = [c for c in completions if c["gradient_norm"] > 0]
    if len(active) < 2:
        return {"active_completions": len(active), "masked_or_uninformative": len(completions) - len(active)}
    lengths = np.array([c["completion_tokens"] for c in active], dtype=float)
    per_advantage = np.array([c["gradient_norm"] / abs(c["advantage"]) for c in active])
    #the binning rule, defined once -> short / long = below / at-or-above the median length of the active completions
    long_half = lengths >= np.median(lengths)
    return {
        "active_completions": len(active),
        "masked_or_uninformative": len(completions) - len(active),
        #positive -> longer completions pull harder on the weights, ~0 -> length does not decide a completion's weight
        "corr_gradient_norm_vs_length": safe_corr(per_advantage, lengths),
        "gradient_norm_short_half": float(per_advantage[~long_half].mean()),
        "gradient_norm_long_half": float(per_advantage[long_half].mean()),
        "long_to_short_ratio": float(per_advantage[long_half].mean() / per_advantage[~long_half].mean()),
        "median_length": float(np.median(lengths)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["results_dir"])
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    #length-normalization study -> two short forks (fork_updates = 8) from the same midpoint, same 8 prompts, seed,
    #reward model, kl beta, clip eps, K and generation cap. the only difference is the divisor of each completion's
    #token sum -> its own length T_k (canonical grpo) or the constant max_completion_length (dr. grpo)
    rows = []
    for loss_type in LOSS_TYPES:
        name = fork_name(loss_type)
        output = str(Path(cfg["output"]).parent / name)
        #a fork that already finished is reused if the job is relaunched
        if (repo_path(output) / "adapter_config.json").exists():
            print(f"[{name}] adapter already exists, skipping training")
        else:
            run_grpo(args.config, output, updates=int(cfg["fork_updates"]), loss_type=loss_type, run_name=name)
        if (out_dir / f"eval_{name}.json").exists():
            print(f"[{name}] evaluation already exists, skipping")
        else:
            run_evaluation(args.config, output, name)

        ev = load_json(out_dir / f"eval_{name}.json")
        log = pd.DataFrame(read_jsonl(out_dir / f"train_log_{name}.jsonl"))
        summary = load_json(out_dir / f"train_summary_{name}.json")
        rows.append({
            "loss_type": loss_type,
            "heldout_reward": ev["reward_mean"],
            "heldout_kl_token": ev["kl_token_mean"],
            "heldout_entropy": ev["entropy_mean"],
            "heldout_length": ev["length_mean"],
            "heldout_length_std": ev["length_std"],
            "heldout_truncation": ev["truncation_rate"],
            "train_reward_mean": log["reward"].mean(),
            "train_length_mean": log["response_length"].mean(),
            "train_grad_norm_mean": log["grad_norm"].mean(),
            "updates": summary["updates"],
            "generated_tokens": summary["generated_tokens"], #the matched budget, reported so it can be checked
            **length_conditioned_stats(read_jsonl(out_dir / f"train_completions_{name}.jsonl")),
        })

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "normalization_comparison.csv", index=False)
    print(table.T.to_string())


if __name__ == "__main__":
    main()
