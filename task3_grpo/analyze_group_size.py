from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from task3_grpo.continue_train import INFORMATIVE_TOLERANCE
from task3_grpo.grpo import group_relative_advantages


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    #the equal-generation partition -> every prompt has 8 cached completions, and all 8 are used for every K.
    #they are cut, in generation_index order, into 8/K consecutive groups of K ->
    #   K = 8 : 1 group per prompt   (completions 0-7)
    #   K = 4 : 2 groups per prompt  (0-3, 4-7)
    #   K = 2 : 4 groups per prompt  (0-1, 2-3, 4-5, 6-7)
    #so the three conditions are built from exactly the same 192 completions and rewards, only the grouping changes.
    #a smaller K buys more groups (more updates) from the same budget, a larger K buys a better baseline per group
    groups = []
    for prompt_key, completions in by_prompt.items():
        completions = completions[:8]
        for start in range(0, 8, k):
            groups.append((prompt_key, completions[start:start + k]))
    return groups


def prompt_difficulty_bins(by_prompt):
    #the binning rule, defined once -> a prompt's difficulty is its mean reward over all 8 cached completions.
    #prompts below the median of that mean are "hard" (the policy scores low on them), the rest are "easy"
    means = {key: float(np.mean([c["reward"] for c in completions[:8]])) for key, completions in by_prompt.items()}
    median = float(np.median(list(means.values())))
    return {key: ("hard" if mean < median else "easy") for key, mean in means.items()}, means, median


def group_table(by_prompt, k, bins, prompt_means):
    #one row per K-sized group with everything the summary needs
    rows = []
    for prompt_key, completions in regroup_equal_generation_budget(by_prompt, k):
        rewards = torch.tensor([c["reward"] for c in completions], dtype=torch.float32)
        #the same helper the training loop uses, so the cached study exercises the corrected advantage code
        advantages = group_relative_advantages(rewards, torch.zeros(k, dtype=torch.long))
        std = float(rewards.std(unbiased=False))
        #the best baseline the cache offers for this prompt is the mean over all 8 completions. a completion's
        #advantage is "right" when its sign agrees with its sign against that baseline
        reference_sign = torch.sign(rewards - prompt_means[prompt_key])
        rows.append({
            "K": k,
            "prompt": prompt_key,
            "difficulty": bins[prompt_key],
            "reward_std": std,
            "informative": std > INFORMATIVE_TOLERANCE,
            "centered_rewards": (rewards - rewards.mean()).tolist(), #the un-normalised group-relative signal r_k - mu_r
            "advantages": advantages.tolist(),
            "baseline_error": float(rewards.mean()) - prompt_means[prompt_key], #group mean against the 8 sample mean
            "sign_agreement": float((torch.sign(advantages) == reference_sign).float().mean()),
        })
    return rows


def summarise(groups, label):
    centered = np.concatenate([g["centered_rewards"] for g in groups])
    return {
        "K": groups[0]["K"],
        "difficulty": label,
        "groups": len(groups),
        "completions": len(centered),
        "informative_rate": float(np.mean([g["informative"] for g in groups])),
        "uninformative_fraction": float(1 - np.mean([g["informative"] for g in groups])),
        "mean_group_reward_std": float(np.mean([g["reward_std"] for g in groups])),
        #variance over completions of r_k - mu_r, the group-relative signal before it is divided by sigma_r.
        #(after the division every informative group has variance exactly 1, so the normalised A_k cannot show this)
        "relative_signal_variance": float(centered.var()),
        #how noisy the group mean is as a baseline -> its spread around the prompt's 8 sample mean (0 for K = 8)
        "baseline_std": float(np.std([g["baseline_error"] for g in groups])),
        "sign_agreement": float(np.mean([g["sign_agreement"] for g in groups])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    bins, prompt_means, median = prompt_difficulty_bins(by_prompt)
    print(f"difficulty rule: prompt mean reward over its 8 completions, hard < {median:.3f} <= easy")

    #no model and no gpu here -> the study only regroups the cached completions and their cached rewards
    summary, all_groups = [], []
    for k in cfg["group_sizes"]:
        groups = group_table(by_prompt, int(k), bins, prompt_means)
        all_groups += groups
        summary.append(summarise(groups, "all"))
        for label in ["hard", "easy"]:
            summary.append(summarise([g for g in groups if g["difficulty"] == label], label))

    out_dir = repo_path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "group_size_study.csv", index=False)
    pd.DataFrame(all_groups).drop(columns=["centered_rewards", "advantages"]).to_csv(out_dir / "group_size_groups.csv", index=False)
    #per prompt view for the qualitative examples -> difficulty, mean reward and the 8 individual rewards
    pd.DataFrame([{
        "prompt": key, "prompt_id": completions[0]["prompt_id"], "difficulty": bins[key], "mean_reward": prompt_means[key],
        "reward_std_k8": float(np.std([c["reward"] for c in completions[:8]])),
        "rewards": [round(c["reward"], 3) for c in completions[:8]],
        "truncated": int(sum(c["clipped_at_max"] for c in completions[:8])),
    } for key, completions in by_prompt.items()]).to_csv(out_dir / "group_size_prompts.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
