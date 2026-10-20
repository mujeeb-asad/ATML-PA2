from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.rl_eval import token_kl, token_kl_difference
from common.metrics import safe_corr
from task3_grpo.analyze_group_size import load_k8_cache, prompt_difficulty_bins, regroup_equal_generation_budget
from task3_grpo.continue_train import INFORMATIVE_TOLERANCE
from task3_grpo.grpo import group_relative_advantages

#report diagnostics for task 3 -> table work only, no model and no gpu. everything is read back from the files the
#training, evaluation and group-size scripts already wrote, and only NEW files are written next to them

POLICIES = ["reference", "midpoint", "standard", "fork_grpo", "fork_dr_grpo"] #every held-out evaluation of the task
RUNS = ["standard", "fork_grpo", "fork_dr_grpo"] #every training run of the task
FORKS = {"grpo": "fork_grpo", "dr_grpo": "fork_dr_grpo"} #loss type -> run name, as in compare_normalization.py
TOLERANCES = [INFORMATIVE_TOLERANCE, 0.05, 0.10, 0.15] #the exact-tie test of the training loop, then looser ones up to
#about the reward model's own per-response noise (sd ~0.15 for identical text scored in different runs)

#same look as the task3 notebook figures
BIN_COLORS = {"all": "#52514e", "hard": "#eb6834", "easy": "#2a78d6"}
LOSS_COLORS = {"grpo": "#2a78d6", "dr_grpo": "#eb6834"}
MUTED = "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2})


def se(x):
    #standard error of a mean over independent prompts
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def generations(results, name):
    return read_jsonl(results / f"eval_{name}_generations.jsonl")


def per_response_kl(records):
    #the kl convention used in the report -> the repo's sampled_kl (mean over tokens of log pi - log ref) applied to ONE
    #response at a time, ie its summed seq_kl divided by its own length, then averaged over responses.
    #every response weighs the same whatever its length, unlike the token-pooled kl_token_mean in eval_*.json
    return np.array([r["seq_kl"] / max(r["response_tokens"], 1) for r in records])


def heldout_table(results):
    #one row per policy on the 200 held-out prompts. the rows share prompts, order, seed and decoding, so differences
    #are paired per prompt and their se is the se of the per-prompt difference
    gens = {name: generations(results, name) for name in POLICIES}
    rewards = {name: np.array([r["reward"] for r in gens[name]]) for name in POLICIES}
    kls = {name: per_response_kl(gens[name]) for name in POLICIES}
    rows = []
    for name in POLICIES:
        ev = load_json(results / f"eval_{name}.json")
        d_ref, d_mid = rewards[name] - rewards["reference"], rewards[name] - rewards["midpoint"]
        rows.append({
            "policy": name,
            "reward": rewards[name].mean(),
            "reward_delta_vs_reference": d_ref.mean(), "reward_delta_vs_reference_se": se(d_ref),
            "reward_delta_vs_midpoint": d_mid.mean(), "reward_delta_vs_midpoint_se": se(d_mid),
            #secondary view -> sampled_kl per response, then averaged over responses
            "kl_per_response": kls[name].mean(), "kl_per_response_se": se(kls[name]),
            "kl_per_response_positive_share": float((kls[name] > 0).mean()),
            #the report's kl -> the course helper pooled over every generated token (equals kl_token_mean), per-prompt se
            "kl_token": token_kl(gens[name])[0],
            "kl_token_se": token_kl(gens[name])[1] if name != "reference" else 0.0,
            "entropy": ev["entropy_mean"],
            "length_mean": ev["length_mean"], "length_std": ev["length_std"],
            "truncation_rate": ev["truncation_rate"],
            "identical_text_vs_reference": int(sum(a["response"] == b["response"]
                                                   for a, b in zip(gens[name], gens["reference"]))),
        })
    #the normalization study's own comparison -> dr. grpo minus canonical grpo, paired per prompt
    d_reward = rewards["fork_dr_grpo"] - rewards["fork_grpo"]
    d_kl = kls["fork_dr_grpo"] - kls["fork_grpo"]
    d_seq = (np.array([r["seq_kl"] for r in gens["fork_dr_grpo"]]) - np.array([r["seq_kl"] for r in gens["fork_grpo"]]))
    kl_token_delta, kl_token_delta_se = token_kl_difference(gens["fork_dr_grpo"], gens["fork_grpo"])
    contrast = {
        "comparison": "fork_dr_grpo minus fork_grpo",
        "reward_delta": float(d_reward.mean()), "reward_delta_se": se(d_reward),
        "kl_token_delta": kl_token_delta, "kl_token_delta_se": kl_token_delta_se, #the report's convention
        "kl_per_response_delta": float(d_kl.mean()), "kl_per_response_delta_se": se(d_kl),
        "kl_sequence_sum_delta": float(d_seq.mean()), "kl_sequence_sum_delta_se": se(d_seq), #the summed convention, for reference
        "identical_text": int(sum(a["response"] == b["response"] for a, b in zip(gens["fork_grpo"], gens["fork_dr_grpo"]))),
    }
    return pd.DataFrame(rows), contrast, kls


def group_size_usable(by_prompt, bins, cap):
    #the group-size study applied with OUR training rule -> a completion that hits the training cap is masked out of the
    #loss (mask_truncated_completions). a group can move the policy only if it is informative (std > tolerance, the
    #training loop's test) AND at least one completion survives the mask with a non-zero advantage.
    #caveat on the cache -> its completion_tokens field is one padded length per PAIR of generations (both members of
    #every pair share the count), so "completion_tokens >= 512" over-counts the completions our 512 cap would mask.
    #the only exact per-completion length signal is clipped_at_max (hit the cache's own 768 cap). two versions ->
    #  usable_rate_768        exact, masking = clipped_at_max, ie the same rule under a 768 cap
    #  usable_rate_512_lower  masking = padded length >= 512, an over-count of masking, so a LOWER bound at 512
    #the true usable rate at 512 lies between the two
    rows = []
    for k in [2, 4, 8]:
        groups = []
        for prompt_key, completions in regroup_equal_generation_budget(by_prompt, k):
            rewards = torch.tensor([c["reward"] for c in completions], dtype=torch.float32)
            advantages = group_relative_advantages(rewards, torch.zeros(k, dtype=torch.long))
            informative = float(rewards.std(unbiased=False)) > INFORMATIVE_TOLERANCE
            entry = {"difficulty": bins[prompt_key], "informative": informative}
            for tag, masked in [("768", [bool(c["clipped_at_max"]) for c in completions]),
                                ("512_lower", [c["completion_tokens"] >= cap for c in completions])]:
                sends_gradient = any((not m) and abs(float(a)) > 0 for m, a in zip(masked, advantages))
                entry[f"usable_{tag}"] = informative and sends_gradient
                entry[f"masked_{tag}"] = sum(masked)
            groups.append(entry)
        frame = pd.DataFrame(groups)
        for label in ["all", "hard", "easy"]:
            sub = frame if label == "all" else frame[frame["difficulty"] == label]
            rows.append({"K": k, "difficulty": label, "groups": len(sub),
                         "informative_rate": sub["informative"].mean(),
                         "usable_rate_768": sub["usable_768"].mean(), #exact under a 768 cap
                         "usable_rate_512_lower": sub["usable_512_lower"].mean(), #lower bound at our 512 cap
                         "masked_share_768": sub["masked_768"].sum() / (len(sub) * k),
                         "masked_share_512_upper": sub["masked_512_lower"].sum() / (len(sub) * k)})
    return pd.DataFrame(rows)


def group_std_without_truncation(by_prompt, bins, cap):
    #is the hard bin's larger spread difficulty or truncation? per prompt reward std over its 8 cached completions,
    #with all of them and without the ones clipped at the 768 cache cap (clipped_at_max is the only exact per-completion
    #length signal in the cache, see group_size_usable)
    rows = []
    for prompt_key, completions in by_prompt.items():
        completions = completions[:8]
        def std(keep):
            rewards = [c["reward"] for c in completions if keep(c)]
            return float(np.std(rewards)) if len(rewards) >= 2 else np.nan #population std, as in the study
        rows.append({"prompt": prompt_key, "difficulty": bins[prompt_key],
                     "std_all": std(lambda c: True),
                     "std_without_clipped_768": std(lambda c: not c["clipped_at_max"]),
                     "clipped_768": sum(c["clipped_at_max"] for c in completions)})
    per_prompt = pd.DataFrame(rows)
    summary = []
    for label in ["all", "hard", "easy"]:
        sub = per_prompt if label == "all" else per_prompt[per_prompt["difficulty"] == label]
        summary.append({"difficulty": label, "prompts": len(sub),
                        "mean_std_all": sub["std_all"].mean(),
                        "mean_std_without_clipped_768": sub["std_without_clipped_768"].mean(), #nan prompts skipped
                        "clipped_768": int(sub["clipped_768"].sum())})
    summary.append({"difficulty": "corr(std_all, clipped_768) over prompts",
                    "mean_std_all": safe_corr(per_prompt["std_all"], per_prompt["clipped_768"])})
    return pd.DataFrame(summary)


def masking_accounting(results):
    #what the 512 mask did in each training run -> completions and generated tokens that carried no gradient, updates
    #that lost every completion, and the zero-sum break on partly masked updates (masked rewards stay in mu and sigma)
    rows, updates = [], []
    for run in RUNS:
        log = pd.DataFrame(read_jsonl(results / f"train_log_{run}.jsonl"))
        comps = pd.DataFrame(read_jsonl(results / f"train_completions_{run}.jsonl"))
        for update, group in comps.groupby("update"):
            masked = group["truncated"]
            if masked.any():
                updates.append({"run": run, "update": int(update), "masked": int(masked.sum()), "K": len(group),
                                "masked_advantages": [round(a, 3) for a in group.loc[masked, "advantage"]],
                                "surviving_advantage_sum": float(group.loc[~masked, "advantage"].sum()),
                                "step_kl": float(log.loc[log["update"] == update, "step_kl"].iloc[0]),
                                "grad_norm": float(log.loc[log["update"] == update, "grad_norm"].iloc[0])})
        all_masked = [u for u in updates if u["run"] == run and u["masked"] == u["K"]]
        tokens = comps["completion_tokens"].sum()
        rows.append({"run": run, "updates": len(log), "completions": len(comps),
                     "masked_completions": int(comps["truncated"].sum()),
                     "masked_completion_share": comps["truncated"].mean(),
                     "generated_tokens": int(tokens),
                     "masked_tokens": int(comps.loc[comps["truncated"], "completion_tokens"].sum()),
                     "masked_token_share": comps.loc[comps["truncated"], "completion_tokens"].sum() / tokens,
                     "all_masked_updates": " ".join(str(u["update"]) for u in all_masked),
                     "all_masked_step_kl": " ".join(f"{u['step_kl']:+.1e}" for u in all_masked),
                     "median_abs_step_kl_other_updates": float(log.loc[~log["update"].isin([u["update"] for u in all_masked]),
                                                                       "step_kl"].abs().median())})
    return pd.DataFrame(rows), pd.DataFrame(updates)


def normalization_diagnostics(results, cap):
    #per update of each normalization fork -> the policy term the loss actually logged, and the sum over unmasked
    #completions of A_k * T_k. at ratio = 1 (policy_epochs = 1) the dr. grpo term is exactly -sum(A_k T_k) / (K * 512),
    #so its sign says whether the surviving advantage mass sits on the long or the short completions of the group
    per_update, per_fork = [], []
    for loss_type, run in FORKS.items():
        log = pd.DataFrame(read_jsonl(results / f"train_log_{run}.jsonl")).set_index("update")
        comps = pd.DataFrame(read_jsonl(results / f"train_completions_{run}.jsonl"))
        for update, group in comps.groupby("update"):
            live = group[~group["truncated"]]
            k = len(group)
            sum_at = float((live["advantage"] * live["completion_tokens"]).sum())
            sum_a_over_t = float((live["advantage"] / live["completion_tokens"].clip(lower=1)).sum())
            predicted = -sum_at / (k * cap) if loss_type == "dr_grpo" else -float(live["advantage"].sum()) / k
            per_update.append({"loss_type": loss_type, "update": int(update), "masked": int(group["truncated"].sum()),
                               "sum_A_times_T": sum_at, "sum_A_over_T": sum_a_over_t,
                               "corr_reward_length_unmasked": safe_corr(live["reward"], live["completion_tokens"]),
                               "logged_policy_term": float(log.loc[update, "policy_loss"]),
                               "predicted_policy_term_at_ratio_1": predicted,
                               "grad_norm": float(log.loc[update, "grad_norm"])})
        #the length-conditioned scale -> canonical divides by T_k, dr. by 512, so for the same completion the two
        #per-completion norms differ by exactly 512 / T_k. its mean over the completions that carried gradient is the
        #expected canonical / dr. ratio of gradient norms
        active = comps[comps["gradient_norm"] > 0]
        weight = (cap / active["completion_tokens"])
        per_fork.append({"loss_type": loss_type, "active_completions": len(active),
                         "mean_512_over_T": weight.mean(),
                         "abs_advantage_weighted_512_over_T": float((weight * active["advantage"].abs()).sum()
                                                                     / active["advantage"].abs().sum()),
                         "train_grad_norm_mean": log["grad_norm"].mean(),
                         "train_grad_norm_mean_without_all_masked": log.loc[log["grad_norm"] > 0, "grad_norm"].mean(),
                         "unmasked_updates_with_negative_policy_term": int(sum(
                             1 for u in per_update if u["loss_type"] == loss_type and u["masked"] == 0
                             and u["logged_policy_term"] < 0)),
                         "unmasked_updates": int(sum(1 for u in per_update if u["loss_type"] == loss_type and u["masked"] == 0))})
    per_fork = pd.DataFrame(per_fork)
    observed = per_fork.set_index("loss_type")["train_grad_norm_mean"]
    per_fork["observed_canonical_over_dr_grad_norm_ratio"] = observed["grpo"] / observed["dr_grpo"]
    return pd.DataFrame(per_update), per_fork


def informativeness_tolerance(results, by_prompt):
    #how many groups count as uninformative if "tied" means a spread below the reward model's own noise instead of an
    #exact tie. the training loop uses std <= 1e-6, the looser cuts are diagnostic only
    rows = []
    log = pd.DataFrame(read_jsonl(results / "train_log_standard.jsonl"))
    for tol in TOLERANCES:
        rows.append({"source": "standard run (K=4, 20 updates)", "tolerance": tol, "groups": len(log),
                     "uninformative": int((log["group_reward_std"] <= tol).sum()),
                     "uninformative_rate": float((log["group_reward_std"] <= tol).mean()),
                     "updates": " ".join(str(u) for u in log.loc[log["group_reward_std"] <= tol, "update"])})
    for k in [2, 4, 8]:
        stds = np.array([np.std([c["reward"] for c in completions])
                         for _, completions in regroup_equal_generation_budget(by_prompt, k)])
        for tol in TOLERANCES:
            rows.append({"source": f"cache K={k}", "tolerance": tol, "groups": len(stds),
                         "uninformative": int((stds <= tol).sum()), "uninformative_rate": float((stds <= tol).mean()),
                         "updates": ""})
    return pd.DataFrame(rows)


def identical_text_noise(results):
    #the same response text, scored by the 8-bit reward model in two different evaluation runs. any reward difference
    #on these pairs is scoring noise, not a policy difference -> the per-response noise floor of the reward
    rows = []
    for a, b in itertools.combinations(POLICIES, 2):
        ga, gb = generations(results, a), generations(results, b)
        diffs = np.array([y["reward"] - x["reward"] for x, y in zip(ga, gb) if x["response"] == y["response"]])
        rows.append({"policy_a": a, "policy_b": b, "identical_texts": len(diffs),
                     "mean_diff": diffs.mean(), "sd_diff": diffs.std(ddof=1), "mean_abs_diff": np.abs(diffs).mean(),
                     "max_abs_diff": np.abs(diffs).max()})
    frame = pd.DataFrame(rows)
    pooled = np.concatenate([np.array([y["reward"] - x["reward"] for x, y in
                                       zip(generations(results, a), generations(results, b)) if x["response"] == y["response"]])
                             for a, b in itertools.combinations(POLICIES, 2)])
    frame.loc[len(frame)] = {"policy_a": "all pairs", "policy_b": "pooled", "identical_texts": len(pooled),
                             "mean_diff": pooled.mean(), "sd_diff": pooled.std(ddof=1),
                             "mean_abs_diff": np.abs(pooled).mean(), "max_abs_diff": np.abs(pooled).max()}
    return frame


def plot_heldout(heldout, figures):
    #held-out reward change and token-level kl (the course convention) of every task 3 policy, each with its per-prompt se
    trained = heldout[heldout["policy"] != "reference"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(trained))[::-1]
    axes[0].errorbar(trained["reward_delta_vs_reference"], y, xerr=2 * trained["reward_delta_vs_reference_se"],
                     fmt="o", color="#2a78d6", ecolor="#9ec5f4", elinewidth=3, markersize=8, capsize=0)
    axes[0].axvline(0, color=MUTED, linewidth=1)
    axes[0].set_title("held-out reward minus reference (± 2 paired se)")
    axes[1].errorbar(trained["kl_token"], y, xerr=2 * trained["kl_token_se"],
                     fmt="o", color="#eb6834", ecolor="#f6c2a8", elinewidth=3, markersize=8, capsize=0)
    axes[1].axvline(0, color=MUTED, linewidth=1)
    axes[1].set_title("token-level kl from the reference (± 2 se)")
    for axis in axes:
        axis.set_yticks(y, trained["policy"])
    fig.tight_layout()
    fig.savefig(figures / "heldout_reward_kl.png", dpi=200)
    plt.close(fig)


def plot_usable(usable, study, figures):
    #the informative-group rate the study reports (exact ties only) next to the share of groups that could actually
    #move the policy once truncated completions are masked, per difficulty bin, at the same 192 generations for every K
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, (frame, column, title) in zip(axes, [(study, "informative_rate", "informative-group rate (exact-tie test)"),
                                                   (usable, "usable_rate_768", "usable groups (solid 768 exact, dashed 512 lower bound)")]):
        for label in ["all", "hard", "easy"]:
            rows = frame[frame["difficulty"] == label].sort_values("K")
            axis.plot(rows["K"], rows[column], color=BIN_COLORS[label], marker="o", markersize=7, label=label)
            if column == "usable_rate_768":
                axis.plot(rows["K"], rows["usable_rate_512_lower"], color=BIN_COLORS[label], marker="o", markersize=5,
                          ls="--", alpha=0.6)
        axis.set_xscale("log", base=2)
        axis.set_xticks([2, 4, 8], ["2", "4", "8"])
        axis.minorticks_off()
        axis.set_xlabel("group size K (192 generations in every case)")
        axis.set_title(title)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, title="prompt difficulty")
    fig.tight_layout()
    fig.savefig(figures / "group_size_usable.png", dpi=200)
    plt.close(fig)


def plot_normalization_overlay(results, figures):
    #both forks' gradient per unit |advantage| against completion length on ONE log axis, so the two normalizations can
    #be read against each other (the existing figure uses a separate panel and scale per fork)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for loss_type, run in FORKS.items():
        comps = pd.DataFrame(read_jsonl(results / f"train_completions_{run}.jsonl"))
        active = comps[comps["gradient_norm"] > 0]
        per_advantage = active["gradient_norm"] / active["advantage"].abs()
        axis.scatter(active["completion_tokens"], per_advantage, s=60, color=LOSS_COLORS[loss_type],
                     edgecolor="#fcfcfb", linewidth=1.2,
                     label=f"{loss_type} (corr with length {safe_corr(per_advantage, active['completion_tokens']):+.2f})")
    axis.set_yscale("log")
    axis.set_xlabel("completion length (tokens)")
    axis.set_ylabel("gradient norm / |advantage|")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "normalization_gradient_vs_length_overlay.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    cap = int(cfg["max_completion_length"]) #the 512 training cap, also the dr. grpo constant

    #1 -> held-out table with the per-response kl convention and paired standard errors
    heldout, contrast, _ = heldout_table(results)
    heldout.to_csv(results / "report_heldout.csv", index=False)
    save_json(results / "report_heldout_dr_vs_canonical.json", contrast)
    print(heldout.round(4).to_string(index=False))
    print(contrast)

    #2 -> group-size study under the training mask, and the hard bin's spread with and without truncated completions
    by_prompt = load_k8_cache(cfg["group_cache"])
    bins, _, _ = prompt_difficulty_bins(by_prompt)
    usable = group_size_usable(by_prompt, bins, cap)
    usable.to_csv(results / "group_size_usable.csv", index=False)
    print(usable.round(3).to_string(index=False))
    std_table = group_std_without_truncation(by_prompt, bins, cap)
    std_table.to_csv(results / "group_std_truncation.csv", index=False)
    print(std_table.round(3).to_string(index=False))

    #3 -> masking accounting per run, plus one row per partly or fully masked update
    masking, masked_updates = masking_accounting(results)
    masking.to_csv(results / "masking_accounting.csv", index=False)
    masked_updates.to_csv(results / "masking_updates.csv", index=False)
    print(masking.round(4).to_string(index=False))
    print(masked_updates.round(4).to_string(index=False))

    #4 -> normalization -> per update A*T mass and policy term, per fork the 512/T_k scale against the grad-norm ratio
    norm_updates, norm_forks = normalization_diagnostics(results, cap)
    norm_updates.to_csv(results / "normalization_diagnostics.csv", index=False)
    norm_forks.to_csv(results / "normalization_scale.csv", index=False)
    print(norm_updates.round(4).to_string(index=False))
    print(norm_forks.round(4).to_string(index=False))

    #5 -> informativeness at looser tolerances
    tolerance = informativeness_tolerance(results, by_prompt)
    tolerance.to_csv(results / "informativeness_tolerance.csv", index=False)
    print(tolerance.round(3).to_string(index=False))

    #6 -> reward-model noise on identical texts across the task 3 evaluations
    noise = identical_text_noise(results)
    noise.to_csv(results / "reward_noise_identical_text.csv", index=False)
    print(noise.round(4).to_string(index=False))

    plot_heldout(heldout, figures)
    plot_usable(usable, pd.read_csv(results / "group_size_study.csv"), figures)
    plot_normalization_overlay(results, figures)


if __name__ == "__main__":
    main()
