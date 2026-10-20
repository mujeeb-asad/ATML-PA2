from __future__ import annotations

import argparse
import itertools

import matplotlib
matplotlib.use("Agg") #files only, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.rl_eval import token_kl
from task2_ppo.analyze_clipping import load_cached_rollouts, pad_rows
from task2_ppo.continue_train import fork_name
from task2_ppo.ppo import compute_gae, normalize_advantages, shaped_rewards

#report diagnostics for task 2 -> cpu only, no model is loaded. everything is recomputed from files already on disk
#(the held-out generations, the training logs, the cached rollout batch) and written to new files next to them

#same palette as task2.ipynb
EPS_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'lines.linewidth': 2})


def se(x):
    #standard error of a mean over independent items (prompts)
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def load_generations(results, name):
    return pd.DataFrame(read_jsonl(results / f"eval_{name}_generations.jsonl"))


def policy_names(cfg):
    #every held-out policy of the task -> anchors, the standard run, then the five forks (eps study + the two extra kl forks)
    eps_forks = [fork_name(float(e), float(cfg["kl_beta"])) for e in cfg["clip_values"]]
    kl_forks = [fork_name(float(cfg["clip_epsilon"]), float(b)) for b in cfg["kl_values"]]
    forks = eps_forks + [f for f in kl_forks if f not in eps_forks]
    return ["reference", "midpoint", "standard"] + forks, forks


#---- 1. held-out table with the manual's kl and entropy conventions
def heldout_table(results, names):
    #kl for the report -> the course sampled_kl helper pooled over every generated token (token_kl, the same number as
    #kl_token_mean in eval_*.json), with a per-prompt standard error. entropy likewise is the token-level mean over all
    #valid response tokens (entropy_mean in eval_*.json). the per-response versions are kept as secondary columns.
    #every policy answers the same 200 prompts in the same order, so differences are paired by prompt
    gens = {name: load_generations(results, name) for name in names}
    ref, mid = gens["reference"], gens["midpoint"]
    rows = []
    for name in names:
        g = gens[name]
        kl = g["seq_kl"] / g["response_tokens"].clip(lower=1)
        d_ref, d_mid = g["reward"] - ref["reward"], g["reward"] - mid["reward"]
        rows.append({
            "policy": name,
            "reward_mean": g["reward"].mean(),
            "reward_delta_vs_reference": d_ref.mean(), "reward_delta_vs_reference_se": se(d_ref),
            "reward_delta_vs_midpoint": d_mid.mean(), "reward_delta_vs_midpoint_se": se(d_mid),
            "kl_token": token_kl(g.to_dict("records"))[0],
            "kl_token_se": token_kl(g.to_dict("records"))[1] if name != "reference" else 0.0, #reference is 0 by construction
            "entropy_token": load_json(results / f"eval_{name}.json")["entropy_mean"],
            "kl_per_response_mean": kl.mean(), "kl_per_response_se": se(kl),
            "kl_per_response_positive_share": float((kl > 0).mean()),
            "entropy_per_response_mean": g["entropy"].mean(), #mean token entropy of each response, averaged over responses
            "length_mean": g["response_tokens"].mean(), "length_std": g["response_tokens"].std(ddof=0),
            "truncation_rate": g["truncated"].mean(),
            "identical_text_vs_reference": int((g["response"] == ref["response"]).sum()),
        })
    return pd.DataFrame(rows), gens


#---- 2. fork replicate analysis
def fork_pairwise(results, gens, runs, n_updates):
    logs = {r: pd.DataFrame(read_jsonl(results / f"train_log_{r}.jsonl")).head(n_updates) for r in runs}
    rows = []
    for a, b in itertools.combinations(runs, 2):
        d = gens[b]["reward"] - gens[a]["reward"]
        la, lb = logs[a], logs[b]
        same_len = la["response_length"].values == lb["response_length"].values
        same_both = same_len & (la["reward"].values == lb["reward"].values) #bit-identical length and raw reward
        rows.append({
            "run_a": a, "run_b": b,
            "heldout_reward_delta_b_minus_a": d.mean(), "heldout_reward_delta_se": se(d),
            "heldout_texts_differ_share": float((gens[a]["response"] != gens[b]["response"]).mean()),
            "train_updates_compared": len(same_len),
            "train_updates_same_length": int(same_len.sum()),
            "train_updates_same_length_and_reward": int(same_both.sum()),
            #first update (1-based) where the rollout length or reward differs, 0 if they never differ
            "train_first_differing_update": int(np.argmax(~same_both) + 1) if (~same_both).any() else 0,
        })
    return pd.DataFrame(rows)


#---- 3. did any ratio leave the clip band
def clip_band_check(results, cfg, runs):
    rows = []
    for run in runs:
        log = pd.DataFrame(read_jsonl(results / f"train_log_{run}.jsonl"))
        eps = float(cfg["clip_epsilon"]) if run == "standard" else float(run.split("eps")[1].split("_")[0])
        epochs = np.array(log["clip_fraction_epochs"].tolist()) #[updates, ppo_epochs], each at the run's own eps
        max_lr = float(log["max_abs_log_ratio"].max()) #after BOTH epochs vs pi_old, the ratio no loss term ever used
        rows.append({
            "run": run, "eps": eps, "updates": len(log),
            "clip_fraction_epoch1_max": float(epochs[:, 0].max()), #ratio = 1 exactly before any step
            "clip_fraction_epoch2_max": float(epochs[:, 1].max()), #the only epoch where the clip could act
            "ratio_mean_dev_max": float((log["ratio_mean"] - 1).abs().max()),
            "post_update_max_abs_log_ratio": max_lr,
            "post_update_ratio_range": f"{np.exp(-max_lr):.4f}-{np.exp(max_lr):.4f}",
            #does the end-of-update ratio (two steps) ever leave the narrowest band, |log rho| > log(1.05)
            "post_update_outside_0.05_band_updates": int((log["max_abs_log_ratio"] > np.log(1.05)).sum()),
        })
    return pd.DataFrame(rows)


def plot_cached_clip(results, figures):
    #the cached rollout study as clip and affected-token fractions against eps -> the controlled pass (pi_old and pi_new
    #both scored here) next to the two diagnostic readings of the cache (direct, and the reference control whose true
    #ratio is exactly 1). symlog axis so the zeros of the controlled pass stay on the plot
    tables = [("controlled pass (pi_new = one 32-step pass)", "cached_clip_study.csv", INK),
              ("direct (cached pi_old)", "cached_clip_direct.csv", '#eb6834'),
              ("reference control (true ratio 1)", "cached_clip_reference_control.csv", '#9ec5f4')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, (column, title) in zip(axes, [("clip_fraction", "clip fraction (ratio outside the band)"),
                                            ("affected_token_fraction", "affected-token fraction (gradient zeroed)")]):
        for label, file, color in tables:
            t = pd.read_csv(results / file)
            axis.plot(t["eps"], 100 * t[column], color=color, marker='o', markersize=7, label=label)
        axis.set_yscale('symlog', linthresh=1e-2)
        axis.set_xscale('log')
        axis.set_xticks([0.05, 0.2, 0.5], ['0.05', '0.2', '0.5'])
        axis.set_xlabel('clip eps')
        axis.set_ylabel('% of response tokens')
        axis.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "cached_clip_fractions.png", dpi=200)
    plt.close(fig)


#---- 4. what the gae advantage is made of on the cached batch
def cached_batch(cfg):
    #the fixed cached batch as padded [32, T] tensors, exactly as analyze_clipping.cached_rollout_study builds it
    cached = load_cached_rollouts(cfg["cached_rollouts"])
    width = max(len(row["old_logprobs"]) for row in cached)
    old = pad_rows([row["old_logprobs"] for row in cached], width)
    ref = pad_rows([row["ref_logprobs"] for row in cached], width)
    values = pad_rows([row["values"] for row in cached], width)
    mask = pad_rows([torch.ones(len(row["old_logprobs"])) for row in cached], width)
    task_reward = torch.tensor([row["effective_terminal_reward"] for row in cached]) #already includes the missing eos penalty
    return old, ref, values, mask, task_reward


def advantage_parts(cfg, old, ref, values, mask, task_reward, beta):
    #gae is linear in the rewards and in the values, so the advantage splits exactly into three parts ->
    #  reward part : the terminal task reward alone (values 0)          = lambda^(T-1-t) * r_task  for gamma = 1
    #  kl part     : the per token kl fee alone (values 0)
    #  critic part : the values alone (rewards 0)                       = the lambda smoothed change in V along the response
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    zeros = torch.zeros_like(values)
    reward_part, _ = compute_gae(shaped_rewards(task_reward, zeros, zeros, mask, 0.0), zeros, mask, gamma, lam)
    kl_part, _ = compute_gae(shaped_rewards(torch.zeros_like(task_reward), old, ref, mask, beta), zeros, mask, gamma, lam)
    critic_part, _ = compute_gae(zeros, values, mask, gamma, lam)
    full, _ = compute_gae(shaped_rewards(task_reward, old, ref, mask, beta), values, mask, gamma, lam)
    assert torch.allclose(full, reward_part + kl_part + critic_part, atol=1e-4), "gae decomposition does not add up"
    return full, reward_part, kl_part, critic_part


def advantage_decomposition(cfg, figures, results):
    old, ref, values, mask, task_reward = cached_batch(cfg)
    full, reward_part, kl_part, critic_part = advantage_parts(cfg, old, ref, values, mask, task_reward, float(cfg["kl_beta"]))
    lengths = mask.sum(-1).long()
    #distance of every token from the last token of its response, d = T - 1 - t
    distance = (lengths[:, None] - 1 - torch.arange(mask.shape[1])[None, :]).float()
    valid = mask.bool()
    d, a = distance[valid].numpy(), full[valid].numpy()
    r = task_reward[:, None].expand_as(full)[valid].numpy() #the rollout's terminal reward, repeated on its tokens
    critic_wins = (critic_part.abs() > reward_part.abs())[valid].numpy()
    weight = (float(cfg["gae_lambda"]) ** distance)[valid].numpy() #the reward's weight on the token, lambda^d

    def summary(sel, label):
        return {"distance_bin": label, "tokens": int(sel.sum()),
                "critic_part_larger_share": float(critic_wins[sel].mean()),
                "mean_reward_weight_lambda_d": float(weight[sel].mean()),
                "corr_advantage_vs_terminal_reward": float(np.corrcoef(a[sel], r[sel])[0, 1])}

    bins = [(0, 10, "0-9"), (10, 50, "10-49"), (50, 100, "50-99"), (100, 10 ** 6, "100+")]
    rows = [summary((d >= lo) & (d < hi), label) for lo, hi, label in bins] + [summary(np.ones_like(d, bool), "all")]
    table = pd.DataFrame(rows)
    table.to_csv(results / "cached_advantage_decomposition.csv", index=False)

    #finer bins for the figure
    edges = list(range(0, 100, 10)) + [100, 150, 200, 300, int(d.max()) + 1]
    fine = [summary((d >= lo) & (d < hi), f"{lo}-{hi - 1}") | {"mid": (lo + min(hi, 350)) / 2} for lo, hi in zip(edges[:-1], edges[1:])]
    fine = pd.DataFrame(fine)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(fine["mid"], 100 * fine["critic_part_larger_share"], color='#2a78d6', marker='o', label='|critic part| > |reward part|')
    axes[0].plot(fine["mid"], 100 * fine["mean_reward_weight_lambda_d"], color=MUTED, linestyle=':', label='reward weight lambda^d (x100)')
    axes[0].set_xlabel('tokens before the end of the response')
    axes[0].set_ylabel('% of tokens')
    axes[0].set_title('what the gae advantage is made of')
    axes[0].legend(frameon=False)
    axes[1].plot(fine["mid"], fine["corr_advantage_vs_terminal_reward"], color='#eb6834', marker='o')
    axes[1].axhline(0, color=MUTED, linewidth=1)
    axes[1].set_xlabel('tokens before the end of the response')
    axes[1].set_ylabel('correlation')
    axes[1].set_title("token advantage vs the rollout's reward")
    fig.tight_layout()
    fig.savefig(figures / "advantage_vs_distance.png", dpi=200)
    plt.close(fig)
    return table, (old, ref, values, mask, task_reward)


def kl_fee_perturbation(cfg, batch, results):
    #how much does the kl price change the advantages ppo actually uses. each fork update holds ONE response, so the
    #whitening is done per rollout here, as in training (normalize_advantages over the valid tokens of the batch)
    old, ref, values, mask, task_reward = batch
    per_beta = {}
    for beta in cfg["kl_values"]:
        full, _, kl_part, _ = advantage_parts(cfg, old, ref, values, mask, task_reward, float(beta))
        per_beta[float(beta)] = (full, kl_part)
    base, _ = per_beta[0.0]
    rows, sums = [], []
    for beta, (full, kl_part) in per_beta.items():
        rel, corr, kl_std = [], [], []
        for i in range(full.shape[0]):
            m = mask[i:i + 1]
            n0 = normalize_advantages(base[i:i + 1], m)[m.bool()]
            nb = normalize_advantages(full[i:i + 1], m)[m.bool()]
            rel.append(float((nb - n0).norm() / n0.norm()))
            corr.append(float(np.corrcoef(nb.numpy(), n0.numpy())[0, 1]))
            kl_std.append(float(kl_part[i][m[0].bool()].std() / full[i][m[0].bool()].std()))
            if beta == float(cfg["kl_beta"]):
                sums.append(float(nb.sum())) #single-rollout whitening -> the response's advantages sum to 0
        rows.append({"kl_beta": beta, "rollouts": full.shape[0],
                     "rel_l2_change_vs_beta0_median": float(np.median(rel)), "rel_l2_change_vs_beta0_max": float(np.max(rel)),
                     "corr_vs_beta0_min": float(np.min(corr)),
                     "kl_part_std_over_advantage_std_median": float(np.median(kl_std)),
                     "kl_part_std_over_advantage_std_max": float(np.max(kl_std))})
    table = pd.DataFrame(rows)
    table.to_csv(results / "kl_fee_perturbation.csv", index=False)
    #the same batch whitened as ONE batch of 32 (the cached study's choice) keeps a per-response offset, for contrast
    full = per_beta[float(cfg["kl_beta"])][0]
    pooled = normalize_advantages(full, mask)
    pooled_means = (pooled.sum(-1) / mask.sum(-1)).numpy()
    whitening = {
        "single_rollout_whitening_max_abs_sum_over_response": float(np.max(np.abs(sums))),
        "batch32_whitening_per_response_mean_range": [float(pooled_means.min()), float(pooled_means.max())],
        "corr_batch32_response_mean_vs_terminal_reward": float(np.corrcoef(pooled_means, task_reward.numpy())[0, 1]),
    }
    save_json(results / "cached_whitening_check.json", whitening)
    return table, whitening


#---- 5. reward model behaviour on the held-out text of every task 2 policy
def rm_length_truncation(gens, figures, results):
    pooled = pd.concat([g.assign(policy=name) for name, g in gens.items()], ignore_index=True)
    #one row per distinct (prompt, text) -> identical texts from different policies are averaged, not counted twice
    distinct = pooled.groupby(["prompt_id", "response"], as_index=False).agg(
        reward=("reward", "mean"), tokens=("response_tokens", "first"), truncated=("truncated", "first"))

    #length slope within the same prompt, finished responses only -> prompt fixed effects by demeaning
    fin = distinct[~distinct["truncated"]].copy()
    fin["x"] = np.log2(fin["tokens"])
    fin = fin[fin.groupby("prompt_id")["x"].transform("count") > 1] #a prompt with one text carries no within-prompt info
    fin["xd"] = fin["x"] - fin.groupby("prompt_id")["x"].transform("mean")
    fin["yd"] = fin["reward"] - fin.groupby("prompt_id")["reward"].transform("mean")
    slope = float((fin["xd"] * fin["yd"]).sum() / (fin["xd"] ** 2).sum())
    resid = fin["yd"] - slope * fin["xd"]
    dof = len(fin) - fin["prompt_id"].nunique() - 1
    slope_se = float(np.sqrt((resid ** 2).sum() / dof / (fin["xd"] ** 2).sum()))

    #truncated minus finished, inside prompts that have both kinds
    per_prompt = distinct.groupby(["prompt_id", "truncated"])["reward"].mean().unstack()
    both = per_prompt.dropna()
    gap = both[True] - both[False]
    rows = [{"statistic": "within_prompt_reward_per_doubling_of_length", "value": slope, "se": slope_se,
             "n": len(fin), "prompts": int(fin["prompt_id"].nunique()),
             "corr": float(np.corrcoef(fin["xd"], fin["yd"])[0, 1])},
            {"statistic": "within_prompt_truncated_minus_finished", "value": float(gap.mean()), "se": se(gap),
             "n": int(len(gap)), "prompts": int(len(gap)), "share_negative": float((gap < 0).mean())},
            {"statistic": "pooled_mean_reward_finished", "value": float(pooled.loc[~pooled["truncated"], "reward"].mean()),
             "n": int((~pooled["truncated"]).sum())},
            {"statistic": "pooled_mean_reward_truncated", "value": float(pooled.loc[pooled["truncated"], "reward"].mean()),
             "n": int(pooled["truncated"].sum())}]
    table = pd.DataFrame(rows)
    table.to_csv(results / "rm_length_truncation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].scatter(fin["xd"], fin["yd"], s=10, color='#2a78d6', alpha=0.35, edgecolor='none')
    xs = np.linspace(fin["xd"].min(), fin["xd"].max(), 10)
    axes[0].plot(xs, slope * xs, color=INK, label=f'{slope:+.2f} reward per doubling')
    axes[0].set_xlabel('log2 length, relative to the prompt mean')
    axes[0].set_ylabel('reward, relative to the prompt mean')
    axes[0].set_title('finished responses, same prompt')
    axes[0].legend(frameon=False)
    axes[1].hist(gap, bins=15, color='#eb6834', edgecolor='#fcfcfb')
    axes[1].axvline(0, color=MUTED, linewidth=1)
    axes[1].axvline(gap.mean(), color=INK, linestyle='--', label=f'mean {gap.mean():+.2f}')
    axes[1].set_xlabel('truncated minus finished reward, same prompt')
    axes[1].set_ylabel('prompts')
    axes[1].set_title('unfinished text (no eos penalty applied)')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "reward_vs_length.png", dpi=200)
    plt.close(fig)
    return table


#---- 6. standard trajectory and critic
def trajectory_trend(results, runs, n_shared):
    log = pd.DataFrame(read_jsonl(results / "train_log_standard.jsonl"))
    x, y = log["update"].to_numpy(float), log["reward"].to_numpy(float)
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    resid = y - y.mean() - slope * xc
    slope_se = float(np.sqrt((resid ** 2).sum() / (len(y) - 2) / (xc ** 2).sum()))
    half = len(y) // 2
    first, second = y[:half], y[half:]
    #updates 1..n_shared answer the same prompt in every run -> between-update variance (prompt identity) against the
    #variance across runs inside an update (sampling + run differences)
    shared = np.array([pd.DataFrame(read_jsonl(results / f"train_log_{r}.jsonl"))["reward"].head(n_shared).to_numpy() for r in runs])
    between = float(shared.mean(0).var(ddof=1))
    within = float(shared.var(0, ddof=1).mean())
    critic = pd.read_csv(results / "standard_critic.csv")
    out = {
        "reward_slope_per_update": slope, "reward_slope_se": slope_se,
        "first_half_mean": float(first.mean()), "second_half_mean": float(second.mean()),
        "half_difference": float(second.mean() - first.mean()),
        "half_difference_se": float(np.sqrt(first.var(ddof=1) / len(first) + second.var(ddof=1) / len(second))),
        "shared_updates": n_shared, "shared_runs": len(runs),
        "between_update_variance": between, "within_update_variance_across_runs": within,
        "prompt_identity_variance_share": between / (between + within),
        "critic_explained_variance_median": float(critic["explained_variance"].median()),
        "critic_explained_variance_range": [float(critic["explained_variance"].min()), float(critic["explained_variance"].max())],
        "critic_corr_value_vs_reward_effective": float(critic["value_mean"].corr(critic["reward_effective"])),
        "critic_value_grad_norm_range": [float(critic["value_grad_norm"].min()), float(critic["value_grad_norm"].max())],
        "critic_value_grad_norm_mean": float(critic["value_grad_norm"].mean()),
        "critic_value_loss_first5_mean": float(critic["value_loss"].head(5).mean()),
        "critic_value_loss_last5_mean": float(critic["value_loss"].tail(5).mean()),
    }
    save_json(results / "trajectory_trend.json", out)
    return out


#---- 7. how differently the reward model scores the SAME text in two evaluation runs
def identical_text_noise(gens, results):
    rows, all_d = [], []
    for a, b in itertools.combinations(gens, 2):
        same = gens[a]["response"] == gens[b]["response"] #same prompt (same row), word-for-word the same response
        d = (gens[b]["reward"] - gens[a]["reward"])[same].to_numpy()
        all_d.append(d)
        if len(d):
            rows.append({"run_a": a, "run_b": b, "n": len(d), "mean": d.mean(), "sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                         "mean_abs": np.abs(d).mean(), "max_abs": np.abs(d).max()})
    d = np.concatenate(all_d)
    #pooled over every pair of runs. a text shared by k runs enters k(k-1)/2 times, so these are not independent draws
    rows.append({"run_a": "all", "run_b": "all", "n": len(d), "mean": d.mean(), "sd": d.std(ddof=1),
                 "mean_abs": np.abs(d).mean(), "max_abs": np.abs(d).max()})
    table = pd.DataFrame(rows)
    table.to_csv(results / "reward_noise_identical_text.csv", index=False)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    names, forks = policy_names(cfg)
    heldout, gens = heldout_table(results, names)
    heldout.to_csv(results / "report_heldout.csv", index=False)
    print(heldout.round(4).to_string(index=False))

    runs = ["standard"] + forks
    n_updates = int(cfg["fork_updates"]) #the standard run shares its first 8 updates (and prompts) with every fork
    pairwise = fork_pairwise(results, gens, runs, n_updates)
    pairwise.to_csv(results / "fork_pairwise.csv", index=False)
    print(pairwise.round(4).to_string(index=False))
    eps_forks = [fork_name(float(e), float(cfg["kl_beta"])) for e in cfg["clip_values"]]
    eps_means = heldout.set_index("policy").loc[eps_forks, "reward_mean"]
    fork_means = heldout.set_index("policy").loc[forks, "reward_mean"]
    spread = {"eps_fork_heldout_reward_spread": float(eps_means.max() - eps_means.min()),
              "all_fork_heldout_reward_spread": float(fork_means.max() - fork_means.min()),
              "fork_pair_texts_differ_range": [float(pairwise["heldout_texts_differ_share"].min()),
                                               float(pairwise["heldout_texts_differ_share"].max())]}
    save_json(results / "fork_replicate_spread.json", spread)
    print(spread)

    band = clip_band_check(results, cfg, runs)
    band.to_csv(results / "clip_band_check.csv", index=False)
    print(band.to_string(index=False))
    plot_cached_clip(results, figures)

    decomposition, batch = advantage_decomposition(cfg, figures, results)
    print(decomposition.round(4).to_string(index=False))
    perturbation, whitening = kl_fee_perturbation(cfg, batch, results)
    print(perturbation.round(5).to_string(index=False))
    print(whitening)

    print(rm_length_truncation(gens, figures, results).round(4).to_string(index=False))
    print(trajectory_trend(results, runs, n_updates))
    noise = identical_text_noise(gens, results)
    print(noise[noise["run_a"] == "all"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
