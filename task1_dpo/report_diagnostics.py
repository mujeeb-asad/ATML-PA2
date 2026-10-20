from __future__ import annotations

import argparse
import itertools

import matplotlib
matplotlib.use("Agg") #no display needed, the figures are only written to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.rl_eval import token_kl

#report diagnostics for task 1 -> pure table work on the files the training and evaluation scripts already saved.
#no model is loaded, nothing is retrained, every number here is recomputed from results/task1_dpo

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]
#same model colours and plot style as task1.ipynb, so the new figures sit next to the old ones
COLORS = {"base": "#52514e", "standard": "#2a78d6", "length_balanced": "#eb6834",
          "beta_0.03": "#1baf7a", "beta_0.1": "#eda100", "beta_0.3": "#e87ba4"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2})


def se(x):
    #standard error of a mean, sample std with ddof 1
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def per_response_kl(records):
    #a secondary view kept for comparison -> sampled_kl applied to ONE response at a time (seq_kl / response_tokens),
    #then averaged over responses. the report uses the manual's convention instead, the course helper pooled over every
    #generated token (token_kl in common/rl_eval.py, the same number as kl_token_mean in eval_*.json)
    return np.array([r["seq_kl"] / max(r["response_tokens"], 1) for r in records])


def paired(a, b):
    #per prompt paired difference b - a on the same prompts, in the same file order, with its standard error
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return float(d.mean()), se(d)


def pertoken_correct(p, beta):
    #per token margin -> the implicit log ratios divided by the response length, so a long response no longer
    #carries more summed log-ratio than a short one. beta > 0 only rescales, so the sign does not depend on it
    chosen = p["chosen_reward"] / beta / max(p["chosen_tokens"], 1)
    rejected = p["rejected_reward"] / beta / max(p["rejected_tokens"], 1)
    return chosen - rejected > 0


def summary_table(cfg, runs, ev, gens, pairs, wl):
    rows = []
    base_gen = gens["base"]
    base_kl = per_response_kl(base_gen)
    for name in runs:
        train = load_json(f"{cfg['results_dir']}/train_summary_{name}.json") if name != "base" else {}
        kl = per_response_kl(gens[name])
        kl_diff, kl_diff_se = paired(base_kl, kl)
        reward_diff, reward_diff_se = paired([r["reward"] for r in base_gen], [r["reward"] for r in gens[name]])
        trunc_base = np.array([r["truncated"] for r in base_gen], dtype=int)
        trunc = np.array([r["truncated"] for r in gens[name]], dtype=int)
        correct = np.array([p["correct"] for p in pairs[name]], dtype=float)
        std_correct = np.array([p["correct"] for p in pairs["standard"]], dtype=float)
        acc_diff, acc_diff_se = paired(std_correct, correct)
        lengths = np.array([r["response_tokens"] for r in gens[name]])
        rows.append({
            "model": name,
            "beta": ev[name]["beta"] if name != "base" else np.nan,
            "train_pairs": train.get("n_pairs", 0),
            "optimizer_steps": train.get("optimizer_steps", 0),
            "heldout_dpo_loss": ev[name]["heldout_dpo_loss"],
            #base == reference, so its margin is exactly 0 and accuracy (margin > 0) is undefined, not 0
            "preference_accuracy": correct.mean() if name != "base" else np.nan,
            "preference_accuracy_se": se(correct) if name != "base" else np.nan,
            #paired over the same 290 pairs, model minus standard
            "accuracy_diff_vs_standard": acc_diff if name not in ("base", "standard") else np.nan,
            "accuracy_diff_vs_standard_se": acc_diff_se if name not in ("base", "standard") else np.nan,
            "margin_mean": ev[name]["heldout_margin_mean"],
            #the report's kl -> the course helper pooled over every generated token, with a per-prompt standard error
            "kl_token": token_kl(gens[name])[0],
            "kl_token_se": token_kl(gens[name])[1] if name != "base" else 0.0,
            "kl_per_response_mean": kl.mean(),
            "kl_per_response_se": se(kl),
            "kl_per_response_diff_vs_base_se": kl_diff_se, #paired, base kl is 0 by construction so the mean equals the row
            "reward_mean": float(np.mean([r["reward"] for r in gens[name]])),
            "reward_diff_vs_base": reward_diff,
            "reward_diff_vs_base_se": reward_diff_se,
            "length_mean": lengths.mean(),
            "length_std": lengths.std(),
            "truncation_rate": trunc.mean(),
            #paired change in how many of the 290 responses hit the 256 cap -> net, and how many flipped each way
            "truncation_net_change_vs_base": int(trunc.sum() - trunc_base.sum()),
            "truncation_newly_truncated": int(((trunc == 1) & (trunc_base == 0)).sum()),
            "truncation_newly_finished": int(((trunc == 0) & (trunc_base == 1)).sum()),
            "word_limit_compliant": int(sum(r["compliant"] for r in wl[name])),
            "word_limit_n": len(wl[name]),
        })
    return pd.DataFrame(rows)


def beta_tables(cfg, beta_runs, ev, gens, pairs):
    max_norm = float(cfg["max_grad_norm"])
    rows = []
    for name in ["standard"] + beta_runs:
        log = read_jsonl(f"{cfg['results_dir']}/train_log_{name}.jsonl")
        margins = np.array([p["margin"] for p in pairs[name]])
        norms = np.array([r["grad_norm"] for r in log])
        rows.append({
            "model": name,
            "beta": ev[name]["beta"],
            "optimizer_steps": len(log),
            "margin_mean": margins.mean(),
            "margin_se": se(margins),
            "preference_accuracy": float(np.mean(margins > 0)),
            "dpo_loss_own_beta": ev[name]["heldout_dpo_loss"],
            #the same saved margins re-scored at one common beta -> if the forks' losses then agree, the loss trend
            #across beta is the arithmetic of beta multiplying the margin, not a difference in what was learned
            "dpo_loss_at_beta_0.1": float(np.mean(np.logaddexp(0.0, -0.1 * margins))), #-log sigmoid(0.1 m), stable form
            "grad_norm_median": float(np.median(norms)), #pre-clip norm, clip_grad_norm_ returns it before clipping
            "frac_steps_clipped": float(np.mean(norms > max_norm)),
            "train_margin_last10": float(np.mean([r["policy_margin"] - r["ref_margin"] for r in log[-10:]])),
            "train_acc_last10": float(np.mean([r["acc"] for r in log[-10:]])),
        })
    pairwise = []
    for a, b in itertools.combinations(["standard"] + beta_runs, 2):
        pairwise.append({
            "model_a": a,
            "model_b": b,
            "margin_corr": safe_corr([p["margin"] for p in pairs[a]], [p["margin"] for p in pairs[b]]),
            #same prompts, same seed stream -> a response is identical unless the two policies flipped a sampled token
            "identical_generations": int(sum(x["response"] == y["response"] for x, y in zip(gens[a], gens[b]))),
            "n_prompts": len(gens[a]),
        })
    return pd.DataFrame(rows), pd.DataFrame(pairwise)


def strata_tables(cfg, runs, ev, pairs):
    rows = []
    #(1) the supplied length-stratified held-out set, standard and length-balanced only (the models analyze_length scored)
    for name in ["standard", "length_balanced"]:
        stratified = read_jsonl(f"{cfg['results_dir']}/stratified_{name}_pairs.jsonl")
        beta = ev[name]["beta"]
        for stratum in STRATA + ["all"]:
            sub = [p for p in stratified if stratum == "all" or p["length_stratum"] == stratum]
            rows.append({"eval_set": "stratified", "model": name, "split": stratum, "n_pairs": len(sub),
                         "accuracy_summed": float(np.mean([p["correct"] for p in sub])),
                         "accuracy_per_token": float(np.mean([pertoken_correct(p, beta) for p in sub]))})
    #(2) the 290 standard held-out pairs for every trained model, split by which response is longer in tokens
    for name in runs:
        if name == "base":
            continue #margin 0 everywhere, nothing to split
        beta = ev[name]["beta"]
        for split, keep in [("chosen_longer", lambda p: p["chosen_tokens"] > p["rejected_tokens"]),
                            ("chosen_shorter", lambda p: p["chosen_tokens"] < p["rejected_tokens"]),
                            ("all", lambda p: True)]:
            sub = [p for p in pairs[name] if keep(p)]
            rows.append({"eval_set": "heldout_290", "model": name, "split": split, "n_pairs": len(sub),
                         "accuracy_summed": float(np.mean([p["correct"] for p in sub])),
                         "accuracy_per_token": float(np.mean([pertoken_correct(p, beta) for p in sub]))})
    return pd.DataFrame(rows)


def implicit_reward_table(runs, ev, pairs):
    #the two implicit log ratios log pi(y)/ref(y) that dpo pushes apart, summed and per token. dpo only needs their
    #gap to grow, so both may fall together, which is what a negative chosen log ratio shows
    rows = []
    for name in runs:
        if name == "base":
            continue
        beta = ev[name]["beta"]
        chosen = np.array([p["chosen_reward"] / beta for p in pairs[name]])
        rejected = np.array([p["rejected_reward"] / beta for p in pairs[name]])
        chosen_tok = chosen / np.array([max(p["chosen_tokens"], 1) for p in pairs[name]])
        rejected_tok = rejected / np.array([max(p["rejected_tokens"], 1) for p in pairs[name]])
        rows.append({
            "model": name,
            "chosen_logratio_mean": chosen.mean(),
            "rejected_logratio_mean": rejected.mean(),
            "common_mode_logratio_mean": ((chosen + rejected) / 2).mean(), #the shared drift of both responses
            "chosen_logratio_per_token_mean": chosen_tok.mean(),
            "rejected_logratio_per_token_mean": rejected_tok.mean(),
            "share_chosen_logratio_negative": float(np.mean(chosen < 0)),
            "share_rejected_logratio_negative": float(np.mean(rejected < 0)),
            "corr_margin_vs_token_gap": safe_corr([p["margin"] for p in pairs[name]],
                                                  [p["chosen_tokens"] - p["rejected_tokens"] for p in pairs[name]]),
        })
    return pd.DataFrame(rows)


def word_limit_tables(runs, wl):
    #compliant samples out of 8 for each of the 10 word-limit prompts, per model
    prompt_ids = list(dict.fromkeys(r["prompt_id"] for r in wl["base"]))
    table = pd.DataFrame({name: [int(sum(r["compliant"] for r in wl[name] if r["prompt_id"] == pid)) for pid in prompt_ids]
                          for name in runs}, index=prompt_ids)
    table.index.name = "prompt_id"
    #paired test base vs standard at the PROMPT level (the 8 samples of one prompt are not independent) ->
    #exact sign-flip test on the 10 per-prompt differences: flip the sign of each difference at random, how often is
    #|mean| at least the observed one. zero differences carry no sign and are kept as zeros
    d = (table["standard"] - table["base"]).to_numpy()
    flips = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    observed = abs(d.mean())
    p_value = float(np.mean(np.abs((flips * d).mean(axis=1)) >= observed - 1e-12))
    test = {
        "comparison": "standard minus base, compliant samples per prompt",
        "per_prompt_difference": d.tolist(),
        "prompts_down": int((d < 0).sum()),
        "prompts_up": int((d > 0).sum()),
        "prompts_unchanged": int((d == 0).sum()),
        "total_compliant_base": int(table["base"].sum()),
        "total_compliant_standard": int(table["standard"].sum()),
        "sign_flip_p_two_sided": p_value,
    }
    return table.reset_index(), test


def identical_text_noise(runs, gens):
    #the same response text, scored by the reward model in two different evaluation runs, should get the same score.
    #the spread of those differences is a measured noise floor for a single reward score
    rows, pooled = [], []
    for a, b in itertools.combinations(runs, 2):
        d = [y["reward"] - x["reward"] for x, y in zip(gens[a], gens[b]) if x["response"] == y["response"]]
        pooled += d
        if d:
            rows.append({"run_a": a, "run_b": b, "n_identical": len(d), "mean_diff": float(np.mean(d)),
                         "sd_diff": float(np.std(d)), "mean_abs_diff": float(np.mean(np.abs(d))),
                         "max_abs_diff": float(np.max(np.abs(d)))})
    rows.append({"run_a": "all", "run_b": "all", "n_identical": len(pooled), "mean_diff": float(np.mean(pooled)),
                 "sd_diff": float(np.std(pooled)), "mean_abs_diff": float(np.mean(np.abs(pooled))),
                 "max_abs_diff": float(np.max(np.abs(pooled)))})
    return pd.DataFrame(rows)


def plot_strata(strata, figures):
    #summed vs per-token preference accuracy per length stratum, one panel per model. the summed margin is what
    #dpo optimises, the per-token one removes the length of each response from it
    frame = strata[(strata["eval_set"] == "stratified") & (strata["split"].isin(STRATA))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    positions = np.arange(len(STRATA))
    for axis, name in zip(axes, ["standard", "length_balanced"]):
        sub = frame[frame["model"] == name].set_index("split").loc[STRATA]
        summed = axis.bar(positions - 0.2, sub["accuracy_summed"], width=0.36, color=COLORS[name], label="summed log ratio")
        per_tok = axis.bar(positions + 0.2, sub["accuracy_per_token"], width=0.36, color=COLORS[name], alpha=0.45,
                           hatch="//", edgecolor="white", label="per-token log ratio")
        axis.bar_label(summed, fmt="%.2f", padding=3, color="#0b0b0b")
        axis.bar_label(per_tok, fmt="%.2f", padding=3, color="#0b0b0b")
        axis.axhline(0.5, color="#52514e", linestyle=":", linewidth=1) #a coin flip
        axis.set_xticks(positions, [stratum.replace("_", " ") for stratum in STRATA])
        axis.set_ylim(0, 1.08)
        axis.set_title(f"{name.replace('_', '-')} model")
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("held-out preference accuracy")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(figures / "strata_summed_vs_pertoken.png", dpi=200)
    plt.close(fig)


def plot_kl(summary, figures):
    #token-level kl from the reference (the course helper's convention) with its standard error, one point per model,
    #base is 0 by construction
    fig, axis = plt.subplots(figsize=(8, 4.2))
    names = summary["model"].tolist()
    for i, row in summary.iterrows():
        axis.errorbar(i, row["kl_token"], yerr=row["kl_token_se"], fmt="o", markersize=8,
                      color=COLORS[row["model"]], capsize=4)
    axis.axhline(0, color="#52514e", linestyle=":", linewidth=1)
    axis.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
    axis.set_ylabel("token mean of log pi - log ref")
    axis.set_title("kl from the reference, token-level course helper (± 1 se)")
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(figures / "token_kl.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["results_dir"])
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    beta_runs = [f"beta_{float(beta):g}" for beta in cfg["betas"]]
    runs = ["base", "standard"] + beta_runs + ["length_balanced"]
    #everything the evaluation scripts saved, one entry per model. every file holds the same prompts in the same order
    ev = {name: load_json(f"{cfg['results_dir']}/eval_{name}.json") for name in runs}
    gens = {name: read_jsonl(f"{cfg['results_dir']}/eval_{name}_generations.jsonl") for name in runs}
    pairs = {name: read_jsonl(f"{cfg['results_dir']}/eval_{name}_pairs.jsonl") for name in runs}
    wl = {name: read_jsonl(f"{cfg['results_dir']}/eval_{name}_word_limit.jsonl") for name in runs}
    for name in runs: #the paired statistics below assume identical prompt order in every file
        assert [r["prompt_id"] for r in gens[name]] == [r["prompt_id"] for r in gens["base"]]
        assert [p["prompt_id"] for p in pairs[name]] == [p["prompt_id"] for p in pairs["base"]]

    #1 -> the report's summary table with the per-response kl convention and paired uncertainties
    summary = summary_table(cfg, runs, ev, gens, pairs, wl)
    summary.to_csv(out_dir / "report_summary.csv", index=False)
    print(summary.round(4).T.to_string())

    #2 -> why beta did not act: gradient scale, clipping, and the loss re-scored at a common beta
    beta_table, beta_pairwise = beta_tables(cfg, beta_runs, ev, gens, pairs)
    beta_table.to_csv(out_dir / "beta_diagnostics.csv", index=False)
    beta_pairwise.to_csv(out_dir / "beta_pairwise.csv", index=False)
    print(beta_table.round(4).to_string(index=False))
    print(beta_pairwise.round(4).to_string(index=False))

    #3 -> summed vs per-token margin by length, and the implicit rewards behind it
    strata = strata_tables(cfg, runs, ev, pairs)
    strata.to_csv(out_dir / "pertoken_strata_accuracy.csv", index=False)
    print(strata.round(4).to_string(index=False))
    implicit = implicit_reward_table(runs, ev, pairs)
    implicit.to_csv(out_dir / "implicit_reward_diagnostics.csv", index=False)
    print(implicit.round(4).to_string(index=False))

    #4 -> word-limit compliance per prompt and the prompt-level paired test
    word_table, word_test = word_limit_tables(runs, wl)
    word_table.to_csv(out_dir / "word_limit_per_prompt.csv", index=False)
    save_json(out_dir / "word_limit_test.json", word_test)
    print(word_table.to_string(index=False))
    print(word_test)

    #5 -> reward-model noise on identical texts
    noise = identical_text_noise(runs, gens)
    noise.to_csv(out_dir / "reward_noise_identical_text.csv", index=False)
    print(noise.round(4).to_string(index=False))

    plot_strata(strata, figures)
    plot_kl(summary, figures)


if __name__ == "__main__":
    main()
