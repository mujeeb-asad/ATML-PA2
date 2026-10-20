from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path

POLICIES = ["sft", "rlvr", "rlaif"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    #pure table work, no model -> joins what evaluate_math (gsm and transfer) and score_perturbations already saved
    gsm = pd.read_csv(outdir / "summary_gsm.csv").set_index("policy")
    transfer = pd.read_csv(outdir / "summary_transfer.csv").set_index("policy")

    #1 -> the policies, in-domain (gsm8k) next to out-of-domain (svamp), with the drop where a drop is meaningful
    rows = []
    for name in POLICIES:
        row = {"policy": name}
        for column in ["exact_accuracy", "format_compliance", "length_mean", "truncation_rate", "win_rate_vs_sft"]:
            row[f"gsm_{column}"] = gsm.loc[name].get(column)
            row[f"transfer_{column}"] = transfer.loc[name].get(column)
        row["accuracy_drop"] = row["gsm_exact_accuracy"] - row["transfer_exact_accuracy"]
        row["format_drop"] = row["gsm_format_compliance"] - row["transfer_format_compliance"]
        row["win_rate_vs_sft_drop"] = (row["gsm_win_rate_vs_sft"] - row["transfer_win_rate_vs_sft"]) if name != "sft" else None
        #each trained policy's accuracy relative to the sft baseline on the same set
        row["gsm_accuracy_gain_over_sft"] = row["gsm_exact_accuracy"] - gsm.loc["sft", "exact_accuracy"]
        row["transfer_accuracy_gain_over_sft"] = row["transfer_exact_accuracy"] - transfer.loc["sft", "exact_accuracy"]
        rows.append(row)
    policies = pd.DataFrame(rows)
    policies.to_csv(outdir / "feedback_policy_comparison.csv", index=False)
    print(policies.set_index("policy").T.round(4).to_string())

    #2 -> failure types of every policy on both sets, as shares of that set
    failure_columns = sorted(c for c in set(gsm.columns) | set(transfer.columns) if c.startswith("failure_"))
    failures = pd.concat([gsm.reindex(columns=failure_columns).assign(dataset="gsm"),
                          transfer.reindex(columns=failure_columns).assign(dataset="transfer")]).fillna(0.0).reset_index()
    failures.to_csv(outdir / "feedback_failure_types.csv", index=False)
    print(failures.round(4).to_string(index=False))

    #3 -> verifier-judge agreement on the policies' own responses, both sets
    agreement = pd.concat([pd.read_csv(outdir / "agreement_gsm.csv"), pd.read_csv(outdir / "agreement_transfer.csv")])
    agreement.to_csv(outdir / "feedback_agreement.csv", index=False)
    print(agreement.round(4).to_string(index=False))

    #4 -> the two feedback mechanisms on the controlled diagnostic set, one row per mechanism
    mechanisms = pd.read_csv(outdir / "diagnostic_sensitivity.csv")
    rewards = pd.read_csv(outdir / "diagnostic_group_rewards.csv").groupby("variant_type")[["rlvr_reward", "rlaif_reward"]].mean()
    for label, column in [("rlvr_exact_verifier", "rlvr_reward"), ("rlaif_pairwise_judge", "rlaif_reward")]:
        for variant_type, value in rewards[column].items():
            mechanisms.loc[mechanisms["mechanism"] == label, f"mean_reward_{variant_type}"] = value
    mechanisms.to_csv(outdir / "feedback_mechanism_comparison.csv", index=False)
    print(mechanisms.set_index("mechanism").T.round(4).to_string())


if __name__ == "__main__":
    main()
