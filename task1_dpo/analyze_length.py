from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json
from common.metrics import safe_corr
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.evaluate import heldout_pair_metrics, run_evaluation
from task1_dpo.train import filter_rows, run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def dataset_length_diagnostics(rows):
    #a property of the DATA, not of any model -> how often the preferred response is the longer one, and how strongly
    #the length gap tracks the preference. the standard file is compared against the balanced one
    gap = np.array([len(r["chosen"][-1]["content"].split()) - len(r["rejected"][-1]["content"].split()) for r in rows])
    return {
        "n_pairs": len(rows),
        "frac_chosen_longer": float(np.mean(gap > 0)),
        "mean_word_gap_chosen_minus_rejected": float(gap.mean()),
        "median_word_gap": float(np.median(gap)),
    }


def stratified_accuracy(config_path, adapter, name, beta):
    #teacher forced preference accuracy on the supplied length-stratified held-out set (82 pairs per stratum before
    #the long-prompt filter), reported separately for each stratum
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_rows(tokenizer, read_jsonl(cfg["paths"]["dpo_length_eval"]), int(cfg["max_sequence_length"]))
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    pairs = heldout_pair_metrics(policy, tokenizer, rows, cfg, beta)
    clear_gpu(policy)
    write_jsonl(f"{cfg['results_dir']}/stratified_{name}_pairs.jsonl", pairs)

    table = []
    for stratum in STRATA:
        sub = [p for p in pairs if p["length_stratum"] == stratum]
        table.append({
            "model": name,
            "stratum": stratum,
            "n_pairs": len(sub),
            "preference_accuracy": float(np.mean([p["correct"] for p in sub])),
            "margin_mean": float(np.mean([p["margin"] for p in sub])),
            "dpo_loss": float(np.mean([p["dpo_loss"] for p in sub])),
        })
    #does the learned margin follow the length gap of the pair? positive -> longer chosen responses get bigger margins
    length_gap = [p["chosen_tokens"] - p["rejected_tokens"] for p in pairs]
    table.append({
        "model": name, "stratum": "all", "n_pairs": len(pairs),
        "preference_accuracy": float(np.mean([p["correct"] for p in pairs])),
        "margin_mean": float(np.mean([p["margin"] for p in pairs])),
        "dpo_loss": float(np.mean([p["dpo_loss"] for p in pairs])),
        "corr_margin_vs_length_gap": safe_corr([p["margin"] for p in pairs], length_gap),
    })
    return table, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--standard-adapter") #defaults to the standard run's output directory
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = cfg["results_dir"]
    beta = float(cfg["beta"])
    standard = args.standard_adapter or cfg["standard_output"]
    balanced = cfg["length_output"]

    balanced_rows = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced_rows))
    print("Length-stratified eval rows:", len(stratified))

    if not (repo_path(standard) / "adapter_config.json").exists():
        raise SystemExit(f"standard dpo adapter missing at {standard}, run task1_dpo.train first")

    #1 -> the length structure of the two TRAINING files (dataset property, no model involved)
    save_json(f"{out_dir}/length_dataset_diagnostics.json", {
        "standard_train": dataset_length_diagnostics(read_jsonl(cfg["paths"]["dpo_standard_train"])),
        "length_balanced_train": dataset_length_diagnostics(balanced_rows),
    })

    #2 -> one extra dpo model, from the same fresh initialization and with the standard recipe (one epoch, beta 0.1),
    #but on the length-balanced file where each of the three strata makes up exactly a third of the pairs
    if (repo_path(balanced) / "adapter_config.json").exists():
        print("[length_balanced] adapter already exists, skipping training")
    else:
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                     output_path=balanced, beta=beta)

    #3 -> full standard evaluation of the balanced model (held-out pairs, generated length, reward, kl, word limits),
    #so it lines up with the standard model's eval_standard.json on the common prompt set
    run_evaluation(args.config, balanced, "length_balanced", beta=beta)

    #4 -> per-stratum preference accuracy, standard vs length-balanced, on the length-stratified held-out set
    table = []
    for name, adapter in [("standard", standard), ("length_balanced", balanced)]:
        rows, dropped = stratified_accuracy(args.config, adapter, name, beta)
        table += rows
    strata = pd.DataFrame(table)
    strata.to_csv(repo_path(f"{out_dir}/length_strata_accuracy.csv"), index=False)
    print(strata.to_string(index=False))

    #5 -> generated length and word-limit compliance on the common prompt set, read back from the two eval files
    compare = []
    for name in ["standard", "length_balanced"]:
        ev = load_json(f"{out_dir}/eval_{name}.json")
        compare.append({
            "model": name,
            "heldout_length_mean": ev["length_mean"],
            "heldout_length_std": ev["length_std"],
            "word_limit_compliance": ev["word_limit"]["compliance_rate"],
            "word_limit_words_mean": ev["word_limit"]["words_mean"],
            "word_limit_tokens_mean": ev["word_limit"]["tokens_mean"],
            "word_limit_tokens_std": ev["word_limit"]["tokens_std"],
        })
    compare = pd.DataFrame(compare)
    compare.to_csv(repo_path(f"{out_dir}/length_generation_compare.csv"), index=False)
    print(compare.to_string(index=False))
    save_json(f"{out_dir}/length_stratified_dropped_prompt_ids.json", dropped)


if __name__ == "__main__":
    main()
