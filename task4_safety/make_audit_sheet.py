from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

POLICIES = ["sft", "dpo", "ppo", "grpo"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def build_audit_sheet(outdir, ids, seed):
    #every policy's response to the audited prompts, read from the GENERATED files (not the judged ones), so no ai
    #label is anywhere near the sheet
    rows = []
    for policy in POLICIES:
        for row in read_jsonl(outdir / f"generated_{policy}.jsonl"):
            if row["xstest_id"] in ids:
                rows.append({"xstest_id": row["xstest_id"], "policy": policy, "prompt": row["prompt"], "response": row["response"]})
    responses = pd.DataFrame(rows)

    #identical answers to the same prompt are audited once -> one sheet row per distinct (prompt, response) text.
    #the rows are shuffled with the fixed seed, and the sheet shows neither the policy nor the benchmark class, so
    #a label cannot be influenced by knowing which model wrote the response or what the benchmark calls the prompt
    unique = responses.drop_duplicates(["xstest_id", "response"])[["xstest_id", "prompt", "response"]]
    unique = unique.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    unique.insert(0, "audit_row", range(1, len(unique) + 1))

    #the key maps every sheet row back to the prompt id and to each policy that produced that exact response
    key = responses.merge(unique[["audit_row", "xstest_id", "response"]], on=["xstest_id", "response"])
    key = key[["audit_row", "xstest_id", "policy"]].sort_values(["audit_row", "policy"])

    sheet = unique[["audit_row", "prompt", "response"]].copy()
    sheet["manual_label"] = ""
    return sheet, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    #the fixed audit subset -> 30 safe and 30 unsafe prompt ids drawn with the course seed
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    sheet_path = outdir / "manual_audit_sheet.csv"
    #a sheet that already carries labels is never overwritten, the hand labels are the one thing that cannot be regenerated
    if sheet_path.exists() and pd.read_csv(sheet_path, dtype=str)["manual_label"].fillna("").str.strip().ne("").any():
        print("Audit sheet already has manual labels, leaving it untouched:", sheet_path)
        return
    sheet, key = build_audit_sheet(outdir, set(ids), int(cfg["seed"]))
    sheet.to_csv(sheet_path, index=False)
    key.to_csv(outdir / "manual_audit_key.csv", index=False)
    print(f"Wrote blind audit sheet with {len(sheet)} distinct responses ({len(key)} policy responses) to {sheet_path}")
    print("Fill the manual_label column with one of SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")
    print("before opening any judged_*.jsonl file, then run task4_safety.evaluate_safety again.")


if __name__ == "__main__":
    main()
