from __future__ import annotations

import argparse
from pathlib import Path

from common.data import load_yaml, repo_path
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training


def beta_run_name(beta):
    return f"beta_{beta:g}" #0.03 -> beta_0.03, 0.1 -> beta_0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])

    #regularization strength study -> three matched short forks that differ ONLY in beta. each one starts from the
    #same fresh base + lora initialization (not from the standard run), sees the same first 600 usable standard pairs
    #in the same seeded order, with the same optimizer, lr, lora config and seed. its budget (600 pairs, 1 epoch) is
    #smaller than the standard run's (all ~1450 pairs), so beta_0.1 here is NOT the same model as the standard run
    for beta in cfg["betas"]:
        name = beta_run_name(float(beta))
        output = str(Path(cfg["standard_output"]).parent / name) #outputs/task1_dpo/beta_0.03 etc, next to the standard run

        #a finished fork is not retrained if the job is relaunched after a crash, the adapter on disk is reused
        if (repo_path(output) / "adapter_config.json").exists():
            print(f"[{name}] adapter already exists, skipping training")
        else:
            run_training(args.config, name, output_path=output, beta=float(beta),
                         max_examples=int(cfg["short_ablation_examples"]))

        #same held-out pairs, prompts and decoding for every fork, so only beta differs between the rows
        run_evaluation(args.config, output, name, beta=float(beta))


if __name__ == "__main__":
    main()
