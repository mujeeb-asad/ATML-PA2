from __future__ import annotations

import argparse

from common.data import load_yaml
from task2_ppo.continue_train import run_fork


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    #reward-overoptimization study -> three short forks from the exact same midpoint policy and critic, same 8 prompts,
    #same seed, same clip eps (0.2). only beta_kl changes, the price per token of leaving the reference policy.
    #beta_kl = 0 removes the penalty entirely, so nothing but the clip holds the policy near the reference
    for beta in cfg["kl_values"]:
        run_fork(args.config, clip_epsilon=float(cfg["clip_epsilon"]), kl_beta=float(beta))


if __name__ == "__main__":
    main()
