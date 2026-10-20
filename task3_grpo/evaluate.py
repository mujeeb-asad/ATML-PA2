from __future__ import annotations

import argparse

from common.data import load_yaml
from common.rl_eval import evaluate_policy


def run_evaluation(config_path: str, adapter: str | None, name: str):
    #the same held-out protocol as ppo (common/rl_eval.py) on the fixed 200 prompt evaluation pool -> learned reward,
    #kl from the reference, entropy and response length, with every response saved. the generation cap is the
    #grpo config's own max_completion_length (512), the cap the supplied midpoint was trained and accepted with
    cfg = load_yaml(config_path)
    return evaluate_policy(cfg, adapter, name, max_new_tokens=int(cfg["max_completion_length"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter") #left out -> evaluates the base model, ie the frozen reference policy
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
