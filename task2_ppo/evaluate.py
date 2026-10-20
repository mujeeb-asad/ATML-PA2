from __future__ import annotations

import argparse

from common.data import load_yaml
from common.rl_eval import evaluate_policy


def run_evaluation(config_path: str, adapter: str | None, name: str):
    #the common held-out protocol (common/rl_eval.py) on the fixed 200 prompt evaluation pool -> learned reward, kl
    #from the reference, entropy, response length, plus every generated response saved for the qualitative examples.
    #evaluation uses the larger 768 token cap from the config, because the supplied midpoint is known to leave a long
    #tail of responses unfinished at the 512 token training cap
    cfg = load_yaml(config_path)
    return evaluate_policy(cfg, adapter, name, max_new_tokens=int(cfg["eval_max_response_length"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter") #left out -> evaluates the base model, ie the frozen reference policy
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
