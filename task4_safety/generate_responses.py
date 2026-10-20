from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(load_xstest(cfg)))
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    #the four frozen policies of the task -> the untouched base model (sft) and the STANDARD dpo / ppo / grpo adapters
    #from tasks 1-3. no beta fork, clipping/kl fork or normalization fork is used here, and nothing in this task feeds
    #back into those runs
    for name, adapter in policy_specs(cfg).items():
        path = outdir / f"generated_{name}.jsonl"
        if path.exists():
            print(f"[{name}] responses already generated, skipping")
            continue
        if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(f"{name} adapter missing at {adapter}, run that task's standard training first")

        #one response per prompt with greedy decoding (do_sample=False) and the common 256 token cap, so the four
        #policies are compared on identical prompts, in the csv's fixed order, with no sampling noise between them
        records = generate_for_policy(cfg, name, batch_size=int(cfg["safety_batch_size"]))
        write_jsonl(path, records)
        print(f"[{name}] {len(records)} responses, mean length {sum(r['response_tokens'] for r in records) / len(records):.1f} tokens")
        clear_gpu() #the policy is released before the next one is loaded


if __name__ == "__main__":
    main()
