from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import mean_response_length, sampled_kl, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import filter_rows, make_collate, pair_logps, reference_pair_logps, to_device


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    #the same >= 768 token prompts training drops are dropped here, so every model is scored on the identical pair set
    rows, dropped = filter_rows(tokenizer, read_jsonl(cfg["paths"]["dpo_standard_eval"]), int(cfg["max_sequence_length"]))
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped_prompt_ids": dropped,
        "tokenizer": tokenizer,
        #adapter None -> the untouched base model, used as the sft reference row (policy == reference, so kl = 0)
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def heldout_pair_metrics(policy, tokenizer, rows, cfg, beta):
    #teacher forced scoring of the fixed held-out pairs, no generation involved. for each pair ->
    #  m = [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)]
    #preference accuracy = fraction with m > 0, held-out dpo loss = mean of -log sigmoid(beta * m)
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(policy.parameters()).device
    size = int(cfg["eval_batch_size"])
    records = []
    for start in range(0, len(rows), size):
        batch_rows = rows[start:start + size]
        chosen, rejected = collate(batch_rows)
        chosen, rejected = to_device(chosen, device), to_device(rejected, device)
        pol_c, pol_r = pair_logps(policy, chosen, rejected)
        ref_c, ref_r = reference_pair_logps(policy, chosen, rejected)
        margin = (pol_c - ref_c) - (pol_r - ref_r)
        loss = -F.logsigmoid(beta * margin)
        for i, row in enumerate(batch_rows):
            records.append({
                "prompt_id": row["prompt_id"],
                "length_stratum": row.get("length_stratum"), #only present in the length-stratified file
                "margin": float(margin[i]),
                "dpo_loss": float(loss[i]),
                "correct": bool(margin[i] > 0),
                #implicit rewards beta * log(pi/ref) of each response, handy for the qualitative examples
                "chosen_reward": float(beta * (pol_c[i] - ref_c[i])),
                "rejected_reward": float(beta * (pol_r[i] - ref_r[i])),
                "chosen_tokens": int(chosen["response_mask"][i].sum()),
                "rejected_tokens": int(rejected["response_mask"][i].sum()),
                "score_chosen": row.get("score_chosen"),
                "score_rejected": row.get("score_rejected"),
            })
    return records


@torch.no_grad()
def generate_and_score(policy, tokenizer, reward, prompts, prompt_ids, cfg, samples_per_prompt=1):
    #samples responses with the fixed decoding config, then scores each one three ways ->
    #token log probs under policy and reference (for the sampled kl), reward model score, and length.
    #the seed is reset first, so every model draws from the same random stream on the same prompts in the same order
    set_seed(int(cfg["seed"]))
    rm_model, rm_tok = reward
    gen_cfg = cfg["generation"]
    max_new = int(cfg["max_generation_tokens"])
    gen_size, score_size = int(cfg["generation_batch_size"]), int(cfg["eval_batch_size"])

    #each prompt repeated samples_per_prompt times, kept next to each other
    jobs = [(pid, p, k) for pid, p in zip(prompt_ids, prompts) for k in range(samples_per_prompt)]
    records, pol_all, ref_all, mask_all = [], [], [], []
    for start in range(0, len(jobs), gen_size):
        chunk = jobs[start:start + gen_size]
        out = batch_generate(
            policy, tokenizer, [p for _, p, _ in chunk],
            max_prompt_length=int(cfg["max_sequence_length"]), #every kept prompt is < 768 tokens, so none gets cut
            max_new_tokens=max_new,
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            do_sample=bool(gen_cfg["do_sample"]),
        )

        #teacher forced log probs of the sampled tokens, in smaller sub batches since the full vocab logits are large
        for s in range(0, len(chunk), score_size):
            sl = slice(s, s + score_size)
            seq, attn, resp = out["sequences"][sl], out["attention_mask"][sl], out["response_ids"][sl]
            pol_logp, _ = response_token_logprobs(policy, seq, attn, out["prompt_width"], resp)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seq, attn, out["prompt_width"], resp)
            #right padded to max_new so the chunks can be stacked into one [n, max_new] tensor for the kl helper
            pad = max_new - pol_logp.shape[1]
            pol_all.append(F.pad(pol_logp, (0, pad)).cpu())
            ref_all.append(F.pad(ref_logp, (0, pad)).cpu())
            mask_all.append(F.pad(out["response_mask"][sl], (0, pad)).cpu())

        rewards = score_reward_pairs(rm_model, rm_tok, [p for _, p, _ in chunk], out["responses"],
                                     max_length=int(cfg["reward_max_length"]))
        for i, (pid, p, k) in enumerate(chunk):
            records.append({
                "prompt_id": pid,
                "sample": k,
                "prompt": p[-1]["content"],
                "response": out["responses"][i],
                "response_tokens": out["response_lengths"][i], #generated tokens after the prompt, padding excluded
                "words": word_count(out["responses"][i]),
                "truncated": out["truncated"][i], #hit the 256 token cap without an eos
                "reward": float(rewards[i]),
            })
        print(f"  generated {min(start + gen_size, len(jobs))}/{len(jobs)}")

    pol_all, ref_all, mask_all = torch.cat(pol_all), torch.cat(ref_all), torch.cat(mask_all)
    #per response sequence kl  sum_t [log pi(y_t) - log ref(y_t)], stored per record for the qualitative examples
    seq_kl = ((pol_all - ref_all) * mask_all).sum(-1)
    for record, kl in zip(records, seq_kl):
        record["seq_kl"] = float(kl)
    return records, pol_all, ref_all, mask_all


def generation_summary(records, pol_all, ref_all, mask_all):
    lengths = np.array([r["response_tokens"] for r in records])
    return {
        #the course sampled kl estimator applied once over every generated token of every response ->
        #token level mean of log pi - log ref, the convention used for every task 1 condition
        "kl_token_mean": float(sampled_kl(pol_all, ref_all, mask_all)),
        #the same estimator summed per response, then averaged over responses -> per sequence kl
        "kl_sequence_mean": float(((pol_all - ref_all) * mask_all).sum(-1).mean()),
        "reward_mean": float(np.mean([r["reward"] for r in records])),
        "reward_std": float(np.std([r["reward"] for r in records])),
        "length_mean": mean_response_length(mask_all),
        "length_std": float(lengths.std()),
        "length_iqr": float(np.percentile(lengths, 75) - np.percentile(lengths, 25)),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "n_responses": len(records),
    }


def word_limit_eval(policy, tokenizer, reward, cfg):
    #the common prompt set from the length study -> 10 prompts each stating an explicit word limit. with only 10
    #prompts one sample each would be very noisy, so word_limit_samples (8) responses are drawn per prompt
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    records, *_ = generate_and_score(
        policy, tokenizer, reward,
        [prompt_messages(r) for r in rows], [r["prompt_id"] for r in rows], cfg,
        samples_per_prompt=int(cfg["word_limit_samples"]),
    )
    for record in records:
        #1.0 if the response stays within the limit parsed out of the prompt text ("at most 40 words" -> 40)
        record["compliant"] = word_limit_compliance(record["prompt"], record["response"])
    summary = {
        "compliance_rate": float(np.mean([r["compliant"] for r in records])),
        "words_mean": float(np.mean([r["words"] for r in records])),
        "words_std": float(np.std([r["words"] for r in records])),
        "tokens_mean": float(np.mean([r["response_tokens"] for r in records])),
        "tokens_std": float(np.std([r["response_tokens"] for r in records])),
        "n_responses": len(records),
    }
    return records, summary


def run_evaluation(config_path: str, adapter: str | None, name: str, beta: float | None = None):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg, tokenizer, policy, reward = bundle["cfg"], bundle["tokenizer"], bundle["policy"], bundle["reward"]
    #the held-out dpo loss depends on beta, so each model is scored with the beta it was trained with.
    #preference accuracy does not (only the sign of the margin matters)
    beta = float(cfg["beta"] if beta is None else beta)
    out_dir = cfg["results_dir"]
    print(f"[eval {name}] adapter {adapter}  beta {beta}  held-out pairs {len(bundle['rows'])}")

    #1 -> held-out preference fit on the fixed eval pairs
    pairs = heldout_pair_metrics(policy, tokenizer, bundle["rows"], cfg, beta)
    write_jsonl(f"{out_dir}/eval_{name}_pairs.jsonl", pairs)

    #2 -> behaviour of the policy itself -> sample a response to every held-out prompt, measure kl, reward, length
    prompts = [prompt_messages_from_preference(r) for r in bundle["rows"]]
    gens, pol_all, ref_all, mask_all = generate_and_score(
        policy, tokenizer, reward, prompts, [r["prompt_id"] for r in bundle["rows"]], cfg
    )
    write_jsonl(f"{out_dir}/eval_{name}_generations.jsonl", gens)

    #3 -> explicit word-limit compliance on the common prompt set
    wl_records, wl_summary = word_limit_eval(policy, tokenizer, reward, cfg)
    write_jsonl(f"{out_dir}/eval_{name}_word_limit.jsonl", wl_records)

    summary = {
        "name": name,
        "adapter": adapter,
        "beta": beta,
        "heldout_dpo_loss": float(np.mean([p["dpo_loss"] for p in pairs])),
        "heldout_preference_accuracy": float(np.mean([p["correct"] for p in pairs])),
        "heldout_margin_mean": float(np.mean([p["margin"] for p in pairs])),
        "n_heldout_pairs": len(pairs),
        **generation_summary(gens, pol_all, ref_all, mask_all),
        "word_limit": wl_summary,
        "decoding": {**cfg["generation"], "max_new_tokens": int(cfg["max_generation_tokens"]), "seed": int(cfg["seed"])},
        "dropped_prompt_ids": bundle["dropped_prompt_ids"],
    }
    save_json(f"{out_dir}/eval_{name}.json", summary)
    print(f"[eval {name}] acc {summary['heldout_preference_accuracy']:.3f}  loss {summary['heldout_dpo_loss']:.4f}  "
          f"kl {summary['kl_token_mean']:.4f}  reward {summary['reward_mean']:.3f}  "
          f"length {summary['length_mean']:.1f}  word-limit {wl_summary['compliance_rate']:.2f}")

    clear_gpu(policy, reward)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter") #left out -> evaluates the untrained base model
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float)
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name, args.beta)


if __name__ == "__main__":
    main()
