from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common.data import prompt_messages, read_jsonl, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean, mean_response_length, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode

#the held-out protocol shared by the online rl tasks (ppo and grpo) -> sample one response to every prompt of the fixed
#evaluation pool with the fixed decoding config, then measure learned reward, kl from the reference, entropy and length


def token_entropy(logits):
    #exact entropy of the policy's next token distribution at every position ->  H_t = -sum_v p(v) log p(v).
    #computed from the full logits, so it does not depend on which token happened to be sampled
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


@torch.no_grad()
def score_generation(policy, out, rows=slice(None)):
    #teacher forces the sampled responses back through the model twice -> with the adapter on (the policy) and with
    #it off (the frozen reference). returns per token log probs of the sampled tokens and the policy's entropy.
    #rows selects a sub batch, the full vocabulary logits are too large to score many long sequences at once
    seq, attn, resp = out["sequences"][rows], out["attention_mask"][rows], out["response_ids"][rows]
    policy_logp, logits = response_token_logprobs(policy, seq, attn, out["prompt_width"], resp)
    entropy = token_entropy(logits)
    del logits
    with reference_mode(policy):
        ref_logp, _ = response_token_logprobs(policy, seq, attn, out["prompt_width"], resp)
    return policy_logp, ref_logp, entropy


def evaluate_policy(cfg: dict, adapter: str | None, name: str, max_new_tokens: int):
    tokenizer = load_tokenizer(cfg["base_model"])
    #adapter None -> the base model itself, ie the frozen reference policy (kl = 0 by construction)
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm_model, rm_tok = load_reward_model(cfg)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    rows = rows[:cfg.get("eval_max_prompts")] #not set in the real configs -> all 200 prompts, the smoke test sets a small number
    gen_cfg = cfg["generation"]
    gen_size, score_size = int(cfg["generation_batch_size"]), int(cfg["eval_batch_size"])
    print(f"[eval {name}] adapter {adapter}  held-out prompts {len(rows)}  cap {max_new_tokens}")

    #same seed, prompts, order and decoding for every policy, so the rows of the results table differ only in the policy
    set_seed(int(cfg["seed"]))
    records, pol_all, ref_all, ent_all, mask_all = [], [], [], [], []
    for start in range(0, len(rows), gen_size):
        chunk = rows[start:start + gen_size]
        prompts = [prompt_messages(r) for r in chunk]
        out = batch_generate(
            policy, tokenizer, prompts,
            max_prompt_length=int(cfg["max_prompt_length"]), #the release's rollout setting, long prompts are cut to 256 tokens
            max_new_tokens=max_new_tokens,
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            do_sample=bool(gen_cfg["do_sample"]),
        )
        for s in range(0, len(chunk), score_size):
            sl = slice(s, s + score_size)
            pol_logp, ref_logp, entropy = score_generation(policy, out, sl)
            #right padded to the cap so every chunk stacks into one [n, cap] tensor for the course kl helper
            pad = max_new_tokens - pol_logp.shape[1]
            pol_all.append(F.pad(pol_logp, (0, pad)).cpu())
            ref_all.append(F.pad(ref_logp, (0, pad)).cpu())
            ent_all.append(F.pad(entropy, (0, pad)).cpu())
            mask_all.append(F.pad(out["response_mask"][sl], (0, pad)).cpu())

        rewards = score_reward_pairs(rm_model, rm_tok, prompts, out["responses"],
                                     max_length=int(cfg.get("reward_max_length", 1280)))
        for i, row in enumerate(chunk):
            records.append({
                "prompt_id": row["prompt_id"],
                "prompt": prompts[i][-1]["content"],
                "response": out["responses"][i],
                "response_tokens": out["response_lengths"][i], #generated tokens after the prompt, padding excluded
                "terminated_with_eos": out["terminated_with_eos"][i],
                "truncated": out["truncated"][i], #reached the cap without producing an eos
                "reward": float(rewards[i]), #raw score of the fixed course reward model
            })
        print(f"  generated {min(start + gen_size, len(rows))}/{len(rows)}", flush=True)

    pol_all, ref_all, ent_all, mask_all = torch.cat(pol_all), torch.cat(ref_all), torch.cat(ent_all), torch.cat(mask_all)
    seq_kl = ((pol_all - ref_all) * mask_all).sum(-1)
    seq_entropy = (ent_all * mask_all).sum(-1) / mask_all.sum(-1).clamp_min(1.0)
    for record, kl, ent in zip(records, seq_kl, seq_entropy):
        record["seq_kl"] = float(kl) #sum over the response of log pi - log ref
        record["entropy"] = float(ent) #mean token entropy of this response

    lengths = np.array([r["response_tokens"] for r in records])
    rewards = np.array([r["reward"] for r in records])
    summary = {
        "name": name,
        "adapter": adapter,
        "n_prompts": len(records),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        #the course sampled kl estimator over every generated token -> token level mean of log pi - log ref
        "kl_token_mean": float(sampled_kl(pol_all, ref_all, mask_all)),
        "kl_sequence_mean": float(seq_kl.mean()),
        "entropy_mean": float(masked_mean(ent_all, mask_all)),
        "length_mean": mean_response_length(mask_all),
        "length_std": float(lengths.std()),
        "length_iqr": float(np.percentile(lengths, 75) - np.percentile(lengths, 25)),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "generated_tokens": int(lengths.sum()),
        "decoding": {**gen_cfg, "max_new_tokens": max_new_tokens, "max_prompt_length": int(cfg["max_prompt_length"]),
                     "seed": int(cfg["seed"])},
    }
    out_dir = cfg["results_dir"]
    write_jsonl(f"{out_dir}/eval_{name}_generations.jsonl", records)
    save_json(f"{out_dir}/eval_{name}.json", summary)
    print(f"[eval {name}] reward {summary['reward_mean']:.3f}  kl {summary['kl_token_mean']:.4f}  "
          f"entropy {summary['entropy_mean']:.3f}  length {summary['length_mean']:.1f}  "
          f"truncated {summary['truncation_rate']:.2f}")

    clear_gpu(policy, rm_model)
    return summary
