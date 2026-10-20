from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task2_ppo.continue_train import run_fork
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def pad_rows(tensors, width):
    #stacks per-rollout 1d tensors of different lengths into one right padded [n_rollouts, width] tensor
    out = torch.zeros(len(tensors), width)
    for i, t in enumerate(tensors):
        out[i, :len(t)] = t.float()
    return out


def rebuild_token_ids(tokenizer, cfg, cached):
    #the cache stores the response as text, not as token ids. the 32 rollouts are the first 32 prompts of the held-out
    #pool, so the exact model input is rebuilt the way the rollout code built it -> the chat-templated prompt cut to
    #max_prompt_length, then the response tokens, then an eos if the response ended with one.
    #(re-tokenising the response text reproduces the stored token count for all 32 rollouts)
    by_id = {row["prompt_id"]: row for row in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    sequences = []
    for row in cached:
        rendered = tokenizer.apply_chat_template(prompt_messages(by_id[row["prompt_id"]]), tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(rendered, truncation=True, max_length=int(cfg["max_prompt_length"]))["input_ids"]
        response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        if row["terminated_with_eos"]:
            response_ids = response_ids + [tokenizer.eos_token_id]
        assert len(response_ids) == len(row["old_logprobs"]), "re-tokenised response does not match the cached rollout"
        sequences.append((prompt_ids, response_ids))
    return sequences


def rollout_logprobs(policy, prompt_ids, response_ids):
    #log pi(a_t|s_t) of the cached response tokens of one rollout under the policy's current weights
    device = next(policy.parameters()).device
    seq = torch.tensor([prompt_ids + response_ids], device=device)
    resp = torch.tensor([response_ids], device=device)
    logp, _ = response_token_logprobs(policy, seq, torch.ones_like(seq), len(prompt_ids), resp)
    return logp[0]


@torch.no_grad()
def batch_logprobs(policy, sequences, width):
    #the same for every cached rollout, stacked into one right padded [n_rollouts, width] cpu tensor
    return pad_rows([rollout_logprobs(policy, p, r).cpu() for p, r in sequences], width)


def cached_rollout_study(cfg):
    cached = load_cached_rollouts(cfg["cached_rollouts"])
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    policy.eval() #dropout off, as in the continuation loop
    sequences = rebuild_token_ids(tokenizer, cfg, cached)

    #the fixed cached batch as padded [32, T] tensors
    width = max(len(row["old_logprobs"]) for row in cached)
    cached_old = pad_rows([row["old_logprobs"] for row in cached], width)
    cached_ref = pad_rows([row["ref_logprobs"] for row in cached], width)
    values = pad_rows([row["values"] for row in cached], width)
    mask = pad_rows([torch.ones(len(row["old_logprobs"])) for row in cached], width)
    task_reward = torch.tensor([row["effective_terminal_reward"] for row in cached]) #already includes the missing eos penalty

    #the advantages a ppo update uses on this batch -> kl shaped rewards from the cached log probs, gae with the cached
    #critic values, then normalised over all valid tokens of the batch
    rewards = shaped_rewards(task_reward, cached_old, cached_ref, mask, float(cfg["kl_beta"]))
    advantages, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    norm_advantages = normalize_advantages(advantages, mask)

    #pi_old -> the supplied midpoint policy re-scored on the cached tokens with this code, on this gpu.
    #the cached old log probs are NOT used as pi_old in the ratio. re-scoring the frozen reference the same way
    #differs from the cached reference log probs by ~0.02 nats per token, although the reference weights cannot have
    #changed -> that is fp16 / batching noise between the machine that wrote the cache and this one, and it is as
    #large as the whole distance between the cached policy and the reference. a ratio taken across that gap would
    #measure numerical noise, not a policy change, so both sides of the ratio are computed here
    old_logp = batch_logprobs(policy, sequences, width)
    with reference_mode(policy):
        ref_logp = batch_logprobs(policy, sequences, width)
    consistency = {
        "midpoint_vs_cached_old_mean_abs_logp_difference": float(masked_mean((old_logp - cached_old).abs(), mask)),
        "midpoint_vs_cached_old_max_abs_logp_difference": float(((old_logp - cached_old) * mask).abs().max()),
        "reference_vs_cached_ref_mean_abs_logp_difference": float(masked_mean((ref_logp - cached_ref).abs(), mask)),
        "reference_vs_cached_ref_max_abs_logp_difference": float(((ref_logp - cached_ref) * mask).abs().max()),
        "cached_old_vs_cached_ref_mean_abs_logp_difference": float(masked_mean((cached_old - cached_ref).abs(), mask)),
        "n_rollouts": len(cached),
        "n_response_tokens": int(mask.sum()),
    }
    print("cache consistency:", consistency)
    #what the naive reading of the cache gives -> the cached old log probs as pi_old, the re-scored midpoint as pi_new
    direct_table = clip_table(old_logp, cached_old, norm_advantages, mask, cfg["clip_values"])
    #the control for it -> the identical computation with the frozen reference on both sides. the true ratio here is
    #exactly 1, so every token this table reports as clipped is numerical noise. if the two tables look alike, the
    #direct table is not measuring a policy change
    control_table = clip_table(ref_logp, cached_ref, norm_advantages, mask, cfg["clip_values"])

    #pi_new -> the midpoint after one pass of policy gradient steps over the cached batch, one optimizer step per
    #rollout as in the continuation loop (prompts_per_update = 1), with the release learning rate and gradient clipping.
    #the pass uses the UNCLIPPED surrogate (eps = inf), so there is a single pi_new that does not depend on eps ->
    #it shows how far this batch wants to move the policy, and each eps is then asked how much of that movement it
    #would have constrained. that is the immediate geometric effect of the clip, before any difference in training
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
    device = next(policy.parameters()).device
    for i, (prompt_ids, response_ids) in enumerate(sequences):
        n = len(response_ids)
        new_logp = rollout_logprobs(policy, prompt_ids, response_ids)
        loss, _, _ = ppo_policy_loss(new_logp, old_logp[i, :n].to(device), norm_advantages[i, :n].to(device),
                                     torch.ones(n, device=device), eps=float("inf"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
        optimizer.step()
    new_logp = batch_logprobs(policy, sequences, width)

    table = clip_table(new_logp, old_logp, norm_advantages, mask, cfg["clip_values"])
    #the spread of the log ratio itself, to read the clip fractions against
    log_ratio = (new_logp - old_logp)[mask.bool()].abs()
    consistency.update({
        "update_abs_log_ratio_mean": float(log_ratio.mean()),
        "update_abs_log_ratio_p99": float(log_ratio.quantile(0.99)),
        "update_abs_log_ratio_max": float(log_ratio.max()),
    })
    clear_gpu(policy)
    return table, direct_table, control_table, consistency


def clip_table(new_logp, old_logp, norm_advantages, mask, eps_values):
    rows = []
    for eps in eps_values:
        loss, ratio, clip_fraction = ppo_policy_loss(new_logp, old_logp, norm_advantages, mask, float(eps))
        #clip fraction counts every token whose ratio left [1-eps, 1+eps]. the affected fraction is the subset where
        #the clip actually changes the objective, ie where the min selects the clipped term ->
        #  A > 0 with rho > 1+eps, or A < 0 with rho < 1-eps. those tokens get zero gradient.
        #a token that left the interval in the harmful direction is clipped but NOT affected, it is still fully penalised
        affected = (((norm_advantages > 0) & (ratio > 1 + eps)) | ((norm_advantages < 0) & (ratio < 1 - eps))).float()
        rows.append({
            "eps": float(eps),
            "clipped_surrogate": -float(loss), #the objective L_clip itself, higher is better
            "unclipped_surrogate": float(masked_mean(ratio * norm_advantages, mask)),
            "clip_fraction": float(clip_fraction),
            "affected_token_fraction": float(masked_mean(affected, mask)),
            "ratio_mean": float(masked_mean(ratio, mask)),
            "ratio_min": float(ratio[mask.bool()].min()),
            "ratio_max": float(ratio[mask.bool()].max()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--cached-only", action="store_true") #only the cached-rollout part, no forks
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = repo_path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Required epsilon values:", cfg["clip_values"])
    table, direct_table, control_table, consistency = cached_rollout_study(cfg)
    table.to_csv(out_dir / "cached_clip_study.csv", index=False)
    direct_table.to_csv(out_dir / "cached_clip_direct.csv", index=False)
    control_table.to_csv(out_dir / "cached_clip_reference_control.csv", index=False)
    save_json(out_dir / "cached_clip_consistency.json", consistency)
    print("cached rollout study, pi_old = re-scored midpoint, pi_new = midpoint after one unclipped pass over the batch")
    print(table.to_string(index=False))
    print("direct reading, pi_old = cached old log probs, pi_new = re-scored midpoint")
    print(direct_table.to_string(index=False))
    print("noise control, cached reference log probs against the re-scored reference (true ratio = 1)")
    print(control_table.to_string(index=False))

    if not args.cached_only:
        #the matched short forks -> same midpoint, prompts, seed and kl beta (0.1), only eps changes
        for eps in cfg["clip_values"]:
            run_fork(args.config, clip_epsilon=float(eps), kl_beta=float(cfg["kl_beta"]))


if __name__ == "__main__":
    main()
