from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters
from common.rl_eval import score_generation
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

INFORMATIVE_TOLERANCE = 1e-6 #a group is informative when its reward std is above this, the eps of the advantage helper


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    #the supplied grpo midpoint adapter, every run and fork starts from it. there is no critic to load
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def collect_group_rollout(policy, tokenizer, prompts, cfg):
    #on-policy sampling -> prompts already holds each prompt K times, so one batch gives K completions per prompt
    gen_cfg = cfg["generation"]
    out = batch_generate(
        policy, tokenizer, prompts,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["max_completion_length"]),
        temperature=float(gen_cfg["temperature"]),
        top_p=float(gen_cfg["top_p"]),
        do_sample=bool(gen_cfg["do_sample"]),
    )
    #generation runs under torch.inference_mode, and tensors created there cannot take part in a backward pass later.
    #cloning turns them into ordinary tensors, the values are unchanged
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in out.items()}


def per_completion_gradient_norms(policy, new_logp, old_logp, advantages, token_mask, ref_logp, eps, loss_type, max_length):
    #the length-conditioned statistic for the normalization study -> the gradient norm each completion contributes to
    #the policy term on its own. completion k's term is the released loss on that one row with beta = 0 (no kl), divided
    #by K because the batch loss averages over the K completions. the graph is kept so the real update can reuse it
    norms = []
    params = trainable_parameters(policy)
    for k in range(new_logp.shape[0]):
        row = slice(k, k + 1)
        term, _ = grpo_policy_loss(new_logp[row], old_logp[row], advantages[row], token_mask[row], ref_logp[row],
                                   eps, 0.0, loss_type, max_length)
        if token_mask[row].sum() == 0 or advantages[k] == 0:
            norms.append(0.0) #a masked completion, or one from an uninformative group, has no gradient at all
            continue
        grads = torch.autograd.grad(term / new_logp.shape[0], params, retain_graph=True)
        norms.append(float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))))
    return norms


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, policy, optimizer = bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    n_updates, per_update, epochs = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["policy_epochs"])
    K = int(cfg["num_generations"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_length = int(cfg["max_completion_length"])

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_log_{run_name}.jsonl"
    completions_path = results_dir / f"train_completions_{run_name}.jsonl"
    log_path.unlink(missing_ok=True) #a rerun overwrites its own logs
    completions_path.unlink(missing_ok=True)
    print(f"[{run_name}] updates {n_updates}  prompts/update {per_update}  K {K}  epochs {epochs}  eps {eps}  "
          f"kl beta {beta}  loss {loss_type}")

    #eval mode for the whole run, so lora dropout is off and the ratio pi_new/pi_old compares the same deterministic
    #function before and after a parameter change (same choice and reason as in the ppo loop)
    policy.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    generated_tokens = 0

    for update in range(n_updates):
        #prompts in file order -> update u always sees the same prompt(s) in every run and fork, which keeps the
        #normalization forks matched in prompts, update budget and generation cap
        rows = bundle["prompt_rows"][update * per_update:(update + 1) * per_update]
        #each prompt K times in a row, group_ids marks which prompt every completion belongs to
        prompts = [prompt_messages(row) for row in rows for _ in range(K)]
        group_ids = torch.arange(len(rows)).repeat_interleave(K)

        #---- 1. K completions per prompt from the current policy, which becomes pi_old for this update
        rollout = collect_group_rollout(policy, tokenizer, prompts, cfg)
        response_mask = rollout["response_mask"]
        seq, attn, resp, width = rollout["sequences"], rollout["attention_mask"], rollout["response_ids"], rollout["prompt_width"]
        device = response_mask.device
        group_ids = group_ids.to(device)

        with torch.no_grad():
            old_logp, ref_logp, entropy = score_generation(policy, rollout)
            rewards = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], prompts,
                                         rollout["responses"], max_length=int(cfg.get("reward_max_length", 1280)))

        #---- 2. group-relative advantages, no critic involved -> each completion against its own prompt's group
        advantages = group_relative_advantages(rewards, group_ids)
        truncated = torch.tensor(rollout["truncated"], device=device)
        #a completion that hit the 512 token cap keeps its reward in the group statistics but is masked from the loss
        token_mask = response_mask
        if cfg.get("mask_truncated_completions", False):
            token_mask = mask_truncated_sequences(response_mask, truncated)

        #within-group reward spread, per prompt group
        group_std = torch.stack([rewards[group_ids == g].std(unbiased=False) for g in range(len(rows))])
        uninformative = (group_std <= INFORMATIVE_TOLERANCE).float()

        #---- 3. policy epochs on this group. with policy_epochs = 1 the single step is taken at ratio = 1, so the
        #clip cannot bind and the gradient is the plain advantage weighted policy gradient plus the kl term
        for epoch in range(epochs):
            new_logp, _ = response_token_logprobs(policy, seq, attn, width, resp)
            loss, stats = grpo_policy_loss(new_logp, old_logp, advantages, token_mask, ref_logp, eps, beta,
                                           loss_type, max_length)
            if epoch == 0:
                completion_grads = per_completion_gradient_norms(policy, new_logp, old_logp, advantages, token_mask,
                                                                 ref_logp, eps, loss_type, max_length)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"])))
            optimizer.step()

        #---- 4. how far the update moved the policy on its own rollout (same stability statistic as ppo)
        with torch.no_grad():
            final_logp, _ = response_token_logprobs(policy, seq, attn, width, resp)
        step_kl = float(masked_mean(old_logp - final_logp, response_mask))

        lengths = response_mask.sum(-1)
        generated_tokens += int(lengths.sum())
        record = {
            "run": run_name,
            "update": update + 1,
            "prompt_ids": [row["prompt_id"] for row in rows],
            "reward": float(rewards.mean()), #mean learned reward over the K completions
            "kl": float(sampled_kl(old_logp, ref_logp, response_mask)), #kl from the reference, course sampled estimator
            "kl_loss_term": float(stats["sampled_kl"]), #the non-negative kl estimate inside the loss
            "group_reward_std": float(group_std.mean()), #mean within-group reward std
            "uninformative_fraction": float(uninformative.mean()), #share of groups with zero reward std
            "policy_loss": float(stats["policy_term"]),
            "loss": loss.item(),
            "grad_norm": grad_norm, #before clipping
            "entropy": float(masked_mean(entropy, response_mask)),
            "response_length": float(lengths.mean()),
            "truncated": float(truncated.float().mean()), #share of completions masked for hitting the cap
            "clip_fraction": float(stats["clip_fraction"]),
            "step_kl": step_kl,
            "generated_tokens": generated_tokens,
            "elapsed_s": elapsed(),
        }
        append_jsonl(log_path, record)

        #one row per sampled completion -> the text for the qualitative examples and the per completion gradient
        #norm for the length-conditioned analysis
        for k in range(len(prompts)):
            append_jsonl(completions_path, {
                "run": run_name,
                "update": update + 1,
                "prompt_id": rows[int(group_ids[k])]["prompt_id"],
                "generation_index": k % K,
                "prompt": prompts[k][-1]["content"],
                "completion": rollout["responses"][k],
                "completion_tokens": int(lengths[k]),
                "truncated": bool(truncated[k]),
                "reward": float(rewards[k]),
                "advantage": float(advantages[k]),
                "group_reward_std": float(group_std[int(group_ids[k])]),
                "gradient_norm": completion_grads[k],
            })
        print(f"[{run_name}] update {update + 1:2d}  reward {record['reward']:+.3f}  group std {record['group_reward_std']:.3f}  "
              f"kl {record['kl']:+.4f}  entropy {record['entropy']:.3f}  len {record['response_length']:.0f}  "
              f"masked {record['truncated']:.2f}  pi_loss {record['policy_loss']:+.4f}  grad {grad_norm:.3f}", flush=True)

    policy.save_pretrained(str(out))

    summary = {
        "run": run_name,
        "loss_type": loss_type,
        "updates": n_updates,
        "prompts_per_update": per_update,
        "num_generations": K,
        "policy_epochs": epochs,
        "clip_epsilon": eps,
        "kl_beta": beta,
        "max_completion_length": max_length,
        "seed": int(cfg["seed"]),
        "generated_tokens": generated_tokens,
        "wall_clock_s": elapsed(),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "output": str(out),
    }
    save_json(results_dir / f"train_summary_{run_name}.json", summary)
    print(f"[{run_name}] done in {summary['wall_clock_s'] / 60:.1f} min, adapter saved to {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
