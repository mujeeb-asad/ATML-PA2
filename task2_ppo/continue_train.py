from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl_eval import score_generation
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    #the supplied midpoint -> a policy lora adapter and the critic that was trained alongside it. every run and every
    #fork starts from these exact two checkpoints
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def collect_rollout(policy, tokenizer, prompts, cfg):
    #on-policy sampling -> the current policy answers the prompts with the release's rollout settings
    gen_cfg = cfg["generation"]
    out = batch_generate(
        policy, tokenizer, prompts,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["max_response_length"]),
        temperature=float(gen_cfg["temperature"]),
        top_p=float(gen_cfg["top_p"]),
        do_sample=bool(gen_cfg["do_sample"]),
    )
    #generation runs under torch.inference_mode, and tensors created there cannot take part in a backward pass later.
    #cloning turns them into ordinary tensors, the values are unchanged
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in out.items()}


def response_values(value_model, out):
    #the critic reads the same token sequence as the policy and outputs one scalar per position. V(s_t) is the value of
    #the state BEFORE response token t is emitted, ie the position that predicts it -> prompt_width - 1 + t. this is
    #the same alignment the policy logits use in response_token_logprobs.
    #this is common.models.token_values with one change -> the hidden states are cast to fp32 before the value head,
    #because the head is kept in fp32 here (see run_ppo)
    base = value_model.get_base_model() #the sequence classification model underneath the peft wrapper
    hidden = base.model(
        input_ids=out["sequences"],
        attention_mask=out["attention_mask"],
        return_dict=True,
        use_cache=False,
    ).last_hidden_state
    values = base.score(hidden.float()).squeeze(-1)
    start = out["prompt_width"] - 1
    return values[:, start:start + out["response_ids"].shape[1]]


def explained_variance(values, returns, mask):
    #1 - var(returns - values) / var(returns) over the valid tokens. 1 = perfect critic, 0 = no better than predicting
    #the mean return, negative = worse than that
    valid = mask.bool()
    target = returns[valid]
    if target.numel() < 2 or target.var() == 0:
        return float("nan")
    return float(1.0 - (target - values[valid]).var() / target.var())


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, policy, value_model = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    policy_optimizer, value_optimizer = bundle["policy_optimizer"], bundle["value_optimizer"]
    n_updates, per_update, epochs = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["ppo_epochs"])
    eps, beta_kl = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_grad_norm = float(cfg["max_grad_norm"])

    log_path = repo_path(cfg["results_dir"]) / f"train_log_{run_name}.jsonl"
    log_path.unlink(missing_ok=True) #a rerun overwrites its own log
    print(f"[{run_name}] updates {n_updates}  prompts/update {per_update}  ppo epochs {epochs}  eps {eps}  kl beta {beta_kl}")

    #the critic's backbone is fp16 and frozen, its lora weights are fp32 (peft does that), but the released loader leaves
    #the trainable value head in fp16. adamw on fp16 weights is unsafe -> the squared gradient average underflows to 0
    #for small gradients and eps = 1e-8 also rounds to 0 in fp16, so the update divides by zero. the head is therefore
    #moved to fp32, same values, just stored with more bits
    for parameter in trainable_parameters(value_model):
        parameter.data = parameter.data.float()

    #the policy stays in eval mode for the whole run, so lora dropout is off. the ratio pi_new/pi_old must compare
    #the same deterministic function before and after a parameter change -> with dropout on, the ratio would already
    #differ from 1 before the first step, and that noise alone would trip the eps = 0.05 clip
    policy.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    generated_tokens = 0

    for update in range(n_updates):
        #prompts are taken in file order -> update u always sees the same prompt(s), in every run and every fork.
        #that is what keeps the short forks matched in prompts, update budget and generation cap
        rows = bundle["prompt_rows"][update * per_update:(update + 1) * per_update]
        prompts = [prompt_messages(row) for row in rows]

        #---- 1. rollout with the current policy, which becomes pi_old for this update
        rollout = collect_rollout(policy, tokenizer, prompts, cfg)
        mask = rollout["response_mask"]
        seq, attn, resp, width = rollout["sequences"], rollout["attention_mask"], rollout["response_ids"], rollout["prompt_width"]

        with torch.no_grad():
            #log pi_old(a_t|s_t), log pi_ref(a_t|s_t) and the policy entropy at every response token
            old_logp, ref_logp, entropy = score_generation(policy, rollout)
            value_model.eval()
            values = response_values(value_model, rollout)
            #the learned reward -> one scalar per response from the frozen reward model
            raw_reward = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], prompts,
                                            rollout["responses"], max_length=int(cfg["reward_max_length"]))

        #a response that hit the 512 token cap without an eos loses missing_eos_penalty, as in the supplied rollouts
        truncated = torch.tensor(rollout["truncated"], device=raw_reward.device, dtype=raw_reward.dtype)
        task_reward = raw_reward - float(cfg["missing_eos_penalty"]) * truncated

        #---- 2. per token rewards and advantages
        #r_t = -beta_kl * (log pi_old - log pi_ref) on every token, plus the task reward on the last one
        rewards = shaped_rewards(task_reward, old_logp, ref_logp, mask, beta_kl)
        advantages, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        #normalised for the policy loss only, the critic regresses on the raw returns
        norm_advantages = normalize_advantages(advantages, mask)

        #---- 3. ppo epochs over this rollout. old_logp, advantages and returns stay frozen, only the networks move
        policy_losses, value_losses, clip_fractions, policy_grads, value_grads = [], [], [], [], []
        for epoch in range(epochs):
            #policy -> recompute the log probs of the SAME sampled tokens under the current weights, then the clipped
            #surrogate. in the first epoch the ratio is exactly 1 (nothing has moved yet), so clipping can only bite from
            #the second epoch on
            new_logp, _ = response_token_logprobs(policy, seq, attn, width, resp)
            policy_loss, ratio, clip_fraction = ppo_policy_loss(new_logp, old_logp, norm_advantages, mask, eps)
            policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_grads.append(float(torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)))
            policy_optimizer.step()

            #critic -> regress V(s_t) onto the returns. it has its own optimizer, value_coef scales its gradient
            value_model.train()
            value_loss = value_mse_loss(response_values(value_model, rollout), returns, mask)
            value_optimizer.zero_grad(set_to_none=True)
            (float(cfg["value_coef"]) * value_loss).backward()
            value_grads.append(float(torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm)))
            value_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            clip_fractions.append(float(clip_fraction))

        #---- 4. how far did this update move the policy on its own rollout (the stability statistic)
        with torch.no_grad():
            final_logp, _ = response_token_logprobs(policy, seq, attn, width, resp)
        log_ratio = (final_logp - old_logp) * mask
        #sampled kl(pi_old || pi_new) = E_old[log pi_old - log pi_new], the tokens were sampled from pi_old
        step_kl = float(masked_mean(old_logp - final_logp, mask))

        generated_tokens += int(mask.sum())
        record = {
            "run": run_name,
            "update": update + 1,
            "prompt_ids": [row["prompt_id"] for row in rows],
            "reward": float(raw_reward.mean()), #mean learned reward, raw reward model score
            "reward_effective": float(task_reward.mean()), #after the missing eos penalty, what ppo actually optimises
            "kl": float(sampled_kl(old_logp, ref_logp, mask)), #kl from the reference, course sampled estimator
            "entropy": float(masked_mean(entropy, mask)),
            "response_length": float(mask.sum(-1).mean()),
            "truncated": float(truncated.mean()),
            "policy_loss": policy_losses[-1],
            "value_loss": value_losses[0], #epoch 0 -> the critic's error on this rollout before it was fitted to it
            "value_loss_final": value_losses[-1],
            "clip_fraction": clip_fractions[-1], #last epoch, the first one is 0 by construction (ratio = 1)
            "grad_norm": sum(policy_grads) / len(policy_grads), #policy gradient norm before clipping, mean over epochs
            "value_grad_norm": sum(value_grads) / len(value_grads),
            "step_kl": step_kl,
            "max_abs_log_ratio": float(log_ratio.abs().max()),
            "ratio_mean": float(masked_mean(ratio, mask)),
            "value_mean": float(masked_mean(values, mask)),
            "return_mean": float(masked_mean(returns, mask)),
            "explained_variance": explained_variance(values, returns, mask),
            "generated_tokens": generated_tokens,
            "elapsed_s": elapsed(),
            "policy_loss_epochs": policy_losses,
            "value_loss_epochs": value_losses,
            "clip_fraction_epochs": clip_fractions,
        }
        append_jsonl(log_path, record)
        print(f"[{run_name}] update {update + 1:2d}  reward {record['reward']:+.3f}  kl {record['kl']:+.4f}  "
              f"entropy {record['entropy']:.3f}  len {record['response_length']:.0f}  clip {record['clip_fraction']:.3f}  "
              f"pi_loss {record['policy_loss']:+.4f}  v_loss {record['value_loss']:.3f}  step_kl {step_kl:+.5f}", flush=True)

    #only the policy adapter is saved. the critic is not needed afterwards, evaluation uses the policy alone and every
    #fork restarts from the supplied critic
    policy.save_pretrained(str(out))

    summary = {
        "run": run_name,
        "updates": n_updates,
        "prompts_per_update": per_update,
        "ppo_epochs": epochs,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "max_response_length": int(cfg["max_response_length"]),
        "seed": int(cfg["seed"]),
        "generated_tokens": generated_tokens,
        "wall_clock_s": elapsed(),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "output": str(out),
    }
    save_json(repo_path(cfg["results_dir"]) / f"train_summary_{run_name}.json", summary)
    print(f"[{run_name}] done in {summary['wall_clock_s'] / 60:.1f} min, adapter saved to {out}")
    return summary


def fork_name(clip_epsilon, kl_beta):
    return f"fork_eps{clip_epsilon:g}_kl{kl_beta:g}" #fork_eps0.2_kl0.1 etc


def run_fork(config_path: str, clip_epsilon: float, kl_beta: float):
    #one short fixed-budget continuation (fork_updates = 8) from the supplied midpoint, followed by the common held-out
    #evaluation. the clipping study and the kl study share the eps 0.2 / beta 0.1 fork, so a fork that already has
    #its adapter and its evaluation on disk is reused instead of being run twice
    from task2_ppo.evaluate import run_evaluation

    cfg = load_yaml(config_path)
    name = fork_name(clip_epsilon, kl_beta)
    output = str(Path(cfg["output"]).parent / name)
    if (repo_path(output) / "adapter_config.json").exists():
        print(f"[{name}] adapter already exists, skipping training")
    else:
        run_ppo(config_path, output, updates=int(cfg["fork_updates"]), clip_epsilon=clip_epsilon, kl_beta=kl_beta, run_name=name)
    if (repo_path(cfg["results_dir"]) / f"eval_{name}.json").exists():
        print(f"[{name}] evaluation already exists, skipping")
    else:
        run_evaluation(config_path, output, name)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
