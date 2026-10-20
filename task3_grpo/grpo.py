from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


#the group-relative advantage from the manual, one scalar per sampled completion ->
#   A_k = (r_k - mu_r) / (sigma_r + eps),   mu_r and sigma_r taken over the K completions of the SAME prompt
#rewards is [n_completions], group_ids[i] says which prompt produced rewards[i].
#the group mean plays the role of ppo's critic: a completion is good only relative to its siblings for that prompt
def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    #fix -> the release took the mean and std over ALL rewards in the batch and ignored group_ids. with more than
    #one prompt in a batch that compares completions of different prompts with each other, so an easy prompt's
    #completions all look good and a hard prompt's all look bad, which is prompt difficulty, not completion quality.
    #the baseline has to be the prompt's own group, so the statistics are computed group by group
    advantages = torch.zeros_like(rewards)
    for group in torch.unique(group_ids):
        members = group_ids == group
        group_rewards = rewards[members]
        mean = group_rewards.mean()
        std = group_rewards.std(unbiased=False) #population std over the K completions, as in the manual's sigma_r
        #a group whose rewards are all equal has std 0 -> every advantage is 0 and the prompt contributes no gradient.
        #that is the uninformative group the manual asks to be counted
        advantages[members] = (group_rewards - mean) / (std + eps)
    return advantages


#the clipped grpo objective for already sampled completions ->
#   L = -E_k[ (1/T_k) * sum_t min(rho_kt * A_k, clip(rho_kt, 1-eps, 1+eps) * A_k) ]  +  beta * KL(pi || pi_ref)
#new_logp / old_logp / ref_logp / token_mask are [K, T], seq_adv is [K]. every token of completion k shares A_k.
#token_mask may be all zero for a completion that was masked out because it hit the maximum generation length
def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    ratio = torch.exp(new_logp - old_logp) #rho_kt = pi_new / pi_old per token
    adv = seq_adv[:, None] #broadcast the per completion advantage over its tokens
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2) #the same pessimistic clipped surrogate as ppo

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        #canonical grpo -> divide by the completion's own length T_k. every completion then weighs the same in the
        #loss whatever its length, so each token of a long completion gets a smaller share than a token of a short one
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        #dr. grpo -> the same constant for every completion. every token then weighs the same, so a completion's
        #total weight grows with its length instead of being fixed
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    #kl(pi || pi_ref) estimated on the sampled tokens with the non-negative form  e^d - d - 1,  d = log pi_ref - log pi.
    #it is 0 when the two agree and grows as the policy drifts from the reference in either direction
    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


#zeroes the whole token mask of every completion that ran into the generation cap without an eos. its reward still
#counts in the group statistics, but an unfinished completion sends no gradient
def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
