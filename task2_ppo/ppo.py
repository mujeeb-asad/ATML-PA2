from __future__ import annotations

import torch

from common.metrics import masked_mean


#token level generalized advantage estimation over the response positions.
#rewards, values, mask are all [batch, response_steps], padding positions have mask = 0.
#   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)          the one step td residual
#   A_t     = sum_k (gamma * lam)^k * delta_{t+k}        computed backwards as  A_t = delta_t + gamma * lam * A_{t+1}
def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    #walk from the last response position back to the first, carrying A_{t+1} in last_adv
    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            #the episode ends at the last valid token, so past it there is nothing to bootstrap from -> V = 0
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid #padding positions get advantage 0
        advantages[:, t] = last_adv

    #the regression target for the critic -> A_t + V(s_t), the lambda-return
    returns = advantages + values
    return advantages, returns


#the per token reward the manual defines ->
#   r_t = r_task * 1[t = T]  -  beta_kl * (log pi(a_t|s_t) - log pi_ref(a_t|s_t))
#every token pays a small penalty for being more likely under the policy than under the reference (a sampled
#estimate of the kl), and the learned reward model score arrives once, on the last valid response token
def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


#the clipped surrogate from the manual ->
#   L_clip = E_t[ min( rho_t * A_t ,  clip(rho_t, 1 - eps, 1 + eps) * A_t ) ],   rho_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
#returns the loss (the negative objective), the ratios, and the fraction of valid tokens whose ratio left [1-eps, 1+eps]
def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    #log probs are subtracted and exponentiated instead of dividing probabilities, which is numerically safer
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage #the plain importance weighted policy gradient objective
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage #the same with the ratio held inside the trust interval

    #fix -> the release took torch.maximum here. the minimum is what makes the surrogate a pessimistic lower bound ->
    #  A > 0 and rho > 1+eps : min picks the clipped term, so pushing an already boosted token further earns nothing
    #  A < 0 and rho < 1-eps : min picks the clipped term, so suppressing an already suppressed token earns nothing
    #  moves in the harmful direction (A > 0 with rho falling, A < 0 with rho rising) stay unclipped and fully penalised
    #the maximum does the opposite on every case: it keeps whichever term looks better, so the gradient never stops
    #rewarding larger ratios and the eps interval no longer limits how far one batch can move the policy
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    #the manual's clip fraction -> valid response tokens whose ratio lies outside [1-eps, 1+eps] before clipping
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(affected, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


#zero mean, unit std over the valid tokens of the batch. only the scale and offset change, the ordering of which
#tokens were better than expected does not, and it keeps the policy gradient size comparable from batch to batch
def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
