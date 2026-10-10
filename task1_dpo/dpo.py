from __future__ import annotations

import torch
import torch.nn.functional as F


#dpo objective from the manual ->
#   L = -E[ log sigmoid( beta * ( log pi(y+|x)/ref(y+|x) - log pi(y-|x)/ref(y-|x) ) ) ]
#every *_logp argument is a [batch] tensor of SEQUENCE log probs, ie the response token log probs summed under
#teacher forcing, so one number per (prompt, response)
def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    #how much more the trainable policy likes y+ over y- ->  log pi(y+) - log pi(y-)
    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    #the same gap under the frozen reference ->  log ref(y+) - log ref(y-)
    ref_margin = (
        ref_chosen_logp
        - ref_rejected_logp
    )

    #regrouping the manual's log ratios gives  [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)]
    #  = policy_margin - ref_margin
    #so the policy is only rewarded for preferring y+ MORE than the reference already did.
    #fix -> the release had  policy_margin + ref_margin  here, which flips the sign of the reference term: it adds the
    #reference's own preference on top instead of measuring the policy against it, so a pair the reference already
    #ranks correctly looks "solved" with no learning at all, and the implicit kl anchor to the reference is lost
    logits = beta * (
        policy_margin - ref_margin
    )

    #-log sigmoid(z) is the bradley terry negative log likelihood that y+ beats y-, averaged over the batch
    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        #the manual's preference accuracy -> fraction of pairs with m_theta = policy_margin - ref_margin > 0.
        #beta > 0 only rescales the margin, so the sign (and this accuracy) does not depend on beta
        "preference_accuracy": (
            (policy_margin - ref_margin) > 0
        ).float().mean().detach(),
    }
