import torch
import torch.nn.functional as F

from tdmpc2.distribution.value_projection import value_to_dist

def value_loss(q_logits, td_target, vmin, vmax):

    num_bins = q_logits[0].shape[-1]

    target_dist = value_to_dist(
        td_target.detach(),
        num_bins=num_bins,
        vmin=vmin,
        vmax=vmax
    ).detach()

    losses = []

    for q in q_logits:
        log_probs = F.log_softmax(q, dim=-1)
        ce = -(target_dist * log_probs).sum(dim=-1).mean()
        losses.append(ce)

    return torch.stack(losses).mean()

