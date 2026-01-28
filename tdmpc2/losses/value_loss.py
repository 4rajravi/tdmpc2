import torch
import torch.nn.functional as F

from tdmpc2.distribution.value_projection import value_to_dist


def value_loss(q_logits, td_target, vmin, vmax):
    """
    q_logits: list of tensors
        each tensor has shape (B, num_bins)
    """

    # infer bin count from first Q head
    num_bins = q_logits[0].shape[-1]

    target_dist = value_to_dist(
        td_target,
        num_bins=num_bins,
        vmin=vmin,
        vmax=vmax
    )

    losses = []

    for q in q_logits:
        log_probs = F.log_softmax(q, dim=-1)
        losses.append(
            -(target_dist * log_probs).sum(dim=-1)
        )

    return torch.stack(losses).mean()
