import torch
import torch.nn.functional as F


def reward_loss(logits, target_dist):
    """
    Cross-entropy loss between predicted
    reward distribution and target distribution.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()
