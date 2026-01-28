import torch
import torch.nn.functional as F


def consistency_loss(z_pred, z_target):
    """
    || z' - sg(z_next) ||^2
    """
    return F.mse_loss(z_pred, z_target.detach())
