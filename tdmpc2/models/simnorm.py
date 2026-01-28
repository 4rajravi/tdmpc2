import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """
    Simplicial normalization from TD-MPC2.

    Splits latent vector into groups and applies
    softmax inside each group.
    """

    def __init__(self, dim, groups, temperature=1.0):
        """
        Args:
            dim: total latent dimension
            groups: number of simplex groups
            temperature: softmax temperature τ
        """
        super().__init__()

        assert dim % groups == 0, \
            "latent_dim must be divisible by simnorm groups"

        self.dim = dim
        self.groups = groups
        self.group_dim = dim // groups
        self.temperature = temperature

    def forward(self, z):
        """
        z: (B, dim)
        returns: (B, dim)
        """
        B = z.shape[0]

        # reshape → (B, groups, group_dim)
        z = z.view(B, self.groups, self.group_dim)

        # softmax per group
        z = F.softmax(z / self.temperature, dim=-1)

        # flatten back
        return z.view(B, self.dim)
