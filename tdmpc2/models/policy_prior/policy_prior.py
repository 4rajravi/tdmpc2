import torch
import torch.nn as nn


class PolicyPrior(nn.Module):
    """
    p(a | z)
    """

    def __init__(self, latent_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, action_dim)
        )

    def forward(self, z):
        return self.net(z)
