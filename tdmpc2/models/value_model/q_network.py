import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Distributional Q-network head.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_bins: int,
        dropout: float = 0.0
    ):
        super().__init__()

        layers = []

        layers.append(
            nn.Linear(latent_dim + action_dim, hidden_dim)
        )
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Mish())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(
            nn.Linear(hidden_dim, num_bins)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, z, a):
        """
        z : (B, latent_dim)
        a : (B, action_dim)
        """
        x = torch.cat([z, a], dim=-1)
        return self.net(x)
