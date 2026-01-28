import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """
    Predicts reward distribution logits.
    Output: categorical distribution over num_bins.
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

        layers = [
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish()
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, num_bins))

        self.net = nn.Sequential(*layers)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        
        return self.net(x)
