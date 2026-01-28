import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    State-based encoder.
    """

    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),

            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs):
        """
        obs: (B, obs_dim)
        """
        return self.net(obs)
