import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    TD-MPC2 pixel encoder.
    """

    def __init__(
        self,
        latent_dim,
        num_channels=32
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, num_channels, 4, stride=2),
            nn.ReLU(),

            nn.Conv2d(num_channels, num_channels * 2, 4, stride=2),
            nn.ReLU(),

            nn.Conv2d(num_channels * 2, num_channels * 4, 4, stride=2),
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(num_channels * 4 * 6 * 6, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, obs):
        """
        obs: (B, 3, H, W)
        """
        obs = obs / 255.0
        x = self.conv(obs)
        return self.fc(x)
