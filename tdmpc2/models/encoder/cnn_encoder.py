import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Generic TD-MPC2 pixel encoder.
    Works for MiniGrid (4ch) and Crafter (3ch).
    """

    def __init__(
        self,
        latent_dim,
        in_channels,          
        num_channels=48      
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 4, stride=2),
            nn.Mish(),

            nn.Conv2d(num_channels, num_channels * 2, 4, stride=2),
            nn.Mish(),

            nn.Conv2d(num_channels * 2, num_channels * 4, 4, stride=2),
            nn.Mish(),
        )

        # ⭐ LazyLinear removes hardcoded spatial size (safer & generic)
        self.fc = nn.Sequential(
            nn.LazyLinear(512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, latent_dim)
        )

        self.in_channels = in_channels


    def forward(self, obs):
        """
        obs: (B, C, H, W)
        """
        if obs.ndim == 4 and obs.shape[-1] == self.in_channels:
            obs = obs.permute(0, 3, 1, 2)

        obs = obs.float() / 255.0
        x = self.conv(obs)
        x = torch.flatten(x, 1)
        return self.fc(x)
