import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianPolicy(nn.Module):
    """
    Stochastic policy prior p(a | z)
    """

    def __init__(
        self,
        latent_dim,
        action_dim,
        log_std_min=-10,
        log_std_max=2
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.Mish()
        )

        self.mean = nn.Linear(512, action_dim)
        self.log_std = nn.Linear(512, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, z):
        h = self.net(z)

        mean = self.mean(h)
        log_std = self.log_std(h)

        log_std = torch.clamp(
            log_std,
            self.log_std_min,
            self.log_std_max
        )

        std = torch.exp(log_std)

        return mean, log_std, std


    def sample(self, z):
        LOG_2PI = 1.8378770664093453
        mean, log_std, std = self(z)

        eps = torch.randn_like(std)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)

        # ---------------- DEBUG PRINT ----------------
        # Print only occasionally or your console will freeze
        if torch.rand(1).item() < 0.01:
            print(
                "[ACTOR]",
                "mean_abs:", action.abs().mean().item(),
                "std:", std.mean().item()
            )
        # ---------------------------------------------

        log_prob = (
            -0.5 * eps.pow(2)
            - log_std
            - 0.5 * LOG_2PI
        ).sum(-1)

        return action, log_prob



    def entropy(self, z):
        _, log_std, _ = self(z)
        return (
            log_std
            + 0.5 * torch.log(
                torch.tensor(2 * 3.141592, device=z.device)
            )
        ).sum(-1)
