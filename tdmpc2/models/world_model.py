import torch.nn as nn

from tdmpc2.models.latent_dynamics.dynamics import LatentDynamics
from tdmpc2.models.reward_model.reward_head import RewardModel
from tdmpc2.models.value_model.q_ensemble import QEnsemble


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model.

    Contains:
      - latent dynamics
      - reward predictor
      - Q-value ensemble

    Policy prior is intentionally NOT included here.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        mlp_dim: int,
        num_bins: int,
        num_q: int,
        dropout: float = 0.0
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # --------------------------------------------------
        # latent dynamics model d(z, a)
        # --------------------------------------------------
        self.dynamics = LatentDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=mlp_dim,
            dropout=dropout
        )

        # --------------------------------------------------
        # reward predictor R(z, a)
        # --------------------------------------------------
        self.reward = RewardModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=mlp_dim,
            num_bins=num_bins,
            dropout=dropout
        )

        # --------------------------------------------------
        # value ensemble Q(z, a)
        # --------------------------------------------------
        self.value = QEnsemble(
            num_q=num_q,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=mlp_dim,
            num_bins=num_bins,
            dropout=dropout
        )

    def forward(self, z, a):
        """
        One-step latent rollout.

        Args:
            z: latent state        (B, latent_dim)
            a: action              (B, action_dim)

        Returns:
            dict with:
              z_next
              reward logits
              value logits
        """

        z_next = self.dynamics(z, a)
        r_logits = self.reward(z, a)
        q_logits = self.value(z, a)

        return {
            "z_next": z_next,
            "reward": r_logits,
            "value": q_logits
        }
