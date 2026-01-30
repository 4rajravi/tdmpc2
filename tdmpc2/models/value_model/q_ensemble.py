import torch.nn as nn
from tdmpc2.models.value_model.q_network import QNetwork


class QEnsemble(nn.Module):
    """
    Ensemble of distributional Q-networks.
    """

    def __init__(
        self,
        num_q: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_bins: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_q = num_q

        self.qs = nn.ModuleList(
            [
                QNetwork(
                    latent_dim=latent_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    num_bins=num_bins,
                    dropout=dropout,
                )
                for _ in range(num_q)
            ]
        )

    def forward(self, z, a):
        """
        Returns list of logits.
        """
        return [q(z, a) for q in self.qs]
