import torch


class DeterministicPlanner:
    """
    Deterministic MPC for evaluation.
    """

    def __init__(self, mpc):
        self.mpc = mpc

    @torch.no_grad()
    def act(self, z):
        """
        No sampling noise.
        No temperature.
        Mean action only.
        """
        return self.mpc.plan(z)
