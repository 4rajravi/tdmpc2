import copy
import torch


class EMATarget:
    """
    Exponential moving average target network.
    """

    def __init__(self, model, tau):
        self.model = model
        self.target = copy.deepcopy(model)
        self.tau = tau

        for p in self.target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self):
        for p, tp in zip(
            self.model.parameters(),
            self.target.parameters()
        ):
            tp.data.mul_(1 - self.tau)
            tp.data.add_(self.tau * p.data)
