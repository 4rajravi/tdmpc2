import torch


def reward_to_dist(
    reward,
    num_bins,
    vmin=-1.0,
    vmax=1.0,
):
    """
    Projects scalar rewards to categorical distribution.
    """
    device = reward.device

    reward = reward.clamp(vmin, vmax)

    atoms = torch.linspace(vmin, vmax, num_bins, device=device)

    delta = (vmax - vmin) / (num_bins - 1)

    b = (reward.unsqueeze(-1) - vmin) / delta
    l = b.floor().long()
    u = b.ceil().long()

    dist = torch.zeros(reward.shape[0], num_bins, device=device)

    l = l.clamp(0, num_bins - 1)
    u = u.clamp(0, num_bins - 1)

    dist.scatter_add_(1, l, (u.float() - b))
    dist.scatter_add_(1, u, (b - l.float()))

    return dist
