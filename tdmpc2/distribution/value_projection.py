import torch


def value_to_dist(
    value,
    num_bins,
    vmin,
    vmax,
):
    """
    Projects scalar TD target to categorical value distribution.
    """

    device = value.device

    value = value.clamp(vmin, vmax)

    atoms = torch.linspace(vmin, vmax, num_bins, device=device)

    delta = (vmax - vmin) / (num_bins - 1)

    b = (value.unsqueeze(-1) - vmin) / delta
    l = b.floor().long()
    u = b.ceil().long()

    dist = torch.zeros(value.shape[0], num_bins, device=device)

    l = l.clamp(0, num_bins - 1)
    u = u.clamp(0, num_bins - 1)

    dist.scatter_add_(1, l, (u.float() - b))
    dist.scatter_add_(1, u, (b - l.float()))

    return dist
