import torch


def value_to_dist(
    value,
    num_bins,
    vmin,
    vmax,
):
    """
    Projects scalar TD target to categorical value distribution.
    Safe C51-style projection used in TD-MPC2.
    """

    device = value.device

    # clamp values to support
    value = value.clamp(vmin, vmax)

    delta = (vmax - vmin) / (num_bins - 1)

    # fractional bin position
    b = (value.unsqueeze(-1) - vmin) / delta

    l = b.floor().long()
    u = b.ceil().long()

    l = l.clamp(0, num_bins - 1)
    u = u.clamp(0, num_bins - 1)

    # create distribution
    dist = torch.zeros(value.shape[0], num_bins, device=device)

    # distribute mass
    dist.scatter_add_(1, l, (u.float() - b))
    dist.scatter_add_(1, u, (b - l.float()))

    # --------------------------------------------------
    # FIX: handle exact-bin case (l == u)
    # --------------------------------------------------
    eq_mask = (u == l)

    if eq_mask.any():
        idx = eq_mask.squeeze(-1)
        dist[idx] = 0.0
        dist[idx, l[idx].squeeze(-1)] = 1.0

    return dist
