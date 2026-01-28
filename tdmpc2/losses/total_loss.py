import torch


def tdmpc2_loss(
    *,
    consistency_losses,
    reward_losses,
    value_losses,
    coeffs,
    rho,
):
    """
    TD-MPC2 total loss.
    """

    loss = 0.0

    # consistency
    for l in consistency_losses:
        loss += coeffs["consistency"] * l

    # reward
    for l in reward_losses:
        loss += coeffs["reward"] * l

    # value
    for l in value_losses:
        loss += coeffs["value"] * l

    return loss
