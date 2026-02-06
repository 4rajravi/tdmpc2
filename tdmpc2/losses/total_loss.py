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
    TD-MPC2 total loss with rho^t weighting.
    """

    loss = 0.0

    H = len(consistency_losses)

    for t in range(H):
        w = rho ** t

        loss += w * (
            coeffs["consistency"] * consistency_losses[t]
            + coeffs["reward"] * reward_losses[t]
            + coeffs["value"] * value_losses[t]
        )

    return loss
