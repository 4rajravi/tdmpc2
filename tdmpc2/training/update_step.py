import torch

from tdmpc2.losses.consistency_loss import consistency_loss
from tdmpc2.losses.reward_loss import reward_loss
from tdmpc2.losses.value_loss import value_loss
from tdmpc2.losses.total_loss import tdmpc2_loss
from tdmpc2.distribution.reward_projection import reward_to_dist


from tdmpc2.training.td_targets import compute_td_target


def update_step(
    *,
    batch,
    encoder,
    simnorm,
    world_model,
    policy_prior,
    target_networks,
    optimizers,
    cfg,
    device
):

    obs, action, reward, next_obs, done = batch

    # --------------------------------------------------
    # move to device
    # --------------------------------------------------
    obs = obs.to(device)
    action = action.to(device)
    reward = reward.to(device)
    next_obs = next_obs.to(device)
    done = done.to(device)

    # --------------------------------------------------
    # HWC → CHW
    # --------------------------------------------------
    if obs.ndim == 4 and obs.shape[-1] == 3:
        obs = obs.permute(0, 3, 1, 2)

    if next_obs.ndim == 4 and next_obs.shape[-1] == 3:
        next_obs = next_obs.permute(0, 3, 1, 2)

    # --------------------------------------------------
    # encode
    # --------------------------------------------------
    z = simnorm(encoder(obs))
    z_next = simnorm(encoder(next_obs))

    # --------------------------------------------------
    # ensure action shape
    # --------------------------------------------------
    if action.ndim == 1:
        action = action.unsqueeze(0)

    if action.shape[0] != z.shape[0]:
        action = action.expand(z.shape[0], -1)

    # --------------------------------------------------
    # world model rollout
    # --------------------------------------------------
    out = world_model(z, action)

    # --------------------------------------------------
    # losses
    # --------------------------------------------------
    loss_cons = consistency_loss(out["z_next"], z_next)

    num_bins = out["reward"].shape[-1]

    target_dist = reward_to_dist(
    reward,
    num_bins=num_bins,
    vmin=cfg.vmin,
    vmax=cfg.vmax
    )

    loss_reward = reward_loss(
        out["reward"],
        target_dist
    )

    with torch.no_grad():
        td_target = compute_td_target(
            reward=reward,
            done=done,
            z_next=z_next,
            policy_prior=policy_prior,
            target_q=target_networks.target,
            cfg=cfg
        )


    loss_value = value_loss(
        out["value"],
        td_target,
        vmin=cfg.vmin,
        vmax=cfg.vmax
    )


    loss = tdmpc2_loss(
        consistency_losses=[loss_cons],
        reward_losses=[loss_reward],
        value_losses=[loss_value],
        coeffs=dict(
            consistency=cfg.consistency_coef,
            reward=cfg.reward_coef,
            value=cfg.value_coef,
            termination=cfg.termination_coef
        ),
        rho=cfg.rho
    )

    # --------------------------------------------------
    # optimize
    # --------------------------------------------------
    optimizers.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        world_model.parameters(),
        cfg.grad_clip_norm
    )

    optimizers.step()

    return {
        "total": loss.item(),
        "consistency": loss_cons.item(),
        "reward": loss_reward.item(),
        "value": loss_value.item()
    }
