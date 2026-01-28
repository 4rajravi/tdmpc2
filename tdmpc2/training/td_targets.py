import torch


def compute_td_target(
    *,
    reward,
    done,
    z_next,
    policy_prior,
    target_q,
    cfg
):
    """
    TD-MPC2 target:

        q_t = r_t + γ (1 - done) Q̄(z_{t+1}, p(z_{t+1}))
    """

    train_cfg = cfg

    with torch.no_grad():

        # --------------------------------------------------
        # next action from policy prior
        # --------------------------------------------------
        a_next, _, _ = policy_prior(z_next)

        # --------------------------------------------------
        # target Q ensemble prediction
        # --------------------------------------------------
        q_logits = target_q(z_next, a_next)

        # --------------------------------------------------
        # double Q (min of two random heads)
        # --------------------------------------------------
        idx = torch.randperm(len(q_logits))[:2]

        q_logits = torch.min(
            q_logits[idx[0]],
            q_logits[idx[1]]
        )

        # --------------------------------------------------
        # distributional expectation
        # --------------------------------------------------
        support = torch.linspace(
            train_cfg.vmin,
            train_cfg.vmax,
            q_logits.shape[-1],
            device=q_logits.device
        )

        q = (torch.softmax(q_logits, dim=-1) * support).sum(dim=-1)

        # --------------------------------------------------
        # TD target
        # --------------------------------------------------
        td_target = reward + train_cfg.discount * (1.0 - done.float()) * q

        return td_target
