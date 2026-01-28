import torch


def policy_loss(
    policy,
    q_ensemble,
    z,
    alpha,
    beta
):
    """
    L = α Q(z, a) − β H(p)
    """

    # sample action
    action, _ = policy.sample(z)

    # evaluate Q ensemble
    q_logits = q_ensemble(z, action)

    # expected value
    q_values = [
        q.mean(dim=-1) for q in q_logits
    ]
    q = torch.stack(q_values).mean(0)

    # entropy bonus
    entropy = policy.entropy(z)

    loss = -(alpha * q - beta * entropy).mean()

    return loss
