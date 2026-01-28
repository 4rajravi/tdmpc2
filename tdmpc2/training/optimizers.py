import torch.optim as optim


def make_optimizers(
    *,
    encoder,
    world_model,
    policy_prior,
    lr
):
    params = []

    params += list(encoder.parameters())
    params += list(world_model.parameters())
    params += list(policy_prior.parameters())

    optimizer = optim.Adam(
        params,
        lr=lr
    )

    return optimizer
