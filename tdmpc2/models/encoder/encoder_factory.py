from tdmpc2.models.encoder.cnn_encoder import CNNEncoder
from tdmpc2.models.encoder.mlp_encoder import MLPEncoder


def make_encoder(obs_cfg, model_cfg, obs_shape):

    if obs_cfg.type == "pixel":
        return CNNEncoder(
            latent_dim=model_cfg.latent_dim,
            num_channels=model_cfg.num_channels
        )

    elif obs_cfg.type == "state":
        return MLPEncoder(
            obs_shape[0],
            model_cfg.latent_dim
        )

    else:
        raise ValueError(f"Unknown obs type {obs_cfg.type}")
