import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Adds the parent directory to the search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from tdmpc2.envs.make_env import make_env
from tdmpc2.models.encoder.encoder_factory import make_encoder
from tdmpc2.models.simnorm import SimNorm


def main():

    # ---------------------------------
    # fake Hydra-style config
    # ---------------------------------
    cfg = OmegaConf.create({
        "env": {
            "name": "minigrid",
            "task": "MiniGrid-Empty-8x8-v0",
            "render": False,
            "seed": 1
        },
        "obs_type": "pixel",
        "latent_dim": 256,
        "simnorm_dim": 8
    })

    # ---------------------------------
    # create env using our pipeline
    # ---------------------------------
    env = make_env(cfg.env)

    obs, _ = env.reset()

    # env already outputs resized image
    obs = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float()

    obs_shape = obs.shape[1:]

    # ---------------------------------
    # encoder + simnorm
    # ---------------------------------
    encoder = make_encoder(cfg, obs_shape)

    simnorm = SimNorm(
        dim=cfg.latent_dim,
        groups=cfg.simnorm_dim
    )

    # ---------------------------------
    # forward pass
    # ---------------------------------
    with torch.no_grad():
        z = encoder(obs)
        z = simnorm(z)

    print("Obs shape:", obs.shape)
    print("Latent shape:", z.shape)
    print(
        "Simplex sums:",
        z.view(1, cfg.simnorm_dim, -1).sum(-1)
    )

    env.close()


if __name__ == "__main__":
    main()
