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
from tdmpc2.models.world_model import WorldModel


def main():

    # ------------------------------------
    # fake hydra-like config
    # ------------------------------------
    cfg = OmegaConf.create({
        "env": {
            "name": "minigrid",
            "task": "MiniGrid-Empty-8x8-v0",
            "render": False,
            "seed": 1
        },
        "obs_type": "pixel",
        "latent_dim": 256,
        "simnorm_dim": 8,
        "action_dim": 7,     # MiniGrid actions
        "num_bins": 101,
        "num_q": 5
    })

    # ------------------------------------
    # create environment
    # ------------------------------------
    env = make_env(cfg.env)
    obs, _ = env.reset()

    obs = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0).float()

    obs_shape = obs.shape[1:]

    # ------------------------------------
    # encoder
    # ------------------------------------
    encoder = make_encoder(cfg, obs_shape)
    simnorm = SimNorm(cfg.latent_dim, cfg.simnorm_dim)

    with torch.no_grad():
        z = simnorm(encoder(obs))

    print("Latent z:", z.shape)

    # ------------------------------------
    # fake action
    # ------------------------------------
    action = torch.randn(1, cfg.action_dim)

    # ------------------------------------
    # world model
    # ------------------------------------
    world_model = WorldModel(
        latent_dim=cfg.latent_dim,
        action_dim=cfg.action_dim,
        num_bins=cfg.num_bins,
        num_q=cfg.num_q
    )

    with torch.no_grad():
        out = world_model(z, action)

    # ------------------------------------
    # print shapes
    # ------------------------------------
    print("\n--- WORLD MODEL OUTPUTS ---")

    print("z_next:", out["z_next"].shape)
    print("reward logits:", out["reward"].shape)

    print("Q ensemble:")
    for i, q in enumerate(out["value"]):
        print(f"  Q{i}:", q.shape)

    print("policy prior:", out["action_prior"].shape)

    env.close()


if __name__ == "__main__":
    main()
