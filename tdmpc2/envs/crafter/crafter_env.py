import crafter
import gymnasium as gym

from tdmpc2.envs.crafter.gym_compat import GymCompatibilityWrapper


def make_crafter_env(cfg):

    if cfg.seed is not None:
        env = crafter.Env(seed=cfg.seed)
    else:
        env = crafter.Env()

    # ✅ convert old gym → gymnasium
    env = GymCompatibilityWrapper(env)

    if hasattr(cfg, "episode_length"):
        env = gym.wrappers.TimeLimit(
            env,
            max_episode_steps=cfg.episode_length
        )

    return env
