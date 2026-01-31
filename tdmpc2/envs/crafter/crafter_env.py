import crafter
import gymnasium as gym

from tdmpc2.envs.crafter.gym_compat import GymCompatibilityWrapper


def make_crafter_env(cfg):

    env = crafter.Env(seed=cfg.seed)

    env = GymCompatibilityWrapper(env)

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=cfg.episode_length
    )

    return env
