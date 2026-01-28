import gymnasium as gym
import minigrid

from tdmpc2.envs.minigrid.wrappers import make_minigrid_wrapped_env


def make_minigrid_env(cfg):

    env = gym.make(
        cfg.task,
        render_mode="rgb_array"
    )

    if cfg.seed is not None:
        env.reset(seed=cfg.seed)

    env = make_minigrid_wrapped_env(
        env,
        size=cfg.image_size
    )

    return env
