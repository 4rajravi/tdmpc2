import crafter
import gymnasium as gym


def make_crafter_env(cfg):
    """
    Crafter environment wrapper.

    Crafter does NOT follow the Gymnasium seeding API,
    so the seed must be passed explicitly.
    """

    # Crafter supports seed at construction
    if cfg.seed is not None:
        env = crafter.Env(seed=cfg.seed)
    else:
        env = crafter.Env()

    # Optional time limit wrapper
    if hasattr(cfg, "episode_length"):
        env = gym.wrappers.TimeLimit(
            env,
            max_episode_steps=cfg.episode_length
        )

    return env
