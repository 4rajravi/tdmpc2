import gymnasium as gym

def make_gym_env(cfg):
    env = gym.make(
        cfg.task,
        render_mode="rgb_array" if cfg.render else None
    )

    if cfg.seed is not None:
        env.reset(seed=cfg.seed)

    return env
