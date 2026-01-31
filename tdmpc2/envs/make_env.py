from tdmpc2.envs.gymnasium.gym_env import make_gym_env
from tdmpc2.envs.minigrid.minigrid_env import make_minigrid_env
from tdmpc2.envs.crafter.crafter_env import make_crafter_env

def make_env(env_cfg):
    """
    Create environment from Hydra env config.

    Expected structure:

    env:
        name: minigrid | crafter | gymnasium
        task: environment id
        obs_type: pixel | state
        image_size: 64
        frame_stack: 1
        episode_length: 1000
        seed: 1
        render: false
    """

    name = env_cfg.name.lower()

    if name == "gymnasium":
        env = make_gym_env(env_cfg)

    elif name == "minigrid":
        env = make_minigrid_env(env_cfg)

    elif name == "crafter":
        env = make_crafter_env(env_cfg)

    else:
        raise ValueError(
            f"Unknown environment '{env_cfg.name}'. "
            f"Valid options: gymnasium | minigrid | crafter"
        )

    return env

