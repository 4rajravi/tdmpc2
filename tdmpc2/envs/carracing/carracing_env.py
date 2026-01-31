import gymnasium as gym
from tdmpc2.envs.carracing.wrappers import CarRacingImageWrapper

def make_carracing_env(
    task="CarRacing-v3",
    image_size=64,
    action_repeat=1,
    seed=None,
    render=False
):

    env = gym.make(
        task,
        render_mode="rgb_array" if render else None,
        continuous=True
    )

    if seed is not None:
        env.reset(seed=seed)

    env = CarRacingImageWrapper(
        env,
        image_size=image_size,
        action_repeat=action_repeat
    )

    return env
