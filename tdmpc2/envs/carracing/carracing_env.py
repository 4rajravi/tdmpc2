# tdmpc2/envs/carracing_env.py

import gymnasium as gym
import numpy as np
import cv2


class make_carracing_env(gym.Wrapper):
    """
    TD-MPC2 compatible CarRacing-v3 environment.
    """

    def __init__(
        self,
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

        super().__init__(env)

        self.image_size = image_size
        self.action_repeat = action_repeat

        # override observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(image_size, image_size, 3),
            dtype=np.uint8
        )

    # --------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        obs = self._process_obs(obs)
        return obs, info

    # --------------------------------------------------
    def step(self, action):

        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.action_repeat):

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        obs = self._process_obs(obs)

        return obs, total_reward, terminated, truncated, info

    # --------------------------------------------------
    def _process_obs(self, obs):
        obs = cv2.resize(
            obs,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )
        return obs
