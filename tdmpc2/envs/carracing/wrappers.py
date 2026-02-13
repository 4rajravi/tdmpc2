import gymnasium as gym
import numpy as np
import cv2


class CarRacingImageWrapper(gym.Wrapper):
    """
    TD-MPC2 compatible CarRacing-v3 wrapper.
    Outputs (H, W, 3) uint8 images.
    """

    def __init__(self, env, image_size=64, action_repeat=1):
        super().__init__(env)

        self.image_size = image_size
        self.action_repeat = action_repeat

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

        # ⭐ IMPORTANT: ensure contiguous memory layout
        obs = np.ascontiguousarray(obs)

        return obs.astype(np.uint8)
