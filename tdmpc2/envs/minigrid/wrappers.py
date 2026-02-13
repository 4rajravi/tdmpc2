import numpy as np
import gymnasium as gym

from minigrid.wrappers import (
    RGBImgObsWrapper,
    ImgObsWrapper
)
from gymnasium.wrappers import ResizeObservation


class DirectionChannelWrapper(gym.ObservationWrapper):
    """
    Adds agent direction as an extra image channel.
    Output shape becomes (H, W, 4).
    """

    def observation(self, obs):
        # After ImgObsWrapper, obs is an RGB image (H, W, 3)
        img = obs

        # Agent direction: 0,1,2,3
        direction = self.unwrapped.agent_dir

        h, w, _ = img.shape

        # Encode direction as pixel intensity
        #dir_value = (direction / 3.0) * 255.0
        dir_value = direction * 64.0

        dir_channel = np.ones((h, w, 1), dtype=img.dtype) * dir_value

        return np.concatenate([img, dir_channel], axis=-1).astype(np.float32)



def make_minigrid_wrapped_env(env, size=64):

    env = RGBImgObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)

    # ⭐ Resize FIRST
    env = ResizeObservation(env, (size, size))

    # ⭐ Then add direction channel (no interpolation)
    env = DirectionChannelWrapper(env)

    return env
