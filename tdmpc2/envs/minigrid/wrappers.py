from minigrid.wrappers import (
    FullyObsWrapper,
    RGBImgObsWrapper,
    ImgObsWrapper
)
from gymnasium.wrappers import ResizeObservation


def make_minigrid_wrapped_env(env, size=64):
    """
    Convert MiniGrid symbolic observations to real RGB images.
    """

    # # full visibility
    # env = FullyObsWrapper(env)

    # IMPORTANT: force pixel rendering
    env = RGBImgObsWrapper(env, tile_size=8)

    # keep only image
    env = ImgObsWrapper(env)

    # resize to CNN-friendly size
    env = ResizeObservation(env, (size, size))

    return env
