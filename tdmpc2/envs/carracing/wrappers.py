import numpy as np
import gymnasium as gym


class CarRacingActionWrapper(gym.ActionWrapper):
    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1],  0.0, 1.0)
        action[2] = np.clip(action[2],  0.0, 1.0)
        return action


def make_carracing_wrapped_env(env):
    # ONLY clip actions
    # NO observation transforms
    return CarRacingActionWrapper(env)
