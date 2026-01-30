import gymnasium as gym


class GymCompatibilityWrapper(gym.Env):
    """
    Wraps old OpenAI Gym environments so they work with Gymnasium >= 1.0
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env):
        self.env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        terminated = bool(done)
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
