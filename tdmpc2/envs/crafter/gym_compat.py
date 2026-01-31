import gymnasium as gym


class GymCompatibilityWrapper(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_mode = "rgb_array"

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass

        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        terminated = bool(done)
        truncated = info.get("TimeLimit.truncated", False)

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
