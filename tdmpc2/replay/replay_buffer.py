import random
import torch


class ReplayBuffer:
    """
    Episode-based replay buffer for TD-MPC2.

    Stores full episodes and samples
    fixed-length sequences for training.
    """

    def __init__(self, capacity, horizon=1):
        self.capacity = capacity
        self.horizon = horizon

        self.episodes = []
        self.current_episode = []

    # ------------------------------------------------
    def __len__(self):
        return sum(len(ep) for ep in self.episodes)

    # ------------------------------------------------
    def add(self, obs, action, reward, next_obs, done):

        self.current_episode.append(
            (
                torch.as_tensor(obs),
                torch.as_tensor(action, dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
                torch.as_tensor(next_obs),
                torch.tensor(done, dtype=torch.float32)

            )
        )

        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []

            if len(self.episodes) > self.capacity:
                self.episodes.pop(0)

    # ------------------------------------------------
    def sample(self, batch_size):
        """
        Sample batch of H-step sequences.

        Returns:
            obs
            action
            reward
            next_obs
            done
        """

        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_next_obs = []
        batch_done = []

        for _ in range(batch_size):

            ep = random.choice(self.episodes)

            idx = random.randint(
                0,
                max(0, len(ep) - self.horizon - 1)
            )

            obs, action, reward, next_obs, done = ep[idx]

            batch_obs.append(obs)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_next_obs.append(next_obs)
            batch_done.append(done)

        return (
            torch.stack(batch_obs),
            torch.stack(batch_action),
            torch.stack(batch_reward),
            torch.stack(batch_next_obs),
            torch.stack(batch_done)
        )
