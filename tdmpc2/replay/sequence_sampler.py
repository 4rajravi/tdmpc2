import random
import torch


class SequenceSampler:
    """
    Samples contiguous H-step trajectories from replay buffer.
    """

    def __init__(self, buffer, horizon):
        self.buffer = buffer
        self.horizon = horizon

    def sample(self, batch_size):
        """
        Returns:
            obs:       (B, H+1, ...)
            actions:   (B, H, action_dim)
            rewards:   (B, H)
            dones:     (B, H)
        """

        episodes = self.buffer.episodes
        batch = []

        while len(batch) < batch_size:
            episode = random.choice(episodes)

            if len(episode) <= self.horizon:
                continue

            start = random.randint(
                0,
                len(episode) - self.horizon - 1
            )

            traj = episode[start : start + self.horizon + 1]
            batch.append(traj)

        # unpack
        obs = []
        actions = []
        rewards = []
        dones = []

        for traj in batch:
            obs.append([t[0] for t in traj])
            actions.append([t[1] for t in traj[:-1]])
            rewards.append([t[2] for t in traj[:-1]])
            dones.append([t[4] for t in traj[:-1]])

        return (
            torch.stack(obs),
            torch.stack(actions),
            torch.tensor(rewards),
            torch.tensor(dones),
        )
