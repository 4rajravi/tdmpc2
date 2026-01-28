import torch
import torch.nn.functional as F


class MPPI:
    """
    Model Predictive Path Integral (MPPI) optimizer.

    All hyperparameters are read from cfg.planner.
    """

    def __init__(self, planner_cfg, action_dim, device):
        """
        Parameters
        ----------
        planner_cfg : DictConfig
            cfg.planner

        action_dim : int
            Action dimension

        device : torch.device
        """

        self.cfg = planner_cfg

        self.H = planner_cfg.horizon
        self.action_dim = action_dim

        self.num_samples = planner_cfg.num_samples
        self.num_elites = planner_cfg.num_elites
        self.temperature = planner_cfg.temperature

        self.min_std = planner_cfg.min_std
        self.max_std = planner_cfg.max_std

        self.num_pi_trajs = planner_cfg.get("num_pi_trajs", 0)

        self.device = device

        # trajectory distribution parameters
        self.mean = torch.zeros(self.H, action_dim, device=device)
        self.std = torch.ones(self.H, action_dim, device=device)

    # planning lifecycle
    def reset(self):
        self.mean.zero_()
        self.std.fill_(1.0)

    def warm_start(self):
        """
        Shift solution by one step.
        """
        self.mean[:-1] = self.mean[1:].clone()
        self.mean[-1].zero_()

    # sampling
    def sample(self, policy_prior=None, z=None):
        """
        Sample action trajectories.

        Mixes:
          - Gaussian samples
          - policy prior trajectories
        """

        # Gaussian samples
        noise = torch.randn(
            self.num_samples,
            self.H,
            self.action_dim,
            device=self.device
        )

        actions = self.mean + self.std * noise

        # policy prior rollouts (TD-MPC2 improvement)
        if (
            policy_prior is not None
            and self.num_pi_trajs > 0
            and z is not None
        ):
            pi_actions = []

            z_pi = z.repeat(self.num_pi_trajs, 1)

            for t in range(self.H):
                a, _ = policy_prior.sample(z_pi)
                pi_actions.append(a)
                z_pi = z_pi 

            pi_actions = torch.stack(pi_actions, dim=1)

            actions[: self.num_pi_trajs] = pi_actions

        return actions

    # distribution update
    def update(self, actions, returns):
        """
        Update mean and std using weighted MPPI update.
        """

        # normalize returns for stability
        returns = returns - returns.max()

        weights = F.softmax(
            returns / self.temperature,
            dim=0
        )

        mean = (weights[:, None, None] * actions).sum(dim=0)

        var = (
            weights[:, None, None]
            * (actions - mean).pow(2)
        ).sum(dim=0)

        std = torch.sqrt(var + 1e-6)

        self.mean = mean
        self.std = std.clamp(
            self.min_std,
            self.max_std
        )
