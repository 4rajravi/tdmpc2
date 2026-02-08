import torch


class MPC:
    """
    TD-MPC2 latent-space Model Predictive Control.
    """

    def __init__(
        self,
        *,
        world_model,
        policy_prior,
        planner_cfg,
        training_cfg,
        mppi
    ):
        self.model = world_model
        self.policy_prior = policy_prior

        self.planner_cfg = planner_cfg
        self.training_cfg = training_cfg

        self.H = planner_cfg.horizon
        self.iterations = planner_cfg.iterations
        self.gamma = training_cfg.discount

        self.mppi = mppi

        self.first_step = True

    # --------------------------------------------------
    @torch.no_grad()
    def plan(self, z0):
        """
        Returns first action of optimized trajectory.
        """

        # ---------------------------------------------
        # TD-MPC2 warm start
        # ---------------------------------------------
        if self.first_step:
            self.mppi.reset()
            self.first_step = False
        else:
            self.mppi.warm_start()

        # ---------------------------------------------
        # MPPI optimization
        # ---------------------------------------------
        for _ in range(self.iterations):

            actions = self.mppi.sample(
                policy_prior=self.policy_prior,
                z=z0
            )

            returns = self.evaluate(z0, actions)

            self.mppi.update(actions, returns)

        # execute first action
        action = self.mppi.mean[0]

        print(
        "[MPC DEBUG]",
        "mean_std:", self.mppi.std.mean().item(),
        "max_std:", self.mppi.std.max().item(),
        "action_norm:", action.norm().item()
    )
        print("[PLAN VALUE]",
      "mean_return:", returns.mean().item(),
      "max_return:", returns.max().item())



        return action

    # --------------------------------------------------
    def evaluate(self, z0, actions):
        """
        Roll out trajectories in latent space.
        """

        B = actions.shape[0]
        z = z0.repeat(B, 1)

        total_return = torch.zeros(B, device=z.device)

        # ---------------------------------------------
        # latent rollout
        # ---------------------------------------------
        for t in range(self.H):

            a = actions[:, t]

            out = self.model(z, a)

            z = out["z_next"]

            r_logits = out["reward"]
            probs = r_logits.softmax(-1)

            num_bins = probs.shape[-1]

            support = torch.linspace(
                self.training_cfg.vmin,
                self.training_cfg.vmax,
                num_bins,
                device=probs.device
            )

            r = (probs * support).sum(-1)


            total_return += (self.gamma ** t) * r

        # ---------------------------------------------
        # terminal value bootstrap
        # ---------------------------------------------
        a_T, _ = self.policy_prior.sample(z)

        q_logits = self.model.value(z, a_T)
        num_bins = q_logits[0].shape[-1]

        support = torch.linspace(
            self.training_cfg.vmin,
            self.training_cfg.vmax,
            num_bins,
            device=z.device
        )

        qs = []
        for q in q_logits:
            probs = q.softmax(-1)
            qs.append((probs * support).sum(-1))

        q = torch.stack(qs, dim=0).min(0)[0]

        total_return += (self.gamma ** self.H) * q

        return total_return
