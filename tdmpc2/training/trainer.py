import torch
import numpy as np

from tdmpc2.replay.replay_buffer import ReplayBuffer
from tdmpc2.training.update_step import update_step
from tdmpc2.training.optimizers import make_optimizers
from tdmpc2.training.target_networks import EMATarget


class Trainer:
    """
    TD-MPC2 training loop.
    """

    def __init__(
        self,
        *,
        env,
        encoder,
        simnorm,
        world_model,
        policy_prior,
        mpc,
        cfg,
        device
    ):
        self.env = env
        self.device = device

        self.encoder = encoder
        self.simnorm = simnorm
        self.world_model = world_model
        self.policy_prior = policy_prior
        self.mpc = mpc

        # configs
        self.train_cfg = cfg.training
        self.env_cfg = cfg.env

        # ------------------------------------------------
        # replay buffer
        # ------------------------------------------------
        self.buffer = ReplayBuffer(
            capacity=self.train_cfg.buffer_size
        )

        # ------------------------------------------------
        # optimizer (encoder + world model + policy)
        # ------------------------------------------------
        self.optimizer = make_optimizers(
            encoder=self.encoder,
            world_model=self.world_model,
            policy_prior=self.policy_prior,
            lr=self.train_cfg.lr
        )

        # ------------------------------------------------
        # EMA target Q network
        # ------------------------------------------------
        self.target_q = EMATarget(
            model=self.world_model.value,
            tau=self.train_cfg.tau
        )

        # ------------------------------------------------
        # training schedule
        # ------------------------------------------------
        self.batch_size = self.train_cfg.batch_size
        self.seed_steps = self.train_cfg.seed_steps
        self.update_every = self.train_cfg.update_every
        self.updates_per_step = self.train_cfg.updates_per_step

        self.total_steps = 0

        self.obs, _ = self.env.reset()

    # --------------------------------------------------
    def train(self, steps):

        for _ in range(steps):

            action_idx, action_vec = self.act(self.obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(action_idx)
            #reward = reward / self.env_cfg.reward_scale
            done = terminated or truncated

            self.buffer.add(
                obs=self.obs,
                action=action_vec,
                reward=reward,
                next_obs=next_obs,
                done=done
            )

            self.obs = next_obs
            self.total_steps += 1

            if done:
                self.obs, _ = self.env.reset()

            # --------------------------------------------------
            # model updates
            # --------------------------------------------------
            if (
                self.total_steps >= self.seed_steps
                and len(self.buffer) >= self.batch_size
                and self.total_steps % self.update_every == 0
            ):
                for _ in range(self.updates_per_step):

                    batch = self.buffer.sample(self.batch_size)

                    update_step(
                        batch=batch,
                        encoder=self.encoder,
                        simnorm=self.simnorm,
                        world_model=self.world_model,
                        target_networks=self.target_q,
                        policy_prior=self.policy_prior,
                        optimizers=self.optimizer,
                        cfg=self.train_cfg,
                        device=self.device
                    )

                    # EMA update
                    self.target_q.update()

    # --------------------------------------------------
    @torch.no_grad()
    def act(self, obs):

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        if obs.ndim == 3:
            obs = obs.permute(2, 0, 1)

        obs = obs.unsqueeze(0)

        # ------------------------------------
        # encode
        # ------------------------------------
        z = self.simnorm(self.encoder(obs))

        # ------------------------------------
        # get continuous action
        # ------------------------------------
        if self.mpc is not None:

            pi_action, _ = self.policy_prior.sample(z)
            mpc_action = self.mpc.plan(z)

            print("[POLICY GAP]",
                (mpc_action - pi_action).abs().mean().item())

            action_vec = mpc_action


        else:
            action_vec, _ = self.policy_prior.sample(z)

        # ------------------------------------
        # discrete environment interface
        # ------------------------------------
        if self.env_cfg.discrete_actions:

            action_idx = int(torch.argmax(action_vec))

            one_hot = torch.zeros(
                self.env.action_space.n,
                device=self.device
            )
            one_hot[action_idx] = 1.0


            return action_idx, one_hot

        else:


            return (
                action_vec.squeeze(0).cpu().numpy(),
                action_vec.squeeze(0)
            )

