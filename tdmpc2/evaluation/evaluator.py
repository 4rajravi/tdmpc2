import os
import json
import numpy as np
import torch

from tdmpc2.evaluation.video_recorder import VideoRecorder
from tdmpc2.evaluation.deterministic_planner import DeterministicPlanner


class Evaluator:
    """
    TD-MPC2 evaluation loop with extended metrics.
    """

    def __init__(
        self,
        *,
        encoder,
        simnorm,
        policy_prior,
        mpc,
        env,
        cfg,
        device,
        csv_logger,
        eval_step,
    ):
        self.encoder = encoder
        self.simnorm = simnorm
        self.policy_prior = policy_prior
        self.mpc = mpc
        self.env = env
        self.device = device

        self.eval_cfg = cfg.evaluation
        self.env_cfg = cfg.env

        self.episodes = self.eval_cfg.eval_episodes
        self.record_video = self.eval_cfg.save_video

        self.success_threshold = getattr(
            self.eval_cfg, "success_threshold", None
        )

        self.save_dir = cfg.work_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.csv = csv_logger
        self.eval_step = eval_step

        self.planner = DeterministicPlanner(mpc=self.mpc)

    # --------------------------------------------------
    @torch.no_grad()
    def run(self):

        print("[EVAL] starting evaluation")

        returns = []
        lengths = []
        successes = []

        for ep in range(self.episodes):

            obs, _ = self.env.reset()
            done = False

            ep_return = 0.0
            ep_len = 0

            recorder = None
            if self.record_video:
                recorder = VideoRecorder(
                    os.path.join(self.save_dir, "videos")
                )

            # --------------------------------------------------
            # rollout
            # --------------------------------------------------
            while not done:

                obs_t = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                if obs_t.ndim == 4 and obs_t.shape[-1] == 3:
                    obs_t = obs_t.permute(0, 3, 1, 2)

                z = self.simnorm(self.encoder(obs_t))

                action = self.planner.act(z)

                if self.env_cfg.discrete_actions:
                    env_action = int(torch.argmax(action))
                else:
                    env_action = action.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, _ = self.env.step(env_action)
                done = terminated or truncated

                ep_return += reward
                ep_len += 1

                if recorder is not None:
                    recorder.record(self.env.render())

            # --------------------------------------------------
            # success metric
            # --------------------------------------------------
            if self.success_threshold is not None:
                success = int(ep_return >= self.success_threshold)
            else:
                success = 0

            # --------------------------------------------------
            # logging
            # --------------------------------------------------
            returns.append(ep_return)
            lengths.append(ep_len)
            successes.append(success)

            self.csv.log_episode(
                eval_step=self.eval_step,
                episode=ep,
                ret=ep_return,
                length=ep_len,
                success=success,
            )

            if recorder is not None:
                recorder.save(f"episode_{ep}.mp4")

            print(
                f"[EVAL] ep={ep} | "
                f"return={ep_return:.1f} | "
                f"length={ep_len} | "
                f"success={success}"
            )

        # --------------------------------------------------
        # evaluation summary
        # --------------------------------------------------
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        min_return = float(np.min(returns))
        max_return = float(np.max(returns))
        mean_length = float(np.mean(lengths))
        success_rate = float(np.mean(successes))

        self.csv.log_summary(
            eval_step=self.eval_step,
            mean_return=mean_return,
            std_return=std_return,
            min_return=min_return,
            max_return=max_return,
            mean_length=mean_length,
            success_rate=success_rate,
        )

        # --------------------------------------------------
        # JSONL history (never overwritten)
        # --------------------------------------------------
        record = {
            "eval_step": self.eval_step,
            "mean": mean_return,
            "std": std_return,
            "min": min_return,
            "max": max_return,
            "mean_length": mean_length,
            "success_rate": success_rate,
        }

        path = os.path.join(self.save_dir, "metrics_history.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print("[EVAL] summary:", record)

        return record
