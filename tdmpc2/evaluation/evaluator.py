import os
import torch

from tdmpc2.evaluation.video_recorder import VideoRecorder
from tdmpc2.evaluation.csv_logger import CSVLogger
from tdmpc2.evaluation.metrics import aggregate_metrics
from tdmpc2.evaluation.deterministic_planner import DeterministicPlanner


class Evaluator:
    """
    TD-MPC2 evaluation loop.

    Uses deterministic planning and optionally records video.
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
    eval_step
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

        self.save_dir = cfg.work_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.csv = csv_logger
        self.eval_step = eval_step


        self.planner = DeterministicPlanner(
            mpc=self.mpc
        )

    # --------------------------------------------------
    @torch.no_grad()
    def run(self):

        print("[EVAL] starting evaluation")

        returns = []

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
            # episode rollout
            # --------------------------------------------------
            while not done:

                obs_t = (
                    torch.as_tensor(obs, device=self.device)
                    .float()
                    .unsqueeze(0)
                )

                # HWC → CHW (pixel observations)
                if obs_t.ndim == 4 and obs_t.shape[-1] == 3:
                    obs_t = obs_t.permute(0, 3, 1, 2)

                # encode
                z = self.simnorm(self.encoder(obs_t))

                # plan action
                action = self.planner.act(z)

                # --------------------------------------------------
                # discrete vs continuous actions
                # --------------------------------------------------
                if self.env_cfg.discrete_actions:
                    env_action = int(torch.argmax(action))
                else:
                    env_action = action.squeeze(0).cpu().numpy()

                # --------------------------------------------------
                # Gymnasium step
                # --------------------------------------------------
                obs, reward, terminated, truncated, _ = self.env.step(env_action)
                done = terminated or truncated

                # --------------------------------------------------
                # bookkeeping
                # --------------------------------------------------
                ep_return += reward
                ep_len += 1

                if recorder is not None:
                    frame = self.env.render()
                    recorder.record(frame)

            # --------------------------------------------------
            # end episode
            # --------------------------------------------------
            if recorder is not None:
                recorder.save(f"episode_{ep}.mp4")

            self.csv.log(
                eval_step=self.eval_step,
                episode=ep,
                ret=ep_return,
                length=ep_len
            )



            returns.append(ep_return)

            print(
                f"[EVAL] episode {ep} | "
                f"return={ep_return:.1f} | "
                f"length={ep_len}"
            )

        metrics = aggregate_metrics(
            returns,
            os.path.join(self.save_dir, "metrics.json")
        )

        print("[EVAL] metrics:", metrics)

        return metrics
