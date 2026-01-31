import os
import csv


class CSVLogger:
    """
    Logs evaluation metrics to CSV.
    """

    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        self.episode_path = os.path.join(save_dir, "eval_episodes.csv")
        self.summary_path = os.path.join(save_dir, "eval_summary.csv")

        self._init_episode_csv()
        self._init_summary_csv()

    # --------------------------------------------------
    def _init_episode_csv(self):
        if not os.path.exists(self.episode_path):
            with open(self.episode_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["eval_step", "episode", "return", "length", "success"]
                )

    # --------------------------------------------------
    def _init_summary_csv(self):
        if not os.path.exists(self.summary_path):
            with open(self.summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "eval_step",
                        "mean_return",
                        "std_return",
                        "min_return",
                        "max_return",
                        "mean_length",
                        "success_rate",
                    ]
                )

    # --------------------------------------------------
    def log_episode(self, *, eval_step, episode, ret, length, success):
        with open(self.episode_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [eval_step, episode, ret, length, success]
            )

    # --------------------------------------------------
    def log_summary(
        self,
        *,
        eval_step,
        mean_return,
        std_return,
        min_return,
        max_return,
        mean_length,
        success_rate,
    ):
        with open(self.summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    eval_step,
                    mean_return,
                    std_return,
                    min_return,
                    max_return,
                    mean_length,
                    success_rate,
                ]
            )
