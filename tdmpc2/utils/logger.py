import csv
import os
from datetime import datetime


class Logger:
    """
    Lightweight CSV logger.

    Used for training metrics.
    """

    def __init__(self, cfg, name="train"):
        """
        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config

        name : str
            train | eval | debug
        """

        self.cfg = cfg
        self.enabled = cfg.logging.save_csv

        if not self.enabled:
            self.file = None
            return

        # --------------------------------------------------
        # directory
        # --------------------------------------------------
        self.log_dir = os.path.join(
            cfg.work_dir,
            "results",
            "logs"
        )

        os.makedirs(self.log_dir, exist_ok=True)

        filename = f"{name}.csv"
        self.path = os.path.join(self.log_dir, filename)

        # --------------------------------------------------
        # open file
        # --------------------------------------------------
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "step",
            "episode_return",
            "episode_length",
            "loss_total",
            "loss_consistency",
            "loss_reward",
            "loss_value"
        ])

    # --------------------------------------------------
    def log(self, **metrics):
        """
        Log arbitrary named metrics.

        Example:
            logger.log(step=1000, episode_return=42.1)
        """

        if not self.enabled:
            return

        row = [
            metrics.get("step"),
            metrics.get("episode_return"),
            metrics.get("episode_length"),
            metrics.get("loss_total"),
            metrics.get("loss_consistency"),
            metrics.get("loss_reward"),
            metrics.get("loss_value"),
        ]

        self.writer.writerow(row)
        self.file.flush()

    # --------------------------------------------------
    def close(self):
        if self.file:
            self.file.close()
