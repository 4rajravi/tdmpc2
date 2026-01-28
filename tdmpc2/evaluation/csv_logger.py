import csv
import os


class CSVLogger:
    """
    Evaluation CSV logger.

    Each row corresponds to one evaluation episode.

    Columns:
        eval_step   → training step when evaluation happened
        episode     → episode index within evaluation
        return      → episode return
        length      → episode length
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # write header only once
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "eval_step",
                    "episode",
                    "return",
                    "length"
                ])

    def log(self, eval_step, episode, ret, length):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                eval_step,
                episode,
                ret,
                length
            ])
