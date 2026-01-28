import json
import numpy as np
import os


def aggregate_metrics(returns, save_path):

    metrics = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns))
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
