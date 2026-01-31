import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_all(work_dir):

    csv_path = os.path.join(work_dir,"eval_csv", "eval_summary.csv")
    plot_dir = os.path.join(work_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    steps = df["eval_step"]
    mean = df["mean_return"]
    std = df["std_return"]

    # --------------------------------------------------
    # 1. Learning curve
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(steps, mean, label="Mean return")
    plt.fill_between(
        steps,
        mean - std,
        mean + std,
        alpha=0.3
    )
    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "learning_curve.png"))
    plt.close()

    # --------------------------------------------------
    # 2. Success rate
    # --------------------------------------------------
    if "success_rate" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(
            steps,
            df["success_rate"],
            marker="o"
        )
        plt.xlabel("Environment steps")
        plt.ylabel("Success rate")
        plt.title("Success Rate")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "success_rate.png"))
        plt.close()

    # --------------------------------------------------
    # 3. Episode length
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(
        steps,
        df["mean_length"],
        marker="o"
    )
    plt.xlabel("Environment steps")
    plt.ylabel("Episode length")
    plt.title("Episode Length")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "episode_length.png"))
    plt.close()

    # --------------------------------------------------
    # 4. Final performance bar
    # --------------------------------------------------
    final_mean = mean.iloc[-1]
    final_std = std.iloc[-1]

    plt.figure(figsize=(4, 5))
    plt.bar(
        ["Final"],
        [final_mean],
        yerr=[final_std],
        capsize=8
    )
    plt.ylabel("Return")
    plt.title("Final Performance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "final_performance.png"))
    plt.close()

    print(f"[PLOT] saved to {plot_dir}")
