import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training(json_path="training/training_returns_baseline.json", smoothing_window=50):
    # Load data
    with open(json_path, "r") as f:
        returns = json.load(f)

    episodes = np.arange(len(returns))

    # Smooth curve using moving average
    smoothed = np.convolve(
        returns,
        np.ones(smoothing_window)/smoothing_window,
        mode='valid'
    )

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, returns, alpha=0.3, label='Raw Returns')
    plt.plot(
        episodes[:len(smoothed)],
        smoothed,
        linewidth=3,
        label=f'Smoothed ({smoothing_window}-episode MA)'
    )

    plt.title("Training Return Curve")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_training()
