import json
import matplotlib.pyplot as plt
import numpy as np

def load_returns(path):
    with open(path, "r") as f:
        return json.load(f)

def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

baseline = load_returns("training/training_returns_baseline.json")
shaped = load_returns("training/training_returns_shaped.json")

episodes_b = np.arange(len(baseline))
episodes_s = np.arange(len(shaped))

plt.figure(figsize=(12, 6))

plt.plot(episodes_b[:len(smooth(baseline))], smooth(baseline), label="Baseline", linewidth=3)
plt.plot(episodes_s[:len(smooth(shaped))], smooth(shaped), label="Reward Shaping", linewidth=3)

plt.title("Baseline vs Reward Shaping – Training Performance")
plt.xlabel("Episode")
plt.ylabel("Smoothed Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
