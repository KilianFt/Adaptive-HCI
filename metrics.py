import numpy as np
from matplotlib import pyplot as plt


def plot_and_mean(rl_reward_history, title):
    if not rl_reward_history:
        return

    plt.title(title)
    plt.plot(rl_reward_history)
    smoothed_rl_reward_history = np.convolve(rl_reward_history, np.ones(100) / 100, mode='valid')
    plt.plot(smoothed_rl_reward_history)
    plt.legend()
    plt.show()
