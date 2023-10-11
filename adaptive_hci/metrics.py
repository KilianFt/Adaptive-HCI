import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from adaptive_hci.utils import predictions_to_onehot

def plot_and_mean(rl_reward_history, title):
    if not rl_reward_history:
        return

    plt.title(title)
    plt.plot(rl_reward_history)
    smoothed_rl_reward_history = np.convolve(rl_reward_history, np.ones(100) / 100, mode='valid')
    plt.plot(smoothed_rl_reward_history)
    plt.legend()
    plt.show()


def get_episode_accuracy(model, observations, optimal_actions):
    model.eval()
    observations = torch.tensor(observations, dtype=torch.float32)
    optimal_actions = torch.tensor(optimal_actions, dtype=torch.float32)
    if model.__class__.__name__ == 'ViT':
        observations.unsqueeze_(axis=1)
    predictions = model(observations)
    onehot_predictions = predictions_to_onehot(predictions.detach().numpy())
    return accuracy_score(optimal_actions, onehot_predictions), f1_score(optimal_actions, onehot_predictions, zero_division=1., average='macro')