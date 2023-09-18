# exploration of online training

# questions to answer
# - do you fine-tune pretrained model or initalize new from scratch?
# - how often is re-init of weights beneficial?
# - how to exploration?
# 

import pickle
import pathlib
import dataclasses

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vit_pytorch import ViT

from datasets import EMGWindowsAdaptattionDataset
from pretrain_model import train_model

def predictions_to_onehot(predictions):
    predicted_labels = np.zeros_like(predictions)
    predicted_labels[predictions > 0.5] = 1
    return predicted_labels


class RLAccuracy():
    def __init__(self, optimal_actions):
        self.optimal_actions = optimal_actions

    def __call__(self, algo, dataset):
        predictions = algo.predict(dataset.observations)
        onehot = predictions_to_onehot(predictions)
        return accuracy_score(self.optimal_actions, onehot)

def get_terminals(episodes, rewards):
    terminals = np.zeros(rewards.shape[0])
    last_terminal_idx = 0
    for e in episodes:
        term_idx = e['rewards'].shape[0] - 1 + last_terminal_idx
        terminals[term_idx] = 1
        last_terminal_idx = term_idx
    return terminals

def get_concatenated_arrays(episodes):
    actions = np.concatenate([predictions_to_onehot(e['actions'].detach().numpy()) \
                                    for e in episodes]).squeeze()
    optimal_actions = np.concatenate([e['optimal_actions'].detach().numpy() for e in episodes])
    observations = np.concatenate([e['user_signals'] for e in episodes]).squeeze()
    rewards = np.concatenate([e['rewards'] for e in episodes]).squeeze()

    terminals = get_terminals(episodes, rewards)

    return observations, actions, optimal_actions, rewards, terminals


def main():
    # load data
    online_data_dir = pathlib.Path('datasets/OnlineData')
    train_filename = online_data_dir / 'episodes_2023-09-18_09-33-03.pkl'
    test_filename = online_data_dir / 'episodes_2023-09-18_09-33-03.pkl'

    device = 'mps'

    with open(train_filename, 'rb') as f:
        train_episodes = pickle.load(f)
        
    with open(test_filename, 'rb') as f:
        test_episodes = pickle.load(f)

    (train_observations,
     train_actions,
     train_optimal_actions,
     train_rewards,
     train_terminals) = get_concatenated_arrays(episodes=train_episodes)

    (test_observations,
     test_actions,
     test_optimal_actions,
     test_rewards,
     test_terminals) = get_concatenated_arrays(episodes=test_episodes)

    dataset_accuracy = accuracy_score(test_optimal_actions, test_actions)
    dataset_f1_score = f1_score(test_optimal_actions, test_actions, average='micro')
    print('initial online accuracy', dataset_accuracy)
    print('initial online F1', dataset_f1_score)

    train_offline_adaption_dataset = EMGWindowsAdaptattionDataset(train_observations, train_optimal_actions)
    test_offline_adaption_dataset = EMGWindowsAdaptattionDataset(train_observations, train_optimal_actions)

    train_dataloader = DataLoader(train_offline_adaption_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_offline_adaption_dataset, batch_size=32, shuffle=True)

    observation_shape = train_observations.shape

    # should I load pretrained model or not?
    model = ViT(
        image_size = observation_shape[2],
        patch_size = observation_shape[1],
        num_classes = train_offline_adaption_dataset.num_unique_labels,
        dim = 64,
        depth = 1,
        heads = 2,
        mlp_dim = 128,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels=1,
    ).to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model, history = train_model(model=model,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 device=device,
                                 epochs=10)

if __name__ == '__main__':
    main()