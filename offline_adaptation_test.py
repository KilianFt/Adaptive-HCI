# exploration of online training

# questions to answer
# - do you fine-tune pretrained model or initalize new from scratch?
# - how often is re-init of weights beneficial?
# - how to exploration?
# 

import pickle
import pathlib
import dataclasses

import wandb
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


def load_fold_list(base_dir, filenames):
    fold_list = []
    for filename in filenames:
        filepath = base_dir / filename
        with open(filepath, 'rb') as f:
            episodes = pickle.load(f)
            fold_list.append(episodes)

    return fold_list


def main():
    device = 'mps'

    run = wandb.init(
        project="adaptive-hci",
        tags=["pretraining"],
    )

    config = {
        'pretrained': False,
        'early_stopping': True,
        'epochs': 20,
        'batch_size': 32,
        'window_size': 200,
        'overlap': 150,
        'model_class': 'ViT',
        'patch_size': 8,
        'dim': 64,
        'depth': 1,
        'heads': 2,
        'mlp_dim': 128,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'channels': 1,
    }

    # TODO try freezing layers

    wandb.config = config

    # load data
    online_data_dir = pathlib.Path('datasets/OnlineData')

    episode_filenames = [
        'episodes_2023-09-18_09-33-03.pkl',
        'episodes_2023-09-18_09-42-15.pkl',
        'episodes_2023-09-18_09-39-11.pkl',
        'episodes_2023-09-18_09-44-40.pkl',
    ]

    artifact = wandb.Artifact(name="offline_adaptattion_data", type="dataset")
    artifact.add_dir(online_data_dir, name='offline_adaptattion_data')
    run.log_artifact(artifact)

    fold_list = load_fold_list(online_data_dir, episode_filenames)

    fold_results = {
        'test_accs': [],
        'test_f1s': [],
        'test_mse': [],
    }

    for fold_idx in range(len(fold_list)):
        print('training fold', fold_idx)
        train_episodes = []
        for sublist in fold_list[:fold_idx] + fold_list[fold_idx + 1:]:
            train_episodes += sublist

        val_episodes = fold_list[fold_idx]

        (train_observations,
        train_actions,
        train_optimal_actions,
        train_rewards,
        train_terminals) = get_concatenated_arrays(episodes=train_episodes)

        (val_observations,
        val_actions,
        val_optimal_actions,
        val_rewards,
        val_terminals) = get_concatenated_arrays(episodes=val_episodes)

        train_offline_adaption_dataset = EMGWindowsAdaptattionDataset(train_observations, train_optimal_actions)
        val_offline_adaption_dataset = EMGWindowsAdaptattionDataset(val_observations, val_optimal_actions)

        train_dataloader = DataLoader(train_offline_adaption_dataset, batch_size=wandb.config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_offline_adaption_dataset, batch_size=wandb.config.batch_size, shuffle=True)

        observation_shape = train_observations.shape

        # should I load pretrained model or not?
        if config['pretrained']:
            print('load pretrained')
            model = torch.load('models/2d_fetch/model_2023-09-18_09-39-11.pkl').to(device=device)
        else:
            model = ViT(
                image_size = wandb.config.window_size,
                patch_size = wandb.config.patch_size,
                num_classes = train_offline_adaption_dataset.num_unique_labels,
                dim = wandb.config.dim,
                depth = wandb.config.depth,
                heads = wandb.config.heads,
                mlp_dim = wandb.config.mlp_dim,
                dropout = wandb.config.dropout,
                emb_dropout = wandb.config.emb_dropout,
                channels = wandb.config.channels,
            ).to(device=device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        model, history = train_model(model=model,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=val_dataloader,
                                    device=device,
                                    epochs=wandb.config.epochs,
                                    early_stopping=wandb.config.early_stopping,)
        
        fold_results['test_accs'].append(history['test_accs'][-1])
        fold_results['test_f1s'].append(history['test_f1s'][-1])
        fold_results['test_mse'].append(history['test_mse'][-1])

        wandb.log(history)
        # wandb.log_artifact(model)

    # log to wandb
    all_folds_mean_acc = np.mean(fold_results['test_accs'])
    all_folds_mean_f1 = np.mean(fold_results['test_f1s'])
    print('4 fold acc', all_folds_mean_acc)
    print('4 fold f1', all_folds_mean_f1)

    wandb.log({
        'all_folds_mean_acc': all_folds_mean_acc,
        'all_folds_mean_f1': all_folds_mean_f1,
    })


    # always train on 

if __name__ == '__main__':
    main()