# exploration of online training

# questions to answer
# - do you fine-tune pretrained model or initalize new from scratch?
# - how often is re-init of weights beneficial?
# - how to exploration?
#

import os
import pathlib
import pickle
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import utils
import wandb
from adaptive_hci.datasets import EMGWindowsAdaptattionDataset
from adaptive_hci.training import train_model

base_configuration = {
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.005,
    'n_frozen_layers': 2,
    'window_size': 600,
    'overlap': 100,
    'model_class': 'ViT',
}


def predictions_to_onehot(predictions):
    predicted_labels = np.zeros_like(predictions)
    predicted_labels[predictions > 0.5] = 1
    return predicted_labels


class RLAccuracy:
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


def main(config=None):
    device = utils.get_device()
    run = wandb.init(tags=["offline_adaptation"], config=config)
    config = wandb.config

    online_data_dir = pathlib.Path('datasets/OnlineData')
    episode_filenames = os.listdir(online_data_dir)

    artifact = wandb.Artifact(name="offline_adaptattion_data", type="dataset")
    artifact.add_dir(online_data_dir, name='offline_adaptattion_data')
    run.log_artifact(artifact)

    fold_list = load_fold_list(online_data_dir, episode_filenames)

    fold_results = {
        'test_accs': [],
        'test_f1s': [],
        'test_mse': [],
        'train_loss': [],
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

        train_dataloader = DataLoader(train_offline_adaption_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_offline_adaption_dataset, batch_size=config.batch_size, shuffle=True)

        model = torch.load('models/pretrained_parameter_search.pt').to(device=device)

        if config.n_frozen_layers >= 1:
            for i, param in enumerate(model.to_patch_embedding.parameters()):
                param.requires_grad = False
            for i, param in enumerate(model.dropout.parameters()):
                param.requires_grad = False

        if config.n_frozen_layers >= 2:
            for layer_idx in range(min((config.n_frozen_layers - 1), 4)):
                for i, param in enumerate(model.transformer.layers[layer_idx].parameters()):
                    param.requires_grad = False

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        model, history = train_model(model=model,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     train_dataloader=train_dataloader,
                                     test_dataloader=val_dataloader,
                                     device=device,
                                     epochs=config.epochs,
                                     logger=wandb.run)

        fold_results['test_accs'].append(history['test_accs'])
        fold_results['test_f1s'].append(history['test_f1s'])
        fold_results['test_mse'].append(history['test_mse'])
        fold_results['train_loss'].append(history['train_loss'])

    all_fold_test_accs = np.array(fold_results['test_accs']).mean(axis=0)
    all_fold_test_f1s = np.array(fold_results['test_f1s']).mean(axis=0)
    all_fold_test_mses = np.array(fold_results['test_mse']).mean(axis=0)
    all_fold_train_losses = np.array(fold_results['train_loss']).mean(axis=0)

    for acc, f1, mse, train_loss in zip(all_fold_test_accs, all_fold_test_f1s, all_fold_test_mses,
                                        all_fold_train_losses):
        wandb.log({
            'all_fold_test_acc': acc,
            'all_folds_test_f1': f1,
            'all_folds_test_mse': mse,
            'all_folds_train_loss': train_loss,
        })

    best_acc_idx = np.argmax(all_fold_test_accs)
    wandb.run.summary["best_accuracy"] = all_fold_test_accs[best_acc_idx]
    wandb.run.summary["best_f1"] = all_fold_test_f1s[best_acc_idx]
    wandb.run.summary["best_mse"] = all_fold_test_mses[best_acc_idx]


def maybe_download_drive_folder():
    destination_folder = "datasets/OnlineData"
    if os.path.exists(destination_folder):
        print("Folder already exists")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    file_ids = [
        "1Sitb0ooo2izvkHQGNQkXTGoDV4CJAnFF",
        "1bIYLJVu-SqHzRnTFxuc1vkzRBs8Ll5Oi",
        "1D7h11vheJ7Oq8Ju4ik8jqBJUocEie-rQ",
        "1EWJdHHZ22xorZEpss-gf5R4cxehEs9pt",
    ]

    for file_id in file_ids:
        download_path = os.path.join(destination_folder, file_id + ".pkl")
        cmd = f"gdown https://drive.google.com/uc?id={file_id} -O {download_path}"
        subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    random_seed = 100
    torch.manual_seed(random_seed)

    maybe_download_drive_folder()

    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'test_acc'},
        'parameters': {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'value': 50},
            'lr': {'max': 0.005, 'min': 0.0001},
            'n_frozen_layers': {'max': 5, 'min': 0},
            'window_size': {'value': 600},
            'overlap': {'values': [50, 100, 150]},
            'model_class': {'value': 'ViT'},
        },
    }

    if sys.gettrace() is not None:
        main(base_configuration)
    else:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="adaptive-hci-offline-adaptation")
        wandb.agent(sweep_id, function=main, count=10)
