import os
import pathlib

import wandb
import torch
import numpy as np
from d3rlpy.datasets import MDPDataset

from adaptive_hci import utils
from adaptive_hci.controllers import SLOnlyController
from adaptive_hci.datasets import get_concatenated_user_episodes, load_online_episodes
from adaptive_hci.metrics import get_episode_accuracy


def simulate_online_adaptation():
    _ = wandb.init(
       tags=["test_online_adaptation"],
    )

    device = utils.get_device()

    # batch_size = 32
    # lr = 1e-3
    # epochs = 1
    # n_frozen_layers = 2
    # train_intervals = 1
    # first_training_episode = 0

    model_path = pathlib.Path('models/pretrained_2023-09-20_22-18-02.pt')
    controller = SLOnlyController(model_path=model_path,
                                  device=device,
                                  lr=wandb.config.lr,
                                  batch_size=wandb.config.batch_size,
                                  epochs=wandb.config.epochs,
                                  n_frozen_layers=wandb.config.n_frozen_layers)

    online_data_dir = pathlib.Path('datasets/AdaptationTest')
    episode_filenames = sorted(os.listdir(online_data_dir))    
    online_data = load_online_episodes(online_data_dir, episode_filenames)

    current_trial_episodes = online_data[0]

    (observations,
     actions,
     optimal_actions,
     rewards,
     terminals) = get_concatenated_user_episodes(episodes=current_trial_episodes)

    rl_dataset = MDPDataset(observations=observations,
                            actions=optimal_actions,
                            rewards=rewards,
                            terminals=terminals)

    # simulate online data replay
    results = []
    for ep_idx, episode in enumerate(rl_dataset.episodes):
        ep_acc, ep_f1 = get_episode_accuracy(controller.policy, episode.observations, episode.actions)

        if ep_idx >= wandb.config.first_training_episode and ep_idx % wandb.config.train_intervals == 0:
            _ = controller.sl_update(episode.observations,
                                episode.actions)

        results.append({
            'accuracy': ep_acc,
            'f1': ep_f1,
        })

    wandb.log({
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
    })
    
def main():
    random_seed = 100
    torch.manual_seed(random_seed)

    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'f1'},
        'parameters': {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'max': 10, 'min': 1},
            'lr': {'max': 0.005, 'min': 0.0001},
            'model_class': {'value': 'ViT'},
            'n_frozen_layers': {'max': 2, 'min': 0},
            'train_intervals': {'max': 5, 'min': 1},
            'first_training_episode': {'max': 10, 'min': 0},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="adaptive-hci-online-test")

    wandb.agent(sweep_id, function=simulate_online_adaptation, count=100)

if __name__ == '__main__':
    main()