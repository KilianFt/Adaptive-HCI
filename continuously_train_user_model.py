import os
import sys
import copy
import pathlib

import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule

import configs
from adaptive_hci.datasets import EMGWindowsAdaptationDataset, \
                                  get_concatenated_user_episodes, \
                                  load_online_episodes

def main(finetuned_model: LightningModule, user_hash, config: configs.BaseConfig) -> LightningModule:

    logger = WandbLogger(project='adaptive_hci',
                         tags=["online_adaptation", user_hash],
                         config=config,
                         name=f"online_adapt_{config}_{user_hash[:15]}")

    # TODO smoke config

    pl_model = copy.deepcopy(finetuned_model)
    pl_model.freeze_layers(config.online_n_frozen_layers)
    pl_model.lr = config.online_lr

    online_data_dir = pathlib.Path('datasets/AdaptationTest')
    episode_filenames = sorted(os.listdir(online_data_dir))    
    online_data = load_online_episodes(online_data_dir, episode_filenames)

    # TODO include other online data?
    current_trial_episodes = online_data[0]

    (observations,
     actions,
     optimal_actions,
     rewards,
     terminals) = get_concatenated_user_episodes(episodes=current_trial_episodes)

    terminal_idxs = torch.argwhere(torch.tensor(terminals)).squeeze()
    observations = torch.tensor(observations)
    optimal_actions = torch.tensor(optimal_actions)

    # FIXME
    is_smoke = isinstance(config, configs.SmokeConfig)
    if is_smoke:
        terminal_idxs = terminal_idxs[:1]

    # simulate online data replay
    results = []
    for ep_idx, terminal_idx in enumerate(terminal_idxs):
        # Can we avoid reloading the dataloader? problem is that max_epochs only works with calling fit once
        trainer = pl.Trainer(limit_train_batches=100,
                    max_epochs=config.online_epochs,
                    log_every_n_steps=1,
                    logger=logger,
                    )
            
        # validation
        val_last_terminal_idx = terminal_idxs[(ep_idx - 1)]
        ep_val_observations = observations[val_last_terminal_idx:terminal_idx]
        ep_val_optimal_actions = optimal_actions[val_last_terminal_idx:terminal_idx]

        val_dataset = EMGWindowsAdaptationDataset(ep_val_observations,
                                                  ep_val_optimal_actions)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=config.online_batch_size,
                                    num_workers=8)
    
        hist = trainer.validate(model=pl_model,
                                dataloaders=val_dataloader)
        
        results.append(hist[0])

        if ep_idx >= config.online_first_training_episode and ep_idx % config.online_train_intervals == 0:
            # training
            # FIXME change way of loading training data, maybe random epochs
            last_batches_to_train = config.online_train_intervals
            last_terminal_idx = terminal_idxs[(ep_idx - last_batches_to_train)]

            ep_observations = observations[last_terminal_idx:terminal_idx]
            ep_optimal_actions = optimal_actions[last_terminal_idx:terminal_idx]

            train_dataset = EMGWindowsAdaptationDataset(ep_observations,
                                                        ep_optimal_actions)
            train_dataloader = DataLoader(train_dataset,
                                        batch_size=config.online_batch_size,
                                        num_workers=8)

            trainer.fit(model=pl_model,
                train_dataloaders=train_dataloader)

    wandb.run.log({
        'mean_accuracy': np.mean([r['val_acc'] for r in results]),
        'mean_f1': np.mean([r['val_f1'] for r in results]),
    })

    return pl_model
    

if __name__ == '__main__':
    random_seed = 100
    torch.manual_seed(random_seed)

    # TODO download data if not present

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

    base_configuration = {
        'batch_size': 16,
        'epochs': 9,
        'lr': 3.5e-3,
        'n_frozen_layers': 2,
        'train_intervals': 4,
        'first_training_episode': 0,
    }

    if sys.gettrace() is not None:
        main(base_configuration)
    else:
        # model_path = pathlib.Path('models/pretrained_2023-09-20_22-18-02.pt')
        # TODO load model and pass to main
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="adaptive-hci-online-test")
        wandb.agent(sweep_id, function=main, count=100)
