import os
import copy
import pathlib
import hashlib
import argparse

import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from d3rlpy.datasets import MDPDataset
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule

import configs
from adaptive_hci.controllers import PLModel
from adaptive_hci.datasets import EMGWindowsAdaptationDataset, \
                                  get_concatenated_user_episodes, \
                                  load_online_episodes

def main(finetuned_model: LightningModule, user_hash, config: configs.BaseConfig, do_sweep: bool = False) -> LightningModule:

    # FIXME load wandb config if sweep
    if do_sweep:
        run = wandb.init()
        print('n frozen', wandb.config.online_n_frozen_layers)

        config = wandb.config
        print('n frozen', config.online_n_frozen_layers)

        # wandb_args = {
        #     'project': 'adaptive_hci',
        #     'tags': ["online_adaptation", user_hash],
        #     'config': wandb.config,
        #     'name': f"online_adapt_{config}_{user_hash[:15]}"
        # }

    wandb_args = {
        'project': 'adaptive_hci',
        'tags': ["online_adaptation", user_hash],
        'config': config,
        'name': f"online_adapt_{config}_{user_hash[:15]}"
    }

    print('before logger')
    logger = WandbLogger(**wandb_args)



    pl_model = copy.deepcopy(finetuned_model)
    pl_model.freeze_layers(config.online_n_frozen_layers)
    pl_model.lr = config.online_lr

    online_data_dir = pathlib.Path('datasets/AdaptationTest')
    episode_filenames = sorted(os.listdir(online_data_dir))    
    online_data = load_online_episodes(online_data_dir, episode_filenames)

    # include other online data?
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

    if config.online_num_episodes is not None:
        all_episodes = rl_dataset.episodes[:config.online_num_episodes]
    else:
        all_episodes = rl_dataset.episodes

    # simulate online data replay
    results = []
    for ep_idx, episode in enumerate(all_episodes):
        # Can we avoid reloading the trainer? problem is that max_epochs only works with calling fit once
        trainer = pl.Trainer(limit_train_batches=100,
                    max_epochs=config.online_epochs,
                    log_every_n_steps=1,
                    logger=logger,
                    )
            
        # validation
        val_dataset = EMGWindowsAdaptationDataset(episode.observations, episode.actions)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=config.online_batch_size,
                                    num_workers=8)
    
        hist = trainer.validate(model=pl_model, dataloaders=val_dataloader)
        results.append(hist[0])


        # training
        if ep_idx >= config.online_first_training_episode and ep_idx % config.online_train_intervals == 0:

            # pick n random past episodes
            if ep_idx > 0:
                rand_episode_idxs = torch.randint(0, ep_idx, size=(config.online_additional_train_episodes,)).unique()
                train_episodes = [all_episodes[r_e_idx] for r_e_idx in rand_episode_idxs]
                train_episodes.append(episode)
            else:
                train_episodes = [episode]

            train_observations = np.concatenate([e.observations for e in train_episodes])
            train_actions = np.concatenate([e.actions for e in train_episodes])
            train_dataset = EMGWindowsAdaptationDataset(train_observations, train_actions)
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=config.online_batch_size,
                                          num_workers=8,
                                          shuffle=True)

            trainer.fit(model=pl_model, train_dataloaders=train_dataloader)

    print('done')
    wandb.run.log({
        'mean_accuracy': np.mean([r['val_acc'] for r in results]),
        'mean_f1': np.mean([r['val_f1'] for r in results]),
    })
    print('end')
    return pl_model
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Adaptive HCI - Fetch')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    random_seed = 100
    torch.manual_seed(random_seed)

    # TODO download data if not present

    pl_model = PLModel.load_from_checkpoint('./adaptive_hci/yp8k1lmf/checkpoints/epoch=0-step=100.ckpt')

    user_hash = hashlib.sha256("Kilian".encode("utf-8")).hexdigest()
    config = configs.SmokeConfig()

    if args.sweep:
        sweep_configuration = {
            'method': 'bayes',
            'name': 'sweep',
            'metric': {'goal': 'maximize', 'name': 'f1'},
            'parameters': {
                'online_batch_size': {'values': [16, 32, 64]},
                'online_epochs': {'max': 10, 'min': 1},
                'online_lr': {'max': 0.005, 'min': 0.0001},
                'online_model_class': {'value': 'ViT'},
                'online_n_frozen_layers': {'max': 2, 'min': 0},
                'online_train_intervals': {'max': 5, 'min': 1},
                'online_first_training_episode': {'max': 10, 'min': 0},
                'online_additional_train_episodes': {'max': 10, 'min': 0},
            },
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project="adaptive-hci-online-test")
        wandb.agent(sweep_id,
                    function=main(finetuned_model=pl_model,
                                    user_hash=user_hash,
                                    config=config,
                                    do_sweep=True),
                    count=1)
    else: 
        main(finetuned_model=pl_model,
            user_hash=user_hash,
            config=config)
