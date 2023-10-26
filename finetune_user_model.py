import os
import pathlib
import sys

import lightning.pytorch as pl
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

import configs
from adaptive_hci.datasets import get_concatenated_user_episodes, load_online_episodes, to_tensor_dataset
from adaptive_hci.utils import maybe_download_drive_folder

base_configuration = {
    'finetune_batch_size': 32,
    'finetune_epochs': 50,
    'finetune_lr': 0.005,
    'finetune_n_frozen_layers': 2,
}

file_ids = [
    "1Sitb0ooo2izvkHQGNQkXTGoDV4CJAnFF",
    "1bIYLJVu-SqHzRnTFxuc1vkzRBs8Ll5Oi",
    "1D7h11vheJ7Oq8Ju4ik8jqBJUocEie-rQ",
    "1EWJdHHZ22xorZEpss-gf5R4cxehEs9pt",
]


def main(model, user_hash, config: configs.BaseConfig):
    logger = WandbLogger(project='adaptive_hci', tags=["offline_adaptation", user_hash], config=config,

                         name=f"finetune_{config}_{user_hash[:15]}")

    online_data_dir = pathlib.Path('datasets/OnlineData')
    maybe_download_drive_folder(online_data_dir, file_ids=file_ids)

    episode_filenames = sorted(os.listdir(online_data_dir))

    artifact = wandb.Artifact(name="offline_adaptattion_data", type="dataset")
    artifact.add_dir(online_data_dir, name='offline_adaptattion_data')
    wandb.run.log_artifact(artifact)

    episode_list = load_online_episodes(online_data_dir, episode_filenames, config.finetune_num_episodes)


    train_episodes = []
    for ep in episode_list[:-1]:
        train_episodes += ep

    if config.finetune_num_episodes is not None:
        train_episodes = train_episodes[:1]

    val_episodes = episode_list[-1]

    (train_observations,
     train_actions,
     train_optimal_actions,
     train_rewards,
     train_terminals) = get_concatenated_user_episodes(episodes=train_episodes)

    (val_observations,
     val_actions,
     val_optimal_actions,
     val_rewards,
     val_terminals) = get_concatenated_user_episodes(episodes=val_episodes)

    train_offline_adaption_dataset = to_tensor_dataset(train_observations, train_optimal_actions)
    val_offline_adaption_dataset = to_tensor_dataset(val_observations, val_optimal_actions)

    dataloader_args = dict(batch_size=config.finetune_batch_size, num_workers=config.finetune_num_workers)

    train_dataloader = DataLoader(train_offline_adaption_dataset, shuffle=True, **dataloader_args)
    val_dataloader = DataLoader(val_offline_adaption_dataset, **dataloader_args)

    model.lr = config.finetune_lr
    model.freeze_layers(config.finetune_n_frozen_layers)

    trainer = pl.Trainer(limit_train_batches=config.limit_train_batches, max_epochs=config.finetune_epochs,
                         log_every_n_steps=1, logger=logger)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return model


if __name__ == '__main__':
    random_seed = 100
    torch.manual_seed(random_seed)

    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'test_acc'},
        'parameters': {
            'finetune_batch_size': {'values': [16, 32, 64]},
            'finetune_epochs': {'value': 50},
            'finetune_lr': {'max': 0.005, 'min': 0.0001},
            'finetune_n_frozen_layers': {'max': 5, 'min': 0},
        },
    }

    if sys.gettrace() is not None:
        main(base_configuration)
    else:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="adaptive-hci-offline-adaptation")
        wandb.agent(sweep_id, function=main, count=10)
