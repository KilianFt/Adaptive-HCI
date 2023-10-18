import os
import pathlib
import subprocess
import sys
import copy

import wandb
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import configs
from adaptive_hci.datasets import EMGWindowsAdaptationDataset, \
                                  get_concatenated_user_episodes, \
                                  load_online_episodes

base_configuration = {
    'batch_size': 32,
    'epochs': 50,
    'lr': 0.005,
    'n_frozen_layers': 2,
    'window_size': 600,
    'overlap': 100,
    'model_class': 'ViT',
}


def main(base_pl_model, user_hash, config: configs.BaseConfig):
    pl_model = copy.deepcopy(base_pl_model)

    logger = WandbLogger(project='adaptive_hci',
                         tags=["offline_adaptation", user_hash],
                         config=config,
                         name=f"finetune_{config}_{user_hash[:15]}")

    online_data_dir = pathlib.Path('datasets/OnlineData')
    episode_filenames = sorted(os.listdir(online_data_dir))

    artifact = wandb.Artifact(name="offline_adaptattion_data", type="dataset")
    artifact.add_dir(online_data_dir, name='offline_adaptattion_data')
    wandb.run.log_artifact(artifact)

    episode_list = load_online_episodes(online_data_dir, episode_filenames)

    # TODO: this should be a parameter of the smoke config, this check is an hack
    # The config should have a parameter like num_folds, which can be None for all or 1,2,3, etc
    is_smoke = isinstance(config, configs.SmokeConfig)
    if is_smoke:
        episode_list = episode_list[:2]

    train_episodes = []
    for ep in episode_list[:-1]:
        train_episodes += ep

    if is_smoke:
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

    train_offline_adaption_dataset = EMGWindowsAdaptationDataset(train_observations, train_optimal_actions)
    val_offline_adaption_dataset = EMGWindowsAdaptationDataset(val_observations, val_optimal_actions)

    dataloader_args = dict(batch_size=config.batch_size, num_workers=8)

    train_dataloader = DataLoader(train_offline_adaption_dataset, shuffle=True, **dataloader_args)
    val_dataloader = DataLoader(val_offline_adaption_dataset, **dataloader_args)
    
    if config.n_frozen_layers >= 1:
        for param in pl_model.model.to_patch_embedding.parameters():
            param.requires_grad = False
        for param in pl_model.model.dropout.parameters():
            param.requires_grad = False

    if config.n_frozen_layers >= 2:
        for layer_idx in range(min((config.n_frozen_layers - 1), 4)):
            for param in pl_model.model.transformer.layers[layer_idx].parameters():
                param.requires_grad = False

    trainer = pl.Trainer(limit_train_batches=100,
                        max_epochs=1,
                        log_every_n_steps=1,
                        logger=logger,
                        )
    trainer.fit(model=pl_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    return pl_model


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