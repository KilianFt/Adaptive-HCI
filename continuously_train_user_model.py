import argparse
import collections
import hashlib
import os
import pathlib
import random

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import configs
from adaptive_hci import datasets
from adaptive_hci.controllers import PLModel
from adaptive_hci.datasets import to_tensor_dataset
from adaptive_hci.utils import maybe_download_drive_folder
from online_adaptation import replay_buffers

file_ids = [
    "1-ZARLHsK1k958Bk2-mlQdrRblLreM8_j",
    "1-jjFfGdP5Y8lUk6_prdUSdRmpfEH0W3w",
    "1-hCAag7xc3_l7u8bHUfUNOTe0j95ZGrz",
]


def get_stored_sessions(config):
    online_data_dir = pathlib.Path('datasets/OnlineAdaptation')
    maybe_download_drive_folder(online_data_dir, file_ids=file_ids)
    episode_filenames = sorted(os.listdir(online_data_dir))
    online_sessions = datasets.load_online_episodes(online_data_dir, episode_filenames, config.online_num_sessions)
    return online_sessions


def sweep_wrapper():
    _ = wandb.init()
    pl_model = PLModel.load_from_checkpoint('./adaptive_hci/yp8k1lmf/checkpoints/epoch=0-step=100.ckpt')
    user_hash = hashlib.sha256("Kilian".encode("utf-8")).hexdigest()
    main(pl_model, user_hash, config=None)


def prepare_data(ep_idx, config, all_episodes, episode_metrics):
    if config.online_adaptive_training:
        per_label_accuracies = np.array([episode_metrics[f'val_acc_label_{label_idx}'][-1] for label_idx in range(config.num_classes)])
        current_episode = datasets.get_adaptive_episode(all_episodes, per_label_accuracies)
    else:
        current_episode = all_episodes[ep_idx]

    observations = [current_episode.observations]
    actions = [current_episode.actions]
    if ep_idx > 0:
        num_samples = min(config.online_additional_train_episodes, len(all_episodes))
        for e in random.sample(all_episodes, num_samples):
            observations.append(e.observations)
            actions.append(e.actions)

    train_observations = np.concatenate(observations)
    train_actions = np.concatenate(actions)
    return to_tensor_dataset(train_observations, train_actions)


def validate_model(trainer, pl_model, val_dataset, config):
    val_dataloader = DataLoader(val_dataset, batch_size=config.online_batch_size,
                                num_workers=config.online_adaptation_num_workers)
    hist, = trainer.validate(model=pl_model, dataloaders=val_dataloader)
    return hist


def train_model(trainer, pl_model, train_dataset, config):
    train_dataloader = DataLoader(train_dataset, batch_size=config.online_batch_size,
                                  num_workers=config.online_adaptation_num_workers, shuffle=True)
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader)  # TODO: we need to track metrics


def process_session(config, current_trial_episodes, logger, pl_model, session_idx):
    all_episodes, num_classes = datasets.get_rl_dataset(current_trial_episodes, config.online_num_episodes)
    episode_metrics = collections.defaultdict(list)
    replay_buffer = replay_buffers.ReplayBuffer(max_size=1_000, num_classes=num_classes)
    trainer = pl.Trainer(limit_train_batches=config.limit_train_batches, max_epochs=0, log_every_n_steps=1,
                         logger=logger)
    for ep_idx, rollout in enumerate(all_episodes):
        trainer.fit_loop.max_epochs += config.online_epochs

        val_dataset = to_tensor_dataset(rollout.observations, rollout.actions)
        validation_metrics = validate_model(trainer, pl_model, val_dataset, config)

        for k, v in validation_metrics.items():
            episode_metrics[k].append(v)

        if ep_idx >= config.online_first_training_episode and ep_idx % config.online_train_intervals == 0:
            train_dataset = prepare_data(ep_idx, config, all_episodes, episode_metrics)
            replay_buffer.extend(train_dataset)

            if len(replay_buffer) > 0:
                train_model(trainer, pl_model, replay_buffer, config)

        wandb.run.log({f'session_{session_idx}/{k}': v for k, v in episode_metrics.items()}, commit=False)


def main(pl_model: LightningModule, user_hash, config: configs.BaseConfig) -> LightningModule:
    if config is None:
        config = wandb.config

    pl_model.freeze_layers(config.online_n_frozen_layers)
    pl_model.lr = config.online_lr

    logger = WandbLogger(project='adaptive_hci', tags=["online_adaptation", user_hash],
                         config=config, name=f"online_adapt_{config}_{user_hash[:15]}")

    online_sessions = get_stored_sessions(config)

    for session_idx, current_trial_episodes in enumerate(online_sessions):
        process_session(config, current_trial_episodes, logger, pl_model, session_idx)

    wandb.run.log({}, commit=True)
    return pl_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Adaptive HCI - Fetch')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    random_seed = 100
    torch.manual_seed(random_seed)
    pl_model = PLModel.load_from_checkpoint('./adaptive_hci/yp8k1lmf/checkpoints/epoch=0-step=100.ckpt')
    user_hash = hashlib.sha256("Kilian".encode("utf-8")).hexdigest()

    if args.sweep:
        sweep_configuration = {
            'method': 'bayes',
            'name': 'sweep',
            'metric': {'goal': 'maximize', 'name': 'mean_f1'},
            'parameters': {
                'limit_train_batches': {'value': 200},
                'online_adaptation_num_workers': {'value': 8},
                'online_batch_size': {'values': [16, 32, 64]},
                'online_epochs': {'max': 10, 'min': 1},
                'online_lr': {'max': 0.005, 'min': 0.0001},
                'online_num_episodes': {'value': None},
                'online_n_frozen_layers': {'max': 2, 'min': 0},
                'online_train_intervals': {'max': 5, 'min': 1},
                'online_first_training_episode': {'max': 10, 'min': 0},
                'online_additional_train_episodes': {'max': 10, 'min': 0},
            },
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project="adaptive-hci-online-test")
        wandb.agent(sweep_id, function=sweep_wrapper, count=2)
    else:
        config = configs.SmokeConfig()
        main(pl_model, user_hash, config)
