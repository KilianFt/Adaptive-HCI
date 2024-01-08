import argparse
import hashlib
import re
import random
from typing import Tuple, Dict, List

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import configs
from adaptive_hci import datasets
from adaptive_hci.controllers import PLModel
from adaptive_hci.datasets import to_tensor_dataset, get_stored_sessions
from adaptive_hci import utils
from online_adaptation import replay_buffers


adaptation_file_ids = [
    "1-ZARLHsK1k958Bk2-mlQdrRblLreM8_j",
    "1-jjFfGdP5Y8lUk6_prdUSdRmpfEH0W3w",
    "1-hCAag7xc3_l7u8bHUfUNOTe0j95ZGrz",
]


def sweep_wrapper():
    _ = wandb.init()
    pl_model = PLModel.load_from_checkpoint('./adaptive_hci/yp8k1lmf/checkpoints/epoch=0-step=100.ckpt')
    user_hash = hashlib.sha256("Kilian".encode("utf-8")).hexdigest()
    main(pl_model, user_hash, config=None)


def extract_per_label_accuracies(episode_metrics):
    pattern = re.compile(r'.*validation/acc_label_(\d+)')
    matching_keys = [key for key in episode_metrics.keys() if pattern.match(key)]
    sorted_keys = sorted(matching_keys, key=lambda key: int(pattern.search(key).group(1)))
    per_label_accuracies = np.array([episode_metrics[key] for key in sorted_keys])
    return per_label_accuracies


def get_unseen_episodes(all_episodes, seen_episodes):
    # Use user signal sum to identify episodes
    seen_user_signal_sums = [e.observations.sum() for e in seen_episodes]
    unseen_episodes = [e for e in all_episodes if e.observations.sum() not in seen_user_signal_sums]
    return unseen_episodes


def prepare_data(ep_idx, config, all_episodes, seen_episodes, episode_metrics):
    if config.online.adaptive_training:
        per_label_accuracies = extract_per_label_accuracies(episode_metrics)
        unseen_episodes = get_unseen_episodes(all_episodes=all_episodes, seen_episodes=seen_episodes)
        current_episode = datasets.get_adaptive_episode(unseen_episodes, per_label_accuracies)
    else:
        current_episode = all_episodes[ep_idx]

    observations = [current_episode.observations]
    actions = [current_episode.actions]
    if ep_idx > 0:
        num_samples = min(config.online.additional_train_episodes, ep_idx)
        for e in random.sample(seen_episodes, num_samples):
            observations.append(e.observations)
            actions.append(e.actions)

    train_observations = np.concatenate(observations)
    train_actions = np.concatenate(actions)
    return to_tensor_dataset(train_observations, train_actions), current_episode


def validate_model(trainer: Trainer, pl_model, val_dataset, config):
    val_dataloader = DataLoader(val_dataset, batch_size=config.online.batch_size,
                                num_workers=config.online.num_workers)
    hist, = trainer.validate(model=pl_model, dataloaders=val_dataloader)
    return hist


def train_model(trainer, pl_model, train_dataset, config):
    train_dataloader = DataLoader(train_dataset, batch_size=config.online.batch_size,
                                  num_workers=config.online.num_workers, shuffle=True)
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader)


def get_training_bools(data, num_samples, do_training=True):
    if not do_training:
        return [False for _ in data]

    if num_samples is not None:
        do_training_list = [i < num_samples for i, _ in enumerate(data)]
    else:
        do_training_list = [True for _ in data]

    return do_training_list


def get_replay_buffer(num_classes, config):
    if config.online.balance_classes:
        replay_buffer = replay_buffers.ClassBalancingReplayBuffer(
            max_size=config.online.buffer_size,
            num_classes=num_classes
        )
    else:
        replay_buffer = replay_buffers.ReplayBuffer(
            max_size=config.online.buffer_size,
            num_classes=num_classes
        )

    return replay_buffer


def process_session(config, current_trial_episodes, logger, pl_model, do_training, shuffle):
    all_episodes, num_classes = datasets.get_rl_dataset(current_trial_episodes, shuffle=shuffle)
    do_training_episodes = get_training_bools(all_episodes, config.online.num_episodes, do_training=do_training)
    replay_buffer = get_replay_buffer(num_classes, config)

    accelerator = utils.get_accelerator(config.config_type)
    trainer = pl.Trainer(max_epochs=0, log_every_n_steps=1, logger=logger,
                         enable_checkpointing=config.save_checkpoints, accelerator=accelerator,
                         gradient_clip_val=config.gradient_clip_val)

    session_val_metrics_list = []
    seen_episodes = []
    for ep_idx, (rollout, is_train_episode) in enumerate(zip(all_episodes, do_training_episodes)):
        trainer.fit_loop.max_epochs += config.online.epochs

        val_dataset = to_tensor_dataset(rollout.observations, rollout.actions)
        validation_metrics = validate_model(trainer, pl_model, val_dataset, config)
        session_val_metrics_list.append(validation_metrics)

        if is_train_episode and ep_idx >= config.online.first_training_episode and ep_idx % config.online.train_intervals == 0:
            train_dataset, current_episode = prepare_data(ep_idx, config, all_episodes, seen_episodes, validation_metrics)
            seen_episodes.append(current_episode)
            replay_buffer.extend(train_dataset)

            if len(replay_buffer) > 0:
                train_model(trainer, pl_model, replay_buffer, config)

    return session_val_metrics_list


def main(pl_model: LightningModule, user_hash, config: configs.BaseConfig) -> Tuple[LightningModule, List[Dict[str, int]]]:
    if config is None:
        config = wandb.config

    pl_model.freeze_layers(config.online.n_frozen_layers)
    pl_model.lr = config.online.lr

    logger = WandbLogger(project='adaptive_hci', tags=["online_adaptation", user_hash],
                         config=config, name=f"online_adapt_{config}_{user_hash}")

    online_sessions = get_stored_sessions(stage="Adaptation", file_ids=adaptation_file_ids)
    do_training_list = get_training_bools(online_sessions, config.online.num_sessions)

    valid_metrics = []
    for session_idx, (current_trial_episodes, do_training) in enumerate(zip(online_sessions, do_training_list)):
        pl_model.metric_prefix = f'{user_hash}/continous/session_{session_idx}/'
        pl_model.step_count = 0
        session_metrics = process_session(config, current_trial_episodes, logger, pl_model, do_training, config.online.shuffle_episodes)
        valid_metrics += session_metrics

    wandb.run.log({}, commit=True)
    return pl_model, valid_metrics


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
