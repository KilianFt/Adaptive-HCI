import os
import pathlib
import sys

import lightning.pytorch as pl
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import configs
from adaptive_hci.datasets import get_concatenated_user_episodes, to_tensor_dataset, get_stored_sessions
from adaptive_hci import utils


finetune_user_ids = [
    "1Sitb0ooo2izvkHQGNQkXTGoDV4CJAnFF",
    "1bIYLJVu-SqHzRnTFxuc1vkzRBs8Ll5Oi",
    "1D7h11vheJ7Oq8Ju4ik8jqBJUocEie-rQ",
    "1EWJdHHZ22xorZEpss-gf5R4cxehEs9pt",
]


def main(model: LightningModule, user_hash, config: configs.BaseConfig) -> LightningModule:
    if not config.finetune.do_finetuning:
        print('Skip finetuning')
        return model

    logger = WandbLogger(project='adaptive_hci', tags=["finetune", user_hash], config=config,
                         name=f"finetune_{config}_{user_hash[:15]}")

    episode_list = get_stored_sessions(stage="Data", num_episodes=config.finetune.num_episodes, file_ids=finetune_user_ids)

    train_episodes = []
    for ep in episode_list[:-1]:
        train_episodes += ep

    val_episodes = episode_list[-1]

    train_observations, _, train_optimal_actions, _, _ = get_concatenated_user_episodes(episodes=train_episodes)

    val_observations, _, val_optimal_actions, _, _ = get_concatenated_user_episodes(episodes=val_episodes)

    train_offline_adaption_dataset = to_tensor_dataset(train_observations, train_optimal_actions)
    val_offline_adaption_dataset = to_tensor_dataset(val_observations, val_optimal_actions)

    dataloader_args = dict(batch_size=config.finetune.batch_size, num_workers=config.finetune.num_workers)

    train_dataloader = DataLoader(train_offline_adaption_dataset, shuffle=True, **dataloader_args)
    val_dataloader = DataLoader(val_offline_adaption_dataset, **dataloader_args)

    model.lr = config.finetune.lr
    model.freeze_layers(config.finetune.n_frozen_layers)
    model.metric_prefix = f'{user_hash}/finetune/'
    model.step_count = 0

    callbacks = utils.get_trainer_callbacks(config.finetune)
    accelerator = utils.get_accelerator(config.config_type)
    trainer = pl.Trainer(max_epochs=config.finetune.epochs, log_every_n_steps=1, logger=logger,
                         enable_checkpointing=config.save_checkpoints, accelerator=accelerator,
                         gradient_clip_val=config.gradient_clip_val, callbacks=callbacks)

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
        raise NotImplementedError
    else:
        raise NotImplementedError
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="adaptive-hci-offline-adaptation")
        wandb.agent(sweep_id, function=main, count=10)
