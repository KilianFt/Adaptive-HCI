import pathlib

import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import configs
from adaptive_hci.datasets import load_emg_writing_data, to_tensor_class_dataset, maybe_download_drive_folder
from adaptive_hci import utils


def load_finetune_dataloader(config):
    file_path = pathlib.Path(__file__).resolve()
    print(file_path)
    # emg_draw_data_dir = file_path.parent / 'datasets' / 'emg_writing_o_l/'
    emg_draw_data_dir = file_path.parent / 'emg_writing_o_l/'
    emg_writing_ids_file = file_path.parent / 'emg_writing_file_names.txt'
    with open(emg_writing_ids_file, 'rb') as f:
        file_ids = f.readlines()
    file_ids = [file_id.decode().strip() for file_id in file_ids]
    print(file_ids)
    maybe_download_drive_folder(emg_draw_data_dir, file_ids)

    observations, actions = load_emg_writing_data(emg_draw_data_dir, window_size=config.window_size, overlap=config.overlap)

    train_observations, val_observations, train_optimal_actions, val_optimal_actions = train_test_split(observations, actions, test_size=0.25)

    train_offline_adaption_dataset = to_tensor_class_dataset(train_observations, train_optimal_actions)
    val_offline_adaption_dataset = to_tensor_class_dataset(val_observations, val_optimal_actions)

    dataloader_args = dict(batch_size=config.finetune.batch_size, num_workers=config.finetune.num_workers)
    train_dataloader = DataLoader(train_offline_adaption_dataset, shuffle=True, **dataloader_args)
    val_dataloader = DataLoader(val_offline_adaption_dataset, **dataloader_args)
    return train_dataloader, val_dataloader


def main(model: LightningModule, user_hash, config: configs.BaseConfig) -> LightningModule:
    if not config.finetune.do_finetuning:
        print('Skip finetuning')
        return model

    logger = WandbLogger(project='adaptive_hci', tags=["finetune", user_hash], config=config,
                         name=f"finetune_{config}_{user_hash[:15]}")

    train_dataloader, val_dataloader = load_finetune_dataloader(config)

    model.lr = config.finetune.lr
    model.freeze_layers(config.finetune.n_frozen_layers)
    model.metric_prefix = f'{user_hash}/finetune/'
    model.step_count = 0

    accelerator = utils.get_accelerator(config.config_type)
    trainer = pl.Trainer(max_epochs=config.finetune.epochs, log_every_n_steps=1, logger=logger,
                         enable_checkpointing=config.save_checkpoints, accelerator=accelerator,
                         gradient_clip_val=config.gradient_clip_val)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return model
