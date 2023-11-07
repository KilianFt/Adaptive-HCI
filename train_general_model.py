import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule

import configs
from adaptive_hci import utils
from adaptive_hci.datasets import CombinedDataset, EMGWindowsDataset
from adaptive_hci.controllers import PLModel
from common import DataSourceEnum, BASE_MODELS


def get_dataset_(config: configs.BaseConfig):
    dataset = get_dataset(config)

    train_dataset, test_dataset = random_split(dataset, [1 - config.pretrain.train_fraction, config.pretrain.train_fraction])

    dataloader_args = dict(batch_size=config.pretrain.batch_size, drop_last=False, num_workers=config.pretrain.num_workers)

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_args)
    test_dataloader = DataLoader(test_dataset, **dataloader_args)
    return train_dataloader, test_dataloader, dataset.num_unique_labels


@utils.disk_cache
def get_dataset(config: configs.BaseConfig):
    data_source = config.data_source
    assert data_source not in (DataSourceEnum.NINA_PRO,), "not implemented, use merged to include it"

    dataset = EMGWindowsDataset(
        data_source,
        split="Train",
        window_size=config.window_size,
        overlap=config.overlap
    )
    if data_source == DataSourceEnum.MERGED:
        ninapro5_train_dataset = EMGWindowsDataset(
            data_source,
            split="train",
            window_size=config.window_size,
            overlap=config.overlap
        )
        ninapro5_test_dataset = EMGWindowsDataset(
            data_source,
            split="test",
            window_size=config.window_size,
            overlap=config.overlap
        )
        ninapro = CombinedDataset(ninapro5_test_dataset, ninapro5_train_dataset)
        dataset = CombinedDataset(dataset, ninapro)
    elif data_source in (DataSourceEnum.MiniMAD, DataSourceEnum.MAD):
        pass
    else:
        raise NotImplementedError(f"Unknown datasource {data_source}")
    return dataset


def main(logger, experiment_config: configs.BaseConfig) -> LightningModule:
    pl_logger = WandbLogger()

    train_dataloader, val_dataloader, n_labels = get_dataset_(experiment_config)

    base_model = BASE_MODELS[experiment_config.base_model_name](experiment_config)

    assert experiment_config.loss in ["MSELoss"], "Only MSELoss is supported for now"

    pl_model = PLModel(base_model, n_labels=n_labels, lr=experiment_config.pretrain.lr, n_frozen_layers=0, threshold=0.5, metric_prefix='pretrain/')
    trainer = pl.Trainer(max_epochs=experiment_config.pretrain.epochs, log_every_n_steps=1, logger=pl_logger,
                         enable_checkpointing=experiment_config.save_checkpoints)
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return pl_model


if __name__ == '__main__':
    smoke_config = configs.SmokeConfig()
    torch.manual_seed(smoke_config.random_seed)
    main(None, smoke_config)
