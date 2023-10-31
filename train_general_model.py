import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule

import configs
from adaptive_hci import utils
from adaptive_hci.datasets import CombinedDataset, EMGWindowsDataset
from adaptive_hci.controllers import EMGViT, PLModel
from common import DataSourceEnum


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


def main(logger, experiment_config: configs.BaseConfig) -> tuple[LightningModule, torch.Tensor]:
    # TODO can we just ignore other logger?
    pl_logger = WandbLogger()

    train_dataloader, val_dataloader, n_labels = get_dataset_(experiment_config)

    vit = EMGViT(
        image_size=experiment_config.window_size,
        patch_size=experiment_config.base_model_config.patch_size,
        num_classes=n_labels,
        dim=experiment_config.base_model_config.dim,
        depth=experiment_config.base_model_config.depth,
        heads=experiment_config.base_model_config.heads,
        mlp_dim=experiment_config.base_model_config.mlp_dim,
        dropout=experiment_config.base_model_config.dropout,
        emb_dropout=experiment_config.base_model_config.emb_dropout,
        channels=experiment_config.base_model_config.channels,
    )

    assert experiment_config.loss in ["MSELoss"], "Only MSELoss is supported for now"

    pl_model = PLModel(vit, n_labels=n_labels, lr=experiment_config.pretrain.lr, n_frozen_layers=0, threshold=0.5)
    trainer = pl.Trainer(max_epochs=experiment_config.pretrain.epochs, log_every_n_steps=1, logger=pl_logger,
                         enable_checkpointing=experiment_config.save_checkpoints)
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    val_metrics = trainer.callback_metrics
    logger.log({f'pretrain/{k}': v for k, v in val_metrics.items()}, commit=False)
    score = val_metrics['validation/f1']

    return pl_model, score


if __name__ == '__main__':
    smoke_config = configs.SmokeConfig()
    torch.manual_seed(smoke_config.random_seed)
    main(None, smoke_config)
