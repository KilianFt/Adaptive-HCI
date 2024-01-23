import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule

import configs
from adaptive_hci import utils
from adaptive_hci.controllers import EMGViT, SingleLabelPlModel
from data_loaders.emg_datasets import get_dataloaders


def main(logger, experiment_config: configs.BaseConfig) -> LightningModule:
    pl_logger = WandbLogger()

    train_dataloader, val_dataloader, n_labels = get_dataloaders(experiment_config)

    vit = EMGViT(
        image_size=experiment_config.window_size,
        patch_size=experiment_config.general_model_config.patch_size,
        num_classes=experiment_config.num_classes,#n_labels,
        dim=experiment_config.general_model_config.dim,
        depth=experiment_config.general_model_config.depth,
        heads=experiment_config.general_model_config.heads,
        mlp_dim=experiment_config.general_model_config.mlp_dim,
        dropout=experiment_config.general_model_config.dropout,
        emb_dropout=experiment_config.general_model_config.emb_dropout,
        channels=experiment_config.general_model_config.channels,
    )

    assert experiment_config.criterion_key in ["mse", "bce", "ce"], "{experiment_config.criterion_key} loss is supported for now"

    pl_model = SingleLabelPlModel(
        vit,
        n_labels=experiment_config.num_classes,
        lr=experiment_config.pretrain.lr,
        n_frozen_layers=0,
        threshold=0.5,
        metric_prefix='pretrain/',
        criterion_key=experiment_config.criterion_key,
    )
    if not experiment_config.pretrain.do_pretraining:
        return pl_model

    callbacks = utils.get_trainer_callbacks(experiment_config.pretrain)
    accelerator = utils.get_accelerator(experiment_config.config_type)
    trainer = pl.Trainer(max_epochs=experiment_config.pretrain.epochs,
                         log_every_n_steps=1,
                         logger=pl_logger,
                         enable_checkpointing=experiment_config.save_checkpoints,
                         accelerator=accelerator,
                         gradient_clip_val=experiment_config.gradient_clip_val,
                         callbacks=callbacks)

    trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return pl_model


if __name__ == '__main__':
    smoke_config = configs.SmokeConfig()
    torch.manual_seed(smoke_config.random_seed)
    main(None, smoke_config)
