import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from vit_pytorch import ViT

import configs
from adaptive_hci import utils
from adaptive_hci.datasets import CombinedDataset, EMGWindowsDataset
from adaptive_hci.training import train_model
from common import DataSourceEnum
from deployment.buddy import buddy_setup


def get_dataset_(config: configs.BaseConfig):
    dataset = get_dataset(config)

    train_dataset, test_dataset = random_split(dataset, [
        1 - config.train_fraction, config.train_fraction])

    dataloader_args = dict(batch_size=config.batch_size, shuffle=True, drop_last=False)

    train_dataloader = DataLoader(train_dataset, **dataloader_args)
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


def main(logger, experiment_config: configs.BaseConfig) -> nn.Module:
    device = utils.get_device()
    print('Using device:', device)

    train_dataloader, test_dataloader, n_labels = get_dataset_(experiment_config)

    model = ViT(
        image_size=experiment_config.window_size,
        patch_size=experiment_config.patch_size,
        num_classes=n_labels,
        dim=experiment_config.dim,
        depth=experiment_config.depth,
        heads=experiment_config.heads,
        mlp_dim=experiment_config.mlp_dim,
        dropout=experiment_config.dropout,
        emb_dropout=experiment_config.emb_dropout,
        channels=experiment_config.channels,
    ).to(device=device)

    assert experiment_config.loss in ["MSELoss"], "Only MSELoss is supported for now"
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=experiment_config.lr)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"pretrained_{timestamp}"  # TODO: replace with wandb.run.name

    model, history = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        model_name=model_name,
        epochs=experiment_config.epochs,
        logger=logger,
        save_checkpoints=experiment_config.save_checkpoints,
    )

    print('Best model epoch', np.argmax(history['test_accs']))

    if experiment_config.save_checkpoints:
        model_save_path = f"models/{model_name}.pt"
        print('Saved model at', model_save_path)
        torch.save(model.cpu(), model_save_path)
    return model


if __name__ == '__main__':
    smoke_config = configs.SmokeConfig()
    torch.manual_seed(smoke_config.random_seed)
    main(smoke_config)
