import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from vit_pytorch import ViT

from adaptive_hci.datasets import CombinedDataset, EMGWindowsDataset
from adaptive_hci.training import train_model
from deployment.buddy import buddy_setup


def get_dataset_(config, dataset_name):
    dataset = get_dataset(config, dataset_name)
    n_labels = dataset.num_unique_labels
    train_ratio = 0.8  # 80% of the data for training
    total_dataset_size = len(dataset)
    train_size = int(train_ratio * total_dataset_size)
    val_size = total_dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    return train_dataloader, test_dataloader, n_labels


def get_dataset(config, name):
    mad_dataset = EMGWindowsDataset(
        'mad',
        window_size=config.window_size,
        overlap=config.overlap
    )
    if name == "ninapro":
        ninapro5_train_dataset = EMGWindowsDataset(
            'ninapro5_train',
            window_size=config.window_size,
            overlap=config.overlap
        )
        ninapro5_test_dataset = EMGWindowsDataset(
            'ninapro5_test',
            window_size=config.window_size,
            overlap=config.overlap
        )
        ninapro = CombinedDataset(ninapro5_test_dataset, ninapro5_train_dataset)
        dataset = CombinedDataset(mad_dataset, ninapro)
    else:
        dataset = mad_dataset
    return dataset


def train_emg_decoder(dataset_name="mad"):
    if torch.cuda.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    config = {
        'pretrained': False,
        'early_stopping': False,
        'epochs': 40,
        'batch_size': 32,
        'lr': 0.0007,
        'window_size': 200,
        'overlap': 150,
        'model_class': 'ViT',
        'patch_size': 8,
        'dim': 64,
        'depth': 1,
        'heads': 2,
        'mlp_dim': 128,
        'dropout': 0.177,
        'emb_dropout': 0.277,
        'channels': 1,
        'random_seed': random_seed,
        'save_checkpoints': False,
    }
    tb = buddy_setup(config)
    config = tb.run.config

    train_dataloader, test_dataloader, n_labels = get_dataset_(config, dataset_name)

    model = ViT(
        image_size=config.window_size,
        patch_size=config.patch_size,
        num_classes=n_labels,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        channels=config.channels,
    ).to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    config.loss = criterion.__class__.__name__
    config.model_class = model.__class__.__name__

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"pretrained_{timestamp}"

    model, history = train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        model_name=model_name,
        epochs=config.epochs,
        wandb_logging=tb,
        save_checkpoints=config.save_checkpoints,
    )

    print('Best model epoch', np.argmax(history['test_accs']))

    if config.save_checkpoints:
        model_save_path = f"models/{model_name}.pt"
        print('Saved model at', model_save_path)
        torch.save(model.cpu(), model_save_path)

    return model


if __name__ == '__main__':
    random_seed = 100
    torch.manual_seed(random_seed)

    train_emg_decoder()
