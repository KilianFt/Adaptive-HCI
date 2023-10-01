import datetime

import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from vit_pytorch import ViT

from datasets import EMGWindowsDataset, CombinedDataset
from training import train_model


def train_emg_decoder(use_ninapro = False):
    device = 'mps'

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
    }

    _ = wandb.init(
        project='adaptive-hci',
        tags=["pretraining", "mad_only"],
        config=config
    )

    mad_dataset = EMGWindowsDataset('mad',
                                    window_size=wandb.config.window_size,
                                    overlap=wandb.config.overlap)

    if use_ninapro:
        ninapro5_train_dataset = EMGWindowsDataset('ninapro5_train',
                                        window_size=wandb.config.window_size,
                                        overlap=wandb.config.overlap)
        ninapro5_test_dataset = EMGWindowsDataset('ninapro5_test',
                                        window_size=wandb.config.window_size,
                                        overlap=wandb.config.overlap)
        
        ninapro = CombinedDataset(ninapro5_test_dataset, ninapro5_train_dataset)
        dataset = CombinedDataset(mad_dataset, ninapro)
    else:
        dataset = mad_dataset

    n_labels = dataset.num_unique_labels

    train_ratio = 0.8  # 80% of the data for training

    total_dataset_size = len(dataset)
    train_size = int(train_ratio * total_dataset_size)
    val_size = total_dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=True)

    model = ViT(
        image_size = wandb.config.window_size,
        patch_size = wandb.config.patch_size,
        num_classes = n_labels,
        dim = wandb.config.dim,
        depth = wandb.config.depth,
        heads = wandb.config.heads,
        mlp_dim = wandb.config.mlp_dim,
        dropout = wandb.config.dropout,
        emb_dropout = wandb.config.emb_dropout,
        channels = wandb.config.channels,
    ).to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)

    wandb.config.loss = criterion.__class__.__name__
    wandb.config.model_class = model.__class__.__name__

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"pretrained_{timestamp}"

    model, history = train_model(model=model,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 device=device,
                                 model_name=model_name,
                                 epochs=wandb.config.epochs,
                                 wandb_logging=True)


    print('Best model epoch', np.argmax(history['test_accs']))

    model_save_path = f"models/{model_name}.pt"
    print('Saved model at', model_save_path)
    torch.save(model.cpu(), model_save_path)

    return model


if __name__ == '__main__':

    random_seed = 100
    torch.manual_seed(random_seed)

    _ = train_emg_decoder()
