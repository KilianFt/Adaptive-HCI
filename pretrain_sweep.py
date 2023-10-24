import datetime

import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from vit_pytorch import ViT

from datasets import EMGWindowsDataset, CombinedDataset
from training import train_model

def train_emg_decoder():
    device = 'mps'

    run = wandb.init(
        tags=["pretraining"],
    )

    mad_dataset = EMGWindowsDataset('mad',
                                    window_size=wandb.config.window_size,
                                    overlap=wandb.config.overlap)
    ninapro5_train_dataset = EMGWindowsDataset('ninapro5_train',
                                    window_size=wandb.config.window_size,
                                    overlap=wandb.config.overlap)
    test_dataset = EMGWindowsDataset('ninapro5_test',
                                    window_size=wandb.config.window_size,
                                    overlap=wandb.config.overlap)


    train_dataset = CombinedDataset(mad_dataset, ninapro5_train_dataset)

    n_labels = train_dataset.num_unique_labels

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

    model, _ = train_model(model=model,
                           optimizer=optimizer,
                           criterion=criterion,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           device=device,
                           epochs=wandb.config.pretraining_epochs,
                           wandb_logging=True)

    return model


if __name__ == '__main__':

    # config = {
    #     'pretrained': False,
    #     'early_stopping': False,
    #     'epochs': 50,
    #     'batch_size': 32,
    #     'lr': 1e-3,
    #     'window_size': 200,
    #     'overlap': 50,
    #     'model_class': 'ViT',
    #     'patch_size': 2,
    #     'dim': 64,
    #     'depth': 1,
    #     'heads': 2,
    #     'mlp_dim': 128,
    #     'dropout': 0.1,
    #     'emb_dropout': 0.1,
    #     'channels': 1,
    #     'random_seed': random_seed,
    # }

    random_seed = 100
    torch.manual_seed(random_seed)

    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'test_acc'},
        'parameters': {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'value': 40},
            'lr': {'max': 0.005, 'min': 0.0001},
            'window_size': {'values': [200, 400, 600]},
            'overlap': {'values': [50, 100, 150]},
            'model_class': {'value': 'ViT'},
            'patch_size': {'values': [2, 4, 8]},
            'dim': {'values': [32, 64, 128]},
            'depth': {'max': 4, 'min': 1},
            'heads': {'max': 6, 'min': 1},
            'mlp_dim': {'values': [64, 128, 256]},
            'dropout': {'max': 0.3, 'min': 0.1},
            'emb_dropout': {'max': 0.3, 'min': 0.1},
            'channels': {'value': 1},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="adaptive-hci")

    wandb.agent(sweep_id, function=train_emg_decoder, count=10)
