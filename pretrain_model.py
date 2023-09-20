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


def train_emg_decoder():
    device = 'mps'

    config = {
        'pretrained': False,
        'early_stopping': False,
        'epochs': 60,
        'batch_size': 16,
        'lr': 0.0007,
        'window_size': 600,
        'overlap': 50,
        'model_class': 'ViT',
        'patch_size': 8,
        'dim': 128,
        'depth': 4,
        'heads': 5,
        'mlp_dim': 256,
        'dropout': 0.25,
        'emb_dropout': 0.137,
        'channels': 1,
        'random_seed': random_seed,
    }

    run = wandb.init(
        project='adaptive-hci',
        tags=["pretraining"],
        config=config
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


    print('best model epoch', np.argmax(history['test_accs']))

    model_save_path = f"models/{model_name}.pt"
    print('Saved model at', model_save_path)
    torch.save(model.cpu(), model_save_path)

    return model


if __name__ == '__main__':

    random_seed = 100
    torch.manual_seed(random_seed)

    train_emg_decoder()
