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

def train_model(model,
                optimizer,
                criterion,
                train_dataloader,
                test_dataloader,
                device,
                epochs=10,
                early_stopping=True,
                wandb_logging=False):

    history = {
        'test_accs': [],
        'test_f1s': [],
        'test_mse': [],
        'losses': [],
    }

    for epoch in range(epochs):

        train_losses = []
        for i, data in enumerate(train_dataloader, 0):
            model.train()

            train_inputs, train_labels = data
            train_inputs = train_inputs.to(device)
            train_labels = train_labels.to(device)

            if model.__class__.__name__ == 'ViT':
                train_inputs.unsqueeze_(axis=1)

            optimizer.zero_grad()

            outputs = model(train_inputs)
            loss = criterion(outputs, train_labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        test_accs = []
        test_f1s = []
        test_mse_list = []
        for data in test_dataloader:
            test_inputs, test_labels = data
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            if model.__class__.__name__ == 'ViT':
                test_inputs.unsqueeze_(axis=1)
            model.eval()
            with torch.no_grad():
                outputs = model(test_inputs)
                predictions = outputs.cpu().squeeze().numpy()

                predicted_onehot = np.zeros_like(predictions)
                predicted_onehot[predictions > 0.5] = 1

            test_labels = test_labels.cpu()
            test_acc = accuracy_score(test_labels, predicted_onehot)
            test_f1 = f1_score(test_labels, predicted_onehot, average='micro')
            test_mse = mean_squared_error(test_labels, predictions)

            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_mse_list.append(test_mse)

        test_mean_accs = np.mean(test_accs)
        test_mean_f1s = np.mean(test_f1s)
        test_mse = np.mean(test_mse_list)

        train_loss = np.mean(train_losses)

        if wandb_logging:
            print('log_wandb')
            wandb.log({
                'test_acc': test_mean_accs,
                'test_f1': test_mean_f1s,
                'test_mse': test_mse,
                'train_loss': train_loss,
            })

        history['test_accs'].append(test_mean_accs)
        history['test_f1s'].append(test_mean_f1s)
        history['test_mse'].append(test_mse)
        history['train_loss'].append(train_loss)

        # early stopping
        if early_stopping and \
                len(history['test_accs']) >= 3 and \
                history['test_accs'][-1] < history['test_accs'][-2] and \
                history['test_accs'][-2] < history['test_accs'][-3]:
            print('early stopping at epoch', epoch)
            break

    print('Finished Training')
    return model, history


def train_emg_decoder():
    print('Training model')
    device = 'mps'

    random_seed = 100
    torch.manual_seed(random_seed)

    # TODO compare losses

    config = {
        'pretrained': False,
        'early_stopping': True,
        'epochs': 20,
        'batch_size': 32,
        'lr': 1e-3,
        'window_size': 200,
        'overlap': 50,
        'model_class': 'ViT',
        'patch_size': 8,
        'dim': 64,
        'depth': 1,
        'heads': 2,
        'mlp_dim': 128,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'channels': 1,
        'random_seed': random_seed,
    }

    run = wandb.init(
        project="adaptive-hci",
        tags=["pretraining"],
        config=config,
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

    # criterion = nn.CrossEntropyLoss()
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
                                 epochs=wandb.config.epochs,
                                 wandb_logging=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = "pretrained_model"
    model_save_path = f"models/{model_name}_{timestamp}.pt"

    print('Saved model at', model_save_path)
    torch.save(model.cpu(), model_save_path)
    # model_state_dict = model.state_dict()
    # torch.save(model_state_dict, 'models/pretrained_vit_onehot_test_state_dict.pt')
    return model


if __name__ == '__main__':
    train_emg_decoder()