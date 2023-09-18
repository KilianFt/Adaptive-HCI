import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from vit_pytorch import ViT

from datasets import EMGWindowsDataset, CombinedDataset

def train_model(model, optimizer, criterion, train_dataloader, test_dataloader, device, epochs=10):

    history = {
        'test_accs': [],
        'test_f1s': [],
        'test_mse': [],
        'losses': [],
    }

    for epoch in range(epochs):

        running_loss = 0.0
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

            running_loss += loss.item()
            history['losses'].append(loss.item())
            if i % 1000 == 999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.8f}')
                running_loss = 0.0

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

        history['test_accs'].append(np.mean(test_accs))
        history['test_f1s'].append(np.mean(test_f1s))
        history['test_mse'].append(np.mean(test_mse_list))

        print('test MSE', np.mean(test_mse_list))
        print('test acc', np.mean(test_accs))
        print('test f1s', np.mean(test_f1s))

    print('Finished Training')
    return model, history

def train_emg_decoder():
    print('Training model')
    device = 'mps'

    mad_dataset = EMGWindowsDataset('mad', overlap=50)
    ninapro5_dataset = EMGWindowsDataset('ninapro5', overlap=50)

    dataset = CombinedDataset(mad_dataset, ninapro5_dataset)

    n_labels = dataset.num_unique_labels

    total_dataset_size = len(dataset)
    train_ratio = 0.8  # 80% of the data for training
    train_size = int(train_ratio * total_dataset_size)
    val_size = total_dataset_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = ViT(
        image_size = 200,
        patch_size = 8,
        num_classes = n_labels,
        dim = 64,
        depth = 1,
        heads = 2,
        mlp_dim = 128,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels=1,
    ).to(device) 

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model, history = train_model(model=model,
                                 optimizer=optimizer,
                                 criterion=criterion,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 device=device,
                                 epochs=10)

    plt.plot(history['test_accs'])
    plt.show()

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