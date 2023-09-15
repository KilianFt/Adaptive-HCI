import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from vit_pytorch import ViT

from datasets import EMGWindowsDataset

if __name__ == '__main__':

    device = 'mps'

    train_dataset = EMGWindowsDataset('mad')
    test_dataset = EMGWindowsDataset('ninapro5')

    n_labels = train_dataset.num_unique_labels

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

    history = {
        'test_accs': [],
        'test_mse': [],
        'losses': [],
    }

    for epoch in range(10):

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
            test_mse = mean_squared_error(test_labels, predictions)

            test_accs.append(test_acc)
            test_mse_list.append(test_mse)

        history['test_accs'].append(np.mean(test_accs))
        history['test_mse'].append(np.mean(test_mse_list))

        print('test MSE', np.mean(test_mse_list))
        print('test acc', np.mean(test_accs))

    print('Finished Training')

plt.plot(history['test_accs'])
plt.show()

torch.save(model.cpu(), 'models/pretrained_vit_onehot_test_model.pt')
model_state_dict = model.state_dict()
torch.save(model_state_dict, 'models/pretrained_vit_onehot_test_state_dict.pt')
