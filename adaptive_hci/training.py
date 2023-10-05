from typing import Optional

import numpy as np
import torch
import tqdm
from sklearn import metrics

import experiment_buddy


def train_model(
        model,
        optimizer,
        criterion,
        device,
        train_dataloader,
        test_dataloader=None,
        model_name=None,
        epochs=10,
        wandb_logging: Optional[experiment_buddy.WandbWrapper] = None,
        save_checkpoints=False,
):
    history = {
        'test_accs': [],
        'test_f1s': [],
        'test_mse': [],
        'train_loss': [],
    }

    for epoch in tqdm.trange(epochs):
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

        train_loss = np.mean(train_losses)

        if test_dataloader is not None:
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
                    predicted_onehot[predictions > 0.7] = 1

                test_labels = test_labels.cpu()
                test_acc = metrics.accuracy_score(test_labels, predicted_onehot)
                test_f1 = metrics.f1_score(test_labels, predicted_onehot, average='micro')
                test_mse = metrics.mean_squared_error(test_labels, predictions)

                test_accs.append(test_acc)
                test_f1s.append(test_f1)
                test_mse_list.append(test_mse)

            test_mean_accs = np.mean(test_accs)
            test_mean_f1s = np.mean(test_f1s)
            test_mse = np.mean(test_mse_list)

            history['test_accs'].append(test_mean_accs)
            history['test_f1s'].append(test_mean_f1s)
            history['test_mse'].append(test_mse)

        history['train_loss'].append(train_loss)

        if wandb_logging:
            wandb_logging.add_scalars({
                'test_acc': test_mean_accs,
                'test_f1': test_mean_f1s,
                'test_mse': test_mse,
                'train_loss': train_loss,
            }, global_step=epoch)
            wandb_logging.dump(epoch)

        if model_name is not None and save_checkpoints:
            model_state_dict_save_path = f"models/{model_name}_state_dict_epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_state_dict_save_path)

    print('Finished Training')
    return model, history
