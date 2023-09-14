import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from vit_pytorch import ViT

from utils import labels_to_onehot


def get_raw_mad_dataset(eval_path, window_length, overlap):
    person_folders = os.listdir(eval_path)

    first_folder = os.listdir(eval_path)[0]
    keys = next(os.walk((eval_path+first_folder)))[1]

    number_of_classes = 7
    size_non_overlap = window_length - overlap

    raw_dataset_dict = {}
    for key in keys:
            
        raw_dataset = {
            'examples': [],
            'labels': [],
        }

        for person_dir in person_folders:
            examples = []
            labels = []
            data_path = eval_path + person_dir + '/' + key
            for data_file in os.listdir(data_path):
                if (data_file.endswith(".dat")):
                    data_read_from_file = np.fromfile((data_path+'/'+data_file), dtype=np.int16)
                    data_read_from_file = np.array(data_read_from_file, dtype=np.float32)

                    dataset_example_formatted = []
                    example = []
                    emg_vector = []
                    for value in data_read_from_file:
                        emg_vector.append(value)
                        if (len(emg_vector) >= 8):
                            if (example == []):
                                example = emg_vector
                            else:
                                example = np.row_stack((example, emg_vector))
                            emg_vector = []
                            if (len(example) >= window_length):
                                example = example.transpose()
                                dataset_example_formatted.append(example)
                                example = example.transpose()
                                example = example[size_non_overlap:]
                    dataset_example_formatted = np.array(dataset_example_formatted)
                    examples.append(dataset_example_formatted)
                    data_file_index = int(data_file.split('classe_')[1][:-4])
                    label = data_file_index % number_of_classes + np.zeros(dataset_example_formatted.shape[0])
                    labels.append(label)

            raw_dataset['examples'].append(np.concatenate(examples))
            raw_dataset['labels'].append(np.concatenate(labels))

        raw_dataset_dict[key] = raw_dataset

    return raw_dataset_dict


def get_mad_windows_dataset(mad_base_dir, window_length, overlap):
    train_path = mad_base_dir + 'PreTrainingDataset/'
    eval_path = mad_base_dir + 'EvaluationDataset/'

    eval_raw_dataset_dict = get_raw_mad_dataset(eval_path, window_length, overlap)
    train_raw_dataset_dict = get_raw_mad_dataset(train_path, window_length, overlap)

    mad_all_windows = eval_raw_dataset_dict['training0']['examples'] + \
                        eval_raw_dataset_dict['Test0']['examples'] + \
                        eval_raw_dataset_dict['Test1']['examples'] + \
                        train_raw_dataset_dict['training0']['examples']

    mad_all_labels = eval_raw_dataset_dict['training0']['labels'] + \
                        eval_raw_dataset_dict['Test0']['labels'] + \
                        eval_raw_dataset_dict['Test1']['labels'] + \
                        train_raw_dataset_dict['training0']['labels']

    # filter by labels
    mad_windows = None
    mad_labels = None
    for mad_p_examples, mad_p_labels in zip(mad_all_windows, mad_all_labels):
        label_map = (mad_p_labels >= 0) & (mad_p_labels <= 4)
        mad_subject_windows = mad_p_examples[label_map]
        subject_labels = mad_p_labels[label_map]

        if mad_windows is None:
            mad_windows = mad_subject_windows
            mad_labels = subject_labels
        else:
            mad_windows = np.concatenate((mad_windows, mad_subject_windows))
            mad_labels = np.concatenate((mad_labels, subject_labels))

    mad_onehot_labels = np.array([labels_to_onehot(label) for label in mad_labels])

    return mad_windows, mad_onehot_labels


def create_windows(X, y, stride, window_length, desired_labels = None):
    features_dataset = {key: [] for key in np.unique(y)}
    last_class_idx = None
    consequetive_features = []

    for class_idx, feature in zip(y, X):
        if class_idx != last_class_idx:
            if consequetive_features:
                features_dataset[class_idx].append(np.array(consequetive_features))
            consequetive_features = [feature]
            last_class_idx = class_idx
        else:
            consequetive_features.append(feature)

    if consequetive_features:
        features_dataset[class_idx].append(np.array(consequetive_features))

    windows = []
    labels = []
    for class_idx, feature_list in features_dataset.items():
        if desired_labels is None or class_idx in desired_labels:
            for consequetive_features in feature_list:
                num_windows = (consequetive_features.shape[0] - window_length) // stride + 1
                for i in range(num_windows):
                    start = i * stride
                    end = start + window_length
                    window = consequetive_features[start:end, :]
                    windows.append(window)
                    labels.append(class_idx)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=int)


def get_ninapro_windows_dataset(ninapro_base_dir, emg_min, emg_max, stride, window_length):
    ninapro_windows = None
    ninapro_labels = None

    ninapro_person_dirs = next(os.walk(ninapro_base_dir))[1]
    for nina_person_dir in ninapro_person_dirs:
        files = os.listdir(ninapro_base_dir + nina_person_dir)
        for file in files:
            if file.endswith('E2_A1.mat'):
                filepath = ninapro_base_dir + nina_person_dir + '/' + file

                ninapro_s1 = loadmat(filepath)

                ninapro_s_x_raw = ninapro_s1['emg'][:, :8]
                ninapro_s_x = np.interp(ninapro_s_x_raw, (emg_min, emg_max), (-1, +1))
                ninapro_s_y = ninapro_s1['restimulus'].squeeze()

                subject_windows, subject_labels = create_windows(X = ninapro_s_x,
                                                                 y = ninapro_s_y,
                                                                 stride = stride,
                                                                 window_length = window_length,
                                                                 desired_labels = [0,13,14,15,16],)

                if ninapro_windows is None:
                    ninapro_windows = subject_windows
                    ninapro_labels = subject_labels
                else:
                    ninapro_windows = np.concatenate((ninapro_windows, subject_windows))
                    ninapro_labels = np.concatenate((ninapro_labels, subject_labels))

    ninapro_windows = ninapro_windows.swapaxes(1,2)

    # replace labels
    label_map = {0: 0,
                13: 2,
                14: 4,
                15: 1,
                16: 3,
                }

    ninapro_mapped_labels = np.vectorize(label_map.get)(ninapro_labels)
    ninapro_onehot_labels = np.array([labels_to_onehot(label) for label in ninapro_mapped_labels])

    return ninapro_windows, ninapro_onehot_labels


class CustomEMGDataset(Dataset):
    def __init__(self, x_samples, y_samples):
        self.x_samples = x_samples
        self.y_samples = y_samples

    def __len__(self):
        return len(self.x_samples)
    
    def __getitem__(self, idx):
        x_tensor = torch.tensor(self.x_samples[idx,:,:])
        y_tensor = torch.tensor(self.y_samples[idx]).type(torch.float32)
        return x_tensor, y_tensor


if __name__ == '__main__':

    mad_base_dir = 'datasets/MyoArmbandDataset/'
    ninapro_base_dir = 'datasets/ninapro/DB5/'

    sampling_rate = 200 # Hz
    window_length = sampling_rate # 1 second
    overlap = 0

    emg_min = -128.
    emg_max = 127.
    stride = window_length - overlap

    device = 'mps'

    mad_windows, mad_onehot_labels = get_mad_windows_dataset(mad_base_dir, window_length, overlap)
    ninapro_windows, ninapro_onehot_labels = get_ninapro_windows_dataset(ninapro_base_dir, emg_min, emg_max, stride, window_length)

    print(ninapro_windows.shape, ninapro_onehot_labels.shape)
    print(mad_windows.shape, mad_onehot_labels.shape)

    # combine mad and ninapro data
    windows = np.concatenate((mad_windows, ninapro_windows))
    labels = np.concatenate((mad_onehot_labels, ninapro_onehot_labels))
    n_labels = len(np.unique(labels))

    train_x, test_x, train_y, test_y = train_test_split(windows, labels)

    train_dataset = CustomEMGDataset(train_x, train_y)
    test_dataset = CustomEMGDataset(test_x, test_y)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = ViT(
        image_size = 200,
        patch_size = 8,
        num_classes = labels.shape[1],
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

    for epoch in range(20):

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
