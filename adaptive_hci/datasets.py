import os

import numpy as np
from scipy.io import loadmat
import torch
from torch.utils import data

from .utils import labels_to_onehot

gesture_names = [
    "rest",
    "index finger flexion",
    "index extension",
    "middle finger flexion",
    "middle finger extension",
    "ring finger flexion",
    "ring finger extension",
    "little finger flexion",
    "little finger extension",
    "thumb adduction",
    "thumb abduction",
    "thumb flexion",
    "thumb extension",
]


def get_raw_mad_dataset(eval_path, window_length, overlap):
    person_folders = os.listdir(eval_path)

    first_folder = os.listdir(eval_path)[0]
    keys = next(os.walk((eval_path + first_folder)))[1]

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
                    data_read_from_file = np.fromfile((data_path + '/' + data_file), dtype=np.int16)
                    data_read_from_file = np.array(data_read_from_file, dtype=np.float32)

                    dataset_example_formatted = []
                    example = None
                    emg_vector = []
                    for value in data_read_from_file:
                        emg_vector.append(value)
                        if (len(emg_vector) >= 8):
                            if example is None:
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


def maybe_download_mad_dataset(mad_base_dir):
    if os.path.exists(mad_base_dir):
        return

    os.makedirs(mad_base_dir, exist_ok=True)
    os.system(f'git clone https://github.com/UlysseCoteAllard/MyoArmbandDataset {mad_base_dir}')


def get_mad_windows_dataset(mad_base_dir, _, window_length, overlap):
    maybe_download_mad_dataset(mad_base_dir)

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


def create_ninapro_windows(X, y, stride, window_length, desired_labels=None):
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


def get_ninapro_windows_dataset(ninapro_base_dir, emg_range, window_length, overlap):
    ninapro_windows = None
    ninapro_labels = None

    stride = window_length - overlap

    ninapro_person_dirs = next(os.walk(ninapro_base_dir))[1]
    for nina_person_dir in ninapro_person_dirs:
        files = os.listdir(ninapro_base_dir + nina_person_dir)
        for file in files:
            if file.endswith('E2_A1.mat'):
                filepath = ninapro_base_dir + nina_person_dir + '/' + file

                ninapro_s1 = loadmat(filepath)

                ninapro_s_x_raw = ninapro_s1['emg'][:, :8]
                ninapro_s_x = np.interp(ninapro_s_x_raw, emg_range, (-1, +1))
                ninapro_s_y = ninapro_s1['restimulus'].squeeze()

                subject_windows, subject_labels = create_ninapro_windows(X=ninapro_s_x,
                                                                         y=ninapro_s_y,
                                                                         stride=stride,
                                                                         window_length=window_length,
                                                                         desired_labels=[0, 13, 14, 15, 16], )

                if ninapro_windows is None:
                    ninapro_windows = subject_windows
                    ninapro_labels = subject_labels
                else:
                    ninapro_windows = np.concatenate((ninapro_windows, subject_windows))
                    ninapro_labels = np.concatenate((ninapro_labels, subject_labels))

    ninapro_windows = ninapro_windows.swapaxes(1, 2)

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


class EMGWindowsDataset(data.Dataset):
    DATASET_DIRS = {
        'ninapro5_train': ('datasets/ninapro/DB5/train/', get_ninapro_windows_dataset),
        'ninapro5_test': ('datasets/ninapro/DB5/test/', get_ninapro_windows_dataset),
        'mad': ('datasets/MyoArmbandDataset/', get_mad_windows_dataset),
    }

    def __init__(self, dataset_name, window_size=200, overlap=0, emg_range=(-128, 127)):
        assert dataset_name in self.DATASET_DIRS, f'Dataset not found, please pick one of {list(self.DATASET_DIRS.keys())}'

        base_dir, load_dataset = self.DATASET_DIRS[dataset_name]

        self.windows, self.labels = load_dataset(base_dir, emg_range, window_size, overlap)

        self.windows = torch.tensor(self.windows, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x_tensor = self.windows[idx, :, :]
        y_tensor = self.labels[idx]
        return x_tensor, y_tensor

    @property
    def num_unique_labels(self):
        return self.labels.shape[1]


class NinaPro1(data.Dataset[data.TensorDataset]):
    def __init__(self):
        try:
            mat = loadmat('ninapro/DB1_s1/S1_A1_E1.mat')
        except FileNotFoundError:
            raise FileNotFoundError("Please download the NinaPro dataset from https://zenodo.org/record/1000116 and "
                                    "extract it into the 'ninapro' folder")

        x = mat['emg']
        y = mat['restimulus'].squeeze(1)

        self.features_size = x.shape[1]
        self.upper_bound = np.max(x, axis=0)
        self.lower_bound = np.min(x, axis=0)
        assert mat["exercise"] == 1
        assert mat["subject"] == 1

        self.class_dataset = {}
        for class_idx in np.unique(y):
            xs = torch.tensor(x[y == class_idx])
            ys = torch.full((len(xs),), class_idx, dtype=torch.long)
            self.class_dataset[class_idx] = data.TensorDataset(xs, ys)

    def __getitem__(self, index) -> data.TensorDataset:
        return self.class_dataset[index]


class CombinedDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Calculate the total length of the combined dataset
        self.total_length = len(self.dataset1) + len(self.dataset2)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            x_tensor = self.dataset1.windows[idx, :, :]
            y_tensor = self.dataset1.labels[idx]
            return x_tensor, y_tensor
        else:
            # Adjust the idx for the second dataset
            idx -= len(self.dataset1)
            x_tensor = self.dataset2.windows[idx, :, :]
            y_tensor = self.dataset2.labels[idx]
            return x_tensor, y_tensor

    @property
    def num_unique_labels(self):
        assert self.dataset1.labels.shape[1] == self.dataset2.labels.shape[1], 'labels of both datasets must match'
        return self.dataset1.labels.shape[1]


class EMGWindowsAdaptattionDataset(data.Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x_tensor = self.windows[idx, :, :]
        y_tensor = self.labels[idx]
        return x_tensor, y_tensor

    @property
    def num_unique_labels(self):
        return self.labels.shape[1]