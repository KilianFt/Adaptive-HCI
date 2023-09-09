import numpy as np
import torch
from torch.utils import data

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


class NinaPro1(data.Dataset[data.TensorDataset]):
    def __init__(self):
        import scipy.io
        mat = scipy.io.loadmat('ninapro/DB1_s1/S1_A1_E1.mat')

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
