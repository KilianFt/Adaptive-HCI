import random
import collections

import numpy as np
import torch
from torch.utils.data import Dataset

base = torch.tensor([2 ** i for i in reversed(range(5))], dtype=torch.float32)


class ReplayBuffer(Dataset):
    def __init__(self, max_size, num_classes):
        self.buffer = collections.deque(maxlen=max_size)

    def extend(self, dataset):
        self.buffer.extend(list(dataset))

    def __getitem__(self, sample_idx):
        return self.buffer[sample_idx]

    def __len__(self):
        return len(self.buffer)


class ClassBalancingReplayBuffer(Dataset):
    def __init__(self, max_size, num_classes):
        self.buffers = {k: collections.deque(maxlen=max_size // num_classes) for k in range(num_classes)}
        self.num_classes = num_classes

    def add(self, observation: torch.Tensor, action: torch.Tensor):
        assert (action.long() == action).all(), "Actions must be integers"
        targets = torch.nonzero(action).flatten().tolist()
        for target in targets:
            self.buffers[target].append(observation)

    def extend(self, dataset):
        xs, ys = zip(*dataset)
        ys = torch.stack(ys)
        assert (ys.long() == ys).all(), "Actions must be integers"
        assert (ys[:, 0] == 0).all(), "0 is supposed to be unused"

        for x, y in dataset:  # TODO: This for loop sucks
            self.add(x, y)

    def __getitem__(self, sample_idx):
        # TODO: this is not ok, nor deterministic
        assert len(self) > 0, "The buffer is not yet filled"
        classes = list(range(self.num_classes))
        classes.remove(0)  # 0 is unused for now.
        class_idx = random.choice(classes)
        class_onehot = torch.zeros(self.num_classes)
        class_onehot[class_idx] = 1
        return random.choice(self.buffers[class_idx]), class_onehot

    def __len__(self):
        # 0 is unused for now.
        return min(len(v) for k, v in self.buffers.items() if k != 0)


if __name__ == '__main__':
    buffer = ReplayBuffer(max_size=1000, num_classes=3)
    for i in range(1000):
        buffer.add(torch.randn(5), torch.tensor([1, 0, 0]))
        buffer.add(torch.randn(5), torch.tensor([0, 1, 0]))
        buffer.add(torch.randn(5), torch.tensor([0, 0, 1]))

    print(len(buffer))
    print(buffer[0])
