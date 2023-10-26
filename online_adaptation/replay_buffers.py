import random
import collections

import numpy as np
import torch
from torch.utils.data import Dataset

base = torch.tensor([2 ** i for i in reversed(range(5))], dtype=torch.float32)


class ReplayBuffer(Dataset):
    def __init__(self, max_size, num_classes):
        self.buffers = {k: collections.deque(maxlen=max_size) for k in range(num_classes)}

    def add(self, observation: torch.Tensor, action: torch.Tensor):
        assert (action.long() == action).all(), "Actions must be integers"
        targets = action.argwhere().flatten().tolist()
        for target in targets:
            self.buffers[target].append(observation)

    def extend(self, dataset):
        for x, y in dataset:  # TODO: This for loop sucks
            self.add(x, y)

    @property
    def num_classes(self):
        return len(self.buffers.values())

    def __getitem__(self, sample_idx):
        # TODO: this is not ok, nor deterministic
        classes = list(self.buffers.keys())
        classes.remove(0) # 0 is unused for now.
        class_idx = random.choice(classes)
        class_onehot = torch.zeros(self.num_classes)
        class_onehot[class_idx] = 1
        return random.choice(self.buffers[class_idx]), class_onehot

    def __len__(self):
        # 0 is unused for now.
        return min(len(v) for k, v in self.buffers.items() if k != 0)


if __name__ == '__main__':
    buffer = ReplayBuffer(max_size=1000, num_classes=3)
    for _ in range(500):
        buffer.add(np.random.rand(5), random.randint(0, 2), random.random(), np.random.rand(5), False)

    observations, actions, rewards, next_observations, dones = buffer.sample(32)
