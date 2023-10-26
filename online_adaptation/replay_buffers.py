import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, num_classes):
        self.buffers = [deque(maxlen=max_size) for _ in range(num_classes)]
        self.class_counters = np.zeros(num_classes)

    def add(self, observations: torch.Tensor, actions: torch.Tensor):
        assert (actions.long() == actions).all(), "Actions must be integers"
        # TODO: Can be sped up by np indexing observations and actions
        for observation, action in zip(observations, actions.long()):
            self.buffers[action.item()].append(observation)

    def extend(self, dataset):
        for experience in dataset:
            self.add(*experience)

    def sample(self, batch_size):
        samples_per_class = batch_size // len(self.buffers)
        remainder = batch_size % len(self.buffers)

        sampled_data = []
        for idx, class_buffer in enumerate(self.buffers):
            samples = random.sample(class_buffer, min(samples_per_class, len(class_buffer)))
            sampled_data.extend(samples)

        # Take care of remainder
        if remainder > 0:
            additional_samples = random.choices(self.buffers, k=remainder)
            for class_buffer in additional_samples:
                if len(class_buffer) > 0:
                    sampled_data.append(random.choice(class_buffer))

        return torch.stack(sampled_data)

    def __len__(self):
        return min(len(b) for b in self.buffers)


if __name__ == '__main__':
    buffer = ReplayBuffer(max_size=1000, num_classes=3)
    for _ in range(500):
        buffer.add(np.random.rand(5), random.randint(0, 2), random.random(), np.random.rand(5), False)

    observations, actions, rewards, next_observations, dones = buffer.sample(32)
