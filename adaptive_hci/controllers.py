import abc
import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.data import DataLoader

from pretrain_model import train_emg_decoder
from training import train_model
from datasets import EMGWindowsAdaptattionDataset
class BaseController:
    @abc.abstractmethod
    def deterministic_forward(self, x) -> torch.Tensor:
        pass

    def sl_update(self, states, optimal_actions):
        device = next(self.policy.parameters()).device
        if states.device != device:
            states = states.to(device)
        if optimal_actions.device != device:
            optimal_actions = optimal_actions.to(device)

        self.policy.train()
        self.optimizer.zero_grad()
        predicted_action = self.policy(states)
        target_action = torch.clip(optimal_actions, -1, 1)

        loss = self.criterion(predicted_action, target_action)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class RLSLController(BaseController, PPO):
    def __init__(self, env):
        super(RLSLController, self).__init__(
            "MlpPolicy",
            env,
            n_steps=50,
            verbose=1,
            tensorboard_log="tmp/fetch_reach_tensorboard/",
            learning_rate=5e-3,
        )

    def deterministic_forward(self, x):
        dist = self.policy.get_distribution(x)
        return dist.distribution.mean


class SLOnlyController(BaseController):
    def __init__(self, model_path, device='cpu', lr=1e-3, batch_size=64, epochs=1, n_frozen_layers=0):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs

        if model_path is not None:
            self.policy = torch.load(model_path).to(self.device)
        else:
            self.policy = train_emg_decoder()

        if n_frozen_layers >= 1:
            for i, param in enumerate(self.policy.to_patch_embedding.parameters()):  
                param.requires_grad = False
            for i, param in enumerate(self.policy.dropout.parameters()):  
                param.requires_grad = False

        if n_frozen_layers >= 2:
            for layer_idx in range(min((n_frozen_layers - 1), 4)):
                for i, param in enumerate(self.policy.transformer.layers[layer_idx].parameters()):
                    param.requires_grad = False

        self.model_name = model_path.split('/')[-1].split('.')[0]

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def deterministic_forward(self, emg_window):
        emg_window_tensor = emg_window.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.policy(emg_window_tensor)
        return outputs
    
    def sl_update(self, states, optimal_actions):
        train_dataset = EMGWindowsAdaptattionDataset(windows=states, labels=optimal_actions)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,)

        # train model
        self.policy, history = train_model(self.policy,
                                    optimizer=self.optimizer,
                                    criterion=self.criterion,
                                    train_dataloader=train_dataloader,
                                    model_name=self.model_name,
                                    device=self.device,
                                    epochs=self.epochs,)

        return history['train_loss'][-1]
