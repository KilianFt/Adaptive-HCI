import numpy as np
import torch
from stable_baselines3 import PPO


class RLSLController(PPO):
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

    def sl_update(self, states, optimal_actions):
        self.policy.train()
        self.policy.optimizer.zero_grad()
        predicted_action = self.deterministic_forward(states)
        target_action = torch.clip(optimal_actions, -1, 1)

        loss = torch.nn.functional.mse_loss(predicted_action, target_action)
        loss.backward()
        self.policy.optimizer.step()
        return loss.item()

class SLOnlyController():
    def __init__(self, model_path, device='cpu', lr=1e-3):
        self.device = device

        if model_path is not None:
            self.policy = torch.load(model_path).to(self.device)
        else:
            raise NotImplementedError
            # TODO train model if not found

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def deterministic_forward(self, emg_window):
        emg_window_tensor = emg_window.unsqueeze(0).to(self.device)
        emg_window_tensor.swapaxes_(2, 3)
        outputs = self.policy(emg_window_tensor)
        return outputs

    def sl_update(self, states, optimal_actions):
        if states.device != self.device:
            states = states.to(self.device)
        if optimal_actions.device != self.device:
            optimal_actions = optimal_actions.to(self.device)

        self.policy.train()
        self.optimizer.zero_grad()
        predicted_action = self.policy(states)
        target_action = torch.clip(optimal_actions, -1, 1)

        loss = self.criterion(predicted_action, target_action)
        loss.backward()
        self.optimizer.step()
        return loss.item()
