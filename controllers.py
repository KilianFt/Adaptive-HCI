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

class SLController():
    def __init__(self):
        self.policy = torch.load('models/pretrained_vit_2dof.pt')
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.criterion = torch.nn.MSELoss()
        self.device = 'cpu'

    def deterministic_forward(self, emg_window):
        emg_window_tensor = emg_window.unsqueeze(0).to(self.device)
        emg_window_tensor.swapaxes_(2, 3)
        predictions = self.policy(emg_window_tensor)
        return predictions.cpu()

    def sl_update(self, states, optimal_actions):
        self.policy.train()
        self.optimizer.zero_grad()
        predicted_action = self.policy(states)
        target_action = torch.clip(optimal_actions, -1, 1)

        loss = self.criterion(predicted_action, target_action)
        loss.backward()
        self.optimizer.step()
        return loss.item()
