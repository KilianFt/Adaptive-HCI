import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


class User:
    def __init__(self, goal):
        self.goal = goal

    def get_signal(self, current_position):
        signal = self.goal - current_position
        signal = torch.tensor([signal], dtype=torch.float32)
        return signal


class Environment:
    def __init__(self, goal):
        self.position = 0
        self.goal = goal

    def step(self, action):
        self.position += action
        position = torch.tensor(self.position, dtype=torch.float32)
        distance_from_goal = torch.abs(position - self.goal)
        done = self.is_done()
        reward = -distance_from_goal
        reward = torch.clamp(reward, -1, 1)
        return self.get_state(), reward, done

    def is_done(self):
        return self.position == self.goal

    def get_state(self):
        return self.position

    def reset(self):
        self.position = 0


class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GaussianPolicy, self).__init__()
        self.mu_head = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
        )
        self.log_std_head = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, state):
        x = state
        mu = self.mu_head(x)

        # Since we have only one learnable parameter let the bias control the variance level
        log_std = self.log_std_head(torch.zeros_like(state))

        # usually, you need to limit the values of log_std
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std


class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CategoricalPolicy, self).__init__()
        self.action_head = torch.nn.Linear(input_dim, num_classes)

    def forward(self, state):
        x = state
        action_logits = self.action_head(x)
        return action_logits


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.policy = CategoricalPolicy(input_dim=1, num_classes=2)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.action_map = torch.tensor((-1, 1))

    def forward(self, x, explore):
        action_logits = self.policy(x)
        if explore:
            action_idx = torch.distributions.Categorical(logits=action_logits).sample()
        else:
            action_idx = torch.argmax(action_logits)
        action_direction = self.action_map[action_idx]
        return action_direction

    def deterministic_forward(self, x):
        mu, _ = self.policy(x)
        return mu

    def rl_update(self, states, actions, rewards):
        """
        Train the RL model using states, actions, and rewards.

        Args:
            states (torch.Tensor): The input states.
            actions (torch.Tensor): The taken actions.
            rewards (torch.Tensor): The rewards.
            epoch (int): The current epoch.

        Returns:
            float: The loss value.
        """
        if not self.training:
            self.train()
        self.optimizer.zero_grad()

        action_logits = self.policy(states)

        # Compute the policy loss
        loss = -(action_logits * rewards).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)
        self.optimizer.step()
        return loss.item()

    def sl_update(self, states, optimal_actions):
        self.train()
        self.optimizer.zero_grad()
        predicted_actions = self.deterministic_forward(states)
        reachable_optimal_action = torch.clamp(optimal_actions, -1, 1)
        loss = torch.nn.functional.mse_loss(predicted_actions, reachable_optimal_action)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def rollout(user, environment, controller, max_steps, explore=True):
    environment.reset()
    states = []
    actions = []
    optimal_actions = []
    rewards = []
    for t in range(max_steps):
        state = environment.get_state()
        user_signal = user.get_signal(state)

        action = controller(user_signal, explore)
        new_state, reward, done = environment.step(action.item())

        states.append(user_signal)
        actions.append(action)
        optimal_actions.append(torch.tensor([environment.goal - state], dtype=torch.float32))
        rewards.append(reward)

        if done:
            break

    states = torch.stack(states)
    actions = torch.stack(actions)
    optimal_actions = torch.stack(optimal_actions)
    rewards = torch.stack(rewards).unsqueeze(-1)
    return states, actions, optimal_actions, rewards


def main():
    user = User(goal=1)
    environment = Environment(goal=1)
    controller = Controller()

    steps = 5
    epochs = 500_000 // steps

    initial_parameters = controller.state_dict()

    rl_losses, rl_reward_history, mus, stds = train_rl(environment, controller, user, steps, epochs)
    controller.load_state_dict(initial_parameters)
    sl_losses, sl_reward_history = train_sl(environment, controller, user, steps, epochs // 10)

    plt.title("mus")
    plt.plot(mus)
    plt.show()
    plt.title("stds")
    stds = np.array(stds)
    plt.plot(stds[:, 0], label="weight")
    plt.plot(stds[:, 1], label="bias")
    plt.legend()
    plt.show()

    plt.title("RL rewards")
    plt.plot(rl_reward_history, label='RL')
    plt.show()
    plt.title("SL rewards")
    plt.plot(sl_reward_history, label='SL Reward')
    plt.show()

    plt.title("RL losses")
    plt.plot(rl_losses, label='RL Loss')
    plt.show()
    plt.title("SL losses")
    plt.plot(sl_losses, label='SL Loss')
    plt.legend()
    plt.show()

    print("done")


def train_rl(environment, controller: Controller, user, steps, epochs):
    rl_reward_history = []
    rl_losses = []
    mus = []
    stds = []
    for epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards = rollout(user, environment, controller, max_steps=steps)
        loss = controller.rl_update(states, actions,  rewards)
        rl_reward_history.append(sum(rewards).item())
        rl_losses.append(loss)
        mus.append(controller.policy.mu_head[0].weight.item())
        stds.append((
            controller.policy.log_std_head.weight.item(),
            controller.policy.log_std_head.bias.item()
        ))
    return rl_losses, rl_reward_history, mus, stds


def train_sl(environment, controller, user, steps, epochs):
    sl_reward_history = []
    sl_losses = []
    for _epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards = rollout(
            user, environment, controller, max_steps=steps, explore=False)
        loss = controller.sl_update(states, optimal_actions)

        sl_reward_history.append(sum(rewards).item())
        sl_losses.append(loss)
    return sl_losses, sl_reward_history


if __name__ == '__main__':
    main()
