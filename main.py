import matplotlib.pyplot as plt
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
            nn.Linear(input_dim, output_dim),
            torch.nn.Tanh()
        )
        self.log_std_head = nn.Linear(input_dim, output_dim)

    def forward(self, state):
        x = state
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        # usually, you need to limit the values of log_std
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.policy = GaussianPolicy(1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x, explore):
        mu, log_std = self.policy(x)
        std = torch.exp(log_std)
        if explore:
            normal = torch.distributions.Normal(mu, std)
            action = normal.sample()
        else:
            action = mu
        return action

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

        # Let's assume that the model predicts mean and log_std of a Gaussian distribution
        mean, log_std = self.policy(states)
        std = torch.exp(log_std)

        # Create a normal distribution with the predicted mean and std
        normal = torch.distributions.Normal(mean, std)

        # Compute log probability of the taken actions
        log_prob = normal.log_prob(actions)

        # Compute the policy loss
        loss = -(log_prob * rewards).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)
        self.optimizer.step()
        return loss.item()

    def sl_update(self, states, optimal_actions):
        self.train()
        self.optimizer.zero_grad()

        predicted_actions = self.deterministic_forward(states)
        loss = torch.nn.functional.mse_loss(predicted_actions, torch.clamp(optimal_actions, -1, 1))
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

    steps = 1  # 00
    epochs = (100_000 // 8) // steps

    initial_parameters = controller.state_dict()

    rl_losses, rl_reward_history = train_rl(environment, controller, user, steps, epochs)
    controller.load_state_dict(initial_parameters)
    sl_losses, sl_reward_history = train_sl(environment, controller, user, steps, epochs)

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
    for epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards = rollout(user, environment, controller, max_steps=steps)
        loss = controller.rl_update(actions, states, rewards)
        rl_reward_history.append(sum(rewards).item())
        rl_losses.append(loss)
    return rl_losses, rl_reward_history


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
