import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from stable_baselines3 import PPO


class User:
    def __init__(self, goal):
        self.goal = goal

    def get_signal(self, current_position):
        signal = self.goal - current_position
        signal = torch.tensor([signal], dtype=torch.float32)
        return signal


class Environment(gym.Env):
    def __init__(self, user, goal):
        self.goal = goal
        self.user = user
        self.position = None
        self.current_steps = None
        self.max_steps = 100
        self.reset

    def step(self, action):
        self.position += action
        position = torch.tensor(self.position, dtype=torch.float32)
        distance_from_goal = torch.abs(position - self.goal)
        done = self.is_done()
        reward = -distance_from_goal
        reward = torch.clamp(reward, -1, 1)
        return self.get_state(), float(reward), done, {}

    def is_done(self):
        return np.abs(self.position - self.goal) < 1 or self.current_steps >= self.max_steps

    def get_state(self):
        return self.user.get_signal(self.position)

    def reset(self, **kwargs):
        super(Environment, self).reset(**kwargs)
        self.position = np.random.randint(-10, 10)
        self.current_steps = 0
        return self.get_state()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-10, high=10, shape=(1,))

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(1,))


class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CategoricalPolicy, self).__init__()
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, state):
        x = state
        action_logits = self.action_head(x)
        return action_logits


class Controller(PPO):
    def __init__(self, env):
        super(Controller, self).__init__(
            "MlpPolicy",
            env,
            n_steps=50,
            verbose=1,
            tensorboard_log="tmp/a2c_cartpole_tensorboard/",
        )
        magnitude = 1.0

        self.index_to_action = np.array([
            -magnitude,
            # -magnitude / 2,
            # magnitude / 2,
            magnitude])
        self.action_to_index = {action: index for index, action in enumerate(self.index_to_action)}
        # self.policy = CategoricalPolicy(input_dim=1, num_classes=len(self.index_to_action))
        # self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def deterministic_forward(self, x):
        dist = self.policy.get_distribution(x)
        return dist.distribution.mean

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

        # Compute the log probabilities of the actions
        log_probabilities = torch.nn.functional.log_softmax(action_logits, dim=1)

        # Select the log probabilities of the actions that were actually taken
        log_probabilities = log_probabilities[range(len(actions)), actions]

        # Compute the loss
        advantage = rewards.squeeze(1) - torch.mean(rewards, dim=1)
        loss = -torch.einsum("i,i->", advantage, log_probabilities)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sl_update(self, states, optimal_actions):
        self.policy.train()
        self.policy.optimizer.zero_grad()
        predicted_action = self.deterministic_forward(states)
        # target_action = torch.argmin(torch.abs(optimal_actions - self.index_to_action), dim=1)
        target_action = torch.clip(optimal_actions, -1, 1)

        loss = torch.nn.functional.mse_loss(predicted_action, target_action)
        loss.backward()
        self.policy.optimizer.step()
        return loss.item()


def rollout(user, environment, controller, max_steps, explore=True):
    s0 = environment.reset()
    states = []
    action_means = []
    optimal_actions = []
    rewards = []
    t = 0
    for t in range(max_steps):
        state = environment.get_state()
        user_signal = user.get_signal(state)

        action_mean = controller.deterministic_forward(user_signal.unsqueeze(0))
        new_state, reward, done, info = environment.step(action_mean.detach().item())

        states.append(user_signal)
        action_means.append(action_mean)
        optimal_actions.append(torch.tensor([state - environment.goal], dtype=torch.float32))
        rewards.append(reward)

        if done:
            break

    states = torch.stack(states)
    action_means = torch.stack(action_means)
    optimal_actions = torch.stack(optimal_actions)
    rewards = torch.tensor(rewards).unsqueeze(-1)
    time_to_goal = t
    return states, action_means, optimal_actions, rewards, time_to_goal


def main():
    user = User(goal=1)
    environment = Environment(user=user, goal=1)
    controller = Controller(env=environment)

    steps = 50
    total_timesteps = 10_000
    initial_parameters = controller.policy.state_dict()
    learner = controller.learn(total_timesteps=total_timesteps)
    learner.logger.dump()

    controller.policy.load_state_dict(initial_parameters)

    sl_losses, sl_reward_history, sl_avg_steps = train_sl(environment, controller, user, steps, total_timesteps)

    plot_and_mean(sl_avg_steps, "SL avg steps")
    plot_and_mean(sl_reward_history, "SL rewards")
    plot_and_mean(sl_losses, "SL losses")

    print("done")


def plot_and_mean(rl_reward_history, title):
    if not rl_reward_history:
        return

    plt.title(title)
    plt.plot(rl_reward_history)  # , label=title)
    smoothed_rl_reward_history = np.convolve(rl_reward_history, np.ones(100) / 100, mode='valid')
    plt.plot(smoothed_rl_reward_history)  # , label=f"{title} smoothed")
    plt.legend()
    plt.show()


def train_rl(controller: Controller, epochs):
    return controller.learn(epochs)


def train_sl(environment, controller, user, steps, epochs):
    sl_reward_history = []
    sl_losses = []
    performances = []

    for _epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards, performance = rollout(
            user, environment, controller, max_steps=steps, explore=False)
        loss = controller.sl_update(states, optimal_actions)

        sl_reward_history.append(torch.mean(rewards))
        performances.append(performance)
        sl_losses.append(loss)
    return sl_losses, sl_reward_history, performances


if __name__ == '__main__':
    main()
