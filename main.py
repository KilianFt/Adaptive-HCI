import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    def __init__(self, user, goal, max_steps):
        self.goal = goal
        self.user = user
        self.position = None
        self.current_steps = None
        self.max_steps = max_steps
        self.reset()

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


class Controller(PPO):
    def __init__(self, env):
        super(Controller, self).__init__(
            "MlpPolicy",
            env,
            n_steps=50,
            verbose=1,
            tensorboard_log="tmp/a2c_cartpole_tensorboard/",
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


def deterministic_rollout(user, environment, controller):
    state = environment.reset()
    states = []
    action_means = []
    optimal_actions = []
    rewards = []
    time_step = 0
    done = False
    while not done:
        user_signal = user.get_signal(state)

        action_mean = controller.deterministic_forward(user_signal.unsqueeze(0))
        new_state, reward, done, info = environment.step(action_mean.detach().item())

        states.append(user_signal)
        action_means.append(action_mean)
        optimal_actions.append(torch.tensor([state - environment.goal], dtype=torch.float32))
        rewards.append(reward)

        state = new_state

    states = torch.stack(states)
    action_means = torch.stack(action_means)
    optimal_actions = torch.stack(optimal_actions)
    rewards = torch.tensor(rewards).unsqueeze(-1)
    return states, action_means, optimal_actions, rewards, time_step


def main():
    user = User(goal=1)
    environment = Environment(user=user, goal=1, max_steps=50)
    controller = Controller(env=environment)

    total_timesteps = 10_000
    initial_parameters = controller.policy.state_dict()
    learner = controller.learn(total_timesteps=total_timesteps)
    learner.logger.dump()

    controller.policy.load_state_dict(initial_parameters)

    sl_losses, sl_reward_history, sl_avg_steps = train_sl(environment, controller, user, total_timesteps)

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


def train_sl(environment, controller, user, epochs):
    sl_reward_history = []
    sl_losses = []
    performances = []

    for _epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards, performance = deterministic_rollout(user, environment, controller)
        loss = controller.sl_update(states, optimal_actions)

        sl_reward_history.append(torch.mean(rewards))
        performances.append(performance)
        sl_losses.append(loss)
    return sl_losses, sl_reward_history, performances


if __name__ == '__main__':
    main()
