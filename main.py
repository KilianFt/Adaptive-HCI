import pickle

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
        return self.get_state(), -distance_from_goal, done

    def is_done(self):
        return self.position == self.goal

    def get_state(self):
        return self.position

    def reset(self):
        self.position = 0


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.fc(x)

    def train_rl(self, states, actions, rewards):
        self.train()
        self.optimizer.zero_grad()

        predicted_actions = self(states)
        loss = self.loss_function(predicted_actions * rewards, actions * rewards)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_sl(self, states, optimal_actions):
        self.train()
        self.optimizer.zero_grad()

        predicted_actions = self(states)
        loss = self.loss_function(predicted_actions, torch.tanh(optimal_actions))
        loss.backward()
        self.optimizer.step()


def rollout(user, environment, controller, max_steps):
    environment.reset()
    states = []
    actions = []
    optimal_actions = []
    rewards = []
    for t in range(max_steps):
        state = environment.get_state()
        user_signal = user.get_signal(state)

        action = controller(user_signal)
        new_state, reward, done = environment.step(action.item())

        states.append(user_signal)
        actions.append(user_signal)
        optimal_actions.append(torch.tensor([environment.goal - state], dtype=torch.float32))
        rewards.append(reward)

        if done:
            break

    path = 'environment_data.pkl'
    with open(path, 'wb') as f:
        pickle.dump([
            torch.stack(states),
            torch.stack(actions),
            torch.stack(optimal_actions),
            torch.stack(rewards).unsqueeze(-1),
        ], f)

    with open(path, 'rb') as f:
        states, actions, optimal_actions, rewards = pickle.load(f)
    return states, actions, optimal_actions, rewards


def main():
    user = User(goal=1)
    environment = Environment(goal=1)
    controller = Controller()

    controller.fc.weight.data = -torch.abs(controller.fc.weight.data)
    reward_history = []

    for _ in tqdm.trange(10_000):
        states, actions, optimal_actions, rewards = rollout(user, environment, controller, max_steps=10)
        reward_history.append(sum(rewards))
        controller.train_sl(states, optimal_actions)

    plt.plot(reward_history)
    plt.show()


if __name__ == '__main__':
    main()
