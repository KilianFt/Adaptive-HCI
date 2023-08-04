import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pygame

class User:
    def __init__(self, goal, middle_pixel):
        self.goal = goal
        self.middle = middle_pixel

    def get_signal(self):
        signal = (pygame.mouse.get_pos()[0] - self.middle) / self.middle
        signal = max(min(signal, 1.), -1.)
        signal = torch.tensor([signal], dtype=torch.float32)
        return signal


class Environment:
    def __init__(self):
        self.position = 0
        self.goal = -1 + np.random.rand() * 2

    def step(self, action):
        self.position += action / 10
        self.position = max(min(self.position, 1.), -1.)
        position = torch.tensor(self.position, dtype=torch.float32)
        distance_from_goal = torch.abs(position - self.goal)
        done = self.is_done()
        return self.get_state(), -distance_from_goal, done

    def is_done(self):
        return abs(self.position - self.goal) < 0.01

    def get_state(self):
        return self.position

    def get_goal(self):
        return self.goal

    def reset(self):
        self.position = 0
        self.goal = -1 + np.random.rand() * 2


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.fc = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.5)

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


def norm_pos_to_pixel(norm_position):
    return 249 * (norm_position + 1)


def rollout(user, environment, controller, screen, clock, max_steps):
    environment.reset()
    states = []
    actions = []
    optimal_actions = []
    rewards = []

    dt = 0
    running = True
    pygame.mouse.set_pos((249,0))

    for t in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.unicode == 'q':
                    running = False

        state = environment.get_state()
        user_signal = user.get_signal()

        action = controller(user_signal)
        new_state, reward, done = environment.step(action.item())

        screen.fill((0,0,0))
        line_y = (screen.get_height() / 2)
        pygame.draw.line(screen,
                        color=(255,255,255),
                        start_pos=(0, line_y),
                        end_pos=(screen.get_width(), line_y),
                        width=5)

        pygame.draw.circle(surface=screen,
                        color=(255,255,255),
                        center=(norm_pos_to_pixel(new_state), line_y),
                        radius=8)

        pygame.draw.circle(surface=screen,
                        color=(0,255,0),
                        center=(norm_pos_to_pixel(environment.get_goal()), line_y),
                        radius=8)

        states.append(user_signal)
        actions.append(user_signal)
        optimal_actions.append(torch.tensor([environment.goal - state], dtype=torch.float32))
        rewards.append(reward)

        if done or not running:
            running = False
            break

        pygame.display.flip()
        dt = clock.tick(10) / 1000


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
    WIDTH = 500
    HEIGHT = 100

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    user = User(goal=1, middle_pixel=(WIDTH - 1) / 2)
    environment = Environment()
    controller = Controller()

    controller.fc.weight.data = -torch.abs(controller.fc.weight.data)
    controller.fc.bias.data = torch.zeros_like(controller.fc.bias.data)
    reward_history = []

    for _ in tqdm.trange(10):
        states, actions, optimal_actions, rewards = rollout(user,
                                                            environment,
                                                            controller,
                                                            screen,
                                                            clock,
                                                            max_steps=100)

        reward_history.append(sum(rewards).numpy()[0])
        controller.train_sl(states, optimal_actions)

    plt.plot(reward_history)
    plt.show()

    pygame.quit()

if __name__ == '__main__':
    main()
