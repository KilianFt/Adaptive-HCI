import pickle

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
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
        reward = -distance_from_goal
        reward = torch.clamp(reward, -1, 1)
        return self.get_state(), reward, done

    def is_done(self):
        return abs(self.position - self.goal) < 0.01

    def get_state(self):
        return self.position

    def get_goal(self):
        return self.goal

    def reset(self):
        self.position = 0
        self.goal = -1 + np.random.rand() * 2


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


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.policy = GaussianPolicy(1, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-1)

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
        reachable_optimal_action = torch.clamp(optimal_actions, -1, 1)
        loss = torch.nn.functional.mse_loss(predicted_actions, reachable_optimal_action)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def norm_pos_to_pixel(norm_position):
    return 249 * (norm_position + 1)


def rollout(user, environment, controller, screen, clock, max_steps, explore=True):
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

        action = controller(user_signal, explore)
        # In SL we have a tanh activation which naturally clips the actions, in RL we have a gaussian distribution
        # this clamp clips the actions before they go into the environment, which means we are not really following
        # the normal distribution, if we don't do that we might sample big actions that destabilize learning by having
        # extreme log_probs.
        action_clip = torch.clamp(action, -1, 1)
        new_state, reward, done = environment.step(action_clip.item())

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
        actions.append(action_clip)
        optimal_actions.append(torch.tensor([environment.goal - state], dtype=torch.float32))
        rewards.append(reward)

        if done or not running:
            running = False
            break

        pygame.display.flip()
        dt = clock.tick(10) / 1000


    states = torch.stack(states)
    actions = torch.stack(actions)
    optimal_actions = torch.stack(optimal_actions)
    rewards = torch.stack(rewards).unsqueeze(-1)
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

    steps = 100
    epochs = 100_000 // steps

    initial_parameters = controller.state_dict()

    # rl_losses, rl_reward_history, mus, stds = train_rl(environment, controller, user, steps, epochs)
    # controller.load_state_dict(initial_parameters)
    sl_losses, sl_reward_history = train_sl(environment,
                                            controller,
                                            user,
                                            screen,
                                            clock,
                                            steps,
                                            5)#epochs // 2)

    # plt.title("mus")
    # plt.plot(mus)
    # plt.show()
    # plt.title("stds")
    # stds = np.array(stds)
    # plt.plot(stds[:, 0], label="weight")
    # plt.plot(stds[:, 1], label="bias")
    # plt.legend()
    # plt.show()

    # plt.title("RL rewards")
    # plt.plot(rl_reward_history, label='RL')
    # plt.show()
    # plt.title("SL rewards")
    # plt.plot(sl_reward_history, label='SL Reward')
    # plt.show()

    # plt.title("RL losses")
    # plt.plot(rl_losses, label='RL Loss')
    # plt.show()
    # plt.title("SL losses")
    # plt.plot(sl_losses, label='SL Loss')
    # plt.legend()
    # plt.show()

    # print("done")
    pygame.quit()

def train_rl(environment, controller: Controller, user, steps, epochs):
    rl_reward_history = []
    rl_losses = []
    mus = []
    stds = []
    for epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards = rollout(user, environment, controller, max_steps=steps)
        loss = controller.rl_update(states, actions, rewards)
        rl_reward_history.append(sum(rewards).item())
        rl_losses.append(loss)
        mus.append(controller.policy.mu_head[0].weight.item())
        stds.append((
            controller.policy.log_std_head.weight.item(),
            controller.policy.log_std_head.bias.item()
        ))
    return rl_losses, rl_reward_history, mus, stds


def train_sl(environment, controller, user, screen, clock, steps, epochs):
    sl_reward_history = []
    sl_losses = []
    for _epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards = rollout(
            user, environment, controller, screen, clock, max_steps=steps, explore=False)
        loss = controller.sl_update(states, optimal_actions)

        sl_reward_history.append(sum(rewards).item())
        sl_losses.append(loss)
    return sl_losses, sl_reward_history


if __name__ == '__main__':
    main()
