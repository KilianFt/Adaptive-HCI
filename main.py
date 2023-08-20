import torch
import tqdm
import pygame
import numpy as np

from controllers import RLSLController
from environment import Environment
from metrics import plot_and_mean
from users import ProportionalUser

import matplotlib
matplotlib.use('module://pygame_matplotlib.backend_pygame')

def norm_pos_to_pixel(norm_position):
    return [249 * (n_pos / 10 + 1) for n_pos in norm_position]

def deterministic_rollout(user, environment, controller, screen, clock):
    state = environment.reset()
    states = []
    action_means = []
    optimal_actions = []
    rewards = []

    dt = 0
    running = True
    pygame.mouse.set_pos((249,0))

    for time_step in range(environment.max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.unicode == 'q':
                    running = False
            
        user_signal = user.get_signal()

        # action_mean = controller.deterministic_forward(user_signal.unsqueeze(0))
        action_mean, _, _ = controller.policy.forward(user_signal.unsqueeze(0), deterministic=True)
        new_state, reward, done, info = environment.step(action_mean.squeeze().detach().numpy())

        screen.fill((0,0,0))
        line_y = (screen.get_height() / 2)
        line_x = (screen.get_width() / 2)

        pygame.draw.line(screen,
                        color=(255,255,255),
                        start_pos=(0, line_y),
                        end_pos=(screen.get_width(), line_y),
                        width=5)

        pygame.draw.line(screen,
                        color=(255,255,255),
                        start_pos=(line_x, 0),
                        end_pos=(line_x, screen.get_height()),
                        width=5)

        pygame.draw.circle(surface=screen,
                        color=(255,255,255),
                        center=norm_pos_to_pixel(new_state),
                        radius=8)

        pygame.draw.circle(surface=screen,
                        color=(0,255,0),
                        center=norm_pos_to_pixel(environment.get_goal()),
                        radius=8)

        states.append(user_signal)
        action_means.append(action_mean)

        optimal_action = np.where((environment.goal > state).astype(int), 1, -1)

        optimal_actions.append(torch.tensor([optimal_action], dtype=torch.float32))
        rewards.append(reward)

        state = new_state
        if done or not running:
            break

        pygame.display.flip()
        dt = clock.tick(10) / 1000

    states = torch.stack(states)
    action_means = torch.stack(action_means)
    optimal_actions = torch.stack(optimal_actions)
    rewards = torch.tensor(rewards).unsqueeze(-1)


    return states, action_means, optimal_actions, rewards, time_step, environment.get_goal()


def train_rl(controller: RLSLController, epochs):
    learner = controller.learn(epochs)
    learner.logger.dump()
    return learner


def train_sl(environment, controller, user, epochs, screen, clock):
    sl_reward_history = []
    sl_reward_sum_history = []
    sl_losses = []
    performances = []
    goals = []

    for _epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards, performance, goal = deterministic_rollout(user, environment, controller, screen, clock)
        loss = controller.sl_update(states, optimal_actions)

        sl_reward_history.append(torch.mean(rewards).detach().item())
        sl_reward_sum_history.append(torch.sum(rewards).detach().item())
        performances.append(performance)
        sl_losses.append(loss)
        goals.append(goal)

    return sl_losses, sl_reward_history, sl_reward_sum_history, performances, goals

def main():
    WIDTH = 500
    HEIGHT = 500

    MAX_STEPS = 100
    total_timesteps = 10 #10_000

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    user = ProportionalUser(goal=1, middle_pixels=(249, 249))
    environment = Environment(user=user, max_steps=MAX_STEPS)
    controller = RLSLController(env=environment)

    initial_parameters = controller.policy.state_dict()
    # train_rl(controller, total_timesteps)

    controller.policy.load_state_dict(initial_parameters)

    sl_losses, sl_reward_history, sl_reward_sum_history, sl_avg_steps, goals = train_sl(environment, controller, user, total_timesteps, screen, clock)

    # sync to wandb
    sl_reward_goal_dist_ratio = [r / abs(g.sum()) if not abs(g.sum()) == 0 else 0 for r, g in zip(sl_reward_sum_history, goals)]

    # plot_and_mean(sl_avg_steps, "SL avg steps")
    plot_and_mean(sl_reward_goal_dist_ratio, "SL return / goal dist")
    # plot_and_mean(sl_losses, "SL losses")

    print("done")

    pygame.quit()


if __name__ == '__main__':
    main()
