import torch
import tqdm
import time
import pygame
import numpy as np
import gymnasium as gym

from controllers import RLSLController
from environment import Environment
from metrics import plot_and_mean
from users import ProportionalUser

def deterministic_rollout(user, environment, controller, max_steps):
    # state = environment.reset()
    reset_info, _ = environment.reset()
    state = reset_info['achieved_goal'][:2]

    states = []
    action_means = []
    optimal_actions = []
    rewards = []

    for time_step in range(max_steps):
        user_signal = user.get_signal()

        # action_mean = controller.deterministic_forward(user_signal.unsqueeze(0))
        action_mean, _, _ = controller.policy.forward(user_signal.unsqueeze(0), deterministic=True)
        # new_state, reward, done, info = environment.step(action_mean.squeeze().detach().numpy())
        obs, _, _, _, info = environment.step(action_mean.squeeze().detach().numpy())
        new_state = obs['achieved_goal'][:2]

        # compute 2D goal (ignore z axis)
        substitute_goal = obs["desired_goal"].copy()
        substitute_goal[2] = obs["achieved_goal"].copy()[2]
        reward = environment.compute_reward(obs["achieved_goal"], substitute_goal, info)
        done = environment.compute_terminated(obs["achieved_goal"], substitute_goal, info)
        truncated = environment.compute_truncated(obs["achieved_goal"], substitute_goal, info)
        
        states.append(user_signal)
        action_means.append(action_mean)

        optimal_action = np.where((substitute_goal[:2] > state).astype(int), 1, -1)
        extended_optimal_actions = np.zeros(4)
        extended_optimal_actions[:2] = optimal_action
        optimal_actions.append(torch.tensor([extended_optimal_actions], dtype=torch.float32))
        rewards.append(reward)
    
        state = new_state
        if done:
            break

        time.sleep(.1)

    states = torch.stack(states)
    action_means = torch.stack(action_means)
    optimal_actions = torch.stack(optimal_actions).squeeze()
    rewards = torch.tensor(rewards).unsqueeze(-1)
    # goal = environment.get_goal()
    goal = substitute_goal[:2]


    return states, action_means, optimal_actions, rewards, time_step, goal


def train_rl(controller: RLSLController, epochs):
    learner = controller.learn(epochs)
    learner.logger.dump()
    return learner


def train_sl(environment, controller, user, epochs, max_steps):
    sl_reward_history = []
    sl_reward_sum_history = []
    sl_losses = []
    performances = []
    goals = []

    for _epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards, performance, goal = deterministic_rollout(user, environment, controller, max_steps)
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

    max_steps = 100
    total_timesteps = 10 #10_000

    # user = ProportionalUser(goal=1, middle_pixels=np.array([249, 249]))

    # max screen pos Point(x=1511, y=981)
    user = ProportionalUser(goal=1, middle_pixels=np.array([755, 470]))

    # environment = Environment(user=user, max_steps=MAX_STEPS)
    environment = gym.make('FetchReachDense-v2', render_mode="human")
    environment.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    controller = RLSLController(env=environment)

    initial_parameters = controller.policy.state_dict()
    # train_rl(controller, total_timesteps)

    controller.policy.load_state_dict(initial_parameters)

    sl_losses, sl_reward_history, sl_reward_sum_history, sl_avg_steps, goals = train_sl(environment, controller, user, total_timesteps, max_steps)

    # sync to wandb
    sl_reward_goal_dist_ratio = [r / abs(g.sum()) if not abs(g.sum()) == 0 else 0 for r, g in zip(sl_reward_sum_history, goals)]

    # plot_and_mean(sl_avg_steps, "SL avg steps")
    # plot_and_mean(sl_reward_goal_dist_ratio, "SL return / goal dist")
    # plot_and_mean(sl_losses, "SL losses")

    print("done")

    # pygame.quit()
    environment.close()


if __name__ == '__main__':
    main()
