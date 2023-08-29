import argparse
import itertools
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import tqdm

from controllers import RLSLController
from environment import XDProjection, EnvironmentWithUser
from metrics import plot_and_mean
from users import MouseProportionalUser


def deterministic_rollout(environment, controller):
    observation, info = environment.reset()
    observation = torch.tensor(observation).unsqueeze(0)

    rollout_goal = info["original_observation"]["desired_goal"]

    states = []
    user_signals = []
    action_means = []
    optimal_actions = []
    rewards = []

    for time_step in itertools.count():
        action_mean = controller.deterministic_forward(observation)
        action = action_mean.squeeze().detach().numpy()
        observation, reward, terminated, truncated, info = environment.step(action)

        observation = torch.tensor(observation).unsqueeze(0)

        optimal_action = info["optimal_action"]

        assert np.all(rollout_goal == info["original_observation"]["desired_goal"]), "Rollout goal changed!"

        states.append(observation)
        user_signals.append(observation)
        action_means.append(action_mean)
        optimal_actions.append(optimal_action)
        rewards.append(reward)

        if terminated or truncated:
            break

    user_signals = torch.stack(user_signals)
    action_means = torch.stack(action_means)
    optimal_actions = torch.tensor(optimal_actions)
    rewards = torch.tensor(rewards).unsqueeze(-1)

    return states, user_signals, action_means, optimal_actions, rewards, time_step, rollout_goal


def train_rl(controller: RLSLController, epochs):
    learner = controller.learn(epochs)
    learner.logger.dump()
    return learner


def train_sl(environment, controller, epochs, do_training=True):
    sl_reward_history = []
    sl_reward_sum_history = []
    sl_losses = []
    episode_durations = []
    goals = []

    if do_training:
        initial_parameters = controller.policy.state_dict()
        if not os.path.exists('models/2d_fetch'):
            os.makedirs('models/2d_fetch')
        torch.save(initial_parameters, 'models/2d_fetch/initial.pt')
    checkpoint_every = max(epochs // 10, 1)

    for epoch in tqdm.trange(epochs):
        states, user_signals, actions, optimal_actions, rewards, episode_duration, goal = deterministic_rollout(
            environment, controller)

        if do_training:
            loss = controller.sl_update(user_signals, optimal_actions)
            if epoch % checkpoint_every == 0:
                parameters = controller.policy.state_dict()
                torch.save(parameters, f'models/2d_fetch/epoch_{str(epoch)}.pt')
        else:
            loss = None

        sl_reward_history.append(torch.mean(rewards).detach().item())
        sl_reward_sum_history.append(torch.sum(rewards).detach().item())
        episode_durations.append(episode_duration)
        sl_losses.append(loss)
        goals.append(goal)

    return sl_losses, sl_reward_history, sl_reward_sum_history, episode_durations, goals


def main():
    parser = argparse.ArgumentParser(prog='Adaptive HCI - Fetch')

    parser.add_argument('--no-training', action='store_true')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()

    do_training = not args.no_training

    max_steps = 100
    total_timesteps = 100
    n_dof = 2

    user = MouseProportionalUser(simulate_user=True)

    environment = gym.make('FetchReachDense-v2', max_episode_steps=max_steps, render_mode="human")
    environment = XDProjection(environment, n_dof = n_dof)
    environment = EnvironmentWithUser(environment, user)

    controller = RLSLController(env=environment)

    if args.model is not None:
        trained_parameters = torch.load(args.model)
        controller.policy.load_state_dict(trained_parameters)

    sl_losses, sl_reward_history, sl_reward_sum_history, sl_avg_steps, goals = train_sl(
        environment, controller, total_timesteps, do_training=do_training)

    results = {
        'sl_losses': sl_losses,
        'sl_reward_history': sl_reward_history,
        'sl_reward_sum_history': sl_reward_sum_history,
        'sl_avg_steps': sl_avg_steps,
        'goals': goals,
    }

    with open('models/2d_fetch/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    sl_reward_goal_dist_ratio = []
    for r, g in zip(sl_reward_sum_history, goals):
        gsum = abs(g.sum())
        if gsum == 0:
            rgs = 0
        else:
            rgs = r / gsum
        sl_reward_goal_dist_ratio.append(rgs)

    plot_and_mean(sl_avg_steps, "SL avg steps")
    plot_and_mean(sl_reward_goal_dist_ratio, "SL return / goal dist")
    plot_and_mean(sl_losses, "SL losses")

    print("done")


if __name__ == '__main__':
    main()
