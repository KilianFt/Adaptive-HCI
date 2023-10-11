import argparse
import itertools
import os
import pickle
import datetime

import gymnasium as gym
import numpy as np
import torch
import tqdm

from adaptive_hci.controllers import RLSLController, SLOnlyController
from adaptive_hci.environment import XDProjection, EnvironmentWithUser
from adaptive_hci.utils import onehot_to_dof
from adaptive_hci.metrics import plot_and_mean
from adaptive_hci import users


def deterministic_rollout(environment, controller):
    observation, info = environment.reset()
    observation = torch.tensor(observation)

    rollout_goal = info["original_observation"]["desired_goal"]

    states = []
    user_signals = []
    action_means = []
    optimal_actions = []
    rewards = []

    for time_step in itertools.count():
        action_mean = controller.deterministic_forward(observation)
        predictions = action_mean.cpu().detach().squeeze().numpy()
        predicted_labels = np.zeros_like(predictions)
        predicted_labels[predictions > 0.7] = 1
        action = onehot_to_dof(predicted_labels)
        action *= 0.5
        observation, reward, terminated, truncated, info = environment.step(action)

        observation = torch.tensor(observation)

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


def train_sl(environment,
             controller,
             epochs,
             n_episodes_per_train=1,
             do_training=True):
    sl_reward_history = []
    sl_reward_sum_history = []
    sl_losses = []
    episode_durations = []
    goals = []

    episodes = []

    if do_training:
        initial_parameters = controller.policy.state_dict()
        if not os.path.exists('../models/2d_fetch'):
            os.makedirs('../models/2d_fetch')
        modelname = 'models/2d_fetch/initial.pt'
        torch.save(initial_parameters, modelname)
    checkpoint_every = max(epochs // 10, 1)

    for epoch in tqdm.trange(epochs):
        states, user_signals, actions, optimal_actions, rewards, episode_duration, goal = deterministic_rollout(
            environment, controller)

        if epoch > 0 and epoch % n_episodes_per_train == 0 and do_training:
            loss = controller.sl_update(user_signals, optimal_actions)
            if epoch % checkpoint_every == 0:
                parameters = controller.policy.state_dict()
                modelname = f'models/2d_fetch/epoch_{str(epoch)}.pt'
                torch.save(parameters, modelname)
        else:
            loss = None

        sl_reward_history.append(torch.mean(rewards).detach().item())
        sl_reward_sum_history.append(torch.sum(rewards).detach().item())
        episode_durations.append(episode_duration)
        sl_losses.append(loss)
        goals.append(goal)

        episode_results = {
            'states': states,
            'user_signals': user_signals,
            'actions': actions,
            'optimal_actions': optimal_actions,
            'rewards': rewards,
            'goal': goal,
            'episode_durations': episode_duration,
            'model_name': controller.model_name,
        }
        episodes.append(episode_results)

    return episodes, sl_losses


def main():
    parser = argparse.ArgumentParser(prog='Adaptive HCI - Fetch')

    parser.add_argument('--no-training', action='store_true')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()

    do_training = not args.no_training

    max_steps = 150
    total_timesteps = 10
    n_dof = 2
    lr = 1e-4
    batch_size = 32
    epochs = 1
    n_frozen_layers = 2
    device = 'cpu'

    controller = SLOnlyController(args.model,
                                  device=device,
                                  lr=lr,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  n_frozen_layers=n_frozen_layers)

    # user = users.FrankensteinProportionalUser()
    user = users.EMGClassificationUser()

    environment = gym.make('FetchReachDense-v2', max_episode_steps=max_steps, render_mode="human")
    environment = XDProjection(environment, n_dof=n_dof)
    environment = EnvironmentWithUser(environment, user)

    # controller = RLSLController(env=environment)

    episodes, sl_losses = train_sl(environment,
                                   controller,
                                   total_timesteps,
                                   do_training=do_training)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_filename = f"datasets/OnlineData/episodes_{timestamp}.pkl"

    with open(results_filename, 'wb') as f:
        pickle.dump(episodes, f)

    resulting_model_filename = f"models/2d_fetch/model_{timestamp}.pkl"
    torch.save(controller.policy.cpu(), resulting_model_filename)
    print('Saved final model at', resulting_model_filename)

    # visualize results
    sl_reward_sum_history = [np.sum(e['rewards'].numpy()) for e in episodes]
    goals = [e['goal'] for e in episodes]

    sl_reward_goal_dist_ratio = []
    for r, g in zip(sl_reward_sum_history, goals):
        gsum = abs(g.sum())
        if gsum == 0:
            rgs = 0
        else:
            rgs = r / gsum
        sl_reward_goal_dist_ratio.append(rgs)

    sl_avg_steps = [e['episode_durations'] for e in episodes]
    plot_and_mean(sl_avg_steps, "SL avg steps")
    plot_and_mean(sl_reward_goal_dist_ratio, "SL return / goal dist")
    plot_and_mean(sl_losses, "SL losses")

    print("done")


if __name__ == '__main__':
    main()
