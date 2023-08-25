import argparse
import itertools
import os
import pickle

import gymnasium as gym
import numpy as np
import torch
import tqdm

from controllers import RLSLController
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
        observation, reward, terminated, truncated, info = environment.step(action_mean.squeeze().detach().numpy())

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
    performances = []
    goals = []

    if do_training:
        initial_parameters = controller.policy.state_dict()
        if not os.path.exists('models/2d_fetch'):
            os.makedirs('models/2d_fetch')
        torch.save(initial_parameters, 'models/2d_fetch/initial.pt')
    checkpoint_every = max(epochs // 10, 1)

    for epoch in tqdm.trange(epochs):
        states, user_signals, actions, optimal_actions, rewards, performance, goal = deterministic_rollout(
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
        performances.append(performance)
        sl_losses.append(loss)
        goals.append(goal)

    return sl_losses, sl_reward_history, sl_reward_sum_history, performances, goals


class TwoDProjection(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "observation": self.observation_space["observation"],
            "desired_goal": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
            "achieved_goal": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
        })
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

    @staticmethod
    def _project_observation(observation):
        observation["desired_goal"] = observation["desired_goal"][:2]
        observation["achieved_goal"] = observation["achieved_goal"][:2]

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._project_observation(observation)
        return observation, info

    def step(self, action):
        real_action = np.zeros(4)
        real_action[:2] = action

        observation, reward, terminated, truncated, info = self.env.step(real_action)
        self._project_observation(observation)
        return observation, reward, terminated, truncated, info


class EnvironmentWithUser(gym.Wrapper):
    """
    This wrapper adds a user to the environment.

    An action goes into the environment, the user observes the environment and returns some features, along with the
    classical transition.
    """

    def __init__(self, env: gym.Env, user: MouseProportionalUser):
        super().__init__(env)
        self.user = user
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

    def reset(self, **kwargs):
        env_obs, env_info = self.env.reset(**kwargs)
        user_observation, user_info = self.user.reset(env_obs, env_info)
        return user_observation, user_info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.user.step(observation, reward, terminated, truncated, info)

    def render(self):
        self.user.think()
        return self.env.render()


def main():
    parser = argparse.ArgumentParser(prog='Adaptive HCI - Fetch')

    parser.add_argument('--no-training', action='store_true')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()

    do_training = not args.no_training

    max_steps = 100
    total_timesteps = 10_000

    user = MouseProportionalUser(simulate_user=True)

    # environment = gym.make('FetchReachDense-v2', render_mode="human", max_episode_steps=100)
    environment = gym.make('FetchReachDense-v2', max_episode_steps=max_steps)
    environment = TwoDProjection(environment)
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

    sl_reward_goal_dist_ratio = [r / abs(g.sum()) if not abs(g.sum()) == 0 else 0 for r, g in
                                 zip(sl_reward_sum_history, goals)]

    plot_and_mean(sl_avg_steps, "SL avg steps")
    plot_and_mean(sl_reward_goal_dist_ratio, "SL return / goal dist")
    plot_and_mean(sl_losses, "SL losses")

    print("done")


if __name__ == '__main__':
    # with open('models/2d_fetch/results.pkl', 'rb') as f:
    #     results = pickle.load(f)

    # sl_losses = results['sl_losses']
    # sl_reward_history = results['sl_reward_history']
    # sl_reward_sum_history = results['sl_reward_sum_history']
    # sl_avg_steps = results['sl_avg_steps']
    # goals = results['goals']

    # sl_reward_goal_dist_ratio = [r / abs(g.sum()) if not abs(g.sum()) == 0 else 0 for r, g in
    #                              zip(sl_reward_sum_history, goals)]

    # plot_and_mean(sl_avg_steps, "SL avg steps")
    # plot_and_mean(sl_reward_goal_dist_ratio, "SL return / goal dist")
    # plot_and_mean(sl_losses, "SL losses")

    # print("done")
    main()
