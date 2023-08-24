import torch
import tqdm
import time
import pickle
import argparse
import numpy as np
import gymnasium as gym
from screeninfo import get_monitors

from controllers import RLSLController
from metrics import plot_and_mean
from users import ProportionalUser


def get_screen_center():
    monitor = get_monitors()[0]  # Assuming the first monitor is the primary
    center_x = monitor.width // 2
    center_y = monitor.height // 2
    return center_x, center_y


def deterministic_rollout(user, environment, controller, max_steps):
    state, _ = environment.reset()

    states = []
    user_signals = []
    action_means = []
    optimal_actions = []
    rewards = []

    for time_step in range(max_steps):
        user_signal = user.get_signal()

        action_mean = controller.deterministic_forward(user_signal.unsqueeze(0))

        # make sure z never moves
        action_mean[0,2] = 0.
        state, _, _, _, info = environment.step(action_mean.squeeze().detach().numpy())
        current_position = state['achieved_goal'][:2]

        # TODO make this part of the environment
        # compute 2D goal (ignore z axis)
        substitute_goal = state["desired_goal"].copy()
        substitute_goal[2] = state["achieved_goal"].copy()[2]
        reward = environment.compute_reward(state["achieved_goal"], substitute_goal, info)
        done = environment.compute_terminated(state["achieved_goal"], substitute_goal, info)
        truncated = environment.compute_truncated(state["achieved_goal"], substitute_goal, info)

        optimal_action = np.where((substitute_goal[:2] > current_position).astype(int), 1, -1)
        extended_optimal_actions = np.zeros(4)
        extended_optimal_actions[:2] = optimal_action

        states.append(state)
        user_signals.append(user_signal)
        action_means.append(action_mean)
        optimal_actions.append(torch.tensor([extended_optimal_actions], dtype=torch.float32))
        rewards.append(reward)
    
        if done:
            break

        time.sleep(.1)

    user_signals = torch.stack(user_signals)
    action_means = torch.stack(action_means)
    optimal_actions = torch.stack(optimal_actions).squeeze()
    rewards = torch.tensor(rewards).unsqueeze(-1)
    goal = substitute_goal[:2]

    return states, user_signals, action_means, optimal_actions, rewards, time_step, goal


def train_rl(controller: RLSLController, epochs):
    learner = controller.learn(epochs)
    learner.logger.dump()
    return learner


def train_sl(environment, controller, user, epochs, max_steps, do_training=True):
    sl_reward_history = []
    sl_reward_sum_history = []
    sl_losses = []
    performances = []
    goals = []

    if do_training:
        initial_parameters = controller.policy.state_dict()
        torch.save(initial_parameters, 'models/2d_fetch/initial.pt')

    for _epoch in tqdm.trange(epochs):
        states, user_signals, actions, optimal_actions, rewards, performance, goal = deterministic_rollout(user, environment, controller, max_steps)
        
        if do_training:
            loss = controller.sl_update(user_signals, optimal_actions)
            parameters = controller.policy.state_dict()
            torch.save(parameters, 'models/2d_fetch/epoch_'+str(_epoch)+'.pt')
        else:
            loss = None

        sl_reward_history.append(torch.mean(rewards).detach().item())
        sl_reward_sum_history.append(torch.sum(rewards).detach().item())
        performances.append(performance)
        sl_losses.append(loss)
        goals.append(goal)

    return sl_losses, sl_reward_history, sl_reward_sum_history, performances, goals

def main():
    parser = argparse.ArgumentParser(prog='Adaptive HCI - Fetch')
    
    parser.add_argument('--no-training', action='store_true')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()

    do_training = not args.no_training

    max_steps = 100
    total_timesteps = 10 #10_000

    monitor_center_x, monitor_center_y = get_screen_center()
    user = ProportionalUser(goal=1, middle_pixels=np.array([monitor_center_x, monitor_center_y]))

    environment = gym.make('FetchReachDense-v2', render_mode="human", max_episode_steps=100)
    environment.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    controller = RLSLController(env=environment)

    # TODO implement RL training
    # initial_parameters = controller.policy.state_dict()
    # train_rl(controller, total_timesteps)

    if args.model is not None:
        trained_parameters = torch.load(args.model)
        controller.policy.load_state_dict(trained_parameters)

    sl_losses, sl_reward_history, sl_reward_sum_history, sl_avg_steps, goals = train_sl(environment, controller, user, total_timesteps, max_steps, do_training=do_training)

    results = {
        'sl_losses': sl_losses,
        'sl_reward_history': sl_reward_history,
        'sl_reward_sum_history': sl_reward_sum_history,
        'sl_avg_steps': sl_avg_steps,
        'goals': goals,
    }

    with open('models/2d_fetch/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # sl_reward_goal_dist_ratio = [r / abs(g.sum()) if not abs(g.sum()) == 0 else 0 for r, g in zip(sl_reward_sum_history, goals)]

    # plot_and_mean(sl_avg_steps, "SL avg steps")
    # plot_and_mean(sl_reward_goal_dist_ratio, "SL return / goal dist")
    # plot_and_mean(sl_losses, "SL losses")

    print("done")
    environment.close()


if __name__ == '__main__':
    main()
