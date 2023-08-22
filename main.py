import torch
import tqdm
import time
import pickle
import argparse
import numpy as np
import gymnasium as gym


from controllers import RLSLController
# from environment import Environment
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
        # make sure z never moves
        action_mean[0,2] = 0.
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
        states, actions, optimal_actions, rewards, performance, goal = deterministic_rollout(user, environment, controller, max_steps)
        
        if do_training:
            loss = controller.sl_update(states, optimal_actions)
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

    # FIXME make this variable by screen
    # max screen pos Point(x=1511, y=981)
    user = ProportionalUser(goal=1, middle_pixels=np.array([755, 470]))

    # environment = Environment(user=user, max_steps=MAX_STEPS)
    environment = gym.make('FetchReachDense-v2', render_mode="human", max_episode_steps=100)
    environment.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    controller = RLSLController(env=environment)

    # FIXME implement RL training
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
