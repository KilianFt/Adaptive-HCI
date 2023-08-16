import torch
import tqdm

from controllers import RLSLController
from environment import Environment
from metrics import plot_and_mean
from users import ProportionalUser


def deterministic_rollout(user, environment, controller):
    state = environment.reset()
    states = []
    action_means = []
    optimal_actions = []
    rewards = []

    for time_step in range(environment.max_steps):
        user_signal = user.get_signal(state)

        action_mean = controller.deterministic_forward(user_signal.unsqueeze(0))
        new_state, reward, done, info = environment.step(action_mean.detach().item())

        states.append(user_signal)
        action_means.append(action_mean)
        optimal_actions.append(torch.tensor([state - environment.goal], dtype=torch.float32))
        rewards.append(reward)

        state = new_state
        if done:
            break

    states = torch.stack(states)
    action_means = torch.stack(action_means)
    optimal_actions = torch.stack(optimal_actions)
    rewards = torch.tensor(rewards).unsqueeze(-1)
    return states, action_means, optimal_actions, rewards, time_step


def train_rl(controller: RLSLController, epochs):
    learner = controller.learn(epochs)
    learner.logger.dump()
    return learner


def train_sl(environment, controller, user, epochs):
    sl_reward_history = []
    sl_losses = []
    performances = []

    for _epoch in tqdm.trange(epochs):
        states, actions, optimal_actions, rewards, performance = deterministic_rollout(user, environment, controller)
        loss = controller.sl_update(states, optimal_actions)

        sl_reward_history.append(torch.mean(rewards))
        performances.append(performance)
        sl_losses.append(loss)
    return sl_losses, sl_reward_history, performances


def main():
    user = ProportionalUser(goal=1)
    environment = Environment(user=user, goal=1, max_steps=50)
    controller = RLSLController(env=environment)

    total_timesteps = 10_000
    initial_parameters = controller.policy.state_dict()
    train_rl(controller, total_timesteps)

    controller.policy.load_state_dict(initial_parameters)

    sl_losses, sl_reward_history, sl_avg_steps = train_sl(environment, controller, user, total_timesteps)

    plot_and_mean(sl_avg_steps, "SL avg steps")
    plot_and_mean(sl_reward_history, "SL rewards")
    plot_and_mean(sl_losses, "SL losses")

    print("done")


if __name__ == '__main__':
    main()
