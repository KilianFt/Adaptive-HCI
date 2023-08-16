import gym
import numpy as np
import torch


class Environment(gym.Env):
    def __init__(self, user, goal, max_steps):
        self.goal = goal
        self.user = user
        self.position = None
        self.current_steps = None
        self.max_steps = max_steps
        self.reset()

    def step(self, action):
        self.position += action
        position = torch.tensor(self.position, dtype=torch.float32)
        distance_from_goal = torch.abs(position - self.goal)
        done = self.is_done()
        reward = -distance_from_goal
        reward = torch.clamp(reward, -1, 1)
        return self.get_state(), float(reward), done, {}

    def is_done(self):
        return np.abs(self.position - self.goal) < 1 or self.current_steps >= self.max_steps

    def get_state(self):
        return self.user.get_signal(self.position)

    def reset(self, **kwargs):
        super(Environment, self).reset(**kwargs)
        self.position = np.random.randint(-10, 10)
        self.current_steps = 0
        return self.get_state()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-10, high=10, shape=(1,))

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(1,))
