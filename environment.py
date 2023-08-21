import gym
import numpy as np
import torch


class Environment(gym.Env):
    def __init__(self, max_steps):
        self.goal = np.random.rand(2) * 10
        self.position = np.zeros(2)
        self.current_steps = None
        self.max_steps = max_steps
        self.reset()

    def step(self, action):
        self.position += action
        self.position = np.clip(self.position, np.array([-10, -10]), np.array([10, 10]))
        distance_from_goal = np.linalg.norm(self.position - self.goal)
        done = self.is_done()
        reward = -distance_from_goal
        reward = np.clip(reward, -1, 1)
        return self.get_state(), float(reward), done, {}

    def is_done(self):
        return np.linalg.norm(self.position - self.goal) < 1 or self.current_steps >= self.max_steps

    def get_state(self):
        return self.position

    def get_goal(self):
        return self.goal

    def reset(self, **kwargs):
        super(Environment, self).reset(**kwargs)
        self.goal = np.random.rand(2) * 10
        self.position = np.zeros(2)
        self.current_steps = 0
        return self.get_state()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,))

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,))
