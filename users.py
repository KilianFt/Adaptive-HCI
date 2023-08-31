import abc
import time

import gymnasium as gym
import numpy as np
import pyautogui
import torch.nn
from screeninfo import get_monitors


def get_screen_center():
    monitor = get_monitors()[0]  # Assuming the first monitor is the primary
    center_x = monitor.width // 2
    center_y = monitor.height // 2
    return center_x, center_y


class ProportionalUserPolicy(torch.nn.Module):
    @staticmethod
    def forward(observation):
        signal = observation["desired_goal"] - observation["achieved_goal"]
        signal = signal.astype(np.float32)
        # only move in one direction at a time
        if np.abs(signal[0]) > np.abs(signal[1]):
            signal[1] = 0
        else:
            signal[0] = 0
        return signal


class BaseUser:
    @abc.abstractmethod
    def reset(self, observation, info) -> (np.ndarray, dict):
        pass

    @abc.abstractmethod
    def step(self, observation, reward, terminated, truncated, info) -> (np.ndarray, float, bool, bool, dict):
        pass

    @staticmethod
    @abc.abstractmethod
    def think() -> None:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self):
        pass


class MouseProportionalUser(BaseUser):
    def __init__(self, simulate_user=False):
        monitor_center_x, monitor_center_y = get_screen_center()
        self.middle_pixels = np.array([monitor_center_x, monitor_center_y])
        # TODO: split the user in two classes based on this
        self.simulate_user = simulate_user
        self.user_policy = ProportionalUserPolicy()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

    def reset(self, observation, info):
        info["original_observation"] = observation
        return self.user_policy(observation), info

    def step(self, observation, reward, terminated, truncated, info):
        if self.simulate_user:
            user_action = self.user_policy(observation)
        else:
            mouse_pos = np.array(pyautogui.position())
            user_action = (mouse_pos - self.middle_pixels) / self.middle_pixels

        user_features = user_action

        info["original_observation"] = observation
        info["optimal_action"] = user_action

        return user_features, reward, terminated, truncated, info

    @staticmethod
    def think():
        time.sleep(0.1)


class FrankensteinProportionalUser(BaseUser):
    def __init__(self):
        self.user_policy = ProportionalUserPolicy()
        self.features_size = features_size = 3
        self.upper_bound = upper_bound = np.float32(1)
        self.lower_bound = lower_bound = np.float32(-1)
        dataset_size = 100
        left_bias, right_bias, up_bias, down_bias = np.zeros((4, features_size))

        self.features_dataset = {
            "left": np.random.uniform(lower_bound, upper_bound, size=(dataset_size, features_size)) + left_bias,
            "right": np.random.uniform(lower_bound, upper_bound, size=(dataset_size, features_size)) + right_bias,
            "up": np.random.uniform(lower_bound, upper_bound, size=(dataset_size, features_size)) + up_bias,
            "down": np.random.uniform(lower_bound, upper_bound, size=(dataset_size, features_size)) + down_bias,
        }

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.features_size,), dtype=np.float32)

    def reset(self, observation, info):
        info["original_observation"] = observation
        user_action = self.user_policy(observation)
        user_features = self.action_to_features(user_action)
        return user_features, info

    def step(self, observation, reward, terminated, truncated, info):
        user_action = self.user_policy(observation)
        user_features = self.action_to_features(user_action)

        info["original_observation"] = observation
        info["optimal_action"] = user_action

        return user_features, reward, terminated, truncated, info

    def action_to_features(self, user_action):
        sample_idx = np.random.randint(0, 100)
        if user_action[0] > 0:
            class_dataset = self.features_dataset["right"]
        elif user_action[0] < 0:
            class_dataset = self.features_dataset["left"]
        elif user_action[1] > 0:
            class_dataset = self.features_dataset["up"]
        elif user_action[1] < 0:
            class_dataset = self.features_dataset["down"]
        else:
            raise ValueError("User action is zero!")
        user_features = class_dataset[sample_idx]
        return user_features
