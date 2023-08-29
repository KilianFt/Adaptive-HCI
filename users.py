import time

import numpy as np
import pyautogui
from screeninfo import get_monitors


def get_screen_center():
    monitor = get_monitors()[0]  # Assuming the first monitor is the primary
    center_x = monitor.width // 2
    center_y = monitor.height // 2
    return center_x, center_y


class MouseProportionalUser:
    def __init__(self, simulate_user=False):
        monitor_center_x, monitor_center_y = get_screen_center()
        self.middle_pixels = np.array([monitor_center_x, monitor_center_y])
        self.goal = None
        self.simulate_user = simulate_user

    def reset(self, observation, info):
        self.goal = observation["desired_goal"]
        info["original_observation"] = observation
        return self._obs_to_features(observation), info

    def step(self, observation, reward, terminated, truncated, info):
        current_position = observation["achieved_goal"]
        goal_state = observation["desired_goal"]

        signal = self._obs_to_features(observation)

        optimal_action = (goal_state - current_position).astype(np.float32)

        info["original_observation"] = observation
        info["optimal_action"] = optimal_action

        return signal, reward, terminated, truncated, info

    @staticmethod
    def think():
        time.sleep(0.1)

    def _obs_to_features(self, observation):
        if self.simulate_user:
            signal = self.goal - observation["achieved_goal"]
        else:
            mouse_pos = np.array(pyautogui.position())
            signal = (mouse_pos - self.middle_pixels) / self.middle_pixels
        return signal
