import numpy as np
import pyautogui
from screeninfo import get_monitors


def get_screen_center():
    monitor = get_monitors()[0]  # Assuming the first monitor is the primary
    center_x = monitor.width // 2
    center_y = monitor.height // 2
    return center_x, center_y


class MouseProportionalUser:
    def __init__(self):
        monitor_center_x, monitor_center_y = get_screen_center()
        self.middle_pixels = np.array([monitor_center_x, monitor_center_y])
        self.goal = None

    def reset(self, observation, info):
        self.goal = observation["desired_goal"]
        info["original_observation"] = observation
        return self._obs_to_features(observation), info

    def step(self, observation, reward, terminated, truncated, info):
        current_position = observation["achieved_goal"]
        substitute_goal = observation["desired_goal"]

        signal = self._obs_to_features(observation)

        optimal_action = np.where((substitute_goal[:2] > current_position).astype(int), 1, -1).astype(np.float32)

        info["original_observation"] = observation
        info["optimal_action"] = optimal_action

        return signal, reward, terminated, truncated, info

    def _obs_to_features(self, _observation):
        mouse_pos = np.array(pyautogui.position())
        signal = (mouse_pos - self.middle_pixels) / self.middle_pixels
        return signal


class ProportionalUser:
    def __init__(self, goal, middle_pixels):
        self.goal = goal
        self.middle_pixels = np.array(middle_pixels)

    def get_signal(self):
        mouse_pos = np.array(pyautogui.position())
        signal = (mouse_pos - self.middle_pixels) / self.middle_pixels
        return signal
