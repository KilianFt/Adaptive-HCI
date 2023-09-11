import abc
import time
import multiprocessing

import gymnasium as gym
import numpy as np
import pyautogui
import torch.nn
from screeninfo import get_monitors
from pyomyo import Myo, emg_mode

import datasets

class_name = [
    "rest",
    "index_flexion",
    "index_extension",
    "middle_flexion",
    "middle_extension",
]


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
        self.dataset = datasets.NinaPro1()
        ds = self.dataset
        self.observation_space_ = gym.spaces.Box(
            low=ds.lower_bound, high=ds.upper_bound, shape=(ds.features_size,), dtype=np.float32)

    @property
    def observation_space(self):
        return self.observation_space_

    def reset(self, observation, info):
        info["original_observation"] = observation
        user_action = self.user_policy(observation)
        user_features = self.action_to_features(user_action)
        return user_features, info

    def think(self) -> None:
        return

    def step(self, observation, reward, terminated, truncated, info):
        user_action = self.user_policy(observation)
        user_features = self.action_to_features(user_action)
        info["original_observation"] = observation
        info["optimal_action"] = user_action

        return user_features, reward, terminated, truncated, info

    def action_to_features(self, user_action):
        if user_action[0] > 0:
            class_dataset = self.dataset[1]
        elif user_action[0] < 0:
            class_dataset = self.dataset[2]
        elif user_action[1] > 0:
            class_dataset = self.dataset[3]
        elif user_action[1] < 0:
            class_dataset = self.dataset[4]
        else:
            raise ValueError("User action is zero!")

        sample_idx = np.random.randint(0, len(class_dataset))
        user_features, _target = class_dataset[sample_idx]
        return user_features



class EMGProportionalUser(BaseUser):
    def __init__(self):
        self.user_policy = ProportionalUserPolicy()

        self.emg_min = -128
        self.emg_max = 127
        self.emg_buffer = []

        # needs 200 sample for first window
        window_size = 200
        overlap = 150
        self.stride = window_size - overlap
        self.n_new_samples = -overlap

        self.q = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.worker, args=(self.q,))
        self.p.start()

        self.observation_space_ = gym.spaces.Box(
            low=-1., high=1., shape=(8,200), dtype=np.float32)

    @staticmethod
    def worker(q):
        m = Myo(mode=emg_mode.RAW)
        m.connect()
        
        def add_to_queue(emg, movement):
            q.put(emg)

        m.add_emg_handler(add_to_queue)
        
        def print_battery(bat):
            print("Battery level:", bat)

        m.add_battery_handler(print_battery)

        # Orange logo and bar LEDs
        m.set_leds([128, 0, 0], [128, 0, 0])
        # Vibrate to know we connected okay
        m.vibrate(1)
        
        """worker function"""
        while True:
            m.run()
        print("Worker Stopped")

    def read_emg_window(self):
        try:
            while self.n_new_samples < self.stride:
                while not(self.q.empty()):
                    emg = list(self.q.get())
                    norm_emg = np.interp(emg, (self.emg_min, self.emg_max), (-1, +1))
                    self.emg_buffer.append(norm_emg)
                    self.n_new_samples += 1

            current_window = np.array(self.emg_buffer[-200:], dtype=np.float32)
            self.n_new_samples = 0
            return current_window

        except KeyboardInterrupt:
            print("Quitting")
            quit()

    @property
    def observation_space(self):
        return self.observation_space_

    def reset(self, observation, info):
        info["original_observation"] = observation
        user_features = self.read_emg_window()
        return user_features, info

    def think(self) -> None:
        return

    def step(self, observation, reward, terminated, truncated, info):
        user_features = self.read_emg_window()
        user_action = self.user_policy(observation)
        info["original_observation"] = observation
        info["optimal_action"] = user_action

        return user_features, reward, terminated, truncated, info
    
    def __del__(self):
        self.p.terminate()
        self.p.join()
