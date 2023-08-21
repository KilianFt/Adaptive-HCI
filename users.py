import torch
import pyautogui

class ProportionalUser:
    def __init__(self, goal, middle_pixels):
        self.goal = goal
        self.middle_pixels = torch.Tensor(middle_pixels)

    def get_signal(self):
        mouse_pos = torch.Tensor(pyautogui.position())
        signal = (mouse_pos - self.middle_pixels) / self.middle_pixels
        return signal
