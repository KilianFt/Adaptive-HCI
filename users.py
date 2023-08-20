import torch
import pygame

class ProportionalUser:
    def __init__(self, goal, middle_pixel):
        self.goal = goal
        self.middle_pixel = middle_pixel

    def get_signal(self):
        signal = (pygame.mouse.get_pos()[0] - self.middle_pixel) / self.middle_pixel
        signal = max(min(signal, 1.), -1.)
        signal = torch.tensor([signal], dtype=torch.float32)
        return signal
