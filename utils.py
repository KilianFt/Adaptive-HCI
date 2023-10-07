import torch


def get_device():
    if torch.cuda.is_available():
        device = 'gpu'
    elif torch.cuda.get_device_name(0) == 'Apple M1':
        device = 'mps'
    else:
        device = 'cpu'
    return device
