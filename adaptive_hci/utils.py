import hashlib
import os
import pickle
from functools import wraps

import numpy as np
import torch
from lightning.pytorch.callbacks import StochasticWeightAveraging


def get_accelerator(config_type='base'):
    return 'mps' # FIXME
    accelerator = 'cuda'
    if config_type == 'smoke':
        accelerator = 'cpu'
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('Cuda not found')
        accelerator = 'cuda'

    return accelerator


def get_trainer_callbacks(stage_config):
    if stage_config.do_swa:
        callbacks = [StochasticWeightAveraging(swa_lrs=stage_config.swa_lrs,
                            swa_epoch_start=stage_config.swa_epoch_start,
                            annealing_epochs=stage_config.annealing_epochs)]
    else:
        callbacks = None

    return callbacks


def labels_to_onehot(label):
    onehot = np.zeros(5)
    onehot[int(label)] = 1
    return onehot


def predictions_to_onehot(predictions):
    predicted_labels = np.zeros_like(predictions)
    predicted_labels[predictions > 0.5] = 1
    return predicted_labels


def disk_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache directory if it doesn't exist
        cache_dir = f'cache/{func.__name__}'
        os.makedirs(cache_dir, exist_ok=True)

        arg_str = pickle.dumps([args, kwargs])
        arg_hash = hashlib.sha256(arg_str).hexdigest()
        cache_file_path = os.path.join(cache_dir, f'{arg_hash}.json')

        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)

        result = func(*args, **kwargs)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(result, f)

        return result

    return wrapper
