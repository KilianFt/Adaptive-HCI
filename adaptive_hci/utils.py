import dataclasses
import hashlib
import json
import os
import pickle
from functools import wraps
from io import BytesIO

import numpy as np
import torch

from adaptive_hci.controllers import PLModel


def labels_to_onehot(label):
    onehot = np.zeros(5)
    onehot[int(label)] = 1
    return onehot


def predictions_to_onehot(predictions):
    predicted_labels = np.zeros_like(predictions)
    predicted_labels[predictions > 0.5] = 1
    return predicted_labels


def onehot_to_dof(onehot_vector):
    onehot_vector = np.array(onehot_vector, dtype=float)
    label_to_dof = np.array([
        [0, 0],
        [0, -1],  # left
        [-1, 0],  # back
        [0, 1],  # right
        [1, 0],  # front
    ])

    dof_cmd = np.dot(onehot_vector, label_to_dof)

    norm = np.linalg.norm(dof_cmd)
    if norm > 0:
        dof_cmd /= norm
    else:
        dof_cmd = np.zeros(2)

    return dof_cmd


def get_device():
    if not torch.cuda.is_available():
        device = 'cpu'
    elif torch.cuda.get_device_name(0) == 'Apple M1':
        device = 'mps'
    else:
        device = 'gpu'
    return device


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def custom_hash(obj):
    if isinstance(obj, PLModel):  # Should be pl.LightningModule but isinstance doesn't work
        buffer = BytesIO()
        torch.save(obj.state_dict(), buffer)
        obj_hash = hashlib.sha256(buffer.getvalue()).hexdigest()
    else:
        arg_str = pickle.dumps(obj)
        obj_hash = hashlib.sha256(arg_str).hexdigest()
    return obj_hash

def disk_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache directory if it doesn't exist
        cache_dir = f'cache/{func.__module__}/{func.__name__}'
        os.makedirs(cache_dir, exist_ok=True)

        arg_hashes = []
        for obj in [*args, *kwargs.values()]:
            arg_hashes.append(custom_hash(obj))

        arg_hash = custom_hash(arg_hashes)
        cache_file_path = os.path.join(cache_dir, f'{arg_hash}.json')

        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)

        result = func(*args, **kwargs)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(result, f)

        return result

    return wrapper
