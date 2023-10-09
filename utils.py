import dataclasses
import hashlib
import json
import os
import pickle
from functools import wraps

import torch


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


def disk_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache directory if it doesn't exist
        cache_dir = f'cache/{func.__name__}'
        os.makedirs(cache_dir, exist_ok=True)

        # Serialize arguments and compute hash
        arg_str = json.dumps([args, kwargs], sort_keys=True, cls=EnhancedJSONEncoder)
        arg_hash = hashlib.sha256(arg_str.encode()).hexdigest()

        cache_file_path = os.path.join(cache_dir, f'{arg_hash}.json')

        if os.path.exists(cache_file_path):
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)

        result = func(*args, **kwargs)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(result, f)

        return result

    return wrapper
