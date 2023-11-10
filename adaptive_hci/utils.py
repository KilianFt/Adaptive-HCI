import hashlib
import os
import pickle
import pathlib
from functools import wraps

import numpy as np

from adaptive_hci.controllers import EMGViT, PLModel

# FIXME
is_slurm_job = os.environ.get("SLURM_JOB_ID") is not None
if is_slurm_job:
    base_data_dir = pathlib.Path('/home/mila/d/delvermm/scratch/adaptive_hci/datasets')
else:
    base_data_dir = pathlib.Path('./datasets')


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

def reload_general_model(config):
    genearal_model_ckpt = base_data_dir.parent / 'general_model.ckpt'

    model = EMGViT(
        image_size=config.window_size,
        patch_size=config.general_model_config.patch_size,
        num_classes=config.num_classes,
        dim=config.general_model_config.dim,
        depth=config.general_model_config.depth,
        heads=config.general_model_config.heads,
        mlp_dim=config.general_model_config.mlp_dim,
        dropout=config.general_model_config.dropout,
        emb_dropout=config.general_model_config.emb_dropout,
        channels=config.general_model_config.channels,
    )
    pl_model = PLModel.load_from_checkpoint(genearal_model_ckpt, model=model)

    return pl_model
