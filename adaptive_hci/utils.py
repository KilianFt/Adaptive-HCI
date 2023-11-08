import hashlib
import logging
import os
import pickle
import subprocess
from functools import wraps

import numpy as np


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


def maybe_download_drive_folder(destination_folder, file_ids):
    destination_folder = destination_folder.as_posix() + '/'
    if os.path.exists(destination_folder):
        print("Folder already exists")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    logging.info("Downloading files from Google Drive")
    for file_id in file_ids:
        logging.info(f"Downloading file with id {file_id} to {destination_folder}")
        cmd = f"gdown https://drive.google.com/uc?id={file_id} -O {destination_folder}"
        subprocess.call(cmd, shell=True)
        print(subprocess.check_output(f"ls {destination_folder}", shell=True).decode())
