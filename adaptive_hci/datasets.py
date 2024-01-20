import dataclasses
import logging
import os
import pathlib
import pickle
import subprocess
import time
import random
from typing import Optional, Any
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils import data
from torch.utils.data import TensorDataset, Dataset

from common import DataSourceEnum
from .utils import labels_to_onehot, predictions_to_onehot

is_slurm_job = os.environ.get("SLURM_JOB_ID") is not None
if is_slurm_job:
    base_data_dir = pathlib.Path('/home/mila/d/delvermm/scratch/adaptive_hci/datasets/')
else:
    base_data_dir = pathlib.Path('datasets/')

gesture_names = [
    "rest",
    "index finger flexion",
    "index extension",
    "middle finger flexion",
    "middle finger extension",
    "ring finger flexion",
    "ring finger extension",
    "little finger flexion",
    "little finger extension",
    "thumb adduction",
    "thumb abduction",
    "thumb flexion",
    "thumb extension",
]


def get_episode_modes(episodes, n_samples_considered: Optional[int] = None):
    if n_samples_considered is not None:
        primary_actions = [extract_primary_action(ep.actions[:n_samples_considered]) for ep in episodes]
    else:
        primary_actions = [extract_primary_action(ep.actions) for ep in episodes]
    return primary_actions


def extract_primary_action(actions):
    unique_actions, counts = np.unique(actions, return_counts=True, axis=0)
    primary_action = unique_actions[np.argmax(counts)]
    return primary_action


def find_closest_episode(episodes, target_row):
    closest_row_index = np.argmin(calculate_action_distances(episodes, target_row))
    return episodes[closest_row_index]


def calculate_action_distances(episodes, target_row):
    primary_actions = get_episode_modes(episodes)
    distances = np.linalg.norm(primary_actions - target_row, axis=1)
    return distances


def get_adaptive_episode(episodes, label_accuracies):
    # Find the episode with the worst label accuracy and retrieve the closest episode based on actions
    worst_label_index = np.argmin(label_accuracies)
    worst_label = np.zeros_like(label_accuracies)
    worst_label[worst_label_index] = 1.0
    closest_episode = find_closest_episode(episodes, worst_label)
    return closest_episode


def to_tensor_dataset(train_observations, train_actions):
    ds = TensorDataset(
        torch.tensor(train_observations, dtype=torch.float32),
        torch.tensor(train_actions, dtype=torch.float32)
    )
    return ds


def get_terminals(episodes, rewards):
    terminals = np.zeros(rewards.shape[0])
    last_terminal_idx = 0
    for e in episodes:
        term_idx = e['rewards'].shape[0] - 1 + last_terminal_idx
        terminals[term_idx] = 1
        last_terminal_idx = term_idx
    return terminals


def get_concatenated_user_episodes(episodes):
    assert len(episodes) > 0, 'Episodes empty'

    # seem to be a problem in sweeps to create list and concatenate them in one line
    # thus we wirst need to create a list and then concatenate (it always fails on the first occasion of np.concat([...]))
    # not on np.concat(my_list)
    action_list = [predictions_to_onehot(e['actions'].detach().numpy()) for e in episodes]
    optimal_actions_list = [e['optimal_actions'].detach().numpy() for e in episodes]
    observations_list = [e['user_signals'] for e in episodes]
    rewards_list = [e['rewards'] for e in episodes]

    assert len(action_list) > 0, 'Action list empty: {action_list}'
    assert len(optimal_actions_list) > 0, 'Optimal action list empty: {optimal_actions_list}'
    assert len(observations_list) > 0, 'Observation list empty: {observations_list}'
    assert len(rewards_list) > 0, 'Reward list empty: {rewards_list}'

    actions = np.concatenate(action_list).squeeze()
    optimal_actions = np.concatenate(optimal_actions_list)
    observations = np.concatenate(observations_list).squeeze()
    rewards = np.concatenate(rewards_list).squeeze()

    terminals = get_terminals(episodes, rewards)

    return observations, actions, optimal_actions, rewards, terminals


@dataclasses.dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray


def split_by_terminal(observations, actions, rewards, terminals):
    episodes = []
    start_idx = 0

    for i, terminal in enumerate(terminals):
        if terminal:
            episodes.append(Episode(
                observations=observations[start_idx:i + 1],
                actions=actions[start_idx:i + 1],
                rewards=rewards[start_idx:i + 1],
                terminals=terminals[start_idx:i + 1]
            ))
            start_idx = i + 1
    return episodes


def get_rl_dataset(current_trial_episodes, online_num_episodes = None, shuffle=False):
    (observations, _, optimal_actions, rewards, terminals) = get_concatenated_user_episodes(current_trial_episodes)

    all_episodes = split_by_terminal(observations, optimal_actions, rewards, terminals)
    if online_num_episodes is not None:
        all_episodes = all_episodes[:online_num_episodes]

    if shuffle:
        print('Shuffling')
        random.shuffle(all_episodes)

    num_classes = optimal_actions.shape[1]
    return all_episodes, num_classes


def get_raw_mad_dataset(eval_path, window_length, overlap):
    person_folders = os.listdir(eval_path)

    first_folder = os.listdir(eval_path)[0]
    keys = next(os.walk((eval_path + first_folder)))[1]

    number_of_classes = 7
    size_non_overlap = window_length - overlap

    raw_dataset_dict = {}
    for key in keys:

        raw_dataset = {
            'examples': [],
            'labels': [],
        }

        for person_dir in person_folders:
            examples = []
            labels = []
            data_path = eval_path + person_dir + '/' + key
            for data_file in os.listdir(data_path):
                if data_file.endswith(".dat"):
                    data_read_from_file = np.fromfile((data_path + '/' + data_file), dtype=np.int16)
                    data_read_from_file = np.array(data_read_from_file, dtype=np.float32)

                    dataset_example_formatted = []
                    example = None
                    emg_vector = []
                    for value in data_read_from_file:
                        emg_vector.append(value)
                        if (len(emg_vector) >= 8):
                            if example is None:
                                example = emg_vector
                            else:
                                example = np.row_stack((example, emg_vector))
                            emg_vector = []
                            if (len(example) >= window_length):
                                example = example.transpose()
                                dataset_example_formatted.append(example)
                                example = example.transpose()
                                example = example[size_non_overlap:]
                    dataset_example_formatted = np.array(dataset_example_formatted)
                    examples.append(dataset_example_formatted)
                    data_file_index = int(data_file.split('classe_')[1][:-4])
                    label = data_file_index % number_of_classes + np.zeros(dataset_example_formatted.shape[0])
                    labels.append(label)

            raw_dataset['examples'].append(np.concatenate(examples))
            raw_dataset['labels'].append(np.concatenate(labels))

        raw_dataset_dict[key] = raw_dataset

    return raw_dataset_dict


def maybe_download_mad_dataset(mad_base_dir):
    if os.path.exists(mad_base_dir):
        return
    print("MyoArmbandDataset not found")

    if os.path.exists(mad_base_dir + '/.lock'):
        print("Waiting for download to finish")
        # wait for download to finish
        while os.path.exists(mad_base_dir + '/.lock'):
            print(".", end="")
            time.sleep(1)
        return

    os.makedirs(mad_base_dir, exist_ok=True)

    # create a lock file to prevent multiple downloads
    os.system(f'touch {mad_base_dir}/.lock')

    print("Downloading MyoArmbandDataset")
    os.system(f'git clone https://github.com/UlysseCoteAllard/MyoArmbandDataset {mad_base_dir}')
    print("Download finished")

    # remove the lock file
    os.system(f'rm {mad_base_dir}/.lock')


def get_mad_windows_dataset(mad_base_dir, _, window_length, overlap):
    maybe_download_mad_dataset(mad_base_dir)

    train_path = mad_base_dir + 'PreTrainingDataset/'
    eval_path = mad_base_dir + 'EvaluationDataset/'

    eval_raw_dataset_dict = get_raw_mad_dataset(eval_path, window_length, overlap)
    train_raw_dataset_dict = get_raw_mad_dataset(train_path, window_length, overlap)

    mad_all_windows = (
            eval_raw_dataset_dict['training0']['examples'] +
            eval_raw_dataset_dict['Test0']['examples'] +
            eval_raw_dataset_dict['Test1']['examples'] +
            train_raw_dataset_dict['training0']['examples']
    )

    mad_all_labels = (
            eval_raw_dataset_dict['training0']['labels'] +
            eval_raw_dataset_dict['Test0']['labels'] +
            eval_raw_dataset_dict['Test1']['labels'] +
            train_raw_dataset_dict['training0']['labels']
    )

    # filter by labels
    mad_windows = None
    mad_labels = None
    for mad_p_examples, mad_p_labels in zip(mad_all_windows, mad_all_labels):
        label_map = (mad_p_labels >= 0) & (mad_p_labels <= 4)
        mad_subject_windows = mad_p_examples[label_map]
        subject_labels = mad_p_labels[label_map]

        if mad_windows is None:
            mad_windows = mad_subject_windows
            mad_labels = subject_labels
        else:
            mad_windows = np.concatenate((mad_windows, mad_subject_windows))
            mad_labels = np.concatenate((mad_labels, subject_labels))

    mad_onehot_labels = np.array([labels_to_onehot(label) for label in mad_labels])
    print("MAD dataset loaded")
    return mad_windows, mad_onehot_labels


def create_ninapro_windows(X, y, stride, window_length, desired_labels=None):
    features_dataset = {key: [] for key in np.unique(y)}
    last_class_idx = None
    consequetive_features = []

    for class_idx, feature in zip(y, X):
        if class_idx != last_class_idx:
            if consequetive_features:
                features_dataset[class_idx].append(np.array(consequetive_features))
            consequetive_features = [feature]
            last_class_idx = class_idx
        else:
            consequetive_features.append(feature)

    if consequetive_features:
        features_dataset[class_idx].append(np.array(consequetive_features))

    windows = []
    labels = []
    for class_idx, feature_list in features_dataset.items():
        if desired_labels is None or class_idx in desired_labels:
            for consequetive_features in feature_list:
                num_windows = (consequetive_features.shape[0] - window_length) // stride + 1
                for i in range(num_windows):
                    start = i * stride
                    end = start + window_length
                    window = consequetive_features[start:end, :]
                    windows.append(window)
                    labels.append(class_idx)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=int)


def get_ninapro_windows_dataset(ninapro_base_dir, emg_range, window_length, overlap, get_raw_labels=False):
    ninapro_windows = None
    ninapro_labels = None

    stride = window_length - overlap

    ninapro_person_dirs = next(os.walk(ninapro_base_dir))[1]
    for nina_person_dir in ninapro_person_dirs:
        files = os.listdir(ninapro_base_dir + nina_person_dir)
        for file in files:
            if file.endswith('E2_A1.mat'):
                filepath = ninapro_base_dir + nina_person_dir + '/' + file

                ninapro_s1 = loadmat(filepath)

                ninapro_s_x_raw = ninapro_s1['emg'][:, :8]
                ninapro_s_x = np.interp(ninapro_s_x_raw, emg_range, (-1, +1))
                ninapro_s_y = ninapro_s1['restimulus'].squeeze()

                desired_labels = None if get_raw_labels else [0, 13, 14, 15, 16]

                subject_windows, subject_labels = create_ninapro_windows(X=ninapro_s_x,
                                                                         y=ninapro_s_y,
                                                                         stride=stride,
                                                                         window_length=window_length,
                                                                         desired_labels=desired_labels, )

                if ninapro_windows is None:
                    ninapro_windows = subject_windows
                    ninapro_labels = subject_labels
                else:
                    ninapro_windows = np.concatenate((ninapro_windows, subject_windows))
                    ninapro_labels = np.concatenate((ninapro_labels, subject_labels))

    ninapro_windows = ninapro_windows.swapaxes(1, 2)

    if get_raw_labels:
        return ninapro_windows, ninapro_labels

    # replace labels
    label_map = {0: 0,
                 13: 2,
                 14: 4,
                 15: 1,
                 16: 3,
                 }

    ninapro_mapped_labels = np.vectorize(label_map.get)(ninapro_labels)
    ninapro_onehot_labels = np.array([labels_to_onehot(label) for label in ninapro_mapped_labels])

    return ninapro_windows, ninapro_onehot_labels


class EMGWindowsDataset(data.Dataset):
    BASE_DIR = pathlib.Path(__file__).parents[1].resolve().as_posix()
    DATASET_DIRS = {
        DataSourceEnum.NINA_PRO: (BASE_DIR+'/datasets/ninapro/DB5', get_ninapro_windows_dataset),
        DataSourceEnum.MAD: (BASE_DIR+'/datasets/MyoArmbandDataset/', get_mad_windows_dataset),
        DataSourceEnum.MiniMAD: (BASE_DIR+'/datasets/MyoArmbandDataset/', get_mad_windows_dataset),
    }
    # Mila server, it's a hack.
    if os.path.exists("/home/mila/d/delvermm/scratch/"):
        for key, value in DATASET_DIRS.items():
            DATASET_DIRS[key] = (os.path.join("/home/mila/d/delvermm/scratch/", value[0]), value[1])

    def __init__(
            self,
            data_source: DataSourceEnum,
            split: str,
            window_size=200,
            overlap=0,
            emg_range=(-128, 127),
    ):
        base_dir, load_dataset = self.DATASET_DIRS[data_source]
        if data_source == DataSourceEnum.NINA_PRO:
            base_dir += "_" + split

        self.windows, self.labels = load_dataset(base_dir, emg_range, window_size, overlap)
        if data_source in (DataSourceEnum.MiniMAD,):
            self.windows = self.windows[:10]
            self.labels = self.labels[:10]

        self.windows = torch.tensor(self.windows, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x_tensor = self.windows[idx, :, :]
        y_tensor = self.labels[idx]
        return x_tensor, y_tensor

    @property
    def num_unique_labels(self):
        return self.labels.shape[1]


class NinaPro1(data.Dataset[data.TensorDataset]):
    def __init__(self):
        try:
            mat = loadmat('ninapro/DB1_s1/S1_A1_E1.mat')
        except FileNotFoundError:
            raise FileNotFoundError("Please download the NinaPro dataset from https://zenodo.org/record/1000116 and "
                                    "extract it into the 'ninapro' folder")

        x = mat['emg']
        y = mat['restimulus'].squeeze(1)

        self.features_size = x.shape[1]
        self.upper_bound = np.max(x, axis=0)
        self.lower_bound = np.min(x, axis=0)
        assert mat["exercise"] == 1
        assert mat["subject"] == 1

        self.class_dataset = {}
        for class_idx in np.unique(y):
            xs = torch.tensor(x[y == class_idx])
            ys = torch.full((len(xs),), class_idx, dtype=torch.long)
            self.class_dataset[class_idx] = data.TensorDataset(xs, ys)

    def __getitem__(self, index) -> data.TensorDataset:
        return self.class_dataset[index]


class CombinedDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Calculate the total length of the combined dataset
        self.total_length = len(self.dataset1) + len(self.dataset2)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            x_tensor = self.dataset1.windows[idx, :, :]
            y_tensor = self.dataset1.labels[idx]
            return x_tensor, y_tensor
        else:
            # Adjust the idx for the second dataset
            idx -= len(self.dataset1)
            x_tensor = self.dataset2.windows[idx, :, :]
            y_tensor = self.dataset2.labels[idx]
            return x_tensor, y_tensor

    @property
    def num_unique_labels(self):
        assert self.dataset1.labels.shape[1] == self.dataset2.labels.shape[1], 'labels of both datasets must match'
        return self.dataset1.labels.shape[1]


def load_files(data_dir, filenames):
    online_episodes_list = []

    for filename in filenames:
        filepath = data_dir / filename
        with open(filepath, 'rb') as f:
            episodes = pickle.load(f)
            online_episodes_list.append(episodes)

    return online_episodes_list


def get_stored_sessions(stage: str, file_ids, num_episodes = None):
    stage = pathlib.Path("Online" + stage)
    data_dir = base_data_dir / stage

    filenames = maybe_download_drive_folder(data_dir, file_ids=file_ids)

    if num_episodes is not None:
        filenames = filenames[:num_episodes]

    online_episodes_list = load_files(data_dir, filenames)

    # check if episodes contain information
    if not online_episodes_list or len(online_episodes_list[0][0]['user_signals']) == 0:
        print('Retrying to load files')
        time.sleep(torch.randint(1, 10, size=(1,)))

        online_episodes_list = load_files(data_dir, filenames)

        assert len(online_episodes_list[0][0]['user_signals']) > 0, \
            f"Could not load episode files in {data_dir}\n filenames {filenames}"

    return online_episodes_list


def maybe_download_drive_folder(destination_folder, file_ids):
    _destination_folder = destination_folder.as_posix() + '/'

    if not os.path.exists(_destination_folder):
        os.makedirs(_destination_folder)

    all_files = os.listdir(_destination_folder)
    filenames = [file for file in all_files if file.endswith(".pkl")]

    if os.path.exists(_destination_folder) and len(filenames) == len(file_ids):
        print("Folder already exists")
        return sorted(filenames)

    logging.info("Downloading files from Google Drive")
    for file_id in file_ids:
        cmd = f"gdown https://drive.google.com/uc?id={file_id} -O {_destination_folder}"
        subprocess.call(cmd, shell=True)

    all_files = os.listdir(_destination_folder)
    filenames = [file for file in all_files if file.endswith(".pkl")]

    assert len(filenames) == len(file_ids), 'Not all files exist {filenames}'

    return sorted(filenames)



# TODO move to own file
# Omniglot
def load_img(fn):
	I = plt.imread(fn)
	I = np.array(I,dtype=bool)
	return I


def load_motor(fn):
	motor = []
	with open(fn,'r') as fid:
		lines = fid.readlines()
	lines = [l.strip() for l in lines]
	for myline in lines:
		if myline =='START': # beginning of character
			stk = []
		elif myline =='BREAK': # break between strokes
			stk = np.array(stk)
			stk[:,1] = -stk[:,1]
			motor.append(stk) # add to list of strokes
			stk = []
		else:
			arr = np.fromstring(myline,dtype=float,sep=',')
			stk.append(arr)
	return motor

def num2str(idx):
	if idx < 10:
		return '0'+str(idx)
	return str(idx)


def get_raw_omniglot_dataset(stroke_dir, img_dir, char_idxs=None):
    # TODO remove img?
    n_characters = len([x for x in img_dir.glob('*')])

    dataset = []
    if char_idxs is None:
        char_iter = range(1, n_characters + 1)
    else:
        char_iter = char_idxs
    for char_idx in char_iter:
        character_id = num2str(char_idx)
        stroke_char_dir = stroke_dir / ('character' + character_id)
        img_char_dir = img_dir / ('character' + character_id)

        stroke_files = [x for x in stroke_char_dir.glob('*')]
        char_id = stroke_files[0].name[:4]
        n_samples = len(stroke_files)

        character_data = defaultdict(list)
        for i in range(1, n_samples+1):
            character_stroke_file = stroke_char_dir / (char_id + '_' + num2str(i) + '.txt')
            character_img_file = img_char_dir / (char_id + '_' + num2str(i) + '.png')
            motor = load_motor(character_stroke_file)
            img = load_img(character_img_file)

            character_data['img'].append(img)
            character_data['motor'].append(motor)

        dataset.append(character_data)
    return dataset


def sequence_to_moves(sequence, canvas_size=30, max_initial_value=120):
    sequence = (sequence / max_initial_value) * canvas_size
    sequence = sequence.astype(np.int32)

    moves = []

    for i in range(len(sequence) - 1):
        point = sequence[i]
        next_point = sequence[i+1]
        # switch x and y
        x_diff = next_point[1] - point[1]
        y_diff = next_point[0] - point[0]
        while abs(x_diff) > 0 or abs(y_diff) > 0:
            if abs(y_diff) > abs(x_diff) or abs(y_diff) == abs(x_diff):
                y_step = -1 if y_diff < 0 else 1
                moves.append([0, y_step])
                y_diff -= y_step
            elif abs(y_diff) < abs(x_diff):
                x_step = -1 if x_diff < 0 else 1
                moves.append([x_step, 0])
                x_diff -= x_step
            else:
                print('what', x_diff, y_diff)

    moves = np.array(moves)
    return moves


def omniglot_dataset_to_moves(dataset, canvas_size=30, max_initial_value=120):
    moves_data = []
    for char_samples in dataset:
        for char_sample in char_samples['motor']:
            for stroke_sample in char_sample:
                sequence = stroke_sample[:,:2]
                moves = sequence_to_moves(sequence, canvas_size=canvas_size,
                                          max_initial_value=max_initial_value)
                moves_data.append(moves)
    return moves_data


def encode_moves(moves_data, move_map):
    encoded_moves_data = []
    for moves in moves_data:
        # filter strokes with less than 5 points
        if moves.shape[0] > 5:
            encoded_moves = torch.zeros(moves.shape[0], dtype=torch.float32)
            for i, point in enumerate(moves):
                label = move_map.index(point.tolist())
                encoded_moves[i] = label

            encoded_moves_data.append(encoded_moves)
    return encoded_moves_data    


def get_omniglot_moves(omniglot_dir: Path, canvas_size: int = 30, max_initial_value: int = 120, char_idxs=None):
    img_dir = omniglot_dir / 'python' / 'images_background' / 'Latin'
    stroke_dir = omniglot_dir / 'python' / 'strokes_background' / 'Latin'
    raw_dataset = get_raw_omniglot_dataset(stroke_dir, img_dir, char_idxs=char_idxs)
    moves_data = omniglot_dataset_to_moves(raw_dataset, canvas_size=canvas_size,
                                           max_initial_value=max_initial_value)

    move_map = [[-1, 0],
                [0, -1],
                [1, 0],
                [0, 1],
                [0, 0]]
    encoded_moves_data = encode_moves(moves_data, move_map)
    return encoded_moves_data


def pad_data(data, pad_token, context_len):
    token_len_plus_label = context_len + 1
    padded_data = []
    for sample in data:
        sample_len = len(sample)
        if sample_len < token_len_plus_label:
            pad_len = token_len_plus_label - sample_len
            padding = torch.zeros(pad_len) + pad_token
            padded_sample = torch.cat((sample, padding))

        else:
            padded_sample = sample[:token_len_plus_label]
        padded_data.append(padded_sample)

    return torch.stack(padded_data).type(torch.long)
    

class OmniglotGridDataset(Dataset):
    def __init__(self, omniglot_dir, context_len=200, pad_token=5, canvas_size=50,
                 max_initial_value=120, eos_token=4, char_idxs=[12, 15]):
        omniglot_dir = Path(omniglot_dir)
        omniglot_data = get_omniglot_moves(
            omniglot_dir,
            canvas_size=canvas_size,
            max_initial_value=max_initial_value,
            char_idxs=char_idxs,
        )

        data_w_stop = [
            torch.cat((x, torch.tensor([eos_token]))).type(torch.long) for x in omniglot_data
        ]
        data_w_pad = pad_data(data_w_stop, pad_token, context_len)

        self.data = data_w_pad
        self.pad_token = pad_token
        self.context_len = context_len
        self.vocab_size = len(data_w_pad.unique())

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.context_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        sample = self.data[index]
        x = sample[:-1].clone()
        y = sample[1:].clone()
        # mask where y is pad
        y[y==self.pad_token] = -1
        return x, y
