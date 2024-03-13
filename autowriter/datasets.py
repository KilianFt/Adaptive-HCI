import os
import requests
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


def maybe_download(file_name: str, path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        strokes_url = f"https://raw.githubusercontent.com/brendenlake/omniglot/master/python/{file_name}_background.zip"
        r = requests.get(strokes_url)
        tmp_file = os.path.join(path, f"{file_name}.zip")
        with open(tmp_file, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
            zip_ref.extractall(path)


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


def switch_directions(traces):
    # replace 0 with 2 and 2 with 0 and 1 with 3 and 3 with 1
    switched_traces = []

    # rotate 180 degrees
    for trace in traces:
        switched_trace = trace.clone()
        switched_trace[trace == 0] = 2
        switched_trace[trace == 2] = 0
        switched_trace[trace == 1] = 3
        switched_trace[trace == 3] = 1
        switched_traces.append(switched_trace)
    # rotate 90 degrees
    for trace in traces:
        switched_trace = trace.clone()
        switched_trace[trace == 0] = 3
        switched_trace[trace == 2] = 1
        switched_trace[trace == 1] = 2
        switched_trace[trace == 3] = 0
        switched_traces.append(switched_trace)
    # rotate 270 degrees
    for trace in traces:
        switched_trace = trace.clone()
        switched_trace[trace == 0] = 1
        switched_trace[trace == 2] = 3
        switched_trace[trace == 1] = 2
        switched_trace[trace == 3] = 0
        switched_traces.append(switched_trace)
    
    return switched_traces


def get_omniglot_moves(omniglot_dir: Path, canvas_sizes: List[int] = [20, 30, 40],
                       max_initial_value: int = 120, char_idxs=None):
    img_dir = omniglot_dir / 'images' / 'images_background' / 'Latin'
    stroke_dir = omniglot_dir / 'traces' / 'strokes_background' / 'Latin'
    raw_dataset = get_raw_omniglot_dataset(stroke_dir, img_dir, char_idxs=char_idxs)

    moves_data = []
    for canvas_size in canvas_sizes:
        moves_data += omniglot_dataset_to_moves(raw_dataset, canvas_size=canvas_size,
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
    def __init__(self, omniglot_dir, context_len=200, pad_token=5, canvas_sizes=[40, 50, 60],
                 max_initial_value=120, eos_token=4, char_idxs=[12, 15], include_switched=True):
        omniglot_dir = Path(omniglot_dir)
        # TODO add automatic download
        img_path = omniglot_dir / 'images'
        traces_path = omniglot_dir / 'traces'
        maybe_download("images", img_path.as_posix())
        maybe_download("strokes", traces_path.as_posix())

        normal_omniglot_data = get_omniglot_moves(
            omniglot_dir,
            canvas_sizes=canvas_sizes,
            max_initial_value=max_initial_value,
            char_idxs=char_idxs,
        )

        if include_switched:
            switched_omniglot_data = switch_directions(normal_omniglot_data)
            omniglot_data = normal_omniglot_data + switched_omniglot_data
        else:
            omniglot_data = normal_omniglot_data

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

    def __getitem__(self, index):
        sample = self.data[index]
        x = sample[:-1].clone()
        y = sample[1:].clone()
        # mask where y is pad
        y[y==self.pad_token] = -1
        return x, y