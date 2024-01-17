import dataclasses
import math
import os
import zipfile
from typing import List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import transformers
from torch.utils.data import Dataset

import constants
from configs import DataSpec

motor_context = Union[np.ndarray, torch.Tensor]


def get_dataloader(data_spec):
    cache_omniglot_dataset(data_spec)
    train_dataset = OmniglotDataset(data_spec, repetitions=data_spec.train_reps)
    return train_dataset


@dataclasses.dataclass
class DataSample:
    token_context: transformers.BatchEncoding
    labels: Optional[torch.Tensor] = None
    image_context: Optional[torch.Tensor] = None
    motor_context: Optional[torch.Tensor] = None  # batch_size x max_char_per_token x points_in_motor_seq x motor_dim

    # next_token_logits: Optional[torch.Tensor]
    def __post_init__(self):
        bs = self.token_context.data['input_ids'].shape[0]
        assert self.labels is None or bs == self.labels.shape[0]
        assert self.image_context is None or bs == self.image_context.shape[0]
        assert self.motor_context is None or bs == self.motor_context.shape[0]


def cache_text_dataset():
    if os.path.exists(constants.TEXT_DATASET_PATH):
        print("Dataset already cached")
        return

    response = requests.get(constants.DATA_URL)
    response.raise_for_status()

    with open(constants.TEXT_DATASET_PATH, 'w') as f:
        f.write(response.text)


def cache_omniglot_dataset(data_spec: DataSpec):
    maybe_download("images", data_spec.img_path)
    maybe_download("strokes", data_spec.traces_path)


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


class OmniglotDataset(Dataset):
    def __init__(self, data_spec: DataSpec, repetitions: Tuple[int, ...]):
        assert max(repetitions) <= 20

        alphabet_name = "Latin"
        self.use_motor_traces = data_spec.use_motor_traces
        file_path = __file__
        base_path = os.path.dirname(file_path)
        self.stroke_dir = os.path.join(base_path, constants.TRACES_PATH, "strokes_background", alphabet_name)
        self.dataset_size = self._calculate_dataset_size()
        self.num_chars = len(os.listdir(self.stroke_dir))
        self.repetitions = repetitions

    def _calculate_dataset_size(self) -> int:
        num_strokes = sum([len(subfolder) for subfolder in os.listdir(self.stroke_dir)])
        return num_strokes

    def _load_motor(self, file_name: str) -> np.ndarray:
        with open(file_name, 'r') as fid:
            lines = [l.strip() for l in fid.readlines()]
        motor = []
        stk = []
        for myline in lines:
            if myline in ['START', 'BREAK']:
                if stk:
                    motor.append(np.array(stk))
                    stk = []
            else:
                stk.append(np.fromstring(myline, dtype=float, sep=','))
        return motor

    def __getitem__(self, char_idx: int):

        repetition_idx, char_idx = divmod(char_idx, self.num_chars)
        strokes = self.char_id_to_sample(char_idx, repetition_idx)
        return strokes

    def char_id_to_sample(self, character_id, rep_idx) -> motor_context:
        assert 0 <= character_id < self.num_chars
        assert 0 <= rep_idx < 20
        character_id_str = f"character{character_id + 1:02d}"
        stroke_file_name = self._get_file_names(character_id_str, rep_idx + 1)
        strokes = self._load_motor(stroke_file_name)
        return strokes

    def _get_file_names(self, character_id, rep_idx):
        stroke_char_dir = os.path.join(self.stroke_dir, character_id)
        fn_example = os.listdir(stroke_char_dir)[0]
        fn_base = fn_example[:fn_example.find('_')]
        fn_stk = os.path.join(stroke_char_dir, f"{fn_base}_{rep_idx:02d}.txt")
        return fn_stk

    def __len__(self):
        return self.dataset_size


def clean_char(char):
    char = char.lower()
    if char.isascii() and char.isalpha():
        return char
    else:
        return ' '


def clean_token(token):
    if token.startswith('<|') and token.endswith('|>'):
        return token
    text = ''.join([clean_char(c) for c in token])
    if text == '':
        text = " "
    return text.lower()


def pad_motor_trace(token_motor_traces: torch.Tensor, eager_rate=1.):
    assert token_motor_traces.shape[1:] == (constants.POINTS_IN_MOTOR_SEQUENCE, 2)  # (num_chars, num_points, 2)
    left_padded_motor_traces = torch.zeros(constants.MAX_CHARS_PER_TOKEN, *token_motor_traces.shape[1:])
    # Calculate the number of characters to keep
    num_chars_to_keep = len(token_motor_traces) * eager_rate
    # If the number of characters to keep is not an integer or is zero, adjust it accordingly
    has_fraction, integer_part = math.modf(num_chars_to_keep)
    should_adjust_last = bool(has_fraction) or integer_part == 0
    num_chars_to_keep = int(num_chars_to_keep) if not should_adjust_last else int(num_chars_to_keep) + 1
    # Slice the motor traces array from the end
    token_motor_traces = token_motor_traces[-num_chars_to_keep:]
    # If adjustment is needed, zero out corresponding steps in the last trace
    if should_adjust_last:
        steps_to_zero = int(has_fraction * token_motor_traces.shape[1])
        token_motor_traces[-1, -steps_to_zero:] = 0
    left_padded_motor_traces[-len(token_motor_traces):] = token_motor_traces
    return left_padded_motor_traces


def preprocess_image(image_and_motor_trace, use_image):
    image, motor_trace = image_and_motor_trace
    image = image.astype(np.float32)
    if use_image:
        image_so_far = process_image(image, motor_trace)
        image_so_far = postprocess_omniglot_image(image_so_far)
    else:
        image_so_far = np.zeros_like(image, dtype=np.float32)
    return image_so_far


class MergeDatasets(Dataset):
    def __init__(self, omniglot_dataset: OmniglotDataset, text_dataset: List[str], ):
        super().__init__()
        self.omniglot_dataset = omniglot_dataset
        self.text_dataset = text_dataset

    def __len__(self):
        return len(self.text_dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        sentence = self.text_dataset[idx]

        text_so_far = []
        motor_contexts = []

        for char in sentence:
            if not char.isalpha():
                continue
            char_strokes = self.omniglot_dataset[char.upper()]

            motor_contexts.append(char_strokes)
            text_so_far.append(char)

            if constants.TEXT_PADDING_ID in text_so_far:
                # return self[idx + 1]
                raise Exception("Found padding char in text so far, returning another sample")

        return motor_contexts, text_so_far


class MemoryCachedMergedDataset(MergeDatasets):
    def __init__(self, omniglot_dataset, text_dataset, token_map):
        super().__init__(omniglot_dataset, text_dataset, token_map)
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def __getitem__(self, idx):
        if self.hits + self.misses > 0:
            print("Hits:", self.hits, "Misses:", self.misses, "Hit Rate:", self.hits / (self.hits + self.misses))
        if idx in self.cache:
            self.hits += 1
            return self.cache[idx]
        else:
            self.misses += 1
            sample = super().__getitem__(idx)
            self.cache[idx] = sample
            print("Cache size in MB:", calc_size(self.cache) / 1024 / 1024)
            return sample


def calc_size(obj):
    size = 0
    for k, v in obj.items():
        if isinstance(v, DataSample):
            size += calc_size(dataclasses.asdict(v))
        elif isinstance(v, dict):
            size += calc_size(v)
        else:
            size += v.element_size() * v.nelement()
    return size


class LineByLineTextDataset(transformers.LineByLineTextDataset):
    def __init__(self, file_path: str, block_size: int):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        self.examples = lines


def main():
    data_spec = DataSpec(
        use_images=False,  # True,
        use_motor_traces=True,
        trace_noise_scale=0.05,
    )
    train_omniglot_dataset, test_omniglot_dataset = get_omniglot_dataset(data_spec)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True,
    )
    for batch in train_dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
