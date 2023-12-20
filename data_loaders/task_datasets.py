import dataclasses
import functools
import math
import os
import random
import zipfile
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
import transformers

import configs
import constants
from configs import DataSpec

# import hyper
# from presets import get_default_tokenizer


tokenizer = None


def get_default_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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


@dataclasses.dataclass
class ProcessingDataSample:
    image_context: Union[np.ndarray, torch.Tensor]
    motor_context: Union[np.ndarray, torch.Tensor]


class FlatteningDataCollator:
    """
    A batch is made of batch_size samples,
     each sample is a dictionary of elements

     token_context: BatchEncoding({
        "input_ids": torch.Tensor, # (batch_size, TOKEN_CONTEXT_LEN)
        "attention_mask": torch.Tensor, # (batch_size, TOKEN_CONTEXT_LEN)
      })
     image_context: ??
     motor_context: torch.Tensor, # (batch_size, MAX_CHARS_PER_TOKEN, POINTS_IN_MOTOR_SEQUENCE, 2)
     labels: torch.Tensor, # (batch_size, )

    This collator flattens the batch into a single dictionary of elements
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # self.llm_collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        # self.collate_with_padding = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, batch):
        elem = batch[0]

        assert elem.keys() == {
            "token_context",
            "image_context",
            "motor_context",
            "labels",
        }, "Batch elements do not match"
        assert all(len(elem) == len(b) for b in batch), "Batch elements do not match"
        assert all(elem.keys() == b.keys() for b in batch), "Batch elements do not match"

        data = {}
        for key in elem:
            rows = [d[key] for d in batch]

            if all(row is None for row in rows):
                continue

            if isinstance(rows[0], transformers.BatchEncoding):
                batched_encoding = {
                    k: torch.cat([d[k] for d in rows], dim=0) for k in rows[0].data.keys()
                }
                data[key] = transformers.BatchEncoding(batched_encoding)
            else:
                data[key] = torch.concat(rows, dim=0)
        return DataSample(**data)


def cache_text_dataset():
    if os.path.exists(constants.TEXT_DATASET_PATH):
        print("Dataset already cached")
        return

    response = requests.get(constants.DATA_URL)
    response.raise_for_status()

    with open(constants.TEXT_DATASET_PATH, 'w') as f:
        f.write(response.text)


def resample_stroke(stroke, num_samples):
    assert len(stroke.shape) == 2, "Stroke must be 2D, not batched"
    x, y, t = stroke.T
    new_t = np.linspace(t[0], t[-1], num_samples)
    return np.stack([np.interp(new_t, t, x), np.interp(new_t, t, y)], axis=1)


def _process_trace(all_traces):
    min_x, min_y = np.min(all_traces, axis=0)

    # y_norm = max_y - min_y
    # y_norm = max(y_norm, 50)
    x_norm = 50
    y_norm = 50

    return np.stack([
        (all_traces[:, 0] - min_x) / x_norm,
        (all_traces[:, 1] - min_y) / y_norm,
    ], axis=1)


def revert_preprocess_trace(all_traces):
    x_norm = 50
    y_norm = 50

    # return np.stack([(all_traces[:, 0] * x_norm), (all_traces[:, 1] * y_norm), ], axis=1)
    all_traces[:, 0] *= x_norm
    all_traces[:, 1] *= y_norm
    return all_traces


example_trace = np.array([[0, 0], [1, 1]])
assert np.allclose(revert_preprocess_trace(_process_trace(example_trace)), example_trace)


class MultimodalTransform:
    def __init__(self, image_transform, trace_transform):
        self.image_transform = image_transform
        self.trace_transform = trace_transform

    def __call__(self, sample: ProcessingDataSample) -> ProcessingDataSample:
        transformed_trace = self.trace_transform(sample.motor_context)
        transformed_image = self.image_transform((sample.image_context, transformed_trace))
        return ProcessingDataSample(
            image_context=transformed_image,
            motor_context=transformed_trace,
        )


# def cache_omniglot_dataset(alphabet_name: str):
#     img_dir = os.path.join(constants.IMG_PATH, alphabet_name)
#     stroke_dir = os.path.join(constants.TRACES_PATH, alphabet_name)
#
#     if not os.path.exists(img_dir):
#         # Download from:
#         # https://github.com/brendenlake/omniglot

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


def get_omniglot_dataset(data_spec: DataSpec, transforms: MultimodalTransform):
    cache_omniglot_dataset(data_spec)
    train_dataset = OmniglotDataset(data_spec, transforms, repetitions=data_spec.train_reps)
    test_dataset = OmniglotDataset(data_spec, transforms, repetitions=data_spec.test_reps)
    return train_dataset, test_dataset


def resample_storkes(motor_traces, total_num_samples):
    return [
        resample_stroke(
            stroke, num_samples=total_num_samples // len(motor_traces)) for stroke in motor_traces
    ]


def _process_image_and_traces(image, all_traces):
    stroke = _process_trace(all_traces)
    image_so_far = process_image(image, stroke)
    return image_so_far, all_traces


def process_image(image, stroke):
    image_so_far = np.zeros_like(image)
    assert image_so_far.shape[0] == image_so_far.shape[1]
    points = np.round(stroke[:, :2] * image_so_far.shape[0]).astype(int)
    points = np.clip(points, 0, image_so_far.shape[0] - 1)
    image_so_far[points[:, 1], points[:, 0]] = 1
    return image_so_far


def pad_trace(all_traces, points_in_motor_sequence):
    motor_traces = np.zeros((points_in_motor_sequence, 2))
    motor_traces_ = np.array(all_traces, dtype=np.float32)
    motor_traces[-len(motor_traces_):] = motor_traces_
    return motor_traces


def _adjust_image_orientation(image_so_far):
    image_so_far = np.rot90(image_so_far, k=2)
    image_so_far = np.fliplr(image_so_far)
    return image_so_far


def postprocess_image_and_traces(image, strokes_for_char, use_image, points_in_motor_sequence):
    motor_trace = process_strokes(strokes_for_char, points_in_motor_sequence)

    if use_image:
        image_so_far = process_image(image, motor_trace)
        image_so_far = postprocess_omniglot_image(image_so_far)
    else:
        image_so_far = np.zeros_like(image)

    return image_so_far, motor_trace


def process_strokes(strokes_for_char, points_in_motor_sequence):
    resampled_motor_traces = resample_storkes(strokes_for_char, total_num_samples=points_in_motor_sequence)
    all_traces = np.concatenate(resampled_motor_traces, axis=0)
    trace = _process_trace(all_traces)
    return trace


def preprocess_trace(strokes_for_char, points_in_motor_sequence):
    motor_traces = process_strokes(strokes_for_char, points_in_motor_sequence)
    motor_trace = pad_trace(motor_traces, points_in_motor_sequence)
    return motor_trace


def postprocess_omniglot_image(image):
    image_so_far = _adjust_image_orientation(image)
    return image_so_far.astype(np.uint8) * 255


class OmniglotDataset(Dataset):
    def __init__(self, data_spec: DataSpec, transforms: MultimodalTransform, repetitions: Tuple[int, ...]):
        assert max(repetitions) <= 20

        alphabet_name = "Latin"
        self.use_images = data_spec.use_images
        self.use_motor_traces = data_spec.use_motor_traces
        file_path = __file__
        base_path = os.path.dirname(file_path)
        self.img_dir = os.path.join(base_path, constants.IMG_PATH, "images_background", alphabet_name)
        self.stroke_dir = os.path.join(base_path, constants.TRACES_PATH, "strokes_background", alphabet_name)
        self.dataset_size = self._calculate_dataset_size()
        self.transforms = transforms
        self.repetitions = repetitions

    def _calculate_dataset_size(self) -> int:
        num_images = sum([len(subfolder) for subfolder in os.listdir(self.img_dir)])
        num_strokes = sum([len(subfolder) for subfolder in os.listdir(self.stroke_dir)])
        assert num_images == num_strokes
        assert num_images > 0
        return num_images

    def _load_motor(self, fn: str) -> np.ndarray:
        with open(fn, 'r') as fid:
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

    @staticmethod
    def _load_img(fn: str) -> np.ndarray:
        return np.array(plt.imread(fn), dtype=bool)

    def __getitem__(self, token: str):
        # TODO: encode somehow the trace repetition, right now we always use the first one
        # token_idx, rep_idx = divmod(idx, self.traces_per_char)
        # rep_idx = 1
        rep_idx = random.choice(self.repetitions)

        token_images = []
        token_traces = []

        if not (self.use_images or self.use_motor_traces):
            return None, None

        if token == '<|endoftext|>':
            token = " "  # TODO: we are aliasing the space character to end of text

        for char in token:
            # TODO keep reviewing the dataset, the intent is to add a doctor writing preprocess
            character_id = ord(char) - ord('a')
            assert 0 <= character_id < 26 or char == ' '

            should_zero_out = False
            if character_id == ord(' ') - ord('a'):
                character_id = 0
                rep_idx = 1
                should_zero_out = True

            raw_sample: ProcessingDataSample = self.char_id_to_sample(character_id, rep_idx)
            data_sample = self.transforms(raw_sample)

            if should_zero_out:
                data_sample.image_context[:] = 0
                data_sample.motor_context[:] = 0

            token_traces.append(data_sample.motor_context)
            token_images.append(data_sample.image_context)

        token_traces = torch.concat(token_traces, dim=0)
        token_images = torch.concat(token_images, dim=0)

        if not self.use_motor_traces:
            token_traces = None
        if not self.use_images:
            token_images = None

        return ProcessingDataSample(
            motor_context=token_traces,
            image_context=token_images,
        )

    def char_id_to_sample(self, character_id, rep_idx) -> ProcessingDataSample:
        character_id_str = f"character{character_id + 1:02d}"
        fn_stk, fn_img = self._get_file_names(character_id_str, rep_idx)
        strokes = self._load_motor(fn_stk)
        image = self._load_img(fn_img)
        return ProcessingDataSample(
            image_context=image,
            motor_context=strokes,
        )

    def _get_file_names(self, character_id, rep_idx):
        img_char_dir = os.path.join(self.img_dir, character_id)
        stroke_char_dir = os.path.join(self.stroke_dir, character_id)
        fn_example = os.listdir(img_char_dir)[0]
        fn_base = fn_example[:fn_example.find('_')]
        fn_stk = os.path.join(stroke_char_dir, f"{fn_base}_{rep_idx:02d}.txt")
        fn_img = os.path.join(img_char_dir, f"{fn_base}_{rep_idx:02d}.png")
        return fn_stk, fn_img

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
    def __init__(
            self,
            omniglot_dataset: OmniglotDataset,
            text_dataset: List[str],
            tokenizer: transformers.PreTrainedTokenizer,
            token_context_len: int,
    ):
        super().__init__()
        self.omniglot_dataset = omniglot_dataset
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer
        self.token_context_length = token_context_len
        self.empty_token, = self.tokenizer.encode(constants.EMPTY_CHAR, add_special_tokens=False)

    def __len__(self):
        return len(self.text_dataset)

    @torch.no_grad()
    def __getitem__(self, idx):
        # Does not return an item but rather a decomposition of a sentence into several items
        sentence = self.text_dataset[idx]

        images = []
        text_so_far = []

        motor_contexts = []
        text_contexts = []
        sentence_tokens = sentence["input_ids"].tolist()

        for token_idx in sentence_tokens:
            token = self.tokenizer.decode(token_idx, clean_up_tokenization_spaces=True).lower()
            token = clean_token(token)

            token_sample: ProcessingDataSample = self.omniglot_dataset[token]

            char_context = np.array(text_so_far)

            if len(char_context) > self.token_context_length:
                char_context = char_context[-self.token_context_length:]
            left_padded_char_context = np.pad(
                char_context, (self.token_context_length - len(char_context), 0), 'constant',
                constant_values=self.tokenizer.pad_token_id)

            if token_sample.image_context is not None:
                img = token_sample.image_context
                assert len(img) <= constants.MAX_CHARS_PER_TOKEN, "too many images for a single token"
                left_padded_images = torch.zeros(constants.MAX_CHARS_PER_TOKEN, *img[1:])
                left_padded_images[-len(img):] = img
                images.append(left_padded_images)

            if token_sample.motor_context is not None:
                if constants.eager_rate < 1:
                    assert token_sample.image_context is None, "eager rate is not supported for images, as it would leak information"
                left_padded_motor_traces = pad_motor_trace(token_sample.motor_context, eager_rate=constants.eager_rate)

                motor_contexts.append(left_padded_motor_traces)

            text_so_far.append(token_idx)

            if constants.TEXT_PADDING_ID in text_so_far:
                print("Found padding token in text so far, returning another sample")
                return self[idx + 1]

            text_contexts.append(left_padded_char_context)

        input_ids = torch.from_numpy(np.array(text_contexts)).to(dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(torch.long)

        token_context = transformers.BatchEncoding({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        batch = DataSample(
            token_context=token_context,
            image_context=torch.stack(images, dim=0) if len(images) > 0 else None,
            motor_context=torch.stack(motor_contexts, dim=0) if len(motor_contexts) > 0 else None,
            labels=torch.tensor(text_so_far, dtype=torch.long),
        )
        return dataclasses.asdict(batch)


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
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, file_path: str, block_size: int):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        ids = batch_encoding["input_ids"]
        self.examples = [{
            "input_ids": torch.tensor(e, dtype=torch.long),
            "text": l

        } for e, l in zip(ids, lines)]


def get_text_dataset(tokenizer, data_spec: DataSpec):
    cache_text_dataset()
    text_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=data_spec.text_dataset_path,
        block_size=128
    )
    if data_spec.max_dataset_size is not None:
        assert len(text_dataset) >= data_spec.max_dataset_size
        text_dataset.examples = text_dataset.examples[:data_spec.max_dataset_size]
        print("Dataset size:", len(text_dataset))
    train_dataset, val_dataset = train_test_split(text_dataset, test_size=data_spec.test_fraction, random_state=42)
    return train_dataset, val_dataset


def get_multimodal_dataset(data_spec: DataSpec, cache_only=False):
    def add_trace_noise(motor_trace):
        noise = np.random.normal(0, data_spec.trace_noise_scale, motor_trace.shape)
        return motor_trace + noise

    multimodal_transforms = MultimodalTransform(
        image_transform=transforms.Compose([
            functools.partial(preprocess_image, use_image=data_spec.use_images),
            transforms.ToPILImage(),
            transforms.Resize((data_spec.image_side, data_spec.image_side)),
            transforms.ToTensor()
        ]),
        trace_transform=transforms.Compose([
            functools.partial(preprocess_trace, points_in_motor_sequence=data_spec.points_in_motor_sequence),
            transforms.ToTensor(),
            add_trace_noise,
        ]))

    tokenizer = get_default_tokenizer()
    text_test_set, text_train_set = get_text_dataset(tokenizer, data_spec)

    # TODO: we are using the same data for train and test, we should split omniglot in two and sample independently
    train_omniglot_dataset, test_omniglot_dataset = get_omniglot_dataset(data_spec, transforms=multimodal_transforms)

    train_set = MergeDatasets(
        train_omniglot_dataset,
        text_train_set,
        tokenizer=tokenizer,
        token_context_len=data_spec.token_context_len,
    )

    test_set = MergeDatasets(
        test_omniglot_dataset,
        text_test_set,
        tokenizer=tokenizer,
        token_context_len=data_spec.token_context_len,
    )

    return train_set, test_set


if __name__ == "__main__":
    data_spec = DataSpec(
        use_images=False,  # True,
        use_motor_traces=True,
        trace_noise_scale=0.05,
    )
    train_dataset, valid_dataset = get_multimodal_dataset(data_spec)
    tokenizer = get_default_tokenizer()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True,
        collate_fn=FlatteningDataCollator(tokenizer),
    )
    for batch in train_dataloader:
        print(batch)
        break
