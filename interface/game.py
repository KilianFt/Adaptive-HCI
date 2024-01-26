import os
import dataclasses
import pickle
import multiprocessing
import random
import string
import sys
from datetime import datetime
from typing import Optional
from collections import deque

import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets
import numpy as np
import qimage2ndarray
from PyQt5.QtWidgets import QApplication
import torch
import torch.nn.functional as F

from adaptive_hci.controllers import EMGViT
import data_loaders.task_datasets
from realtime_pred_test import worker
from autowriter.plotting import label_to_move
from autowriter.datasets import OmniglotGridDataset
from autowriter.mingpt.model import GPT

from configs import BaseConfig

# Constants
WIDTH, HEIGHT = 500, 500
ALLOWED_TASKS = [ord(c) - ord('a') for c in 'lo']
SUGGESTION_WIDTH = 5
X_OFFSET = 30 # to make sure that there is space between stroke prompt and border

event_code_map = {
    PyQt5.QtGui.QMouseEvent.MouseButtonPress: PyQt5.QtGui.QTabletEvent.TabletPress,
    PyQt5.QtGui.QMouseEvent.MouseButtonRelease: PyQt5.QtGui.QTabletEvent.TabletRelease,
    PyQt5.QtGui.QMouseEvent.MouseMove: PyQt5.QtGui.QTabletEvent.TabletMove,
    PyQt5.QtGui.QTabletEvent.TabletPress: PyQt5.QtGui.QTabletEvent.TabletPress,
    PyQt5.QtGui.QTabletEvent.TabletRelease: PyQt5.QtGui.QTabletEvent.TabletRelease,
    PyQt5.QtGui.QTabletEvent.TabletMove: PyQt5.QtGui.QTabletEvent.TabletMove,
}


@dataclasses.dataclass
class GameState:
    pen: tuple[int, int]
    emg: np.ndarray
    label: Optional[np.ndarray] = None


emg_queue = multiprocessing.Queue()


def get_emg():
    if not emg_queue.empty():
        return emg_queue.get()
    else:
        return np.zeros(8)  # Return zero array if no data is available


size = 3
emg_size = 5

def load_models(emg_decoder_state_dict_file, auto_writer_state_dict_file, config):
    emg_decoder_state_dict = torch.load(emg_decoder_state_dict_file)
    emg_decoder = EMGViT(
        image_size=config.window_size,
        patch_size=config.general_model_config.patch_size,
        num_classes=config.num_classes,#n_labels,
        dim=config.general_model_config.dim,
        depth=config.general_model_config.depth,
        heads=config.general_model_config.heads,
        mlp_dim=config.general_model_config.mlp_dim,
        dropout=config.general_model_config.dropout,
        emb_dropout=config.general_model_config.emb_dropout,
        channels=config.general_model_config.channels,
    )
    emg_decoder.load_state_dict(emg_decoder_state_dict)
    emg_decoder.eval()

    train_dataset = OmniglotGridDataset((config.auto_writer.omniglot_dir),
                                        context_len=config.auto_writer.context_len,
                                        char_idxs=config.auto_writer.character_idxs)

    model_config = GPT.get_default_config()
    model_config.model_type = config.auto_writer.gpt_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    auto_writer = GPT(model_config)

    auto_writer_state_dict = torch.load(auto_writer_state_dict_file, map_location=torch.device('cpu'))
    auto_writer.load_state_dict(auto_writer_state_dict)
    auto_writer.eval()
    return emg_decoder, auto_writer


class Interface(PyQt5.QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.canvas = np.zeros((WIDTH, HEIGHT), dtype=np.uint8)
        config = BaseConfig()

        self.dataloader_iter = data_loaders.task_datasets.get_omniglot_dataset()
        self.traces = {i: [[], ] for i in ALLOWED_TASKS}

        game_state, new_task = self.reset()
        self.current_task = new_task
        self.game_state = game_state
        self.trace_count = 0
        self.label_history = deque(maxlen=config.auto_writer.context_len)
        self.emg_buffer = deque(maxlen=config.window_size)
        self.n_new_samples = -config.overlap

        self.predict_mode = True
        if self.predict_mode:
            auto_writer_state_dict_file = './models/draw_gpt_state_dict_o_l.pt'
            emg_decoder_state_dict_file = './models/finetuned_emg_decoder_state_dict.pt'
            self.emg_decoder, self.auto_writer = load_models(emg_decoder_state_dict_file,
                                                             auto_writer_state_dict_file,
                                                             config)
            self.emg_pos = np.array([WIDTH/2, HEIGHT/2])
            self.step_size = 10 # px
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.out_folder = f"./datasets/emg_writing/{dt_string}/"
        os.makedirs(self.out_folder)

    def _predict_emg(self):
        emg_tensor = torch.tensor(self.emg_buffer, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.label_history, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            emg_pred_raw = self.emg_decoder(emg_tensor)
            emg_pred = F.sigmoid(emg_pred_raw)
            auto_pred_raw, _ = self.auto_writer(label_tensor)
            auto_pred = F.sigmoid(auto_pred_raw)

        emg_next_token_probs = emg_pred[0,:4]
        if len(self.label_history) > 0:
            auto_next_token_probs = auto_pred[0,-1,:4]
        else:
            auto_next_token_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        print(emg_next_token_probs)
        print(auto_next_token_probs)
        label = ((emg_next_token_probs + auto_next_token_probs) / 2).argmax().item()
        print(label)
        self.label_history.append(label)
        move = np.array(label_to_move(label)) * self.step_size
        print(move)
        self.emg_pos += move
        print(self.emg_pos)
        return int(self.emg_pos[0]), int(self.emg_pos[1])

    def _transition(self, pen_x, pen_y, pen_is_down):
        emg = get_emg()
        self.emg_buffer.append(emg)
        self.n_new_samples += 1
        label = None
        if pen_is_down:
            if self.predict_mode and self.n_new_samples > 50:
                emg_x, emg_y = self._predict_emg()
                self.canvas[emg_y - emg_size:emg_y + emg_size, emg_x - emg_size:emg_x + emg_size] = 0
                self.n_new_samples = 0

            self.canvas[pen_y - size:pen_y + size, pen_x - size:pen_x + size] = 100
        new_state = GameState(pen=(pen_x, pen_y), emg=emg, label=label)
        return new_state

    def init_ui(self):
        self.setWindowTitle("OCR")
        self.resize(WIDTH, HEIGHT)

    def event(self, a0: PyQt5.QtCore.QEvent):
        if isinstance(a0, PyQt5.QtCore.QEvent) and a0.type() in event_code_map:
            new_code = event_code_map[a0.type()]
            a1 = MockEvent(a0, new_code)
            self.tabletEvent(a1)
        return super().event(a0)

    def tabletEvent(self, event):
        pen_x = event.pos().x()
        pen_y = event.pos().y()
        terminate = False

        if event.type() == PyQt5.QtGui.QTabletEvent.TabletPress:
            pen_is_down = True
        elif event.type() == PyQt5.QtGui.QTabletEvent.TabletMove:
            pen_is_down = True
        elif event.type() == PyQt5.QtGui.QTabletEvent.TabletRelease:
            pen_is_down = False
            terminate = True
        else:
            print("Ignoring", event, event.type())
            return

        event.accept()
        self.update()

        new_state = self._transition(pen_x, pen_y, pen_is_down)
        self.traces[self.current_task][-1].append(new_state)

        if terminate:
            new_state, current_task = self.reset()
            self.traces[current_task].append([])
            out_file = self.out_folder + f"traces{self.trace_count}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(self.traces, f)
            self.trace_count += 1

    def reset(self):
        current_task = random.choice(ALLOWED_TASKS)
        normalized_char_strokes = self._sample_stroke(current_task)

        self.canvas.fill(255)
        for stroke in normalized_char_strokes:
            for x, y in stroke:
                x = x + X_OFFSET
                self.canvas[y:y + SUGGESTION_WIDTH, x:x + SUGGESTION_WIDTH] = 0

        return GameState(pen=(0, 0), emg=np.array([])), current_task

    def _sample_stroke(self, current_task):
        char_strokes = self.dataloader_iter[current_task]
        all_points = np.concatenate(char_strokes)
        min_dataset = np.min(all_points, axis=0)
        max_dataset = np.max(all_points, axis=0)

        def resize(stroke):
            z = (max_dataset - min_dataset)
            normalized = (stroke - min_dataset) / z

            # Shrink by 30%
            shrink_factor = 0.7  # 70% of the original size
            # Calculate the center offset to keep the stroke centered after shrinking
            center_offset = (1 - shrink_factor) / 2
            # Apply shrinkage and re-center
            shrunk_normalized = normalized * shrink_factor + center_offset
            shrunk_normalized[:, 1] = 1 - shrunk_normalized[:, 1]
            return (shrunk_normalized * self.canvas.shape).astype(np.int32)

        normalized_char_strokes = [resize(c) for c in char_strokes]
        return normalized_char_strokes

    def paintEvent(self, event):
        painter = PyQt5.QtGui.QPainter(self)
        _, _, w, h = self.rect().getRect()

        if self.canvas.shape != (w, h):
            raise ValueError(f"Canvas shape {self.canvas.shape} does not match widget shape {(w, h)}")

        qImg = qimage2ndarray.array2qimage(self.canvas)
        qImg = PyQt5.QtGui.QPixmap(qImg)
        painter.drawPixmap(self.rect(), qImg)

        if self.current_task is not None:
            text = "Write: " + string.ascii_uppercase[self.current_task]
        else:
            text = "Task not initialized yet"

        painter.setRenderHint(PyQt5.QtGui.QPainter.TextAntialiasing)
        painter.drawText(self.rect(), PyQt5.QtCore.Qt.AlignTop | PyQt5.QtCore.Qt.AlignLeft, text)


class MockEvent:
    def __init__(self, evt, new_code):
        self.evt = evt
        self._new_code = new_code

    def type(self):
        return self._new_code

    def pos(self):
        return self.evt.pos()

    def accept(self):
        self.evt.accept()


def main():
    myo_process = multiprocessing.Process(target=worker, args=(emg_queue,))
    myo_process.start()

    app = QApplication(sys.argv)
    interface = Interface()
    interface.show()
    app.exec()


if __name__ == "__main__":
    main()
