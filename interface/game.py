import dataclasses
import pickle
import multiprocessing
import random
import string
import sys

import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets
import numpy as np
import qimage2ndarray
from PyQt5.QtWidgets import QApplication

import data_loaders.task_datasets
from realtime_pred_test import worker

# Constants
WIDTH, HEIGHT = 500, 500
ALLOWED_TASKS = [ord(c) - ord('a') for c in 'io']
SUGGESTION_WIDTH = 5

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


emg_queue = multiprocessing.Queue()


def get_emg():
    if not emg_queue.empty():
        return emg_queue.get()
    else:
        return np.zeros(8)  # Return zero array if no data is available


size = 5


class Interface(PyQt5.QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.canvas = np.zeros((WIDTH, HEIGHT), dtype=np.uint8)

        self.dataloader_iter = data_loaders.task_datasets.get_omniglot_dataset()
        self.traces = {i: [[], ] for i in ALLOWED_TASKS}

        game_state, new_task = self.reset()
        self.current_task = new_task
        self.game_state = game_state

    def _transition(self, pen_x, pen_y, pen_is_down):
        if pen_is_down:
            self.canvas[pen_y - size:pen_y + size, pen_x - size:pen_x + size] = 0
        new_state = GameState(pen=(pen_x, pen_y), emg=get_emg())
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
            with open("traces.pkl", "wb") as f:
                pickle.dump(self.traces, f)

    def reset(self):
        current_task = random.choice(ALLOWED_TASKS)
        normalized_char_strokes = self._sample_stroke(current_task)

        self.canvas.fill(255)
        for stroke in normalized_char_strokes:
            for x, y in stroke:
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
