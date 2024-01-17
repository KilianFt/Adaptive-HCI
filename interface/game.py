import dataclasses
import random
import string
import sys
from typing import Tuple

import PyQt5.QtCore
import PyQt5.QtGui
import PyQt5.QtWidgets
import numpy as np
import qimage2ndarray
from PyQt5.QtWidgets import QApplication

import data_loaders.task_datasets
from configs import DataSpec

NUM_TASKS = 26
width, height = 500, 500


@dataclasses.dataclass
class GameState:
    pen: Tuple[int, int]
    task: int = 0
    lifted: bool = False


class Game:
    def __init__(self, width, height, size):
        self.goal = None
        self.pen_is_down = False
        self.pen_x = 0
        self.pen_y = 0
        self.size = size
        self.world = np.ones((width, height), dtype=np.uint8) * 255
        self.reset()

    def update(self, pen_x, pen_y, pen_is_down):
        self.pen_is_down = pen_is_down
        self.pen_x = pen_x
        self.pen_y = pen_y
        if self.pen_is_down:
            self.world[
            self.pen_y - self.size:self.pen_y + self.size,
            self.pen_x - self.size:self.pen_x + self.size
            ] = 0

    def reset(self):
        self.world.fill(255)
        self.goal = random.choice(range(NUM_TASKS))

    def step(self, terminate) -> GameState:
        if terminate:
            self.reset()

        state = self.get_state()
        return state

    def get_state(self) -> GameState:
        return GameState(
            pen=(self.pen_x, self.pen_y),
            task=self.goal,
            lifted=not self.pen_is_down,
        )


class Interface(PyQt5.QtWidgets.QWidget):
    def __init__(self, parent=None):
        self.app = QApplication(sys.argv)
        super().__init__(parent)
        self.traces = [[] for _ in range(NUM_TASKS)]
        self.game_state = Game(width, height, size=2)
        self.resize(width, height)
        self.move(-9, 0)
        self.metadata = {}
        self.setWindowTitle("OCR")

        self.dataloader_iter = data_loaders.task_datasets.get_dataloader(data_spec=DataSpec())
        self.text = "not reset yet"
        self.reset(self.game_state.goal)

    def exec(self):
        self.show()
        self.app.exec_()

    def event(self, a0: PyQt5.QtCore.QEvent):
        event_type = a0.type()
        if isinstance(a0, PyQt5.QtGui.QPlatformSurfaceEvent):
            if event_type not in (217,):
                raise NotImplementedError(a0)
        elif isinstance(a0, PyQt5.QtCore.QEvent):
            if event_type not in (
                    PyQt5.QtCore.QEvent.WindowTitleChange,
                    PyQt5.QtCore.QEvent.WinIdChange,
                    PyQt5.QtCore.QEvent.WindowIconChange,
                    PyQt5.QtCore.QEvent.Polish,
                    PyQt5.QtGui.QMoveEvent.Move,
                    PyQt5.QtGui.QResizeEvent.Resize,
                    PyQt5.QtGui.QShowEvent.Show,
                    PyQt5.QtCore.QEvent.CursorChange,
                    PyQt5.QtCore.QEvent.ShowToParent,
                    PyQt5.QtCore.QEvent.PolishRequest,
                    PyQt5.QtCore.QEvent.UpdateLater,
                    PyQt5.QtCore.QEvent.UpdateRequest,
                    PyQt5.QtCore.QEvent.Paint,
                    PyQt5.QtCore.QEvent.WindowActivate,
                    PyQt5.QtCore.QEvent.ActivationChange,
                    PyQt5.QtGui.QInputMethodQueryEvent.InputMethodQuery,
                    PyQt5.QtCore.QEvent.WindowDeactivate,
                    PyQt5.QtGui.QEnterEvent.Enter,
                    PyQt5.QtGui.QHelpEvent.ToolTip,
                    PyQt5.QtGui.QKeyEvent.KeyPress,
                    PyQt5.QtGui.QKeyEvent.KeyRelease,
                    PyQt5.QtGui.QKeyEvent.ShortcutOverride,
                    PyQt5.QtGui.QKeyEvent.Leave,
                    PyQt5.QtGui.QMouseEvent.MouseButtonDblClick,
            ):
                for attr in dir(a0):
                    type_num = getattr(PyQt5.QtCore.QEvent, attr)
                    if type_num == event_type:
                        print(attr)
                        break

                if event_type == PyQt5.QtGui.QMouseEvent.MouseButtonPress:
                    e = MockEvent(PyQt5.QtGui.QTabletEvent.TabletPress, a0.x(), a0.y())
                    self.tabletEvent(e)
                elif event_type == PyQt5.QtGui.QMouseEvent.MouseButtonRelease:
                    e = MockEvent(PyQt5.QtGui.QTabletEvent.TabletRelease, a0.x(), a0.y())
                    self.tabletEvent(e)
                elif event_type == PyQt5.QtGui.QMouseEvent.MouseMove:
                    e = MockEvent(PyQt5.QtGui.QTabletEvent.TabletMove, a0.x(), a0.y())
                    self.tabletEvent(e)
                elif event_type == 216:
                    print("216, idk what this is")
                else:
                    raise NotImplementedError(a0)
        else:
            raise NotImplementedError(a0)
        return super().event(a0)

    def tabletEvent(self, event):
        pen_x = event.pos().x()
        pen_y = event.pos().y()

        if event.type() == PyQt5.QtGui.QTabletEvent.TabletPress:
            pen_is_down = True
        elif event.type() == PyQt5.QtGui.QTabletEvent.TabletMove:
            pen_is_down = True
        elif event.type() == PyQt5.QtGui.QTabletEvent.TabletRelease:
            pen_is_down = False
        else:
            print("Ignoring", event, event.type())
            return
        self.game_state.update(pen_x, pen_y, pen_is_down)
        self.text = f" goal: {self.game_state.goal}"

        event.accept()
        self.update()

        terminate = False
        if event.type() == PyQt5.QtGui.QTabletEvent.TabletRelease:
            terminate = True

        old_task = self.game_state.goal
        state = self.game_state.step(terminate)
        new_task = self.game_state.goal

        self.traces[old_task][-1].append(state)
        if terminate:
            self.reset(new_task)

    def reset(self, next_task):
        self.text = f"goal: {self.game_state.goal}"
        self.traces[next_task].append([])
        char_strokes = self.dataloader_iter[next_task]
        char_strokes = [c[:, :2] for c in char_strokes]
        all_points = np.concatenate(char_strokes)

        min_dataset = np.min(all_points, axis=0)
        max_dataset = np.max(all_points, axis=0)
        world_size = self.game_state.world.shape

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
            return (shrunk_normalized * world_size).astype(np.int32)

        normalized_char_strokes = [resize(c) for c in char_strokes]
        trace_width = 5

        for stroke in normalized_char_strokes:
            for x, y in stroke:
                self.game_state.world[y:y + trace_width, x:x + trace_width] = 0

    def paintEvent(self, event):
        painter = PyQt5.QtGui.QPainter(self)
        _, _, w, h = self.rect().getRect()
        world = self.game_state.world

        if world.shape != (w, h):
            self.game_state = Game(w, h, size=2)
            world = self.game_state.world

        qImg = qimage2ndarray.array2qimage(world)
        qImg = PyQt5.QtGui.QPixmap(qImg)
        painter.drawPixmap(self.rect(), qImg)
        text = "Write: " + string.ascii_uppercase[self.game_state.goal]

        painter.setRenderHint(PyQt5.QtGui.QPainter.TextAntialiasing)
        painter.drawText(self.rect(), PyQt5.QtCore.Qt.AlignTop | PyQt5.QtCore.Qt.AlignLeft, text)


class MockEvent:
    def __init__(self, code, x, y):
        self.code = code
        self._x = x
        self._y = y

    def type(self):
        return self.code

    def pos(self):
        return MockEvent(None, self._x, self._y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def accept(self):
        pass


def main():
    interface = Interface()
    interface.exec()


if __name__ == "__main__":
    main()
