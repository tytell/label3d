"""Bottom bar: frame navigation buttons, seekbar, frame counter."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

from label3d.state import ProjectState


class FrameControls(QWidget):
    """Frame navigation controls — buttons, slider, keyboard shortcuts."""

    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self._state = state

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # Buttons
        self._btn_start = QPushButton("⏮")
        self._btn_prev_big = QPushButton("⏪")
        self._btn_prev = QPushButton("◀")
        self._btn_next = QPushButton("▶")
        self._btn_next_big = QPushButton("⏩")
        self._btn_end = QPushButton("⏭")

        for btn in [
            self._btn_start, self._btn_prev_big, self._btn_prev,
            self._btn_next, self._btn_next_big, self._btn_end,
        ]:
            btn.setFixedWidth(36)
            layout.addWidget(btn)

        # Frame label
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumWidth(100)
        layout.addWidget(self._frame_label)

        # Seekbar
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        layout.addWidget(self._slider, stretch=1)

        # Connect buttons
        self._btn_start.clicked.connect(state.go_to_start)
        self._btn_end.clicked.connect(state.go_to_end)
        self._btn_prev.clicked.connect(lambda: state.step_frame(-1))
        self._btn_next.clicked.connect(lambda: state.step_frame(1))
        self._btn_prev_big.clicked.connect(lambda: state.step_frame(-4))
        self._btn_next_big.clicked.connect(lambda: state.step_frame(4))

        # Connect slider
        self._slider.valueChanged.connect(self._on_slider_changed)

        # Connect state signals
        state.frame_changed.connect(self._on_frame_changed)

        # Keyboard shortcuts (registered on parent window via shortcut context)
        self._setup_shortcuts()

    def set_range(self, max_frame: int):
        """Set the valid frame range."""
        self._slider.setMaximum(max_frame)
        self._update_label()

    def _on_frame_changed(self, frame: int):
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._update_label()

    def _on_slider_changed(self, value: int):
        self._state.current_frame = value

    def _update_label(self):
        self._frame_label.setText(
            f"{self._state.current_frame} / {self._state.max_frame}"
        )

    def _setup_shortcuts(self):
        """Register keyboard shortcuts for frame navigation."""
        shortcuts = [
            (Qt.Key.Key_Right, lambda: self._state.step_frame(1)),
            (Qt.Key.Key_Left, lambda: self._state.step_frame(-1)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Right),
             lambda: self._state.step_frame(4)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Left),
             lambda: self._state.step_frame(-4)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.ALT | Qt.Key.Key_Right),
             lambda: self._state.step_frame(100)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Modifier.ALT | Qt.Key.Key_Left),
             lambda: self._state.step_frame(-100)),
            (Qt.Key.Key_Home, self._state.go_to_start),
            (Qt.Key.Key_End, self._state.go_to_end),
        ]

        for key, callback in shortcuts:
            shortcut = QShortcut(key, self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(callback)
