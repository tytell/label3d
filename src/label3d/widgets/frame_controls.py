"""Bottom bar: frame navigation buttons, seekbar, frame counter."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from label3d.frames import shown_frame
from label3d.state import ProjectState
from label3d.widgets.frame_slider import FrameSlider


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
        self._frame_label = QLabel(f"{shown_frame(0)} / {shown_frame(0)}")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumWidth(100)
        layout.addWidget(self._frame_label)

        # Seekbar with per-camera label indicators
        self._slider = FrameSlider()
        layout.addWidget(self._slider, stretch=1)

        # Connect buttons
        self._btn_start.clicked.connect(state.go_to_start)
        self._btn_end.clicked.connect(state.go_to_end)
        self._btn_prev.clicked.connect(lambda: state.step_frame(-1))
        self._btn_next.clicked.connect(lambda: state.step_frame(1))
        self._btn_prev_big.clicked.connect(lambda: state.step_frame(-4))
        self._btn_next_big.clicked.connect(lambda: state.step_frame(4))

        # Connect slider
        self._slider.frame_changed.connect(self._on_slider_changed)

        # Connect state signals
        state.frame_changed.connect(self._on_frame_changed)

        # Keyboard shortcuts (registered on parent window via shortcut context)
        self._setup_shortcuts()

    def set_cameras(self, cameras: list[str]) -> None:
        """Inform the slider of the ordered camera list (sets up rows)."""
        self._slider.set_cameras(cameras)

    def set_range(self, max_frame: int) -> None:
        """Set the valid frame range."""
        self._slider.set_range(max_frame)
        self._update_label()

    def update_marks(self, marks: dict[str, dict[int, str]]) -> None:
        """Update labeled-frame tick marks on the slider."""
        self._slider.set_marks(marks)

    def update_new_marks(self, marks: dict[int, str]) -> None:
        """Update the slider's row of frames with new points."""
        self._slider.set_new_marks(marks)

    def _on_frame_changed(self, frame: int):
        self._slider.set_value(frame)
        self._update_label()

    def _on_slider_changed(self, value: int):
        self._state.current_frame = value

    def _update_label(self):
        self._frame_label.setText(
            f"{shown_frame(self._state.current_frame)} / {shown_frame(self._state.max_frame)}"
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
