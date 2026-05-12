"""GenericInputController — keyboard/mouse simulation via DirectInput.

Uses pydirectinput (SendInput API) which works with DirectX games.
Falls back to pyautogui for non-DirectX applications.

Action spec format:
    ('key',        'w')           — hold key for one frame
    ('key_tap',    'e')           — instant tap (no hold)
    ('mouse',      'left')        — left click
    ('mouse',      'right')       — right click
    ('mouse_move', (dx, dy))      — relative mouse movement in pixels
"""

from __future__ import annotations
import numpy as np


class GenericInputController:

    def __init__(
        self,
        actions:       dict[int, tuple],
        audio_channels: int = 32,
        hold_ticks:    int  = 1,
    ) -> None:
        self._actions       = actions
        self.audio_channels = audio_channels
        self._hold_ticks    = hold_ticks
        self._pending_release: list[str] = []
        self._sound_buf = np.zeros(audio_channels, dtype=np.float32)

        # Try pydirectinput first (DirectX games), fall back to pyautogui
        try:
            import pydirectinput
            pydirectinput.PAUSE = 0.0
            self._di  = pydirectinput
            self._pag = None
        except ImportError:
            try:
                import pyautogui
                pyautogui.PAUSE = 0.0
                pyautogui.FAILSAFE = False
                self._pag = pyautogui
                self._di  = None
            except ImportError:
                raise ImportError(
                    "pip install pydirectinput  # for DirectX games\n"
                    "pip install pyautogui      # fallback for other apps")

    # ------------------------------------------------------------------

    def release_held(self) -> None:
        """Release all keys held from the previous tick. Call at tick start."""
        for key in self._pending_release:
            try:
                if self._di:
                    self._di.keyUp(key)
                else:
                    self._pag.keyUp(key)
            except Exception:
                pass
        self._pending_release.clear()

    def execute(self, motor_idx: int) -> None:
        """Execute the action bound to motor_idx."""
        spec = self._actions.get(motor_idx)
        if spec is None:
            return

        kind = spec[0]

        if kind == 'key':
            key = spec[1]
            if self._di:
                self._di.keyDown(key)
            else:
                self._pag.keyDown(key)
            self._pending_release.append(key)
            # Efference copy: encode which key was pressed into audio buffer
            slot = motor_idx % self.audio_channels
            self._sound_buf[slot] = min(1.0, self._sound_buf[slot] + 0.5)

        elif kind == 'key_tap':
            key = spec[1]
            if self._di:
                self._di.press(key)
            else:
                self._pag.press(key)

        elif kind == 'mouse':
            button = spec[1]
            if self._di:
                if button == 'left':
                    self._di.click()
                elif button == 'right':
                    self._di.rightClick()
            else:
                if button == 'left':
                    self._pag.click()
                elif button == 'right':
                    self._pag.rightClick()
            slot = motor_idx % self.audio_channels
            self._sound_buf[slot] = min(1.0, self._sound_buf[slot] + 0.7)

        elif kind == 'mouse_move':
            dx, dy = spec[1]
            if self._di:
                self._di.moveRel(dx, dy, relative=True)
            else:
                self._pag.moveRel(dx, dy)

    def pop_sound(self) -> np.ndarray:
        buf = self._sound_buf.copy()
        self._sound_buf[:] = 0.0
        return buf
