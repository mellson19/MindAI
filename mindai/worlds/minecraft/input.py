"""Motor-index → Minecraft action mapping.

Motor commands follow the same biological model as other worlds:
    - Each motor neuron index maps to a muscle group / action
    - Proprioceptive efference copy (unique audio signature) is injected
      BEFORE the action so STDP can link the motor pattern to its expected
      sensory consequence (Wolpert 1995)

Action table (12 actions)
--------------------------
idx  | action       | description
-----|--------------|------------------------------------------
0    | move_forward | W key
1    | move_back    | S key
2    | strafe_left  | A key
3    | strafe_right | D key
4    | jump         | Space
5    | sneak        | Shift (hold)
6    | attack       | Left mouse
7    | use          | Right mouse
8    | cam_left     | yaw -= 5°
9    | cam_right    | yaw += 5°
10   | cam_up       | pitch -= 5°
11   | cam_down     | pitch += 5°

Audio band allocation (32 bands assumed, matches TrainingWorld convention)
--------------------------------------------------------------------------
[0:2]  proprioception — forward
[2:4]  proprioception — back
[4:6]  proprioception — left
[6:8]  proprioception — right
[8:10] proprioception — jump
[10:12] proprioception — sneak
[12:14] proprioception — attack
[14:16] proprioception — use
[16:20] proprioception — camera (combined band for all 4 cam actions)
[20:32] world ambient (injected by world, not here)
"""

from __future__ import annotations

import numpy as np

from mindai.worlds.minecraft.mod_client import ModClient


# Action names sent to the mod via POST /action
ACTION_NAMES = [
    'move_forward',
    'move_back',
    'strafe_left',
    'strafe_right',
    'jump',
    'sneak',
    'attack',
    'use',
    'cam_left',
    'cam_right',
    'cam_up',
    'cam_down',
]

# Proprioceptive band for each action (slice into 32-band audio buffer)
_PROP_BANDS = [
    slice(0,  2),   # 0 forward
    slice(2,  4),   # 1 back
    slice(4,  6),   # 2 strafe_left
    slice(6,  8),   # 3 strafe_right
    slice(8,  10),  # 4 jump
    slice(10, 12),  # 5 sneak
    slice(12, 14),  # 6 attack
    slice(14, 16),  # 7 use
    slice(16, 18),  # 8 cam_left
    slice(18, 20),  # 9 cam_right
    slice(20, 22),  # 10 cam_up
    slice(22, 24),  # 11 cam_down
]

# Camera yaw/pitch delta per cam action (degrees)
_CAM_DELTA = 5.0


class InputController:
    """Translates brain motor_idx into Minecraft inputs via the Fabric mod.

    Parameters
    ----------
    client:
        Active ModClient instance.
    audio_channels:
        Must match Brain's sensory_layout['audio'].  Default 32.
    """

    def __init__(self, client: ModClient, audio_channels: int = 32) -> None:
        self._client        = client
        self._audio_channels = audio_channels
        self._sound_buffer  = np.zeros(audio_channels, dtype=np.float32)

    def execute(self, motor_idx: int) -> None:
        """Send action to mod and inject proprioceptive efference copy."""
        # Efference copy BEFORE the action (motor command → predictable sensation)
        self._emit_proprioception(motor_idx)

        if motor_idx >= len(ACTION_NAMES):
            return

        action = ACTION_NAMES[motor_idx]

        if action in ('cam_left', 'cam_right', 'cam_up', 'cam_down'):
            delta_map = {
                'cam_left':  ('yaw',   -_CAM_DELTA),
                'cam_right': ('yaw',   +_CAM_DELTA),
                'cam_up':    ('pitch', -_CAM_DELTA),
                'cam_down':  ('pitch', +_CAM_DELTA),
            }
            axis, value = delta_map[action]
            self._client.send_action(f'rotate_{axis}', value)
        else:
            self._client.send_action(action)

    def _emit_proprioception(self, motor_idx: int) -> None:
        if motor_idx < len(_PROP_BANDS):
            band = _PROP_BANDS[motor_idx]
            start = band.start or 0
            stop  = min(band.stop, self._audio_channels)
            if start < stop:
                self._sound_buffer[start:stop] = 0.9

    def pop_sound(self) -> np.ndarray:
        """Return and clear proprioceptive buffer (called each tick by world)."""
        out = np.clip(self._sound_buffer, 0.0, 1.0)
        self._sound_buffer = np.zeros(self._audio_channels, dtype=np.float32)
        return out
