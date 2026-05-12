"""MinecraftWorld — World implementation backed by a running Minecraft instance.

Architecture overview
---------------------
Vision:
    FovealRetina captures the Minecraft window via mss, applies
    eccentricity-based Gaussian blur (cortical magnification factor),
    and downsamples to a flat float32 array matching the brain's vision neurons.
    The brain sees the game exactly as a human does — no block-level access,
    no structured world state.

Homeostasis:
    Health and food are read from the Fabric mod's /state endpoint every tick.
    They are converted to deficit signals [0=optimal, 1=max distress] and
    returned by get_homeostatic_signals().  The brain has no direct knowledge
    of HP values — only the abstract feeling of pain and hunger, identical to
    how a human perceives their own body.

    Pain mapping:  pain_deficit = (1 - health/max_health)^1.5
        The exponent makes mild damage feel mild (quadratic onset) and
        severe damage feel severe, matching nociceptor response curves.

    Hunger mapping: hunger_deficit = (max_food - food) / max_food

Motor:
    InputController maps motor_idx (0–11) to keyboard/mouse actions via
    POST /action.  Proprioceptive efference copy is injected into the
    sound buffer before each action.

Protocol compliance:
    MinecraftWorld implements the full World ABC — it can be swapped with
    GridWorld or TrainingWorld in main.py without touching library code.
"""

from __future__ import annotations

import numpy as np

from mindai.worlds.base import World
from mindai.worlds.minecraft.retina import FovealRetina
from mindai.worlds.minecraft.mod_client import ModClient
from mindai.worlds.minecraft.input import InputController


class MinecraftWorld(World):
    """Minecraft environment adapter.

    Parameters
    ----------
    window_title:
        Substring matched against window titles to locate the game window.
    grid_h, grid_w:
        Vision output grid dimensions (neurons = grid_h * grid_w * 5).
    audio_channels:
        Must match Brain's sensory_layout['audio'].
    mod_url:
        URL of the Fabric mod HTTP server.
    mod_timeout:
        Per-request timeout for mod communication (seconds).
    """

    def __init__(
        self,
        window_title:   str   = 'minecraft',
        vision_size:    int   = 2880,
        audio_channels: int   = 32,
        mod_url:        str   = 'http://localhost:25576',
        mod_timeout:    float = 0.5,
    ) -> None:
        self.audio_channels = audio_channels

        self._retina  = FovealRetina(
            window_title=window_title,
            vision_size=vision_size,
        )
        self._client  = ModClient(base_url=mod_url, timeout=mod_timeout)
        self._input   = InputController(self._client, audio_channels)

        # Wait for mod server before starting the brain loop
        self._client.wait_for_connection()

        # Ambient sound buffer (world ambient — e.g. injected by future
        # dimension/biome detection logic)
        self._ambient_buffer = np.zeros(audio_channels, dtype=np.float32)
        self._last_agent_vocalization = np.zeros(audio_channels, dtype=np.float32)

        # Track previous state for death detection
        self._prev_alive = True

        # Expose minimal fields that brain.py / UI may read
        self.world_tick    = 0
        self.isolation_ticks = 0
        self.inventory     = 'empty'
        self.agent_pos     = [0, 0]

        # Debug-only: human player state from mod (not injected into brain)
        self.debug_human_action: str | None = None
        self.debug_human_distance: float = float('inf')

        # Cache last mod state so we don't double-fetch per tick
        self._state: dict = {}

    # ------------------------------------------------------------------
    # Required World protocol
    # ------------------------------------------------------------------

    def get_sensory_retina(self, num_nodes: int) -> np.ndarray:
        return self._retina.get_visual_array()

    def execute_action(self, motor_idx: int) -> dict:
        self.world_tick += 1
        self._input.execute(motor_idx)
        # World feedback is entirely encoded via homeostatic signals —
        # the immediate energy/water/stress delta is zero here because the
        # consequence of an action (health change from fall damage, eating,
        # etc.) is reflected in the NEXT tick's mod state.
        return {'energy': 0.0, 'water': 0.0, 'stress': 0.0}

    # ------------------------------------------------------------------
    # Homeostasis
    # ------------------------------------------------------------------

    def get_homeostatic_signals(self) -> dict:
        self._state = self._client.get_state()
        state = self._state

        health     = float(state.get('health',     20.0))
        max_health = float(state.get('max_health', 20.0))
        food       = float(state.get('food',       20.0))
        max_food   = float(state.get('max_food',   20.0))

        # Update 2D agent pos proxy for UI
        self.agent_pos = [int(state.get('z', 0)), int(state.get('x', 0))]

        health_ratio  = health / max(max_health, 1.0)
        pain_deficit  = (1.0 - health_ratio) ** 1.5
        hunger_deficit = max(0.0, (max_food - food) / max(max_food, 1.0))

        # Debug-only fields — stored for display, never injected into brain
        self.debug_human_action = state.get('observed_human_action', '') or None
        dist = float(state.get('human_distance', -1.0))
        self.debug_human_distance = dist if dist >= 0 else float('inf')

        return {
            'hunger': float(np.clip(hunger_deficit, 0.0, 1.0)),
            'pain':   float(np.clip(pain_deficit,   0.0, 1.0)),
        }

    def is_alive(self) -> bool:
        return bool(self._state.get('alive', True))

    # ------------------------------------------------------------------
    # Sound
    # ------------------------------------------------------------------

    def pop_world_sound(self) -> np.ndarray:
        # Proprioceptive efference copy from InputController
        prop = self._input.pop_sound()
        out  = np.clip(self._ambient_buffer + prop, 0.0, 1.0)
        self._ambient_buffer = np.zeros(self.audio_channels, dtype=np.float32)
        return out

    def add_sound(self, source_pos, sound_vector: np.ndarray) -> None:
        self._ambient_buffer = np.clip(
            self._ambient_buffer + np.asarray(sound_vector), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Vocalization (echo feedback)
    # ------------------------------------------------------------------

    @property
    def last_agent_vocalization(self) -> np.ndarray:
        return self._last_agent_vocalization

    @last_agent_vocalization.setter
    def last_agent_vocalization(self, value: np.ndarray) -> None:
        self._last_agent_vocalization = value

    def receive_vocalization(self, vec: np.ndarray) -> None:
        self._last_agent_vocalization = vec

    # ------------------------------------------------------------------
    # Human / social
    # ------------------------------------------------------------------

    def get_distance_to_human(self) -> float:
        return self.debug_human_distance

    def process_human_input(self, keys_pressed: dict) -> None:
        pass

    def get_render_string(self) -> str:
        s    = self._state
        dist = self.debug_human_distance
        dist_str = f"{dist:.1f}m" if dist != float('inf') else "—"
        act  = self.debug_human_action or "—"
        return (f"MC  hp={s.get('health',0):.1f}/{s.get('max_health',0):.0f}  "
                f"food={s.get('food',0):.0f}  "
                f"pos=({s.get('x',0):.0f},{s.get('y',0):.0f},{s.get('z',0):.0f})  "
                f"human={dist_str} obs={act}")
