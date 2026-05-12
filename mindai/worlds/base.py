from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class World(ABC):
    """Abstract contract between a Brain and its environment.

    Implementors must supply sensory input and consume motor output.  All other
    hooks are optional — the default implementations are no-ops that return
    neutral values so that headless / minimal worlds work without override.
    """

    # --- required interface --------------------------------------------------

    @abstractmethod
    def get_sensory_retina(self, num_nodes: int) -> np.ndarray:
        """Return a 1-D float32 array representing the current visual field.

        The array is written into activity[0 : len(array)].  Length may vary
        between ticks but should remain stable across a single run.
        """

    @abstractmethod
    def execute_action(self, motor_idx: int) -> dict:
        """Apply motor_idx to the world and return metabolic consequences.

        Returns:
            dict with keys ``energy`` (float), ``water`` (float),
            ``stress`` (float).  Positive energy/water = consumed resources,
            positive stress = pain added.
        """

    # --- optional interface --------------------------------------------------

    def process_human_input(self, keys_pressed: dict) -> None:
        """Apply keyboard input for a human player present in the world."""

    def pop_world_sound(self) -> np.ndarray:
        """Return and clear the current ambient sound vector (32 bands)."""
        return np.zeros(32, dtype=np.float32)

    def add_sound(self, source_pos, sound_vector: np.ndarray) -> None:
        """Inject a sound event at source_pos into the world sound buffer."""

    def human_interact(self) -> None:
        """Trigger a direct human→agent interaction event."""

    @property
    def last_agent_vocalization(self) -> np.ndarray:
        """Last vocal output produced by the brain (32 bands, echo feedback)."""
        return np.zeros(32, dtype=np.float32)

    @last_agent_vocalization.setter
    def last_agent_vocalization(self, value: np.ndarray) -> None:
        pass

    def receive_vocalization(self, vec: np.ndarray) -> None:  # noqa: ARG002
        """Receive the brain's vocal output for echo feedback or social cues."""

    def get_distance_to_human(self) -> float:
        """Manhattan distance from agent to human player. Returns inf if no human."""
        return float('inf')

    def get_homeostatic_signals(self) -> dict:
        """Return normalized deficit signals keyed by sensory channel name.

        Convention: 1.0 = maximum distress/deficit, 0.0 = optimal.

        Keys must match channel names registered in the Brain's sensory_layout.
        The brain injects each value directly into the corresponding sensory
        channel slice every tick — no hardcoding of hunger/thirst/pain in brain.py.

        Example return value::

            {'hunger': 0.7, 'thirst': 0.3, 'pain': 0.1}
        """
        return {}

    def is_alive(self) -> bool:
        """Return False to trigger death, save deletion, and loop exit."""
        return True
