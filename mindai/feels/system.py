"""FeelingSystem — collection of Feel objects attached to a Brain.

The FeelingSystem is the bridge between the world and the brain's sensory
channels.  It holds named feelings, updates their raw values from world signals
each tick, and injects the post-curve sensations into the sensory array.

Architecture role::

    World.get_homeostatic_signals()
            ↓ raw signals dict
    FeelingSystem.update(signals)
            ↓ applies psychophysical curves
    FeelingSystem.inject(raw_array, layout)
            ↓ writes to sensory neuron channels
    Brain neural computation

This means the brain is completely agnostic to what "hunger" or "pain" means
in the physical world — it only receives a normalized activation pattern that
it learns to respond to via Hebbian STDP, exactly as the biological brain does.

Usage::

    from mindai.feels import FeelingSystem, Feel
    from mindai.feels.curves import power, quadratic

    feels = FeelingSystem()
    feels.add(Feel('pain',   channel='pain',   curve=power(1.5)))
    feels.add(Feel('hunger', channel='hunger', curve=quadratic))
    feels.add(Feel('cold',   channel='thirst', curve='sqrt'))   # custom mapping

    brain = Brain(...)
    brain.attach(feels)
    brain.run(world)

    # Or set manually from user code (e.g. in a custom World subclass):
    feels['pain'].set(1.0 - player_health / max_health)
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from mindai.feels.feel import Feel


class FeelingSystem:
    """Container and injector for Feel objects.

    Parameters
    ----------
    feels:
        Optional initial list of Feel objects (can also be added later).
    """

    def __init__(self, feels: list[Feel] | None = None) -> None:
        self._feels: dict[str, Feel] = {}
        if feels:
            for f in feels:
                self.add(f)

    # ------------------------------------------------------------------
    # Building the system
    # ------------------------------------------------------------------

    def add(self, feel: Feel) -> 'FeelingSystem':
        """Add a Feel to this system.  Returns self for chaining."""
        self._feels[feel.name] = feel
        return self

    def __getitem__(self, name: str) -> Feel:
        """Access a Feel by name: ``feels['pain'].set(0.7)``."""
        return self._feels[name]

    def __contains__(self, name: str) -> bool:
        return name in self._feels

    def __iter__(self) -> Iterator[Feel]:
        return iter(self._feels.values())

    def __len__(self) -> int:
        return len(self._feels)

    # ------------------------------------------------------------------
    # Tick interface (called by Brain each tick)
    # ------------------------------------------------------------------

    def update(self, signals: dict) -> None:
        """Update all feel raw values from a world signals dict.

        Each Feel with a ``source`` callable reads its own signal.
        Feels without a source only update if their name is a key in signals.

        Parameters
        ----------
        signals:
            Dict from ``world.get_homeostatic_signals()`` — keys are signal
            names (e.g. 'pain', 'hunger'), values are raw [0, 1] floats.
        """
        for feel in self._feels.values():
            feel.update(signals)

    def inject(self, raw_array: np.ndarray, layout) -> None:
        """Write post-curve sensation values into the sensory neuron array.

        Parameters
        ----------
        raw_array:
            Mutable numpy array of shape (num_neurons,).  Modified in-place.
        layout:
            ``SensoryLayout`` instance from the Brain.
        """
        for feel in self._feels.values():
            if layout.has(feel.channel):
                raw_array[layout.slice(feel.channel)] = feel.sensation

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, float]:
        """Return current sensation values keyed by feel name."""
        return {name: feel.sensation for name, feel in self._feels.items()}

    def wellbeing(self) -> float:
        """Overall wellbeing [0=max distress, 1=perfect].

        Computed as 1 minus the mean of all sensation values.
        A brain with all feels at zero is in optimal condition.
        """
        if not self._feels:
            return 1.0
        return float(np.clip(
            1.0 - np.mean([f.sensation for f in self._feels.values()]),
            0.0, 1.0
        ))

    def dominant(self) -> Feel | None:
        """Return the Feel with the highest current sensation, or None."""
        if not self._feels:
            return None
        return max(self._feels.values(), key=lambda f: f.sensation)

    def __repr__(self) -> str:
        parts = [f'{f.name}={f.sensation:.2f}' for f in self._feels.values()]
        return f"FeelingSystem({', '.join(parts)})"
