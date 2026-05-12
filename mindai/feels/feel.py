"""Feel — a single named sensation with a psychophysical curve.

A Feel wraps a raw signal [0, 1] (set by the world or user code) and exposes
a ``sensation`` value [0, 1] after applying a psychophysical curve.

This separates *what is happening in the world* from *how the brain perceives it*.
The brain never receives raw "health = 4/20".  It receives the felt sensation:
    raw = 1 - 4/20 = 0.8  →  power(1.5)(0.8) = 0.716 → injected into pain neurons.

Usage examples::

    from mindai.feels import Feel, FeelingSystem
    from mindai.feels.curves import power, quadratic, threshold

    feels = FeelingSystem()

    # Pain: nociceptor curve (mild damage barely felt, severe = overwhelming)
    feels.add(Feel('pain',   channel='pain',   curve=power(1.5)))

    # Hunger: quadratic — mild hunger comfortable, starvation urgent
    feels.add(Feel('hunger', channel='hunger', curve=quadratic))

    # Thirst: logarithmic — even small thirst is noticeable
    feels.add(Feel('thirst', channel='thirst', curve='log'))

    # Pain from Minecraft health
    feels.add(Feel(
        'pain',
        channel='pain',
        curve=power(1.5),
        source=lambda signals: signals.get('pain', 0.0),
    ))

    brain.attach(feels)
    brain.run(world)

    # Or set manually each tick from anywhere:
    feels['pain'].set(1.0 - health / max_health)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mindai.feels.curves import resolve, CurveFn


class Feel:
    """A single perceptual channel: maps a raw signal through a curve.

    Parameters
    ----------
    name:
        Unique identifier (e.g. 'pain', 'hunger', 'cold').
    channel:
        Brain sensory layout channel name to inject into (e.g. 'pain').
        Must match a key in the Brain's ``sensory_layout``.
    curve:
        Psychophysical transfer function.  Accepts:
        - A string name: ``'linear'``, ``'quadratic'``, ``'sqrt'``, ``'log'``
        - A callable ``f(x: float) -> float`` from ``mindai.feels.curves``
        - ``None`` → linear (identity)
    source:
        Optional callable ``(signals: dict) -> float`` that reads the raw
        signal from a world state dict.  If provided, ``FeelingSystem.update()``
        calls it automatically each tick.  If None, set manually via
        ``feel.set(value)``.
    amplitude:
        Scale factor applied after the curve (default 1.0).  Allows tuning
        how strongly this feeling fills its sensory channel.
    """

    def __init__(
        self,
        name:      str,
        channel:   str,
        curve:     object = 'linear',
        source:    Optional[Callable[[dict], float]] = None,
        amplitude: float = 1.0,
    ) -> None:
        self.name      = name
        self.channel   = channel
        self.amplitude = float(amplitude)
        self._curve_fn: CurveFn = resolve(curve)
        self._source   = source
        self._raw:  float = 0.0

    # ------------------------------------------------------------------

    def set(self, value: float) -> None:
        """Manually set the raw signal [0, 1].

        Call this from world code or a connector when no ``source`` callable
        was provided::

            feels['pain'].set(1.0 - health / max_health)
        """
        self._raw = float(np.clip(value, 0.0, 1.0))

    def update(self, signals: dict) -> None:
        """Update raw value from world signals dict (called by FeelingSystem)."""
        if self._source is not None:
            raw = self._source(signals)
            self.set(raw)
        elif self.name in signals:
            self.set(signals[self.name])

    @property
    def raw(self) -> float:
        """Raw input signal before the psychophysical curve."""
        return self._raw

    @property
    def sensation(self) -> float:
        """Felt intensity after applying curve and amplitude scaling."""
        return float(np.clip(self._curve_fn(self._raw) * self.amplitude,
                             0.0, 1.0))

    def __repr__(self) -> str:
        return (f"Feel('{self.name}', channel='{self.channel}', "
                f"raw={self._raw:.3f}, sensation={self.sensation:.3f}, "
                f"curve={self._curve_fn.__name__})")
