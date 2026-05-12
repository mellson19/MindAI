"""Sensation curves — map raw signal [0,1] to felt intensity [0,1].

Biological basis
----------------
Sensory neurons don't respond linearly to stimulus intensity.  Weber-Fechner
law and Stevens' power law describe how perceived intensity scales with physical
magnitude.  Pain, hunger, and thirst each have distinct psychophysical curves.

Available curves
----------------
linear      : f(x) = x                  — proportional (no distortion)
quadratic   : f(x) = x²                 — mild signals barely felt; severe = intense
sqrt        : f(x) = √x                 — small changes very noticeable; diminishing returns
power(n)    : f(x) = xⁿ                 — generalised Stevens' law
sigmoid(k)  : f(x) = 1/(1+e^(-k(x-0.5))) — sharp threshold; neutral midpoint
log_scaled  : f(x) = log(1+9x)/log(10)  — loud sounds, Weber-Fechner
threshold(t): 0 below t; linear above    — nociceptor firing threshold
step(t)     : 0 below t; 1 above         — binary (rare in biology, use sparingly)

All curves accept and return values in [0, 1].
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

CurveFn = Callable[[float], float]


# ---------------------------------------------------------------------------
# Curve implementations
# ---------------------------------------------------------------------------

def linear(x: float) -> float:
    """Proportional — no perceptual distortion."""
    return float(np.clip(x, 0.0, 1.0))


def quadratic(x: float) -> float:
    """Stevens' exponent ~2: mild deficit barely felt, severe = overwhelming.

    Biological match: hunger, thirst — mild deprivation is comfortable,
    severe deprivation triggers urgent action.
    """
    return float(np.clip(x * x, 0.0, 1.0))


def sqrt(x: float) -> float:
    """Stevens' exponent ~0.5: small changes very noticeable.

    Biological match: sound loudness, skin temperature.
    """
    return float(np.clip(math.sqrt(max(0.0, x)), 0.0, 1.0))


def power(n: float) -> CurveFn:
    """Generalised Stevens' power law.

    Parameters
    ----------
    n:
        Exponent.  n>1 → suppresses mild signals (pain gate).
                   n<1 → amplifies mild signals (touch sensitivity).

    Examples
    --------
    ``power(1.5)`` — nociceptor response curve (mild injury barely felt,
    severe injury overwhelming).  Used in MinecraftWorld for health→pain.
    ``power(0.67)`` — brightness perception (CIE lightness formula).
    """
    def _curve(x: float) -> float:
        return float(np.clip(x ** n, 0.0, 1.0))
    _curve.__name__ = f'power({n})'
    return _curve


def sigmoid(sharpness: float = 10.0, midpoint: float = 0.5) -> CurveFn:
    """Logistic curve — sharp transition around midpoint.

    Biological match: action potential threshold; all-or-nothing firing.
    High sharpness → binary-like.  Low sharpness → smooth S-curve.

    Parameters
    ----------
    sharpness:
        Steepness of the transition (default 10).
    midpoint:
        Inflection point in [0, 1] (default 0.5).
    """
    def _curve(x: float) -> float:
        return float(np.clip(1.0 / (1.0 + math.exp(-sharpness * (x - midpoint))),
                             0.0, 1.0))
    _curve.__name__ = f'sigmoid(k={sharpness}, m={midpoint})'
    return _curve


def log_scaled(x: float) -> float:
    """Weber-Fechner law: f(x) = log(1+9x) / log(10).

    Biological match: auditory loudness, olfactory intensity.
    Rapid rise at low values, slow saturation at high values.
    """
    return float(np.clip(math.log(1.0 + 9.0 * max(0.0, x)) / math.log(10.0),
                         0.0, 1.0))


def threshold(t: float) -> CurveFn:
    """Zero below threshold t; linear from t→1 mapped to 0→1 above.

    Biological match: nociceptor activation threshold — no pain signal is
    sent to the brain until stimulus exceeds the firing threshold.
    """
    def _curve(x: float) -> float:
        if x <= t:
            return 0.0
        return float(np.clip((x - t) / max(1.0 - t, 1e-9), 0.0, 1.0))
    _curve.__name__ = f'threshold({t})'
    return _curve


def step(t: float) -> CurveFn:
    """Binary step: 0 below t, 1 above.  Use sparingly — rarely biological."""
    def _curve(x: float) -> float:
        return 1.0 if x >= t else 0.0
    _curve.__name__ = f'step({t})'
    return _curve


# ---------------------------------------------------------------------------
# Resolve curve from name or callable
# ---------------------------------------------------------------------------

_NAMED: dict[str, CurveFn] = {
    'linear':     linear,
    'quadratic':  quadratic,
    'sqrt':       sqrt,
    'log':        log_scaled,
    'log_scaled': log_scaled,
}


def resolve(curve) -> CurveFn:
    """Accept a curve name (str), a callable, or None (→ linear)."""
    if curve is None:
        return linear
    if callable(curve):
        return curve
    if isinstance(curve, str):
        if curve not in _NAMED:
            raise ValueError(
                f"Unknown curve name '{curve}'. "
                f"Available: {list(_NAMED)}. "
                f"Or pass a callable from mindai.feels.curves.")
        return _NAMED[curve]
    raise TypeError(f"curve must be str or callable, got {type(curve)}")
