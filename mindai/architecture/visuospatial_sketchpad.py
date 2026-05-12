"""Visuospatial Sketchpad — spatial working memory.

Biological basis (Baddeley 1986, 2000; Logie 1995):
  Component of working memory specialised for holding and manipulating
  spatial and visual information.

  Neural substrate:
    Right hemisphere parietal cortex (posterior parietal, area 7a/LIP):
      spatial layout, object positions, egocentric frames.
    Right dorsolateral PFC (area 46): maintenance and manipulation.
    Visual areas V3A, V4, MT/V5: feature retention.

  Two sub-components (Logie 1995):
    1. Visual cache: stores static visual form/colour information
       (right occipital → right PFC, ~2 seconds capacity).
    2. Inner scribe: processes spatial and movement information;
       also drives mental rotation.

  Capacity: ~3–4 visual objects (Luck & Vogel 1997).
  Interference: spatial secondary tasks disrupt it; verbal tasks do not.

  Link to navigation:
    Hippocampal place cells + parietal spatial map → allocentric navigation.
    Spatial working memory: the parietal "mental scratchpad" for route planning.

Implementation:
  Maintains a sliding buffer of recent spatial frames.
  Each frame: centroid positions of active visual clusters.
  On update: computes spatial change (optical flow analog).
  Output: spatial_change (motion energy), spatial_load (how full the buffer is).
"""

from __future__ import annotations
import numpy as np
from collections import deque


_CAPACITY = 4   # object slots (Luck & Vogel 1997)
_DECAY    = 0.9  # per-tick retention (~2 s at 10 Hz → 20 ticks × 0.9^20 ≈ 0.12)


class VisuospatialSketchpad:

    def __init__(self, vision_width: int = 24, vision_height: int = 24):
        self.vision_width  = vision_width
        self.vision_height = vision_height

        # Circular buffer of recent spatial frames (object centroids as 2D coords)
        self._frame_buffer: deque = deque(maxlen=_CAPACITY)

        # Current retained spatial pattern (decays over time)
        self.retained: np.ndarray = np.zeros(
            (vision_height, vision_width), dtype=np.float32)

        # Output signals
        self.spatial_change: float = 0.0   # motion energy across frames
        self.spatial_load:   float = 0.0   # how full the buffer is

    def update(self, visual_frame: np.ndarray) -> dict:
        """Retain new visual frame in spatial working memory.

        visual_frame: 1D array (vision_width × vision_height channels, or any
                      flat visual input).  Reshaped to 2D grid.

        Returns:
          spatial_change : float [0,1] — spatial motion/change across buffer
          spatial_load   : float [0,1] — buffer fullness
          retained       : np.ndarray — retained spatial pattern
        """
        n = self.vision_width * self.vision_height
        if len(visual_frame) >= n:
            frame = visual_frame[:n].reshape(self.vision_height, self.vision_width)
        else:
            frame = np.zeros((self.vision_height, self.vision_width), dtype=np.float32)
            flat = visual_frame[:len(visual_frame)]
            frame.flat[:len(flat)] = flat

        # Decay retained pattern and add new frame
        self.retained = np.clip(self.retained * _DECAY + frame * (1.0 - _DECAY), 0.0, 1.0)

        # Compute spatial change vs previous frame
        if self._frame_buffer:
            prev = self._frame_buffer[-1]
            diff = np.abs(frame - prev)
            self.spatial_change = float(np.clip(np.mean(diff) * 10.0, 0.0, 1.0))
        else:
            self.spatial_change = 0.0

        self._frame_buffer.append(frame.copy())
        self.spatial_load = float(len(self._frame_buffer) / _CAPACITY)

        return {
            'spatial_change': self.spatial_change,
            'spatial_load':   self.spatial_load,
            'retained':       self.retained,
        }
