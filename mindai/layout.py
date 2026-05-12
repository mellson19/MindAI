from __future__ import annotations


class SensoryLayout:
    """Maps channel names to neuron index ranges.

    Each channel is either auto-assigned (contiguous) or placed at an
    explicit absolute position (``(start, end)`` tuple).

    Channel size can be specified as:
    * ``int``   — absolute number of neurons
    * ``float`` — fraction of total ``num_neurons`` (e.g. ``0.01`` = 1%)
    * ``(start, end)`` — absolute indices (may overlap, e.g. mirror_neurons)

    Fractional sizes are resolved at construction time via ``num_neurons``.
    This mirrors cortical scaling: as total neuron count grows, each area
    receives proportionally more processing neurons — exactly as in biological
    allometric scaling (Finlay & Darlington 1995).

    Example::

        layout = SensoryLayout.from_channels(
            num_neurons=12000,
            vision_size=2880,
            sensory={
                'hunger': 0.005,   # 0.5% → 60 neurons at 12K, 500 at 100K
                'pain':   0.01,    # 1.0% → 120 neurons at 12K, 1000 at 100K
                'audio':  32,      # absolute — matches audio hardware channels
            },
            motor={
                'motor':          0.015,
                'mirror_neurons': (0, 10),
            },
        )
    """

    def __init__(self, channels: dict[str, tuple[int, int]]) -> None:
        self._ch = channels

    def start(self, channel: str) -> int:
        return self._ch[channel][0]

    def end(self, channel: str) -> int:
        return self._ch[channel][1]

    def size(self, channel: str) -> int:
        s, e = self._ch[channel]
        return e - s

    def slice(self, channel: str) -> slice:
        s, e = self._ch[channel]
        return slice(s, e)

    def has(self, channel: str) -> bool:
        return channel in self._ch

    def min_neurons(self) -> int:
        """Minimum num_neurons required to hold every channel."""
        return max(e for _, e in self._ch.values())

    @classmethod
    def from_channels(
        cls,
        vision_size: int,
        sensory: dict[str, int | float | tuple[int, int]],
        motor:   dict[str, int | float | tuple[int, int]],
        num_neurons: int = 0,
    ) -> SensoryLayout:
        """Build a layout from channel definitions.

        Parameters
        ----------
        vision_size:
            Number of vision neurons (always placed first at index 0).
        sensory:
            Non-vision sensory channels.  Values: int (absolute), float
            (fraction of num_neurons), or (start,end) tuple (absolute).
        motor:
            Motor channels.  Same value types as sensory.
        num_neurons:
            Total neuron count — required when any channel size is a float.
        """
        def resolve(spec: int | float | tuple, num_neurons: int) -> int | tuple:
            if isinstance(spec, tuple):
                return spec
            if isinstance(spec, float):
                return max(4, int(round(spec * num_neurons)))
            return int(spec)

        channels: dict[str, tuple[int, int]] = {}
        offset = 0

        channels['vision'] = (0, vision_size)
        offset = vision_size

        for name, spec in sensory.items():
            r = resolve(spec, num_neurons)
            if isinstance(r, tuple):
                channels[name] = r
                offset = max(offset, r[1])
            else:
                channels[name] = (offset, offset + r)
                offset += r

        motor_start = offset
        for name, spec in motor.items():
            r = resolve(spec, num_neurons)
            if isinstance(r, tuple):
                rel_start, rel_end = r
                channels[name] = (motor_start + rel_start, motor_start + rel_end)
                offset = max(offset, motor_start + rel_end)
            else:
                channels[name] = (offset, offset + r)
                offset += r

        if 'motor' not in motor:
            channels['motor'] = (motor_start, offset)

        return cls(channels)
