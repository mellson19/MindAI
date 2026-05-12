import math
import numpy as np


class BiologicalClock:

    def __init__(self, cycle_length_ticks: int = 2400):
        self.cycle_length = cycle_length_ticks
        self.tick_counter = 0
        self.adenosine    = 0.0
        self.melatonin    = 0.0
        self.is_awake     = True

        # Cortisol Awakening Response (Clow 2010): HPA anticipatory rise
        # 20-30 min before habitual wake time
        self.car_cortisol_boost:   float = 0.0
        self._car_ticks_remaining: int   = 0

    def trigger_cortisol_awakening_response(self) -> None:
        self._car_ticks_remaining = 30

    def update_clock(self, energy_spent: float) -> bool:
        self.tick_counter += 1
        self.adenosine += energy_spent * 0.01
        time_of_day = self.tick_counter % self.cycle_length / self.cycle_length
        self.melatonin = (math.sin(time_of_day * 2 * math.pi - math.pi / 2) + 1.0) / 2.0
        sleep_pressure = self.adenosine * 0.5 + self.melatonin * 0.5

        if self.is_awake and sleep_pressure > 0.8:
            self.is_awake = False
            self._trigger_sleep_onset()

        # CAR: fire when approaching wake threshold from sleep side
        if not self.is_awake and sleep_pressure < 0.3 and self._car_ticks_remaining == 0:
            self.trigger_cortisol_awakening_response()

        # Decay CAR boost linearly over 30 ticks
        if self._car_ticks_remaining > 0:
            self._car_ticks_remaining -= 1
            self.car_cortisol_boost = 0.15 * (self._car_ticks_remaining / 30.0)
        else:
            self.car_cortisol_boost = 0.0

        # NOTE: wake decision is now owned by SleepCycle._maybe_advance_phase()
        # (natural wake only at REM exit). The clock only signals sleep onset;
        # brain.py calls clock.is_awake = True and clock.adenosine = 0.0 on wake.
        return self.is_awake

    @property
    def melatonin_suppression_factor(self) -> float:
        """Melatonin > 0.4 gently suppresses awake ACh baseline."""
        return float(np.clip((self.melatonin - 0.4) / 0.6, 0.0, 1.0))

    def _trigger_sleep_onset(self) -> None:
        pass
