import math

class BiologicalClock:

    def __init__(self, cycle_length_ticks: int=2400):
        self.cycle_length = cycle_length_ticks
        self.tick_counter = 0
        self.adenosine = 0.0
        self.melatonin = 0.0
        self.is_awake = True

    def update_clock(self, energy_spent: float):
        self.tick_counter += 1
        self.adenosine += energy_spent * 0.01
        time_of_day = self.tick_counter % self.cycle_length / self.cycle_length
        self.melatonin = (math.sin(time_of_day * 2 * math.pi - math.pi / 2) + 1.0) / 2.0
        sleep_pressure = self.adenosine * 0.5 + self.melatonin * 0.5
        if self.is_awake and sleep_pressure > 0.8:
            self.is_awake = False
            self._trigger_sleep_onset()
        elif not self.is_awake and sleep_pressure < 0.2:
            self.is_awake = True
            self.adenosine = 0.0
        return self.is_awake

    def _trigger_sleep_onset(self):
        pass