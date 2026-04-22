import numpy as np

class EndocrineSystem:

    def __init__(self, auditory_cortex_size=32):
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.noradrenaline = 0.1
        self.cortisol = 0.0
        self.oxytocin = 0.0
        self.endorphins = 0.0
        self.adrenaline = 0.0
        self.stress_accumulator = 0.0
        self.boredom = 0.0
        self.sound_to_dopamine_weights = np.zeros(auditory_cortex_size)
        self.sound_to_fear_weights = np.zeros(auditory_cortex_size)

    def update_state(self, global_arousal: float, layer23_error_spikes: float, raw_pain_signal: float, energy_ratio: float, water_ratio: float, auditory_spikes: np.ndarray=None):
        learned_dopamine = 0.0
        learned_fear = 0.0
        if auditory_spikes is not None:
            learned_dopamine = np.dot(auditory_spikes, self.sound_to_dopamine_weights)
            learned_fear = np.dot(auditory_spikes, self.sound_to_fear_weights)
            innate_dopamine_trigger = 1.0 if energy_ratio + water_ratio > 1.5 else 0.0
            if innate_dopamine_trigger > 0:
                self.sound_to_dopamine_weights += auditory_spikes * 0.01
            if raw_pain_signal > 0.1:
                self.sound_to_fear_weights += auditory_spikes * 0.05
            self.sound_to_dopamine_weights *= 0.999
            self.sound_to_fear_weights *= 0.999
        gut_satisfaction = (energy_ratio + water_ratio) / 2.0
        vagal_tone = np.clip(gut_satisfaction, 0.0, 1.0)
        self.serotonin = self.serotonin * 0.9 + vagal_tone * 0.1
        felt_pain = raw_pain_signal * (1.0 - self.endorphins * 0.8)
        fear_dampener = 1.0 - self.oxytocin * 0.8
        panic_signal = layer23_error_spikes * 0.1 + felt_pain + learned_fear
        self.noradrenaline = np.clip(self.noradrenaline * 0.9 + panic_signal * fear_dampener * (1.0 - vagal_tone), 0.1, 1.0)
        if felt_pain > 0.5 and self.noradrenaline > 0.6:
            self.adrenaline = min(1.0, self.adrenaline + 0.3)
        else:
            self.adrenaline *= 0.9
        if self.noradrenaline > 0.7:
            self.stress_accumulator += 0.01 * (1.0 - self.oxytocin)
        else:
            self.stress_accumulator = max(0.0, self.stress_accumulator - 0.005)
        self.cortisol = np.clip(self.cortisol * 0.999 + self.stress_accumulator * 0.001, 0.0, 1.0)
        if learned_dopamine > 0.5:
            self.dopamine = min(1.0, self.dopamine + 0.2)
        else:
            self.dopamine = max(0.1, self.dopamine * 0.98)
        self.endorphins *= 0.95
        self.oxytocin = max(0.0, self.oxytocin * 0.998)
        is_safe_and_satiated = vagal_tone > 0.7 and felt_pain < 0.1 and (layer23_error_spikes < 2.0)
        if is_safe_and_satiated:
            self.boredom = min(1.0, self.boredom + 0.005)
        else:
            self.boredom *= 0.8

    def trigger_social_bonding(self):
        self.oxytocin = min(1.0, self.oxytocin + 0.1)
        self.dopamine = min(1.0, self.dopamine + 0.05)
        self.cortisol = max(0.0, self.cortisol - 0.02)

    def trigger_endorphin_rush(self):
        self.endorphins = 1.0
        self.dopamine = 1.0
        self.adrenaline = 0.0
        self.noradrenaline *= 0.2

    def get_plasticity_multiplier(self) -> float:
        learning_capacity = max(0.0, 1.0 - self.cortisol)
        return (self.dopamine * 1.5 + self.serotonin * 0.5) * learning_capacity * (1.0 + self.endorphins)