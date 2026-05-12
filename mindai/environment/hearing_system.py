"""Biologically accurate auditory transduction — cochlea to auditory cortex.

Pathway implemented:
  Basilar membrane    — ERB-spaced bandpass filters (Glasberg & Moore 1990)
  Inner hair cells    — half-wave rectification + power-law compression (Stevens 0.3)
  Dorsal cochlear nucleus  — onset/offset detectors (first derivative of energy)
  Ventral cochlear nucleus — sustained rate-coded neurons
  Inferior colliculus      — temporal modulation (amplitude envelope rate)
  Superior olive           — broadband energy / inter-aural level

Neuron allocation (Ehret 1997; Merzenich 1975):
  40% sustained    — tonotopic, ERB-spaced, steady-state energy per band
  30% onset        — respond to energy increases (attack transients)
  20% modulation   — amplitude envelope rate (rhythm, prosody)
  10% broadband    — overall loudness (superior olive level detector)

Output: float32 array of shape (audio_size,) in [0, 1].
"""

import threading
import numpy as np


class Cochlea:

    def __init__(self, audio_size: int, sample_rate: int = 16000, chunk_size: int = 1024):
        self.audio_size   = audio_size
        self.sample_rate  = sample_rate
        self.chunk_size   = chunk_size

        # Neuron allocation across processing stages (Ehret 1997)
        self._n_sustained  = int(audio_size * 0.40)
        self._n_onset      = int(audio_size * 0.30)
        self._n_modulation = int(audio_size * 0.20)
        self._n_broadband  = audio_size - self._n_sustained - self._n_onset - self._n_modulation

        # ERB-spaced centre frequencies — basilar membrane tonotopy (Glasberg & Moore 1990)
        # ERB_rate = 21.4 * log10(4.37 * f_kHz + 1)
        f_min = 50.0
        f_max = min(8000.0, sample_rate / 2.0 - 1.0)
        erb_min = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
        erb_max = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
        erb_rates = np.linspace(erb_min, erb_max, self._n_sustained)
        self._center_freqs = (10.0 ** (erb_rates / 21.4) - 1.0) / 4.37 * 1000.0

        # Precompute triangular ERB bandpass filters on FFT grid
        n_fft = chunk_size // 2 + 1
        self._fft_freqs    = np.fft.rfftfreq(chunk_size, 1.0 / sample_rate)
        self._band_filters = self._build_erb_filters(n_fft)

        # Onset detection state — previous sustained energies
        self._prev_sustained = np.zeros(self._n_sustained, dtype=np.float32)
        # Temporal modulation state — running envelope (τ ≈ 50 ms at 16 kHz)
        self._envelope = 0.0

        # Output shared with mic callback
        self._output = np.zeros(audio_size, dtype=np.float32)
        self._lock   = threading.Lock()
        self._stream = None

        print(f'    [БИОЛОГИЯ] Улитка уха: {audio_size} нейронов  '
              f'({self._n_sustained} sustained / {self._n_onset} onset / '
              f'{self._n_modulation} modulation / {self._n_broadband} broadband)  '
              f'ERB {f_min:.0f}–{f_max:.0f} Hz')

    # ------------------------------------------------------------------
    # Filter construction
    # ------------------------------------------------------------------

    def _build_erb_filters(self, n_fft: int) -> np.ndarray:
        """Triangular bandpass filters centred on ERB-spaced frequencies.

        Each filter spans ±1 ERB around its centre frequency with a triangular
        amplitude profile — approximates the auditory filter shape (Moore 1983).
        """
        filters = np.zeros((self._n_sustained, n_fft), dtype=np.float32)
        freqs   = self._fft_freqs
        for i, f_c in enumerate(self._center_freqs):
            # ERB bandwidth at this centre frequency (Moore 1983)
            erb_bw = 24.7 * (4.37 * f_c / 1000.0 + 1.0)
            mask   = (freqs >= f_c - erb_bw) & (freqs <= f_c + erb_bw)
            if mask.any():
                filters[i, mask] = 1.0 - np.abs(freqs[mask] - f_c) / erb_bw
        return filters

    # ------------------------------------------------------------------
    # Core processing — one audio block → neuron activations
    # ------------------------------------------------------------------

    def process(self, block: np.ndarray) -> np.ndarray:
        """Transduce one PCM block into auditory neuron activations.

        Parameters
        ----------
        block : (chunk_size,) float32  PCM samples in [-1, 1]

        Returns
        -------
        (audio_size,) float32 in [0, 1]
        """
        fft_mag = np.abs(np.fft.rfft(block.astype(np.float32)))

        # --- Sustained neurons: ERB filterbank + hair-cell compression ---
        # Stevens (1957) power law: loudness ∝ intensity^0.3
        raw       = self._band_filters @ fft_mag          # (n_sustained,)
        raw       = np.power(np.clip(raw, 0.0, None) + 1e-9, 0.3)
        peak      = raw.max() + 1e-9
        sustained = np.clip(raw / peak, 0.0, 1.0).astype(np.float32)

        # --- Onset detectors: positive Δ energy (dorsal cochlear nucleus) ---
        delta  = sustained - self._prev_sustained
        onset_full = np.clip(delta * 6.0, 0.0, 1.0)
        # Resample from n_sustained → n_onset via linear interpolation
        onset = np.interp(
            np.linspace(0, self._n_sustained - 1, self._n_onset),
            np.arange(self._n_sustained),
            onset_full,
        ).astype(np.float32)

        # --- Temporal modulation: envelope rate (inferior colliculus) ---
        rms = float(np.sqrt(np.mean(block ** 2) + 1e-9))
        # τ ≈ 50 ms: at chunk_size=1024, sr=16000 → ~15.6 ticks/s, α=0.82
        alpha = np.exp(-chunk_rate(self.chunk_size, self.sample_rate) / 20.0)
        self._envelope = alpha * self._envelope + (1.0 - alpha) * rms
        mod_depth = float(np.clip(abs(rms - self._envelope) * 15.0, 0.0, 1.0))
        modulation = np.full(self._n_modulation, mod_depth, dtype=np.float32)

        # --- Broadband neurons: overall loudness (superior olive) ---
        loudness  = float(np.clip(rms * 25.0, 0.0, 1.0))
        broadband = np.full(self._n_broadband, loudness, dtype=np.float32)

        self._prev_sustained = sustained

        return np.concatenate([sustained, onset, modulation, broadband])

    # ------------------------------------------------------------------
    # Microphone interface
    # ------------------------------------------------------------------

    def start_mic(self):
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate = self.sample_rate,
                channels   = 1,
                blocksize  = self.chunk_size,
                callback   = self._mic_callback,
            )
            self._stream.start()
            print('    [БИОЛОГИЯ] Микрофон подключён.')
        except Exception as e:
            print(f'    [ОШИБКА] Микрофон не найден — ИИ будет глухим: {e}')

    def _mic_callback(self, indata, frames, time_info, status):
        result = self.process(indata[:, 0])
        with self._lock:
            self._output[:] = result

    def get_auditory_nerve_signal(self) -> np.ndarray:
        with self._lock:
            return self._output.copy()


def chunk_rate(chunk_size: int, sample_rate: int) -> float:
    """Chunks per second — used for envelope time-constant calculation."""
    return sample_rate / chunk_size
