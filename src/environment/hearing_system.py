import numpy as np
import sounddevice as sd

class Cochlea:

    def __init__(self, num_bands=32, sample_rate=16000, chunk_size=1024):
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.current_spectrum = np.zeros(num_bands)
        self.is_listening = False
        try:
            self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize=self.chunk_size, callback=self.audio_callback)
            self.stream.start()
            print('    [БИОЛОГИЯ] Кохлеарный аппарат (Микрофон) успешно подключен.')
        except Exception as e:
            print(f'    [ОШИБКА] Микрофон не найден. ИИ будет глухим: {e}')
            self.stream = None

    def audio_callback(self, indata, frames, time, status):
        if not self.is_listening:
            self.current_spectrum.fill(0.0)
            return
        audio_window = indata[:, 0]
        fft_complex = np.fft.rfft(audio_window)
        fft_mags = np.abs(fft_complex)
        fft_mags = fft_mags[1:]
        bin_size = len(fft_mags) // self.num_bands
        new_spectrum = np.zeros(self.num_bands)
        for i in range(self.num_bands):
            band_mag = np.mean(fft_mags[i * bin_size:(i + 1) * bin_size])
            new_spectrum[i] = np.clip(band_mag * 5.0, 0.0, 1.0)
        self.current_spectrum = new_spectrum

    def get_auditory_nerve_signal(self) -> np.ndarray:
        return self.current_spectrum.copy()