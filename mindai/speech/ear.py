"""Ear — speech → text via faster-whisper.

User talks → microphone (browser via WebRTC) → 16 kHz PCM → Whisper
→ text → tokenizer → brain input queue.

faster-whisper is preferred (4-5× faster than openai-whisper, lower
VRAM via CTranslate2). Falls back to openai-whisper if not installed.

Languages: configured English-first by default since the project corpus
is English. Set language=None for auto-detect.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

try:
    from faster_whisper import WhisperModel
    _HAVE_FW = True
except ImportError:
    _HAVE_FW = False

try:
    import whisper as _openai_whisper
    _HAVE_OAI = True
except ImportError:
    _HAVE_OAI = False


class Ear:
    """Speech-to-text. Default: faster-whisper 'base' model, English."""

    def __init__(
        self,
        model_size: str = 'base',
        language:   str | None = 'en',
        device:     str = 'auto',
    ):
        self.language   = language
        self._model_size = model_size
        self._model = None
        self._backend = None

        if _HAVE_FW:
            try:
                dev = device
                if dev == 'auto':
                    try:
                        import torch
                        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                    except ImportError:
                        dev = 'cpu'
                self._model   = WhisperModel(model_size, device=dev,
                                             compute_type='int8' if dev == 'cpu' else 'float16')
                self._backend = 'faster-whisper'
            except Exception as e:
                print(f'>>> Ear faster-whisper init failed: {e}')

        if self._model is None and _HAVE_OAI:
            try:
                self._model   = _openai_whisper.load_model(model_size)
                self._backend = 'openai-whisper'
            except Exception as e:
                print(f'>>> Ear openai-whisper init failed: {e}')

        if self._model is None:
            print('>>> Ear: install faster-whisper (preferred) or openai-whisper')

    @property
    def available(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------

    def transcribe(self, pcm_or_path) -> str:
        """Transcribe int16 PCM array (16 kHz) OR a path to a wav/mp3."""
        if not self.available:
            return ''

        # Normalise input → float32 mono 16 kHz
        if isinstance(pcm_or_path, (str, Path)):
            audio = _load_audio(str(pcm_or_path), 16000)
        else:
            pcm = np.asarray(pcm_or_path)
            if pcm.dtype == np.int16:
                audio = pcm.astype(np.float32) / 32768.0
            else:
                audio = pcm.astype(np.float32)

        if audio.size == 0:
            return ''

        try:
            if self._backend == 'faster-whisper':
                segments, _ = self._model.transcribe(
                    audio, language=self.language, beam_size=1)
                return ' '.join(s.text for s in segments).strip()
            else:
                result = self._model.transcribe(
                    audio, language=self.language or 'en')
                return result.get('text', '').strip()
        except Exception as e:
            print(f'>>> Ear transcription failed: {e}')
            return ''


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio file as float32 mono at target_sr."""
    p = Path(path)
    if p.suffix.lower() == '.wav':
        with wave.open(str(p), 'rb') as w:
            sr   = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        audio = data.astype(np.float32) / 32768.0
    else:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(str(p))
            seg = seg.set_channels(1).set_sample_width(2).set_frame_rate(target_sr)
            audio = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            sr = target_sr
        except Exception:
            return np.zeros(0, dtype=np.float32)

    if sr != target_sr:
        n_dst = int(len(audio) * target_sr / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n_dst),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio
