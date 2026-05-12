"""VocalApparatus — text→audio synthesis with a fixed VoiceID.

Backend: edge-tts (Microsoft Edge text-to-speech, free, no API key).
Falls back to pyttsx3 (offline, OS-native) if edge-tts unavailable.

Design
------
The brain produces a token stream via motor neurons. AgentWorld decodes
that stream to text. This module turns the text into audio with the
brain's FIXED voice — same pitch/rate/voice every time, defined by
VoiceID.

API kept synchronous for easy integration with the brain loop;
internally a worker thread handles async edge-tts calls.

Output is raw PCM (16 kHz, mono, int16) suitable for both:
  - WebSocket streaming to the browser (Web GUI plays back)
  - File save (.wav)
  - Cochlea re-injection (the brain hears its own voice — phonological loop)
"""

from __future__ import annotations

import asyncio
import io
import threading
from pathlib import Path

import numpy as np

from mindai.speech.voice_id import VoiceID

try:
    import edge_tts
    _HAVE_EDGE = True
except ImportError:
    _HAVE_EDGE = False

try:
    import pyttsx3
    _HAVE_PYTTSX = True
except ImportError:
    _HAVE_PYTTSX = False


class VocalApparatus:
    """Text → audio with a brain's fixed VoiceID."""

    def __init__(self, voice_id: VoiceID, sample_rate: int = 16000):
        self.voice_id   = voice_id
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        if not (_HAVE_EDGE or _HAVE_PYTTSX):
            print('>>> VocalApparatus: install edge-tts or pyttsx3 for speech')

    # ------------------------------------------------------------------
    # Public — synchronous API
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> np.ndarray:
        """Return mono int16 PCM at self.sample_rate.

        Empty array if synthesis unavailable / empty text.
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16)

        if _HAVE_EDGE:
            try:
                return self._edge_synthesize(text)
            except Exception as e:
                print(f'>>> VocalApparatus edge-tts failed: {e}')

        if _HAVE_PYTTSX:
            try:
                return self._pyttsx_synthesize(text)
            except Exception as e:
                print(f'>>> VocalApparatus pyttsx3 failed: {e}')

        return np.zeros(0, dtype=np.int16)

    def synthesize_to_file(self, text: str, path: str | Path) -> Path:
        """Save synthesised audio to a .wav file."""
        pcm = self.synthesize(text)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        import wave
        with wave.open(str(out), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm.tobytes())
        return out

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _edge_synthesize(self, text: str) -> np.ndarray:
        """edge-tts → MP3 bytes → PCM via miniaudio/pydub."""
        async def _gen() -> bytes:
            comm = edge_tts.Communicate(
                text,
                voice = self.voice_id.base_voice,
                pitch = self.voice_id.edge_tts_pitch,
                rate  = self.voice_id.edge_tts_rate,
            )
            buf = io.BytesIO()
            async for chunk in comm.stream():
                if chunk['type'] == 'audio':
                    buf.write(chunk['data'])
            return buf.getvalue()

        with self._lock:
            mp3_bytes = asyncio.run(_gen())

        return _mp3_to_pcm(mp3_bytes, self.sample_rate)

    def _pyttsx_synthesize(self, text: str) -> np.ndarray:
        """pyttsx3 fallback — saves to temp wav then loads."""
        import tempfile, wave
        with self._lock:
            engine = pyttsx3.init()
            engine.setProperty('rate', int(180 * self.voice_id.rate))
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                path = tmp.name
            engine.save_to_file(text, path)
            engine.runAndWait()
        with wave.open(path, 'rb') as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            sr  = w.getframerate()
        Path(path).unlink(missing_ok=True)
        if sr != self.sample_rate:
            pcm = _resample(pcm, sr, self.sample_rate)
        return pcm


# ---------------------------------------------------------------------------
# Audio decoding helpers
# ---------------------------------------------------------------------------

def _mp3_to_pcm(mp3_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode MP3 bytes to mono int16 PCM at target_sr.

    Tries miniaudio → pydub → av in that order.
    """
    # miniaudio (fastest, pure decode)
    try:
        import miniaudio
        decoded = miniaudio.decode(mp3_bytes,
                                   output_format=miniaudio.SampleFormat.SIGNED16,
                                   nchannels=1, sample_rate=target_sr)
        return np.array(decoded.samples, dtype=np.int16)
    except Exception:
        pass

    # pydub (uses ffmpeg)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format='mp3')
        seg = seg.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        return np.frombuffer(seg.raw_data, dtype=np.int16)
    except Exception:
        pass

    # av (PyAV) fallback
    try:
        import av
        container = av.open(io.BytesIO(mp3_bytes))
        frames = []
        for frame in container.decode(audio=0):
            frames.append(frame.to_ndarray())
        if not frames:
            return np.zeros(0, dtype=np.int16)
        audio = np.concatenate(frames, axis=-1).astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        # Normalise → int16
        audio = (audio / max(1.0, np.max(np.abs(audio))) * 32000).astype(np.int16)
        # If sample rate differs, resample (very crude)
        sr = container.streams.audio[0].rate
        if sr != target_sr:
            audio = _resample(audio, sr, target_sr)
        return audio
    except Exception:
        pass

    print('>>> Cannot decode MP3 — install miniaudio or pydub+ffmpeg')
    return np.zeros(0, dtype=np.int16)


def _resample(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resampler. Adequate for speech."""
    if src_sr == dst_sr or len(pcm) == 0:
        return pcm
    n_dst = int(len(pcm) * dst_sr / src_sr)
    idx   = np.linspace(0, len(pcm) - 1, n_dst).astype(np.int64)
    return pcm[idx]
