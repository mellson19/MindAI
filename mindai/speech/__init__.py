"""Speech I/O — TTS (vocal apparatus) and STT (ear).

A MindAI brain has no anatomical larynx, so it would otherwise mimic
every voice it hears. We solve this by giving each brain a fixed,
deterministic VoiceID at birth (derived from a stable random seed).
The VoiceID picks a base TTS voice + pitch/rate offsets that stay
constant for the brain's lifetime — biologically: fixed vocal
apparatus (vocal fold length, oral cavity geometry).

What the brain learns is WHAT to say (token sequence) and WHEN to
emphasise (intonation, via SSML markers from punctuation/cap-tokens).

Public API:
    VocalApparatus  — text → audio, fixed voice from VoiceID
    Ear             — audio → text via Whisper
    VoiceID         — deterministic voice fingerprint
"""

from mindai.speech.voice_id        import VoiceID
from mindai.speech.vocal_apparatus import VocalApparatus
from mindai.speech.ear             import Ear

__all__ = ['VoiceID', 'VocalApparatus', 'Ear']
