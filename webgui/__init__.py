"""MindAI Web GUI — modern browser-based chat interface.

Features
--------
* Drag-and-drop image / video / audio upload — fed straight into the
  brain's sensory channels (FovealRetina + Cochlea).
* Live token stream — brain output appears character-by-character as
  it's generated.
* Two-way voice chat — browser captures mic via WebRTC, server uses
  Whisper STT, brain generates a response, edge-tts synthesises it
  with the brain's fixed VoiceID, audio streamed back to browser.
* Live brain telemetry — surprise, mood, dopamine/cortisol, sleep
  phase, memory novelty.

Run:
    python -m webgui                       # localhost:8765
    python -m webgui --port 8888 --share   # public ngrok tunnel
"""
