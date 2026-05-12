"""ActiveBrain — manages the single in-memory brain and its world.

Only ONE chat is active at a time. Switching chats:
    1. saves current brain into its model directory
    2. tears down the old brain.run() thread
    3. loads target chat's model, starts a new brain.run() thread

Cloud chats don't load a local brain — instead they hold a CloudBridge.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from mindai import Brain
from mindai.worlds.agent_world import AgentWorld
from mindai.neurochemistry.neuromodulators import EndocrineSystem
from mindai.speech import VoiceID, VocalApparatus, Ear

from webgui import models as model_registry


# Brain configuration — kept as module-level so a fresh chat reuses sizes
NUM_NEURONS     = 400_000
SYNAPSE_DENSITY = 0.001
DATA_DIR_DEFAULT = Path('data')


def _build_layout(num_neurons: int) -> dict:
    return {
        'vision': (round(num_neurons * 0.00576) // 5) * 5,
        'audio':  max(64, int(num_neurons * 0.00154)),
        'hunger': int(num_neurons * 0.005),
        'pain':   int(num_neurons * 0.010),
        'token':  int(num_neurons * 0.00819) * 2,
    }


def _discover_data(data_dir: Path) -> dict:
    sources: dict[str, str] = {}
    for key, name in [('text', 'corpus.txt'), ('qa', 'qa.txt')]:
        p = data_dir / name
        if p.exists():
            sources[key] = str(p)
    for key, name in [('images', 'images'), ('video', 'video'), ('audio', 'audio')]:
        p = data_dir / name
        if p.is_dir() and any(p.iterdir()):
            sources[key] = str(p)
    return sources


# ---------------------------------------------------------------------------

class ActiveBrain:
    """Holds (or proxies to) the currently active brain for the open chat."""

    def __init__(self):
        self.chat_id:   str | None  = None
        self.mode:      str | None  = None       # 'local' | 'cloud'
        self.model_id:  str | None  = None
        # Local mode state
        self.brain:     Brain      | None = None
        self.world:     AgentWorld | None = None
        self.thread:    threading.Thread | None = None
        self._stop_evt: threading.Event | None  = None
        # Cloud mode state
        self.cloud:     object | None = None     # CloudBridge instance
        # Speech (always local — TTS runs near the GUI for low latency)
        self.voice:     VoiceID         | None = None
        self.vocal:     VocalApparatus  | None = None
        self.ear:       Ear             | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def is_open(self) -> bool:
        return self.chat_id is not None

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    def open_chat(
        self,
        chat_id:   str,
        mode:      str,
        model_id:  str,
        cloud_url: str | None = None,
        on_token:     callable | None = None,
        on_telemetry: callable | None = None,
        on_voice:     callable | None = None,
    ) -> dict:
        """Activate a chat. Saves and tears down any previous one."""
        with self._lock:
            self.close_chat_locked()

            self.chat_id  = chat_id
            self.mode     = mode
            self.model_id = model_id

            if mode == 'local':
                self._open_local(model_id)
                # Voice for local chat — load from model dir
                save_dir = str(model_registry.get_model_dir(model_id))
                self.voice = VoiceID.load_or_create(save_dir)
                self.vocal = VocalApparatus(self.voice)
                self.ear   = Ear(model_size='base', language='en')
                voice_payload = self.voice.to_dict()
            else:
                # Cloud — bridge to remote server
                from webgui.cloud_bridge import CloudBridge
                self.cloud = CloudBridge(
                    url       = cloud_url or '',
                    model_id  = model_id,
                    on_token  = on_token,
                    on_telemetry = on_telemetry,
                    on_voice  = on_voice,
                )
                ack = self.cloud.connect()
                voice_payload = ack.get('voice', {})
                if voice_payload:
                    # Keep a local VoiceID so /tts can synthesise the same voice locally
                    seed = voice_payload.get('seed', 0)
                    self.voice = VoiceID(seed=seed)
                    if 'base_voice' in voice_payload:
                        self.voice.base_voice = voice_payload['base_voice']
                    if 'pitch_shift' in voice_payload:
                        self.voice.pitch_shift = float(voice_payload['pitch_shift'])
                    if 'rate' in voice_payload:
                        self.voice.rate = float(voice_payload['rate'])
                    self.vocal = VocalApparatus(self.voice)
                    self.ear   = Ear(model_size='base', language='en')

            model_registry.touch_model(model_id)
            return {'voice': voice_payload}

    def _open_local(self, model_id: str) -> None:
        save_dir = model_registry.get_model_dir(model_id)
        save_dir.mkdir(parents=True, exist_ok=True)

        sources = _discover_data(DATA_DIR_DEFAULT)
        layout  = _build_layout(NUM_NEURONS)

        self.world = AgentWorld(
            text_corpus  = sources.get('text'),
            images_dir   = sources.get('images'),
            video_dir    = sources.get('video'),
            qa_corpus    = sources.get('qa'),
            audio_source = sources.get('audio'),
            vision_size  = layout['vision'],
            audio_size   = layout['audio'],
            interactive  = False,
            tokenizer    = 'auto',
        )
        self.brain = Brain(
            num_neurons     = NUM_NEURONS,
            sensory_layout  = layout,
            motor_layout    = self.world.motor_layout,
            device          = 'auto',
            save_path       = str(save_dir),
            num_actions     = self.world.tokenizer.vocab_size,
            synapse_density = SYNAPSE_DENSITY,
        )
        self.brain.attach(EndocrineSystem())

        if (save_dir / 'brain.json').exists():
            try:
                self.brain.load(str(save_dir))
            except Exception as e:
                print(f'>>> Brain load failed for {model_id}: {e}')

        self._stop_evt = threading.Event()
        self.thread    = threading.Thread(
            target=self._local_runner, daemon=True)
        self.thread.start()

    def _local_runner(self) -> None:
        try:
            self.brain.run(self.world, headless=True,
                           save_path=str(model_registry.get_model_dir(self.model_id)))
        except Exception as e:
            print(f'>>> Brain loop exited: {type(e).__name__}: {e}')

    # ------------------------------------------------------------------

    def close_chat(self) -> None:
        with self._lock:
            self.close_chat_locked()

    def close_chat_locked(self) -> None:
        """Must be called with _lock held."""
        if self.mode == 'local' and self.brain is not None:
            try:
                self.brain.save(str(model_registry.get_model_dir(self.model_id)))
            except Exception as e:
                print(f'>>> Save failed: {e}')
            # Stopping brain.run() cleanly is hard — it has no abort flag.
            # We let the thread keep running into a dead world (set is_alive=False).
            try:
                self.world._alive = False
            except Exception:
                pass
        elif self.mode == 'cloud' and self.cloud is not None:
            try:
                self.cloud.close()
            except Exception:
                pass

        self.chat_id  = None
        self.mode     = None
        self.model_id = None
        self.brain    = None
        self.world    = None
        self.thread   = None
        self.cloud    = None

    # ------------------------------------------------------------------
    # Chat actions — uniform interface for both modes
    # ------------------------------------------------------------------

    def send_prompt(self, text: str) -> None:
        if self.mode == 'local' and self.world is not None:
            self.world.inject_prompt(text)
        elif self.mode == 'cloud' and self.cloud is not None:
            self.cloud.send_prompt(text)

    def send_media(self, path: Path, kind: str) -> dict:
        """kind: 'image' | 'video' | 'audio' | 'voice' (recorded mic)"""
        if self.mode == 'local' and self.world is not None:
            if kind in ('image', 'video'):
                self.world.inject_image(path)
            elif kind == 'voice':
                # STT → prompt
                if self.ear and self.ear.available:
                    text = self.ear.transcribe(str(path))
                    if text:
                        self.world.inject_prompt(text)
                        return {'transcribed': text}
            elif kind == 'audio':
                self.world.inject_audio(path)
            return {}
        elif self.mode == 'cloud' and self.cloud is not None:
            if kind == 'voice' and self.ear and self.ear.available:
                text = self.ear.transcribe(str(path))
                if text:
                    self.cloud.send_prompt(text)
                    return {'transcribed': text}
            else:
                data = path.read_bytes()
                self.cloud.send_upload(path.name, data, kind)
        return {}

    def clear_media(self) -> None:
        if self.mode == 'local' and self.world is not None:
            self.world.inject_image(None)
        elif self.mode == 'cloud' and self.cloud is not None:
            self.cloud.clear_media()

    def save(self) -> int:
        if self.mode == 'local' and self.brain is not None:
            self.brain.save(str(model_registry.get_model_dir(self.model_id)))
            return self.brain.tick
        if self.mode == 'cloud' and self.cloud is not None:
            self.cloud.save()
        return 0

    # ------------------------------------------------------------------
    # Telemetry / output (local only — cloud streams via CloudBridge)
    # ------------------------------------------------------------------

    def get_local_telemetry(self) -> dict | None:
        if self.mode != 'local' or self.brain is None:
            return None
        b = self.brain
        chem = b._chemistry
        return {
            'tick':           b.tick,
            'mood':           b.mood,
            'surprise':       round(float(b.surprise), 3),
            'dopamine':       round(float(getattr(chem, 'dopamine', 0.5)), 3),
            'cortisol':       round(float(getattr(chem, 'cortisol', 0.0)), 3),
            'noradrenaline':  round(float(getattr(chem, 'noradrenaline', 0.1)), 3),
            'acetylcholine':  round(float(getattr(chem, 'acetylcholine', 0.5)), 3),
            'serotonin':      round(float(getattr(chem, 'serotonin', 0.5)), 3),
            'oxytocin':       round(float(getattr(chem, 'oxytocin', 0.0)), 3),
            'pag_mode':       getattr(b, '_pag', None).mode if hasattr(b, '_pag') else 'rest',
            'sleep':          b._sleep.is_sleeping,
            'sleep_phase':    str(b._sleep.current_phase.name) if b._sleep.is_sleeping else 'awake',
        }

    def get_local_output(self) -> str:
        if self.mode != 'local' or self.world is None:
            return ''
        try:
            return self.world.get_current_output()
        except Exception:
            return ''
