"""MindAI GPU Server — runs the FULL brain on a remote GPU via WebSocket.

Architecture
------------
The brain runs server-side using `brain.run(RemoteWorldProxy)` — every brain
module is active (PFC, amygdala, BG three-factor, sleep consolidation, all 15
neuromodulators, neurogenesis). The world (mic, files, stdin) stays on the
user's local machine and answers RPCs over the websocket.

This means weights trained via the server are biologically equivalent to
weights trained locally — same learning, just heavier compute on a GPU.

Setup (friend's machine / Colab)
--------------------------------
    pip install fastapi uvicorn websockets numpy torch scipy msgpack msgpack-numpy
    python server.py                     # binds 0.0.0.0:8000
    python server.py --ngrok             # auto-tunnel via pyngrok
    python server.py --ngrok --token T   # ngrok with auth token

Usage (your machine)
--------------------
    python main_agent.py --remote ws://FRIEND_IP:8000
    python main_agent.py --remote ws://abc123.ngrok.io
    python main_agent.py --download ws://FRIEND_IP:8000   # pull weights
"""

import argparse
import asyncio
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from mindai import Brain
from mindai.neurochemistry.neuromodulators import EndocrineSystem
from mindai.worlds.remote_world import RemoteWorldProxy, AsyncWSBridge

# ---------------------------------------------------------------------------
# Configuration — neurons match main_agent.py for weight compatibility
# ---------------------------------------------------------------------------

_SAVE_DIR        = 'savegame_brain'
_NUM_NEURONS     = 1_500_000
_SYNAPSE_DENSITY = 0.0002
_CLOCK_SCALE     = 0.05

# ---------------------------------------------------------------------------

app    = FastAPI(title='MindAI GPU Server')
_brain: Brain | None = None
_brain_lock = threading.Lock()


def _build_brain(sensory_layout: dict, motor_layout: dict) -> Brain:
    """Build (or reuse) the brain with layouts received from the client."""
    global _brain
    with _brain_lock:
        if _brain is not None:
            return _brain

        brain = Brain(
            num_neurons     = _NUM_NEURONS,
            sensory_layout  = sensory_layout,
            motor_layout    = motor_layout,
            device          = 'auto',
            save_path       = _SAVE_DIR,
            num_actions     = 455,
            synapse_density = _SYNAPSE_DENSITY,
        )
        brain.attach(EndocrineSystem())
        brain._clock_energy_scale = _CLOCK_SCALE

        if Path(_SAVE_DIR + '/brain.json').exists():
            try:
                brain.load(_SAVE_DIR)
                print(f'>>> Loaded brain (tick={brain.tick})')
            except Exception as e:
                print(f'>>> Load failed ({e}) — starting fresh')
        else:
            print('>>> Fresh brain')

        _brain = brain
        return _brain


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get('/')
def status():
    b = _brain
    if b is None:
        return {'status': 'waiting', 'loaded': False}
    return {
        'status':   'running',
        'loaded':   True,
        'tick':     b.tick,
        'surprise': round(b.surprise, 4),
        'mood':     b.mood,
        'neurons':  _NUM_NEURONS,
        'device':   str(b._device),
    }


@app.post('/save')
def save_brain():
    if _brain is None:
        return {'status': 'not loaded'}
    _brain.save(_SAVE_DIR)
    return {'status': 'saved', 'tick': _brain.tick}


# ---------------------------------------------------------------------------
# Models registry (for Web GUI Cloud mode)
# ---------------------------------------------------------------------------

@app.get('/models')
def get_models_endpoint():
    """List savegame_brain checkpoints available on this server."""
    from pathlib import Path as _P
    import json as _json
    out = []
    root = _P('models')
    # New layout — models/<id>/
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            meta_p  = d / 'meta.json'
            brain_p = d / 'brain.json'
            voice_p = d / 'voice.json'
            try:
                meta  = _json.loads(meta_p.read_text())  if meta_p.exists()  else {}
                brain = _json.loads(brain_p.read_text()) if brain_p.exists() else {}
                voice = _json.loads(voice_p.read_text()) if voice_p.exists() else {}
            except Exception:
                meta, brain, voice = {}, {}, {}
            out.append({
                'id':          d.name,
                'name':        meta.get('name', d.name),
                'tick':        brain.get('tick', 0),
                'num_neurons': brain.get('num_neurons', _NUM_NEURONS),
                'mood':        brain.get('mood', 'unknown'),
                'voice':       (voice.get('override') or voice).get('base_voice'),
            })
    # Legacy single-dir mode
    if not out and Path(_SAVE_DIR + '/brain.json').exists():
        try:
            brain = json.loads(Path(_SAVE_DIR + '/brain.json').read_text())
            out.append({
                'id':          'default',
                'name':        'Default',
                'tick':        brain.get('tick', 0),
                'num_neurons': brain.get('num_neurons', _NUM_NEURONS),
                'mood':        brain.get('mood', 'unknown'),
            })
        except Exception:
            pass
    return out


@app.get('/weights/download')
def download_weights():
    import io, zipfile
    from fastapi.responses import StreamingResponse
    p = Path(_SAVE_DIR)
    if not p.exists():
        return JSONResponse({'error': 'no savegame'}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in p.rglob('*'):
            if f.is_file():
                zf.write(f, f.relative_to(p))
    buf.seek(0)
    return StreamingResponse(buf, media_type='application/zip',
        headers={'Content-Disposition': 'attachment; filename=savegame_brain.zip'})


# ---------------------------------------------------------------------------
# /chat — Web-GUI Cloud-mode endpoint
# Brain + AgentWorld both live on the server. Client sends prompts/media,
# server streams token output. msgpack frames.
# ---------------------------------------------------------------------------

@app.websocket('/chat')
async def chat_socket(ws: WebSocket):
    import msgpack, msgpack_numpy
    msgpack_numpy.patch()
    from pathlib import Path as _P
    import io
    from mindai import Brain
    from mindai.worlds.agent_world import AgentWorld
    from mindai.neurochemistry.neuromodulators import EndocrineSystem
    from mindai.speech.voice_id import VoiceID

    await ws.accept()
    print('>>> [/chat] client connected')

    async def _send(obj):
        await ws.send_bytes(msgpack.packb(obj, use_bin_type=True))

    async def _recv():
        m = await ws.receive()
        b = m.get('bytes') or (m.get('text') or '').encode('utf-8')
        if not b: raise ConnectionError('closed')
        return msgpack.unpackb(b, raw=False)

    try:
        hello = await _recv()
    except Exception as e:
        print(f'>>> [/chat] handshake failed: {e}')
        await ws.close(); return

    if hello.get('op') != 'hello':
        await ws.close(); return

    model_id = hello.get('model_id', 'default')
    model_dir = _P('models') / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build brain + AgentWorld with default sensory sizes
    layout = {
        'vision': (round(_NUM_NEURONS * 0.00576) // 5) * 5,
        'audio':  max(64, int(_NUM_NEURONS * 0.00154)),
        'hunger': int(_NUM_NEURONS * 0.005),
        'pain':   int(_NUM_NEURONS * 0.010),
        'token':  int(_NUM_NEURONS * 0.00819) * 2,
    }
    world = AgentWorld(
        vision_size=layout['vision'], audio_size=layout['audio'],
        interactive=False, tokenizer='auto',
    )
    brain = Brain(
        num_neurons=_NUM_NEURONS, sensory_layout=layout,
        motor_layout=world.motor_layout, device='auto',
        save_path=str(model_dir), num_actions=world.tokenizer.vocab_size,
        synapse_density=_SYNAPSE_DENSITY,
    )
    brain.attach(EndocrineSystem())
    if (model_dir / 'brain.json').exists():
        try: brain.load(str(model_dir))
        except Exception as e: print(f'>>> load failed: {e}')

    voice = VoiceID.load_or_create(str(model_dir))
    await _send({'op': 'hello_ack', 'model_id': model_id, 'voice': voice.to_dict()})

    # Start brain in worker thread
    done = threading.Event()
    def _runner():
        try:
            brain.run(world, headless=True, save_path=str(model_dir))
        except Exception as e:
            print(f'>>> brain exited: {e}')
        finally:
            done.set()
    threading.Thread(target=_runner, daemon=True).start()

    loop = asyncio.get_running_loop()

    # Pumps
    async def _telemetry():
        while not done.is_set():
            await asyncio.sleep(0.5)
            chem = brain._chemistry
            try:
                await _send({
                    'op':'telemetry',
                    'tick': brain.tick, 'mood': brain.mood,
                    'surprise': round(float(brain.surprise), 3),
                    'dopamine':       round(float(getattr(chem, 'dopamine', 0.5)), 3),
                    'cortisol':       round(float(getattr(chem, 'cortisol', 0.0)), 3),
                    'noradrenaline':  round(float(getattr(chem, 'noradrenaline', 0.1)), 3),
                    'acetylcholine':  round(float(getattr(chem, 'acetylcholine', 0.5)), 3),
                    'serotonin':      round(float(getattr(chem, 'serotonin', 0.5)), 3),
                    'oxytocin':       round(float(getattr(chem, 'oxytocin', 0.0)), 3),
                    'pag_mode':       getattr(brain, '_pag', None).mode if hasattr(brain, '_pag') else 'rest',
                    'sleep':          brain._sleep.is_sleeping,
                    'sleep_phase':    str(brain._sleep.current_phase.name) if brain._sleep.is_sleeping else 'awake',
                })
            except Exception: return

    async def _output():
        last = ''
        while not done.is_set():
            await asyncio.sleep(0.15)
            try:
                text = world.get_current_output() or ''
            except Exception:
                continue
            if text != last:
                chunk = text[len(last):] if text.startswith(last) else text
                last = text
                if chunk:
                    try: await _send({'op': 'token_chunk', 'text': chunk})
                    except Exception: return

    asyncio.create_task(_telemetry())
    asyncio.create_task(_output())

    try:
        while not done.is_set():
            msg = await _recv()
            op  = msg.get('op')
            if op == 'prompt':
                world.inject_prompt(msg.get('text', ''))
            elif op == 'upload':
                fname = msg.get('filename', 'upload.bin')
                data  = msg.get('data', b'')
                kind  = msg.get('kind', 'image')
                tmp = _P('uploads_chat'); tmp.mkdir(exist_ok=True)
                fp  = tmp / f'{int(time.time()*1000)}_{fname}'
                fp.write_bytes(data if isinstance(data, (bytes, bytearray)) else bytes(data))
                if kind in ('image', 'video'):
                    world.inject_image(fp)
                elif kind == 'audio':
                    world.inject_audio(fp)
            elif op == 'clear_media':
                world.inject_image(None)
            elif op == 'save':
                brain.save(str(model_dir))
                await _send({'op': 'saved', 'tick': brain.tick})
    except (WebSocketDisconnect, ConnectionError):
        pass
    finally:
        try: brain.save(str(model_dir))
        except Exception: pass
        print(f'>>> [/chat] session over (tick {brain.tick})')


# ---------------------------------------------------------------------------
# /ws — legacy --remote endpoint (brain on server, world on client)
# ---------------------------------------------------------------------------

@app.websocket('/ws')
async def brain_socket(ws: WebSocket):
    await ws.accept()
    print('>>> Client connected — waiting for handshake...')

    bridge = AsyncWSBridge(ws, asyncio.get_running_loop())
    try:
        hello = await bridge.recv_hello()
    except Exception as e:
        print(f'>>> Handshake failed: {e}')
        await ws.close()
        return

    sensory_layout = hello['sensory_layout']
    motor_layout   = hello['motor_layout']
    size           = hello.get('size', 40)
    print(f'>>> Layouts received — sensory={sensory_layout} motor={motor_layout}')

    # Validate brain compatibility (if already loaded with different layout)
    brain = _build_brain(sensory_layout, motor_layout)
    proxy = RemoteWorldProxy(bridge, sensory_layout, motor_layout, size=size)

    print('>>> Starting brain.run() with remote world — full pipeline active')
    t0 = time.time()

    # brain.run() is sync; run it in a worker thread so the asyncio loop
    # remains free to service the websocket.
    done_evt = threading.Event()
    def _runner():
        try:
            brain.run(proxy, headless=True, save_path=_SAVE_DIR)
        except Exception as e:
            print(f'>>> brain.run() exited: {type(e).__name__}: {e}')
        finally:
            done_evt.set()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    try:
        # Idle here while the brain thread runs; periodically poll status.
        # If the client disconnects, recv() will raise WebSocketDisconnect
        # in the AsyncWSBridge → brain.run() will see is_alive() == False (RuntimeError)
        # → loop exits → done_evt fires.
        while not done_evt.is_set():
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    finally:
        elapsed = time.time() - t0
        print(f'>>> Session over ({elapsed:.0f}s, tick={brain.tick}) — saving...')
        try:
            brain.save(_SAVE_DIR)
            print(f'>>> Saved (tick={brain.tick})')
        except Exception as e:
            print(f'>>> Save failed: {e}')


# ---------------------------------------------------------------------------
# Auto-ngrok tunnel
# ---------------------------------------------------------------------------

def _start_ngrok(port: int, token: str | None) -> str | None:
    try:
        from pyngrok import conf, ngrok
    except ImportError:
        print('>>> pip install pyngrok  — для авто-туннеля')
        return None

    if token:
        conf.get_default().auth_token = token
    try:
        tunnel = ngrok.connect(port, 'http')
        public = tunnel.public_url.replace('https://', 'wss://').replace('http://', 'ws://')
        print(f'\n>>> NGROK туннель открыт:')
        print(f'>>> URL для друга: {public}')
        print(f'>>> Команда:       python main_agent.py --remote {public}\n')
        return public
    except Exception as e:
        print(f'>>> ngrok ошибка: {e}')
        return None


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',  default='0.0.0.0')
    parser.add_argument('--port',  type=int, default=8000)
    parser.add_argument('--ngrok', action='store_true',
                        help='auto-открыть публичный туннель через pyngrok')
    parser.add_argument('--token', default=None,
                        help='ngrok auth-token (или переменная NGROK_AUTHTOKEN)')
    args = parser.parse_args()

    print(f'\n>>> MindAI GPU Server (full brain pipeline)')
    print(f'>>> Neurons : {_NUM_NEURONS:,}  Density: {_SYNAPSE_DENSITY}')
    print(f'>>> Локально: ws://{args.host}:{args.port}/ws')

    if args.ngrok:
        import os
        token = args.token or os.environ.get('NGROK_AUTHTOKEN')
        _start_ngrok(args.port, token)

    print('>>> Жду подключения клиента (мозг построится из layouts клиента)\n')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
