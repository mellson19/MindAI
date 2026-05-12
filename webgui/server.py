"""MindAI Web GUI server — chats, models, Local | Cloud modes.

Endpoints
---------
HTTP
    GET  /                       — index.html
    GET  /static/*               — static assets
    GET  /models                 — list saved models (local)
    POST /models                 — create model        body: {name, base?}
    POST /models/{id}/rename     — rename               body: {name}
    DELETE /models/{id}          — delete
    GET  /remote/models?url=...  — list models on a Colab server
    GET  /chats                  — list chats
    POST /chats                  — create chat         body: {name, mode, model_id, cloud_url?}
    GET  /chats/{id}             — full chat (with messages)
    POST /chats/{id}/open        — make this chat active
    POST /chats/{id}/prompt      — send text prompt    body: {text}
    POST /chats/{id}/upload      — upload media       multipart/form
    POST /chats/{id}/save        — persist active model
    POST /chats/{id}/rename      — rename chat
    DELETE /chats/{id}           — delete chat
    POST /tts                    — synthesise text   body: {text}
    GET  /voice                  — current voice + options
    POST /voice                  — override voice    body: {base_voice, pitch_shift, rate}

WebSocket
    /ws                          — telemetry + token streaming for active chat
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from mindai.speech.voice_id import list_voices

from webgui import models as model_registry
from webgui import chats   as chat_registry
from webgui.active_brain   import ActiveBrain
from webgui.cloud_bridge   import list_remote_models


# ---------------------------------------------------------------------------

_HERE     = Path(__file__).resolve().parent
_STATIC   = _HERE / 'static'
_UPLOADS  = _HERE / 'uploads'
_UPLOADS.mkdir(exist_ok=True)

_active_brain = ActiveBrain()
_ws_clients: set[WebSocket] = set()
_main_loop: asyncio.AbstractEventLoop | None = None


# ---------------------------------------------------------------------------
# WS broadcast helpers (called from sync threads via run_coroutine_threadsafe)
# ---------------------------------------------------------------------------

async def _broadcast(payload: dict) -> None:
    if not _ws_clients:
        return
    msg = json.dumps(payload)
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


def _broadcast_threadsafe(payload: dict) -> None:
    if _main_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(payload), _main_loop)


# ---------------------------------------------------------------------------
# Telemetry + output pumps
# ---------------------------------------------------------------------------

async def _telemetry_pump():
    """Local-mode telemetry — cloud telemetry comes through CloudBridge callbacks."""
    while True:
        await asyncio.sleep(0.5)
        if not _active_brain.is_open() or _active_brain.mode != 'local':
            continue
        t = _active_brain.get_local_telemetry()
        if t is not None:
            await _broadcast({'op': 'telemetry', **t})


async def _output_pump():
    """Local-mode token streaming."""
    last = ''
    while True:
        await asyncio.sleep(0.15)
        if not _active_brain.is_open() or _active_brain.mode != 'local':
            last = ''
            continue
        text = _active_brain.get_local_output() or ''
        if text != last:
            # Naive diff — works for append-only output text
            if text.startswith(last):
                chunk = text[len(last):]
            else:
                chunk = text
            last = text
            if chunk:
                await _broadcast({'op': 'token_chunk',
                                  'chat_id': _active_brain.chat_id,
                                  'text': chunk})


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title='MindAI Web GUI')


@app.on_event('startup')
async def _on_startup():
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    model_registry.ensure_default_model()
    asyncio.create_task(_telemetry_pump())
    asyncio.create_task(_output_pump())


@app.get('/')
def index():
    return FileResponse(_STATIC / 'index.html')


app.mount('/static', StaticFiles(directory=_STATIC), name='static')


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get('/models')
def get_models():
    return model_registry.list_models()


@app.post('/models')
def post_model(payload: dict):
    name = (payload or {}).get('name', '').strip() or 'Untitled'
    base = (payload or {}).get('base')
    return model_registry.create_model(name, base_id=base)


@app.post('/models/{model_id}/rename')
def rename_model(model_id: str, payload: dict):
    if not model_registry.rename_model(model_id, (payload or {}).get('name', '').strip()):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return {'ok': True}


@app.delete('/models/{model_id}')
def delete_model(model_id: str):
    if _active_brain.is_open() and _active_brain.model_id == model_id:
        return JSONResponse({'error': 'model is in use by active chat'},
                            status_code=409)
    if not model_registry.delete_model(model_id):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return {'ok': True}


@app.get('/remote/models')
def get_remote_models(url: str):
    return list_remote_models(url)


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

@app.get('/chats')
def get_chats():
    return chat_registry.list_chats()


@app.post('/chats')
def post_chat(payload: dict):
    name      = (payload or {}).get('name', '').strip() or 'New chat'
    mode      = (payload or {}).get('mode', 'local')
    model_id  = (payload or {}).get('model_id') or 'default'
    cloud_url = (payload or {}).get('cloud_url')
    try:
        chat = chat_registry.create_chat(name, mode, model_id, cloud_url)
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    return chat


@app.get('/chats/{chat_id}')
def get_chat(chat_id: str):
    chat = chat_registry.get_chat(chat_id)
    if chat is None:
        return JSONResponse({'error': 'not found'}, status_code=404)
    return chat


@app.post('/chats/{chat_id}/open')
def open_chat(chat_id: str):
    chat = chat_registry.get_chat(chat_id)
    if chat is None:
        return JSONResponse({'error': 'not found'}, status_code=404)
    try:
        info = _active_brain.open_chat(
            chat_id   = chat_id,
            mode      = chat['mode'],
            model_id  = chat['model_id'],
            cloud_url = chat.get('cloud_url'),
            on_token     = lambda text: _broadcast_threadsafe(
                {'op': 'token_chunk', 'chat_id': chat_id, 'text': text}),
            on_telemetry = lambda d:    _broadcast_threadsafe(
                {'op': 'telemetry', **{k: v for k, v in d.items() if k != 'op'}}),
            on_voice     = lambda v:    _broadcast_threadsafe(
                {'op': 'voice_info', 'voice': v}),
        )
        return {'ok': True, 'chat_id': chat_id, **info}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.post('/chats/{chat_id}/prompt')
def chat_prompt(chat_id: str, payload: dict):
    if _active_brain.chat_id != chat_id:
        return JSONResponse({'error': 'chat not active'}, status_code=409)
    text = (payload or {}).get('text', '').strip()
    if not text:
        return {'ok': False}
    chat_registry.add_message(chat_id, 'user', text)
    _active_brain.send_prompt(text)
    return {'ok': True}


@app.post('/chats/{chat_id}/upload')
async def chat_upload(chat_id: str, file: UploadFile = File(...)):
    if _active_brain.chat_id != chat_id:
        return JSONResponse({'error': 'chat not active'}, status_code=409)
    safe = (file.filename or 'upload.bin').replace('/', '_').replace('\\', '_')
    out  = _UPLOADS / f'{int(time.time()*1000)}_{safe}'
    with open(out, 'wb') as fh:
        fh.write(await file.read())
    ext = out.suffix.lower()
    if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
        kind = 'image'
    elif ext in {'.mp4', '.mov', '.mkv', '.webm', '.avi'}:
        kind = 'video'
    elif ext in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}:
        kind = 'audio'
    elif ext == '.webm':
        kind = 'voice'
    else:
        kind = 'audio'

    # Heuristic: small webm = mic recording
    if file.filename and 'voice' in file.filename:
        kind = 'voice'

    info = _active_brain.send_media(out, kind)
    chat_registry.add_message(
        chat_id, 'user',
        info.get('transcribed', '') or f'[{kind}] {file.filename}',
        media=str(out))
    return {'ok': True, 'kind': kind, **info}


@app.post('/chats/{chat_id}/save')
def chat_save(chat_id: str):
    if _active_brain.chat_id != chat_id:
        return JSONResponse({'error': 'chat not active'}, status_code=409)
    tick = _active_brain.save()
    return {'ok': True, 'tick': tick}


@app.post('/chats/{chat_id}/rename')
def chat_rename(chat_id: str, payload: dict):
    name = (payload or {}).get('name', '').strip()
    if not name or not chat_registry.rename_chat(chat_id, name):
        return JSONResponse({'error': 'not found or empty'}, status_code=400)
    return {'ok': True}


@app.delete('/chats/{chat_id}')
def chat_delete(chat_id: str):
    if _active_brain.chat_id == chat_id:
        _active_brain.close_chat()
    if not chat_registry.delete_chat(chat_id):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return {'ok': True}


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------

@app.get('/voice')
def voice_info():
    if _active_brain.voice is None:
        return {'available': False, 'options': list_voices()}
    return {
        'available': True,
        'current':   _active_brain.voice.to_dict(),
        'options':   list_voices(),
    }


@app.post('/voice')
def set_voice(payload: dict):
    if _active_brain.voice is None or _active_brain.vocal is None:
        return JSONResponse({'error': 'no active chat'}, status_code=409)
    valid = {v['id'] for v in list_voices()}
    bv = (payload or {}).get('base_voice')
    if bv:
        if bv not in valid:
            return JSONResponse({'error': f'unknown voice {bv}'}, status_code=400)
        _active_brain.voice.base_voice = bv
    if 'pitch_shift' in payload:
        _active_brain.voice.pitch_shift = max(-12.0, min(12.0,
                                                         float(payload['pitch_shift'])))
    if 'rate' in payload:
        _active_brain.voice.rate = max(0.5, min(2.0, float(payload['rate'])))
    if _active_brain.mode == 'local' and _active_brain.model_id:
        _active_brain.voice.save_override(
            str(model_registry.get_model_dir(_active_brain.model_id)))
    return {'ok': True, 'current': _active_brain.voice.to_dict()}


@app.post('/tts')
async def tts(payload: dict):
    text = (payload or {}).get('text', '').strip()
    if not text or _active_brain.vocal is None:
        return Response(status_code=204)
    try:
        tmp = _UPLOADS / f'tts_{int(time.time()*1000)}.wav'
        _active_brain.vocal.synthesize_to_file(text, tmp)
        data = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return Response(content=data, media_type='audio/wav')
    except Exception as e:
        return Response(status_code=500, content=str(e))


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        await ws.send_text(json.dumps({
            'op': 'state',
            'active_chat': _active_brain.chat_id,
            'voice': _active_brain.voice.to_dict() if _active_brain.voice else None,
        }))
        while True:
            await ws.receive_text()   # currently only used as keepalive
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',  default='127.0.0.1')
    parser.add_argument('--port',  type=int, default=8765)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--token', default=None)
    args = parser.parse_args()

    print(f'>>> MindAI Web GUI on http://{args.host}:{args.port}/')

    if args.share:
        try:
            from pyngrok import conf, ngrok
            if args.token:
                conf.get_default().auth_token = args.token
            elif os.environ.get('NGROK_AUTHTOKEN'):
                conf.get_default().auth_token = os.environ['NGROK_AUTHTOKEN']
            tunnel = ngrok.connect(args.port, 'http')
            print(f'>>> Public URL: {tunnel.public_url}')
        except ImportError:
            print('>>> pip install pyngrok  for --share')

    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
