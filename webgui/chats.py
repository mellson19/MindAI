"""Chat registry — chats/<chat_id>.json.

Each chat is bound to ONE model and ONE mode (local | cloud) at creation
time. Switching mode/model is not allowed because the model's neurons
diverge during inference (every reply changes the synaptic weights).

Chat file layout:
    {
        "id":          "<uuid>",
        "name":        "...",
        "mode":        "local" | "cloud",
        "model_id":    "<id>",
        "cloud_url":   "wss://..."  (cloud only),
        "created":     <epoch>,
        "updated":     <epoch>,
        "messages":    [{"role":"user|brain","text":"...","media":null}, ...]
    }
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

CHATS_DIR = Path('chats')


def _ensure_root() -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)


def _path(chat_id: str) -> Path:
    return CHATS_DIR / f'{chat_id}.json'


def list_chats() -> list[dict]:
    _ensure_root()
    out: list[dict] = []
    for p in sorted(CHATS_DIR.glob('*.json'),
                    key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        out.append({
            'id':            d.get('id', p.stem),
            'name':          d.get('name', p.stem),
            'mode':          d.get('mode', 'local'),
            'model_id':      d.get('model_id'),
            'cloud_url':     d.get('cloud_url'),
            'created':       d.get('created', 0),
            'updated':       d.get('updated', 0),
            'message_count': len(d.get('messages', [])),
        })
    return out


def get_chat(chat_id: str) -> dict | None:
    p = _path(chat_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def create_chat(name: str, mode: str, model_id: str,
                cloud_url: str | None = None) -> dict:
    _ensure_root()
    if mode not in ('local', 'cloud'):
        raise ValueError(f'unknown mode: {mode}')
    if mode == 'cloud' and not cloud_url:
        raise ValueError('cloud mode requires cloud_url')
    chat = {
        'id':        uuid.uuid4().hex[:12],
        'name':      name or 'New chat',
        'mode':      mode,
        'model_id':  model_id,
        'cloud_url': cloud_url,
        'created':   int(time.time()),
        'updated':   int(time.time()),
        'messages':  [],
    }
    _path(chat['id']).write_text(json.dumps(chat, indent=2))
    return chat


def add_message(chat_id: str, role: str, text: str,
                media: str | None = None) -> bool:
    chat = get_chat(chat_id)
    if chat is None:
        return False
    chat['messages'].append({
        'role':  role,
        'text':  text,
        'media': media,
        'ts':    int(time.time()),
    })
    chat['updated'] = int(time.time())
    _path(chat_id).write_text(json.dumps(chat, indent=2))
    return True


def rename_chat(chat_id: str, new_name: str) -> bool:
    chat = get_chat(chat_id)
    if chat is None:
        return False
    chat['name'] = new_name
    _path(chat_id).write_text(json.dumps(chat, indent=2))
    return True


def delete_chat(chat_id: str) -> bool:
    p = _path(chat_id)
    if p.exists():
        p.unlink()
        return True
    return False
