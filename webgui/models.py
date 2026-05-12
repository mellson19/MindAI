"""Model registry — each saved brain lives in models/<id>/.

A model directory contains:
    brain.json        — metadata (tick, neurons, mood, etc.)
    weights.npz       — sparse synapse matrix
    voice.json        — fixed voice fingerprint for this brain
    meta.json         — registry metadata (display name, created, last_used)

Each chat is bound to ONE model. Multiple chats may share a model — they
continue training the same brain (its neurons keep mutating). To get an
independent learning trajectory, "duplicate" a model first.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

MODELS_DIR = Path('models')


def _ensure_root() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_id(name: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in name).lower()[:40]


def _meta_path(model_dir: Path) -> Path:
    return model_dir / 'meta.json'


def _read_meta(model_dir: Path) -> dict:
    p = _meta_path(model_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _write_meta(model_dir: Path, meta: dict) -> None:
    _meta_path(model_dir).write_text(json.dumps(meta, indent=2))


def _read_brain_json(model_dir: Path) -> dict:
    p = model_dir / 'brain.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _read_voice_json(model_dir: Path) -> dict:
    p = model_dir / 'voice.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _dir_size_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob('*') if f.is_file())


# ---------------------------------------------------------------------------

def list_models() -> list[dict]:
    """All saved models with metadata."""
    _ensure_root()
    out: list[dict] = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta  = _read_meta(d)
        brain = _read_brain_json(d)
        voice = _read_voice_json(d)
        out.append({
            'id':           d.name,
            'name':         meta.get('name', d.name),
            'created':      meta.get('created', 0),
            'last_used':    meta.get('last_used', 0),
            'tick':         brain.get('tick', 0),
            'num_neurons':  brain.get('num_neurons', 0),
            'mood':         brain.get('mood', 'unknown'),
            'voice':        voice.get('override', voice).get('base_voice') if voice else None,
            'size_mb':      round(_dir_size_bytes(d) / 1_048_576, 1),
            'has_weights':  (d / 'weights.npz').exists(),
        })
    return out


def get_model_dir(model_id: str) -> Path:
    return MODELS_DIR / model_id


def model_exists(model_id: str) -> bool:
    return get_model_dir(model_id).is_dir()


def create_model(display_name: str, base_id: str | None = None) -> dict:
    """Create a new empty (or cloned) model directory.

    If base_id is given, copies its weights/voice into the new directory.
    Otherwise the directory is created empty — weights will appear on
    first save() of the active brain.
    """
    _ensure_root()
    base_name = _safe_id(display_name) or 'model'
    model_id  = base_name
    n = 1
    while model_exists(model_id):
        n += 1
        model_id = f'{base_name}-{n}'

    new_dir = get_model_dir(model_id)
    new_dir.mkdir(parents=True, exist_ok=True)

    if base_id and model_exists(base_id):
        src = get_model_dir(base_id)
        for fname in ('brain.json', 'weights.npz', 'voice.json'):
            f = src / fname
            if f.exists():
                shutil.copy2(f, new_dir / fname)

    _write_meta(new_dir, {
        'name':       display_name,
        'created':    int(time.time()),
        'last_used':  0,
        'cloned_from': base_id if base_id else None,
    })
    return {'id': model_id, 'name': display_name}


def touch_model(model_id: str) -> None:
    """Mark model as recently used."""
    if not model_exists(model_id):
        return
    meta = _read_meta(get_model_dir(model_id))
    meta['last_used'] = int(time.time())
    _write_meta(get_model_dir(model_id), meta)


def delete_model(model_id: str) -> bool:
    d = get_model_dir(model_id)
    if d.is_dir():
        shutil.rmtree(d)
        return True
    return False


def rename_model(model_id: str, new_display_name: str) -> bool:
    if not model_exists(model_id):
        return False
    meta = _read_meta(get_model_dir(model_id))
    meta['name'] = new_display_name
    _write_meta(get_model_dir(model_id), meta)
    return True


# ---------------------------------------------------------------------------
# Default model — auto-created on first run if absent
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = 'default'

def ensure_default_model() -> str:
    """Make sure a 'default' model entry exists. Returns its id."""
    _ensure_root()
    d = get_model_dir(DEFAULT_MODEL_ID)
    if not d.is_dir():
        d.mkdir(parents=True, exist_ok=True)
        _write_meta(d, {
            'name':      'Default',
            'created':   int(time.time()),
            'last_used': 0,
        })
    return DEFAULT_MODEL_ID
