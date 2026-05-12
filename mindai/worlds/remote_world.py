"""Remote world proxy — brain on a GPU server, world local to the user.

The brain runs on a remote GPU. The world (microphone, files, stdin, retina)
stays on the user's local machine. They communicate over one persistent
WebSocket using msgpack-numpy for binary efficiency.

    Server (GPU)             ──msgpack RPC──►   Client (local)
      brain.run(proxy)        ◄──────────────    AgentWorld
      proxy.foo() → RPC                          serve_world(ws)

The proxy implements the full World interface; every call is forwarded.
The server runs the COMPLETE awake/sleep loop (PFC, amygdala, BG, sleep
consolidation, neuromodulators, neurogenesis) — same as local.

Wire format — msgpack dicts, ndarrays via msgpack-numpy:
    Server → Client:  {"op":"call", "method":"...", "args":[...]}
    Client → Server:  {"op":"ret", "v": <result>}      # success
                      {"op":"err", "msg": "..."}       # exception
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

try:
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()
except ImportError:
    msgpack = None


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------

def _require_msgpack() -> None:
    if msgpack is None:
        raise RuntimeError(
            'msgpack and msgpack-numpy required:\n'
            '    pip install msgpack msgpack-numpy')


def _pack(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def _unpack(data) -> dict:
    if isinstance(data, str):
        data = data.encode('utf-8')
    return msgpack.unpackb(data, raw=False)


def _send(ws, obj: dict) -> None:
    ws.send(_pack(obj), opcode=0x2)


def _recv(ws) -> dict:
    return _unpack(ws.recv())


async def _asend(ws, obj: dict) -> None:
    await ws.send_bytes(_pack(obj))


async def _arecv(ws) -> dict:
    msg = await ws.receive()
    data = msg.get('bytes') or msg.get('text')
    if data is None:
        raise ConnectionError('websocket closed')
    return _unpack(data)


# ---------------------------------------------------------------------------
# Server-side proxy — looks like a World, forwards every method to client
# ---------------------------------------------------------------------------

class RemoteWorldProxy:
    """World stub on the server. Every method = one RPC over the websocket."""

    def __init__(self, sync_call, sensory_layout: dict, motor_layout: dict,
                 size: int = 40, agent_pos=(0, 0), human_pos=(0, 0)):
        _require_msgpack()
        self._call           = sync_call
        self.sensory_layout  = sensory_layout
        self.motor_layout    = motor_layout
        self.size            = size
        self.agent_pos       = list(agent_pos)
        self.human_pos       = list(human_pos)
        self.isolation_ticks = 0
        self.last_agent_vocalization = np.zeros(
            motor_layout.get('vocalization', 0), dtype=np.float32)

    # World API — every method is one RPC ----------------------------------

    def get_homeostatic_signals(self) -> dict[str, float]:
        return self._call('get_homeostatic_signals')

    def get_sensory_retina(self, n: int) -> np.ndarray:
        return np.asarray(self._call('get_sensory_retina', n), dtype=np.float32)

    def get_distance_to_human(self) -> float:
        return float(self._call('get_distance_to_human'))

    def get_proprioception(self) -> np.ndarray:
        v = self._call('get_proprioception')
        return np.asarray(v, dtype=np.float32) if v is not None else np.zeros(0, np.float32)

    def receive_motor_pattern(self, motor: np.ndarray) -> None:
        self._call('receive_motor_pattern', np.asarray(motor, dtype=np.float32))

    def receive_gaze(self, fx: float, fy: float) -> None:
        self._call('receive_gaze', float(fx), float(fy))

    def receive_vocalization(self, vocal: np.ndarray) -> None:
        self._call('receive_vocalization', np.asarray(vocal, dtype=np.float32))

    def execute_action(self, idx: int) -> dict:
        return self._call('execute_action', int(idx))

    def is_alive(self) -> bool:
        return bool(self._call('is_alive'))

    def process_human_input(self, keys: dict) -> None:
        self._call('process_human_input', dict(keys))

    def pop_world_sound(self) -> np.ndarray:
        v = self._call('pop_world_sound')
        return np.asarray(v, dtype=np.float32) if v is not None else np.zeros(32, np.float32)

    def add_sound(self, pos, sound: np.ndarray) -> None:
        self._call('add_sound', list(pos), np.asarray(sound, dtype=np.float32))

    def human_interact(self) -> None:
        self._call('human_interact')


# ---------------------------------------------------------------------------
# Client-side host — answers RPCs from a real local World
# ---------------------------------------------------------------------------

_WORLD_METHODS = frozenset({
    'get_homeostatic_signals', 'get_sensory_retina', 'get_distance_to_human',
    'get_proprioception',      'receive_motor_pattern', 'receive_gaze',
    'receive_vocalization',    'execute_action',        'is_alive',
    'process_human_input',     'pop_world_sound',       'add_sound',
    'human_interact',
})


def _default_for(method: str, args: list) -> Any:
    if method == 'get_homeostatic_signals':
        return {'pain': 0.0, 'hunger': 0.0, 'thirst': 0.0}
    if method == 'get_sensory_retina':
        return np.zeros(args[0] if args else 1, dtype=np.float32)
    if method == 'get_distance_to_human':
        return float('inf')
    if method == 'is_alive':
        return True
    if method == 'pop_world_sound':
        return np.zeros(32, dtype=np.float32)
    if method == 'execute_action':
        return {'energy': 0.0, 'water': 0.0, 'stress': 0.0}
    return None


def serve_world(ws, world) -> None:
    """Block until the websocket closes, answering RPCs from `world`.

    Sends a `hello` handshake first (with layouts) so the server can build
    the brain. Then loops on incoming method calls.
    """
    _require_msgpack()

    _send(ws, {
        'op':             'hello',
        'sensory_layout': dict(world.sensory_layout),
        'motor_layout':   dict(world.motor_layout),
        'size':           int(getattr(world, 'size', 40)),
    })

    while True:
        try:
            msg = _recv(ws)
        except Exception:
            return

        if msg.get('op') != 'call':
            continue

        method = msg.get('method')
        args   = msg.get('args', [])

        if method not in _WORLD_METHODS:
            _send(ws, {'op': 'err', 'msg': f'unknown method {method}'})
            continue

        try:
            fn = getattr(world, method, None)
            result = fn(*args) if fn is not None else _default_for(method, args)
        except Exception as e:
            _send(ws, {'op': 'err', 'msg': f'{type(e).__name__}: {e}'})
            continue

        _send(ws, {'op': 'ret', 'v': result})


# ---------------------------------------------------------------------------
# Bridge async FastAPI WebSocket → sync brain.run()
# ---------------------------------------------------------------------------

class AsyncWSBridge:
    """Adapts a FastAPI async WebSocket into a sync `call(method, *args)`.
    The brain runs in a worker thread; RPCs are marshalled through
    asyncio.run_coroutine_threadsafe back into the server's event loop.
    """

    def __init__(self, ws, loop):
        self._ws   = ws
        self._loop = loop

    def __call__(self, method: str, *args) -> Any:
        fut = asyncio.run_coroutine_threadsafe(
            self._async_call(method, list(args)), self._loop)
        return fut.result()

    async def _async_call(self, method: str, args: list) -> Any:
        await _asend(self._ws, {'op': 'call', 'method': method, 'args': args})
        reply = await _arecv(self._ws)
        if reply.get('op') == 'err':
            raise RuntimeError(reply.get('msg', 'remote error'))
        return reply.get('v')

    async def recv_hello(self) -> dict:
        msg = await _arecv(self._ws)
        if msg.get('op') != 'hello':
            raise RuntimeError(f'expected hello, got {msg.get("op")}')
        return msg
