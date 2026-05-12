"""CloudBridge — proxy from Web GUI to a remote MindAI brain on Colab.

In Cloud mode the local Web GUI does NOT load a brain. It connects via
WebSocket to a Colab-hosted server.py instance and forwards prompts /
media uploads. Token output streams back as it is generated remotely.

The remote brain runs the FULL pipeline (PFC, BG, sleep, neurogenesis)
just like local — its neurons keep mutating across the conversation.

Wire format
-----------
We extend server.py's protocol with a thin RPC layer for chat use:

    Client → Server (msgpack):
        {"op":"hello",     "model_id":"<id>"}                # pick model
        {"op":"prompt",    "text":"...."}
        {"op":"upload",    "filename":"...", "data":<bytes>, "kind":"image|video|audio"}
        {"op":"clear_media"}
        {"op":"save"}

    Server → Client:
        {"op":"hello_ack", "model_id":"<id>", "voice":<voice_dict>}
        {"op":"models",    "list":[...]}            (if requested)
        {"op":"token_chunk","text":"..."}
        {"op":"telemetry", ...}                     (every 500 ms)
        {"op":"saved", "tick":N}
"""

from __future__ import annotations

import asyncio
import threading
from typing import Callable

try:
    import websocket
    _HAVE_WS = True
except ImportError:
    _HAVE_WS = False

try:
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()
    _HAVE_MSGPACK = True
except ImportError:
    _HAVE_MSGPACK = False


def _normalise_url(url: str) -> str:
    url = url.strip().rstrip('/')
    url = url.replace('http://', 'ws://').replace('https://', 'wss://')
    if not url.startswith('ws'):
        url = 'wss://' + url
    if not url.endswith('/chat'):
        url += '/chat'
    return url


class CloudBridge:
    """Synchronous WS proxy. One bridge per active cloud chat."""

    def __init__(
        self,
        url:        str,
        model_id:   str,
        on_token:   Callable[[str], None] | None      = None,
        on_telemetry: Callable[[dict], None] | None   = None,
        on_voice:   Callable[[dict], None] | None     = None,
    ):
        if not _HAVE_WS or not _HAVE_MSGPACK:
            raise RuntimeError(
                'pip install websocket-client msgpack msgpack-numpy')

        self.url       = _normalise_url(url)
        self.model_id  = model_id
        self._on_token     = on_token
        self._on_telemetry = on_telemetry
        self._on_voice     = on_voice
        self._ws: websocket.WebSocket | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    # ------------------------------------------------------------------

    def connect(self) -> dict:
        """Open the WS, send hello, return server's hello_ack payload."""
        self._ws = websocket.create_connection(self.url, timeout=15,
                                               enable_multithread=True)
        self._send({'op': 'hello', 'model_id': self.model_id})
        ack = self._recv()
        if ack.get('op') != 'hello_ack':
            raise RuntimeError(f'unexpected handshake: {ack}')
        if self._on_voice and ack.get('voice'):
            self._on_voice(ack['voice'])
        # Start receiver thread
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        return ack

    # ------------------------------------------------------------------
    # Outgoing
    # ------------------------------------------------------------------

    def send_prompt(self, text: str) -> None:
        self._send({'op': 'prompt', 'text': text})

    def send_upload(self, filename: str, data: bytes, kind: str) -> None:
        self._send({'op': 'upload', 'filename': filename,
                    'data': data, 'kind': kind})

    def clear_media(self) -> None:
        self._send({'op': 'clear_media'})

    def save(self) -> None:
        self._send({'op': 'save'})

    def close(self) -> None:
        self._closed = True
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------------

    def _send(self, obj: dict) -> None:
        if not self._ws:
            return
        try:
            self._ws.send(msgpack.packb(obj, use_bin_type=True), opcode=0x2)
        except Exception:
            self._closed = True

    def _recv(self) -> dict:
        data = self._ws.recv()
        if isinstance(data, str):
            data = data.encode('utf-8')
        return msgpack.unpackb(data, raw=False)

    def _recv_loop(self) -> None:
        while not self._closed:
            try:
                msg = self._recv()
            except Exception:
                self._closed = True
                return
            op = msg.get('op')
            if   op == 'token_chunk' and self._on_token:
                self._on_token(msg.get('text', ''))
            elif op == 'telemetry'   and self._on_telemetry:
                self._on_telemetry(msg)
            elif op == 'voice_info'  and self._on_voice:
                self._on_voice(msg.get('voice', msg))


# ---------------------------------------------------------------------------
# Helper: list models on a remote Colab server (HTTP GET, separate from WS)
# ---------------------------------------------------------------------------

def list_remote_models(url: str, timeout: float = 10.0) -> list[dict]:
    """HTTP GET <base>/models — returns server's available models."""
    import urllib.request, json as _json
    base = url.rstrip('/').replace('ws://', 'http://').replace('wss://', 'https://')
    if base.endswith('/chat'):
        base = base[:-5]
    if base.endswith('/ws'):
        base = base[:-3]
    try:
        with urllib.request.urlopen(base + '/models', timeout=timeout) as r:
            return _json.loads(r.read())
    except Exception as e:
        print(f'>>> list_remote_models failed: {e}')
        return []
