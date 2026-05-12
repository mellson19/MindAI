"""HTTP client for the MindAI Fabric mod.

The Fabric mod runs an HTTP server on localhost:25576.
This module provides a thin Python client to read game state and send actions.

Endpoints
---------
GET  /state   → JSON: health, max_health, food, max_food, x, y, z,
                       yaw, pitch, alive (bool), dimension
POST /action  → JSON body: {"action": str, "value": float (optional)}
               Returns 200 OK on success.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Optional


_DEFAULT_URL = 'http://localhost:25576'


class ModClient:
    """Lightweight synchronous HTTP client for the Fabric mod.

    Parameters
    ----------
    base_url:
        URL of the mod HTTP server (default localhost:25576).
    timeout:
        Per-request timeout in seconds.
    retry_delay:
        Seconds to sleep between reconnect attempts on connection failure.
    """

    def __init__(
        self,
        base_url:    str   = _DEFAULT_URL,
        timeout:     float = 0.5,
        retry_delay: float = 2.0,
    ) -> None:
        self.base_url    = base_url.rstrip('/')
        self.timeout     = timeout
        self.retry_delay = retry_delay
        self._last_state: dict = {
            'health': 20.0, 'max_health': 20.0,
            'food': 20.0,   'max_food':   20.0,
            'x': 0.0, 'y': 64.0, 'z': 0.0,
            'yaw': 0.0, 'pitch': 0.0,
            'alive': True,
            'dimension': 'overworld',
        }
        self._connected = False

    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return latest game state dict.  Returns cached value on timeout."""
        try:
            req = urllib.request.Request(f'{self.base_url}/state')
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode('utf-8')
            self._last_state = json.loads(raw)
            self._connected  = True
        except (urllib.error.URLError, OSError):
            self._connected = False
        return self._last_state

    def send_action(self, action: str, value: float = 0.0) -> bool:
        """POST an action to the mod.

        Parameters
        ----------
        action:
            One of the action names listed in InputController.ACTION_NAMES.
        value:
            Optional float (e.g. yaw/pitch delta for camera actions).

        Returns True if the server acknowledged, False on connection failure.
        """
        body = json.dumps({'action': action, 'value': value}).encode('utf-8')
        req = urllib.request.Request(
            f'{self.base_url}/action',
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout):
                return True
        except (urllib.error.URLError, OSError):
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def wait_for_connection(self, max_wait: float = 60.0) -> None:
        """Block until the mod server responds or max_wait seconds elapse."""
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            self.get_state()
            if self._connected:
                print('>>> Fabric mod connected.')
                return
            print(f'>>> Waiting for Minecraft mod on {self.base_url}…')
            time.sleep(self.retry_delay)
        raise TimeoutError(
            f'Could not connect to Minecraft mod at {self.base_url} '
            f'after {max_wait}s')
