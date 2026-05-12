# MindAI on Google Colab — Inference + Web UI guide

Run a MindAI brain on a free Colab T4 GPU and chat with it from the **Web UI**
on your local computer (Cloud mode). Files (photos / videos / audio) stay on
your machine and are uploaded **only encrypted** through TLS to Colab.

---

## ⚠️ Privacy reality check

End-to-end encryption with inference on Colab is **physically impossible** —
the GPU must see decoded sensory arrays in RAM to do STDP. What we *can*
guarantee:

| Threat | Status |
|---|---|
| ngrok / ISP / Wi-Fi sees your data | ✅ TLS (`wss://…`) — encrypted in transit |
| Files on Colab disk | ✅ Server only writes `uploads_chat/` short-lived buffers, deleted on process exit |
| Google Cloud snapshots Colab RAM | ❌ unavoidable for any SaaS GPU |

If you need stricter privacy, run `server.py` on your own GPU PC and tunnel via
**Tailscale** instead of ngrok — see [§ Tailscale alternative](#alternative-tailscale-without-cloud).

---

## Two Colab modes

| Mode | Brain runs | World runs | Use when |
|---|---|---|---|
| **Cloud chat** *(this guide)* | Colab | Colab | Chat with the brain via Web UI; data lives on Colab |
| **Remote-GPU training** | Colab | Your PC | You have a big local dataset; just want GPU compute |

Both expose `server.py` on different endpoints (`/chat` vs `/ws`).
This guide covers Cloud chat.

---

## Setup — step by step

### 1. Local PC — install Web UI dependencies

```powershell
.venv\Scripts\activate
pip install -r requirements-webgui.txt
pip install msgpack msgpack-numpy websocket-client    # for Cloud bridge
```

### 2. ngrok account (free)

1. Sign up: https://dashboard.ngrok.com/signup
2. Copy auth-token: https://dashboard.ngrok.com/get-started/your-authtoken
3. Without a token the tunnel closes after 2 hours. With it — unlimited.

### 3. Open Colab

1. Open `colab_server.ipynb` (`File → Upload notebook` in Colab).
2. **Runtime → Change runtime type → GPU (T4)**.
3. **Cell 1** — clones repo, installs deps.
4. **Cell 2** — paste `NGROK_AUTHTOKEN`.
5. **Cell 3** — pre-create some named models (default + custom).
6. **Cell 4** — launches the server. Output shows:

   ```
   >>> URL для друга: wss://abc123.ngrok-free.app
   >>> Команда:       python main_agent.py --remote wss://abc123.ngrok-free.app
   ```

   Copy the `wss://abc…ngrok-free.app` URL.

### 4. Local PC — start the Web UI

```powershell
python main_agent.py --gui
# or directly:
python -m webgui --port 8765
```

Open http://127.0.0.1:8765/

### 5. Create a Cloud chat

1. Click **New chat** in the sidebar.
2. Name it (e.g., "Alice on Colab").
3. Switch the mode to **Cloud**.
4. Paste the Colab `wss://…` URL.
5. Click **refresh models from cloud →** — the dropdown populates with the
   models you created on Colab in Cell 3.
6. Pick a model. Click **Create**.

The Web UI connects via WebSocket to Colab. Token output streams back as
the brain generates it. Drag-drop photo/video/audio anywhere — files
upload to Colab inside the same TLS tunnel.

---

## How chats are bound to models

MindAI is **dynamic** — every reply changes the brain's synaptic weights.
A chat is therefore tied to the specific model it was created with:

- You **cannot** change the model or mode of an existing chat.
- Two chats can use the same model, but they share its weight trajectory
  (the brain keeps learning across both).
- For an isolated branch, click **Duplicate** in the Models tab before
  starting a new chat.

The **Models** tab on the local Web UI shows local models (saved on your PC).
For Cloud chats, the model lives on Colab — its name appears in the chat
sidebar but Colab's `/models` is the source of truth.

---

## Saving Cloud models

When the Cloud chat is open, click **save** in the top bar — Colab persists
the brain to `models/<id>/` immediately. To survive a Colab runtime reset,
copy the whole `models/` directory to Google Drive (see Cell 5 of the
notebook):

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r models /content/drive/MyDrive/mindai_models
```

To pull weights back to your local PC for offline use:

```powershell
python main_agent.py --download wss://abc123.ngrok-free.app
```

This downloads `savegame_brain.zip` containing every model directory.

---

## Alternative: Tailscale without cloud

If you don't trust Google with the in-RAM data:

### Option A — your second PC with GPU

1. Install Tailscale on both your laptop and the GPU machine, log in to the
   same account: https://tailscale.com/download
2. On the GPU PC: `python server.py` (no `--ngrok`).
3. On the laptop: open the Web UI, **New chat → Cloud**,
   paste `ws://100.x.y.z:8000` (Tailscale IP).
4. Trafic is end-to-end encrypted via WireGuard. Tailscale's coordination
   servers see only key exchange, not your data.

### Option B — Tailscale + Colab

```bash
!curl -fsSL https://tailscale.com/install.sh | sh
!sudo tailscaled --tun=userspace-networking &
!sudo tailscale up --authkey=tskey-...
```

Removes ngrok from the chain — but Colab itself still sees decrypted data
in RAM (fundamental limitation).

---

## What flows over the network

Per tick (≈50 ms on Colab T4):

```
Local Web UI → Colab:
    msgpack {"op":"prompt","text":"hello"}
    msgpack {"op":"upload","filename":"...","data":<bytes>,"kind":"image"}

Colab → Local Web UI:
    msgpack {"op":"hello_ack","model_id":"alice","voice":{...}}
    msgpack {"op":"token_chunk","text":"..."}
    msgpack {"op":"telemetry", tick, mood, dopamine, cortisol, ...}
```

Files (`*.png`, `*.mp4`, `*.wav`) leave your PC encoded inside the TLS tunnel.
Colab decodes them, feeds through `FovealRetina` / `Cochlea`, then deletes
the file from `uploads_chat/` after processing.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `>>> List remote models failed` in Web UI | Make sure URL is `wss://…ngrok-free.app` (no `/chat` suffix). Also Cell 4 must still be running on Colab. |
| Tunnel disconnects after 2 hours | Set `NGROK_AUTHTOKEN` (Cell 2). |
| Tokens don't appear | Check Cell 4 output for `>>> [/chat] client connected`. If absent, the Web UI didn't reach Colab — verify the URL. |
| `pip install` fails on Colab | Re-run Cell 1; transient PyPI issues are common on Colab. |
| Models tab on local Web UI doesn't show Colab models | The local Models tab shows only **local** models. Cloud-side models appear in the dropdown when creating a new Cloud chat. |

---

## Security checklist

- [x] `wss://` (NOT `ws://`) — TLS mandatory
- [x] `NGROK_AUTHTOKEN` not committed to GitHub
- [x] Don't share the `wss://…ngrok` URL — anyone with it can chat with your brain
- [x] Stop the Colab runtime when done — frees Colab's RAM
- [x] Back up `models/` to Google Drive periodically
- [x] For sensitive data — use Tailscale + your own GPU instead of Colab
