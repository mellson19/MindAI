// MindAI Web GUI — Local | Cloud chat client

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  activeChatId: null,
  activeChatMeta: null,
  chats: [],
  models: [],
  currentTab: 'chats',
  newChatMode: 'local',
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

const api = {
  getChats:  () => fetch('/chats').then(r => r.json()),
  newChat:   (b) => fetch('/chats', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(b) }).then(r => r.json()),
  openChat:  (id) => fetch(`/chats/${id}/open`, { method:'POST' }).then(r => r.json()),
  getChat:   (id) => fetch(`/chats/${id}`).then(r => r.json()),
  delChat:   (id) => fetch(`/chats/${id}`, { method:'DELETE' }),
  sendPrompt: (id, text) => fetch(`/chats/${id}/prompt`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text }) }),
  uploadFile: (id, file) => { const fd = new FormData(); fd.append('file', file); return fetch(`/chats/${id}/upload`, { method:'POST', body: fd }).then(r => r.json()); },
  saveChat:   (id) => fetch(`/chats/${id}/save`, { method:'POST' }).then(r => r.json()),

  getModels: () => fetch('/models').then(r => r.json()),
  newModel:  (b) => fetch('/models', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(b) }).then(r => r.json()),
  delModel:  (id) => fetch(`/models/${id}`, { method:'DELETE' }),
  remoteModels: (url) => fetch(`/remote/models?url=${encodeURIComponent(url)}`).then(r => r.json()),

  getVoice: () => fetch('/voice').then(r => r.json()),
  setVoice: (b) => fetch('/voice', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(b) }).then(r => r.json()),
  tts:      (text) => fetch('/tts', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text }) }),
};

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

let ws = null;
let currentBrainBubble = null;
let lastKnownChatId = null;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen  = () => { $('liveDot').className = 'dot live'; $('connState').textContent = 'connected'; };
  ws.onclose = () => {
    $('liveDot').className = 'dot';
    $('connState').textContent = 'reconnecting…';
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    let m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    if      (m.op === 'telemetry')   updateMetrics(m);
    else if (m.op === 'token_chunk') appendBrainChunk(m.text, m.chat_id);
    else if (m.op === 'voice_info')  refreshVoice();
    else if (m.op === 'state')       handleStateSync(m);
  };
}
connect();

function handleStateSync(m) {
  if (m.active_chat) lastKnownChatId = m.active_chat;
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    state.currentTab = btn.dataset.tab;
    if (state.currentTab === 'chats') {
      $('viewChat').classList.remove('hidden');
      $('viewModels').classList.add('hidden');
    } else {
      $('viewChat').classList.add('hidden');
      $('viewModels').classList.remove('hidden');
      renderModels();
    }
  });
});

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

function updateMetrics(m) {
  $('m-tick').textContent     = (m.tick || 0).toLocaleString();
  $('m-mood').textContent     = m.mood     || '—';
  $('m-surprise').textContent = m.surprise ?? '—';
  $('m-da').textContent       = m.dopamine ?? '—';
  $('m-cort').textContent     = m.cortisol ?? '—';
  $('m-state').textContent    = m.sleep ? m.sleep_phase : (m.pag_mode || 'awake');
  $('liveDot').className      = m.sleep ? 'dot sleep' : 'dot live';
}

// ---------------------------------------------------------------------------
// Chats list
// ---------------------------------------------------------------------------

async function refreshChats() {
  state.chats = await api.getChats();
  renderChats();
}

function renderChats() {
  const list = $('chatsList');
  list.innerHTML = '';
  for (const c of state.chats) {
    const div = document.createElement('div');
    div.className = 'chat-item' + (c.id === state.activeChatId ? ' active' : '');
    div.innerHTML = `
      <span class="ci-name">${escapeHtml(c.name)}</span>
      <span class="ci-mode ${c.mode}">${c.mode}</span>
      <button class="ci-del" title="Delete">✕</button>
    `;
    div.onclick = (e) => {
      if (e.target.classList.contains('ci-del')) return;
      openChat(c.id);
    };
    div.querySelector('.ci-del').onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete chat "${c.name}"?`)) return;
      await api.delChat(c.id);
      if (state.activeChatId === c.id) {
        state.activeChatId = null;
        renderActiveChat();
      }
      refreshChats();
    };
    list.appendChild(div);
  }
}

async function openChat(chatId) {
  state.activeChatId = chatId;
  state.activeChatMeta = state.chats.find(c => c.id === chatId);
  renderChats();
  $('chat').innerHTML = '';
  currentBrainBubble = null;

  // Load history
  const chat = await api.getChat(chatId);
  if (chat && chat.messages) {
    for (const m of chat.messages) {
      if (m.role === 'user')  addUserMsg(m.text);
      else if (m.role === 'brain') addBrainMsg(m.text);
    }
  }

  // Tell server to switch
  const r = await api.openChat(chatId);
  if (r.error) { toast('open failed: ' + r.error); return; }

  $('activeChatName').textContent = chat?.name || chatId;
  const badge = $('activeChatBadge');
  badge.className = 'badge ' + (chat?.mode || '');
  badge.textContent = chat?.mode || '';

  refreshVoice();
}

function renderActiveChat() {
  if (!state.activeChatId) {
    $('activeChatName').textContent = 'No chat selected';
    $('activeChatBadge').textContent = '';
    $('chat').innerHTML = `<div class="welcome"><h1>Hello.</h1><p class="muted">Select a chat or click <b>New chat</b>.</p></div>`;
  }
}

// ---------------------------------------------------------------------------
// New chat modal
// ---------------------------------------------------------------------------

const ncModal     = $('newChatModal');
const ncName      = $('ncName');
const ncCloudUrl  = $('ncCloudUrl');
const ncCloudWrap = $('ncCloudUrlWrap');
const ncSelect    = $('ncModelSelect');

$('btnNewChat').onclick = async () => {
  ncName.value = '';
  ncCloudUrl.value = '';
  state.newChatMode = 'local';
  document.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === 'local'));
  ncCloudWrap.classList.add('hidden');
  await populateModelSelect('local');
  ncModal.classList.add('open');
  ncName.focus();
};

document.querySelectorAll('.mode-btn').forEach(b => {
  b.onclick = async () => {
    document.querySelectorAll('.mode-btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    state.newChatMode = b.dataset.mode;
    if (state.newChatMode === 'cloud') {
      ncCloudWrap.classList.remove('hidden');
      ncSelect.innerHTML = '<option>Enter cloud URL and click refresh →</option>';
    } else {
      ncCloudWrap.classList.add('hidden');
      await populateModelSelect('local');
    }
  };
});

$('ncFetchRemote').onclick = async () => {
  const url = ncCloudUrl.value.trim();
  if (!url) return toast('enter cloud URL first');
  ncSelect.innerHTML = '<option>Loading…</option>';
  try {
    const list = await api.remoteModels(url);
    ncSelect.innerHTML = '';
    if (!list.length) {
      ncSelect.innerHTML = '<option value="default">default (server has no models endpoint)</option>';
      return;
    }
    for (const m of list) {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = `${m.name} — ${(m.num_neurons || 0).toLocaleString()} neurons, tick ${(m.tick || 0).toLocaleString()}`;
      ncSelect.appendChild(o);
    }
  } catch (e) {
    toast('fetch failed: ' + e);
  }
};

async function populateModelSelect(mode) {
  if (mode === 'local') {
    state.models = await api.getModels();
    ncSelect.innerHTML = '';
    for (const m of state.models) {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = `${m.name} — tick ${m.tick.toLocaleString()}, ${m.num_neurons.toLocaleString()} neurons`;
      ncSelect.appendChild(o);
    }
    if (state.models.length === 0) {
      ncSelect.innerHTML = '<option value="default">default (will be created on first save)</option>';
    }
  }
}

$('ncCancel').onclick = () => ncModal.classList.remove('open');
$('ncCreate').onclick = async () => {
  const name = ncName.value.trim() || 'New chat';
  const mode = state.newChatMode;
  const model_id = ncSelect.value || 'default';
  const cloud_url = mode === 'cloud' ? ncCloudUrl.value.trim() : null;

  if (mode === 'cloud' && !cloud_url) return toast('cloud URL required');

  const chat = await api.newChat({ name, mode, model_id, cloud_url });
  if (chat.error) return toast('create failed: ' + chat.error);

  ncModal.classList.remove('open');
  await refreshChats();
  openChat(chat.id);
};

// ---------------------------------------------------------------------------
// New model modal
// ---------------------------------------------------------------------------

const nmModal = $('newModelModal');
const nmName  = $('nmName');
const nmBase  = $('nmBaseSelect');

$('btnNewModel').onclick = async () => {
  nmName.value = '';
  const models = await api.getModels();
  nmBase.innerHTML = '<option value="">— None (fresh brain) —</option>';
  for (const m of models) {
    const o = document.createElement('option');
    o.value = m.id; o.textContent = m.name;
    nmBase.appendChild(o);
  }
  nmModal.classList.add('open');
  nmName.focus();
};

$('nmCancel').onclick = () => nmModal.classList.remove('open');
$('nmCreate').onclick = async () => {
  const name = nmName.value.trim();
  if (!name) return toast('name required');
  await api.newModel({ name, base: nmBase.value || null });
  nmModal.classList.remove('open');
  renderModels();
};

// ---------------------------------------------------------------------------
// Models view
// ---------------------------------------------------------------------------

async function renderModels() {
  const list = $('modelsList');
  list.innerHTML = '<p class="muted small">Loading…</p>';
  const models = await api.getModels();
  state.models = models;
  list.innerHTML = '';
  for (const m of models) {
    const card = document.createElement('div');
    card.className = 'model-card';
    const lastUsed = m.last_used ? new Date(m.last_used * 1000).toLocaleDateString() : 'never';
    const created  = m.created   ? new Date(m.created  * 1000).toLocaleDateString() : '—';
    card.innerHTML = `
      <h3>${escapeHtml(m.name)}</h3>
      <div class="stats">
        <span>id<br/><strong>${escapeHtml(m.id)}</strong></span>
        <span>neurons<br/><strong>${(m.num_neurons || 0).toLocaleString()}</strong></span>
        <span>tick<br/><strong>${(m.tick || 0).toLocaleString()}</strong></span>
        <span>size<br/><strong>${m.size_mb} MB</strong></span>
        <span>created<br/><strong>${created}</strong></span>
        <span>last used<br/><strong>${lastUsed}</strong></span>
        <span>mood<br/><strong>${escapeHtml(m.mood || '—')}</strong></span>
        <span>voice<br/><strong>${escapeHtml(m.voice || '—')}</strong></span>
      </div>
      <div class="actions">
        <button class="ghost small" data-action="clone">Duplicate</button>
        <button class="ghost small" data-action="delete">Delete</button>
      </div>
    `;
    card.querySelector('[data-action="clone"]').onclick = async () => {
      const name = prompt('Name for the copy:', m.name + ' (copy)');
      if (!name) return;
      await api.newModel({ name, base: m.id });
      renderModels();
    };
    card.querySelector('[data-action="delete"]').onclick = async () => {
      if (!confirm(`Delete model "${m.name}"? This cannot be undone.`)) return;
      const r = await api.delModel(m.id);
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        toast(j.error || 'delete failed');
      }
      renderModels();
    };
    list.appendChild(card);
  }
  if (!models.length) {
    list.innerHTML = '<p class="muted">No models yet. Click <b>New model</b> above.</p>';
  }
}

// ---------------------------------------------------------------------------
// Chat bubbles
// ---------------------------------------------------------------------------

function clearWelcome() { document.querySelector('.welcome')?.remove(); }

function addUserMsg(text, mediaUrl, mediaKind) {
  clearWelcome();
  const div = document.createElement('div');
  div.className = 'msg user';
  if (text) div.textContent = text;
  if (mediaUrl && mediaKind) {
    let m;
    if (mediaKind === 'image') m = document.createElement('img');
    else if (mediaKind === 'video') { m = document.createElement('video'); m.controls = true; }
    else { m = document.createElement('audio'); m.controls = true; }
    m.src = mediaUrl;
    m.className = 'media-thumb';
    div.appendChild(m);
  }
  $('chat').appendChild(div);
  scrollChat();
}

function addBrainMsg(text) {
  clearWelcome();
  const div = document.createElement('div');
  div.className = 'msg brain';
  div.textContent = text;
  $('chat').appendChild(div);
  scrollChat();
}

function addSystemMsg(text) {
  clearWelcome();
  const div = document.createElement('div');
  div.className = 'msg system';
  div.textContent = text;
  $('chat').appendChild(div);
  scrollChat();
}

function startBrainBubble() {
  clearWelcome();
  const div = document.createElement('div');
  div.className = 'msg brain streaming';
  $('chat').appendChild(div);
  currentBrainBubble = div;
  scrollChat();
  return div;
}

function appendBrainChunk(text, chatId) {
  if (chatId && chatId !== state.activeChatId) return;
  if (!currentBrainBubble) startBrainBubble();
  currentBrainBubble.textContent = (currentBrainBubble.textContent || '') + text;
  scrollChat();
  scheduleSpeak();
}

let speakTimer = null;
function scheduleSpeak() {
  clearTimeout(speakTimer);
  speakTimer = setTimeout(() => {
    if (currentBrainBubble) {
      currentBrainBubble.classList.remove('streaming');
      const finalText = currentBrainBubble.textContent || '';
      currentBrainBubble = null;
      if (finalText.trim()) speakText(finalText);
    }
  }, 1500);
}

function scrollChat() { $('chat').scrollTop = $('chat').scrollHeight; }

// ---------------------------------------------------------------------------
// Composer
// ---------------------------------------------------------------------------

$('btnSend').addEventListener('click', sendPrompt);
$('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendPrompt(); }
});
$('prompt').addEventListener('input', () => {
  $('prompt').style.height = 'auto';
  $('prompt').style.height = Math.min(200, $('prompt').scrollHeight) + 'px';
});

function sendPrompt() {
  const text = $('prompt').value.trim();
  if (!text) return;
  if (!state.activeChatId) return toast('select or create a chat first');
  addUserMsg(text);
  api.sendPrompt(state.activeChatId, text);
  $('prompt').value = '';
  $('prompt').style.height = 'auto';
}

$('btnSave').addEventListener('click', async () => {
  if (!state.activeChatId) return;
  const r = await api.saveChat(state.activeChatId);
  if (r.ok) toast(`saved at tick ${r.tick}`);
});

$('btnClear').addEventListener('click', async () => {
  if (!state.activeChatId) return;
  await fetch(`/chats/${state.activeChatId}/upload`, { method: 'POST' }).catch(() => {});
  toast('cleared');
});

// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------

$('btnAttach').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', () => {
  if ($('fileInput').files[0]) uploadFile($('fileInput').files[0]);
  $('fileInput').value = '';
});

let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; $('dropzone').classList.add('active'); });
window.addEventListener('dragleave', (e) => { e.preventDefault(); dragDepth = Math.max(0, dragDepth - 1); if (dragDepth === 0) $('dropzone').classList.remove('active'); });
window.addEventListener('dragover',  (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault(); dragDepth = 0;
  $('dropzone').classList.remove('active');
  const f = e.dataTransfer.files[0];
  if (f) uploadFile(f);
});

async function uploadFile(file) {
  if (!state.activeChatId) return toast('open a chat first');
  const url = URL.createObjectURL(file);
  let kind = 'audio';
  if (file.type.startsWith('image/')) kind = 'image';
  else if (file.type.startsWith('video/')) kind = 'video';
  addUserMsg(file.name, url, kind);
  const j = await api.uploadFile(state.activeChatId, file);
  if (j.transcribed) addSystemMsg(`heard: "${j.transcribed}"`);
}

// ---------------------------------------------------------------------------
// Microphone
// ---------------------------------------------------------------------------

const btnMic = $('btnMic');
let mediaRec = null, chunks = [];

async function startRecording() {
  if (!state.activeChatId) return toast('open a chat first');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRec = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    chunks = [];
    mediaRec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    mediaRec.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const file = new File([blob], 'voice.webm', { type: 'audio/webm' });
      const j = await api.uploadFile(state.activeChatId, file);
      if (j.transcribed) addUserMsg(j.transcribed);
    };
    mediaRec.start();
    btnMic.classList.add('recording');
  } catch (e) { toast('microphone blocked'); }
}

function stopRecording() {
  if (mediaRec && mediaRec.state === 'recording') mediaRec.stop();
  btnMic.classList.remove('recording');
}

btnMic.addEventListener('mousedown',   startRecording);
btnMic.addEventListener('touchstart',  (e) => { e.preventDefault(); startRecording(); });
btnMic.addEventListener('mouseup',     stopRecording);
btnMic.addEventListener('mouseleave',  stopRecording);
btnMic.addEventListener('touchend',    stopRecording);

// ---------------------------------------------------------------------------
// Voice picker
// ---------------------------------------------------------------------------

const popover     = $('voicePopover');
const voiceSelect = $('voiceSelect');
const pitchSlider = $('pitchSlider');
const pitchVal    = $('pitchVal');
const rateSlider  = $('rateSlider');
const rateVal     = $('rateVal');

async function refreshVoice() {
  try {
    const j = await api.getVoice();
    if (!j.options) return;
    voiceSelect.innerHTML = '';
    for (const o of j.options) {
      const opt = document.createElement('option');
      opt.value = o.id;
      opt.textContent = `${o.name} — ${o.accent}, ${o.gender}`;
      voiceSelect.appendChild(opt);
    }
    if (j.current) {
      voiceSelect.value     = j.current.base_voice;
      pitchSlider.value     = j.current.pitch_shift.toFixed(1);
      pitchVal.textContent  = formatPitch(pitchSlider.value);
      rateSlider.value      = j.current.rate.toFixed(2);
      rateVal.textContent   = `${(+rateSlider.value).toFixed(2)}x`;
    }
  } catch (e) { /* ignore */ }
}

const formatPitch = (v) => `${v >= 0 ? '+' : ''}${(+v).toFixed(1)} Hz`;
pitchSlider.addEventListener('input', () => pitchVal.textContent = formatPitch(pitchSlider.value));
rateSlider .addEventListener('input', () => rateVal.textContent  = `${(+rateSlider.value).toFixed(2)}x`);

$('btnVoice').addEventListener('click', (e) => {
  e.stopPropagation(); popover.classList.toggle('open');
});
document.addEventListener('click', (e) => {
  if (popover.classList.contains('open') && !popover.contains(e.target) && e.target.id !== 'btnVoice') {
    popover.classList.remove('open');
  }
});

async function applyVoice() {
  const r = await api.setVoice({
    base_voice:  voiceSelect.value,
    pitch_shift: parseFloat(pitchSlider.value),
    rate:        parseFloat(rateSlider.value),
  });
  if (!r.ok) return toast(r.error || 'voice update failed');
  toast(`voice: ${r.current.base_voice}`);
}

$('voiceApply').addEventListener('click', async () => { await applyVoice(); popover.classList.remove('open'); });
$('voicePreview').addEventListener('click', async () => { await applyVoice(); speakText('Hi. This is how I sound.'); });

// ---------------------------------------------------------------------------
// TTS playback
// ---------------------------------------------------------------------------

let ttsAudio = null;
async function speakText(text) {
  try {
    if (ttsAudio) { ttsAudio.pause(); ttsAudio = null; }
    const r = await api.tts(text);
    if (!r.ok) return;
    const blob = await r.blob();
    ttsAudio = new Audio(URL.createObjectURL(blob));
    ttsAudio.play().catch(() => {});
  } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

function toast(msg) { addSystemMsg(msg); }

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

(async () => {
  await refreshChats();
  if (state.chats.length > 0) {
    openChat(state.chats[0].id);
  }
})();
