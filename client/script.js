// Patient-IA — cliente mínimo: login + chat + informe de cohorte
const API = `${window.location.origin}/api/v1`;
const TOKEN_KEY = 'patientia_token';
const USER_KEY = 'patientia_user';

let currentDatasetId = null;

const $ = (id) => document.getElementById(id);

function getToken() { return localStorage.getItem(TOKEN_KEY); }
function authHeaders(extra = {}) {
  const t = getToken();
  return t ? { ...extra, Authorization: `Bearer ${t}` } : extra;
}

function showApp() {
  $('login-view').classList.add('hidden');
  $('app-view').classList.remove('hidden');
  $('who').textContent = localStorage.getItem(USER_KEY) || '';
}
function showLogin() {
  $('app-view').classList.add('hidden');
  $('login-view').classList.remove('hidden');
}

// ------------------------------- Auth ---------------------------------------
async function login(username, password) {
  const res = await fetch(`${API}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error('Credenciales inválidas');
  const data = await res.json();
  const token = data.data.access_token;
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, data.data.username);
}

function logout() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  currentDatasetId = null;
  showLogin();
}

async function verifySession() {
  if (!getToken()) return false;
  try {
    const res = await fetch(`${API}/auth/me`, { headers: authHeaders() });
    return res.ok;
  } catch { return false; }
}

// ------------------------------- Dataset ------------------------------------
async function uploadDataset(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API}/datasets/upload`, { method: 'POST', headers: authHeaders(), body: form });
  if (res.status === 401) { logout(); throw new Error('Sesión caducada'); }
  if (!res.ok) throw new Error('Error subiendo el dataset');
  const data = await res.json();
  const info = data.data.dataset_info;
  currentDatasetId = info.id || info.dataset_id;
  return info;
}

// ------------------------------- Report -------------------------------------
async function generateReport() {
  if (!currentDatasetId) return;
  const model = $('model-type').value;
  const num = parseInt($('num-samples').value, 10) || 50;
  $('report-status').textContent = '⏳ Ejecutando pipeline (analyzer → generator → validator → evaluator → simulator)…';
  $('report-btn').disabled = true;
  try {
    const res = await fetch(`${API}/report`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ dataset_id: currentDatasetId, model_type: model, num_samples: num }),
    });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Error generando el informe');
    }
    const data = await res.json();
    $('report-frame').srcdoc = data.data.report_html;
    $('report-status').textContent = `✅ Informe generado (${data.data.num_samples} pacientes, agentes: ${(data.data.agents || []).length}).`;
  } catch (e) {
    $('report-status').textContent = `❌ ${e.message}`;
  } finally {
    $('report-btn').disabled = false;
  }
}

// -------------------------------- Chat --------------------------------------
function appendMsg(text, who) {
  const div = document.createElement('div');
  div.className = `msg ${who}`;
  div.textContent = text;
  $('chat-log').appendChild(div);
  $('chat-log').scrollTop = $('chat-log').scrollHeight;
}

async function sendChat(message) {
  appendMsg(message, 'user');
  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ message, context: currentDatasetId ? { dataset_id: currentDatasetId } : {} }),
    });
    if (res.status === 401) { logout(); return; }
    const data = await res.json();
    const reply = (data.data && data.data.chat_response && data.data.chat_response.message) || 'Sin respuesta.';
    appendMsg(reply, 'assistant');
  } catch (e) {
    appendMsg(`Error: ${e.message}`, 'assistant');
  }
}

// ------------------------------- Wiring -------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
  if (await verifySession()) showApp(); else showLogin();

  $('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    $('login-error').textContent = '';
    try {
      await login($('username').value.trim(), $('password').value);
      showApp();
    } catch (err) {
      $('login-error').textContent = err.message;
    }
  });

  $('logout-btn').addEventListener('click', logout);

  $('upload-btn').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    $('dataset-status').textContent = '⏳ Subiendo…';
    try {
      const info = await uploadDataset(file);
      $('dataset-status').textContent = `✅ ${info.filename} — ${info.rows} filas × ${info.columns} columnas`;
      $('dataset-status').classList.remove('muted');
      $('report-btn').disabled = false;
    } catch (err) {
      $('dataset-status').textContent = `❌ ${err.message}`;
    }
  });

  $('report-btn').addEventListener('click', generateReport);

  $('chat-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const val = $('chat-input').value.trim();
    if (!val) return;
    $('chat-input').value = '';
    sendChat(val);
  });
});
