// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

// ── Char counter ─────────────────────────────────────────────────────────────
const input = document.getElementById('msg-input');
input.addEventListener('input', () => {
  document.getElementById('char-count').textContent = `${input.value.length} / 2000`;
});

// ── Scan ─────────────────────────────────────────────────────────────────────
async function runScan() {
  const text = input.value.trim();
  if (!text) return;

  setLoading(true);
  hideError();
  hideResult();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model: 'ensemble' }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Request failed');
    }

    const data = await res.json();
    renderResult(data);
  } catch (e) {
    showError(e.message.includes('Failed to fetch')
      ? 'Cannot reach the API. Make sure the server is running at ' + API_BASE
      : e.message);
  } finally {
    setLoading(false);
  }
}

function renderResult(data) {
  const isSpam = data.is_spam;
  const prob   = data.spam_probability;
  const pct    = Math.round(prob * 100);

  const card = document.getElementById('result-card');
  card.className = 'result-card ' + (isSpam ? 'spam' : 'ham');

  document.getElementById('verdict-icon').textContent = isSpam ? '🚨' : '✅';
  document.getElementById('verdict-text').textContent = isSpam ? 'Spam detected' : 'Looks clean';
  document.getElementById('verdict-sub').textContent  = isSpam
    ? 'Classified as spam by the ensemble'
    : 'No spam signals found — classified as ham';

  document.getElementById('prob-big').textContent = pct + '%';
  document.getElementById('threshold-label').textContent =
    `Threshold: ${Math.round(data.ensemble_threshold * 100)}%`;

  // Animate bar after paint
  requestAnimationFrame(() => {
    document.getElementById('prob-bar').style.width = pct + '%';
  });

  // Individual model scores
  if (data.roberta) {
    const rp = Math.round(data.roberta.spam_probability * 100);
    document.getElementById('roberta-prob').textContent = rp + '%';
    document.getElementById('roberta-prob').className = 'm-prob ' + (data.roberta.is_spam ? 'col-spam' : 'col-ham');
    document.getElementById('roberta-verdict').textContent = data.roberta.is_spam ? '🚨 Spam' : '✅ Ham';
    document.getElementById('roberta-verdict').className = 'm-verdict ' + (data.roberta.is_spam ? 'col-spam' : 'col-ham');
  }

  if (data.electra) {
    const ep = Math.round(data.electra.spam_probability * 100);
    document.getElementById('electra-prob').textContent = ep + '%';
    document.getElementById('electra-prob').className = 'm-prob ' + (data.electra.is_spam ? 'col-spam' : 'col-ham');
    document.getElementById('electra-verdict').textContent = data.electra.is_spam ? '🚨 Spam' : '✅ Ham';
    document.getElementById('electra-verdict').className = 'm-verdict ' + (data.electra.is_spam ? 'col-spam' : 'col-ham');
  }

  // Animate in
  requestAnimationFrame(() => { card.classList.add('visible'); });
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('scan-btn').disabled = on;
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('btn-label').textContent  = on ? 'Scanning…' : 'Scan message';
}

function hideResult() {
  const card = document.getElementById('result-card');
  card.classList.remove('visible');
  document.getElementById('prob-bar').style.width = '0%';
}

function clearAll() {
  input.value = '';
  document.getElementById('char-count').textContent = '0 / 2000';
  hideResult();
  hideError();
}

function showError(msg) {
  const el = document.getElementById('error-banner');
  el.textContent = '⚠ ' + msg;
  el.style.display = 'block';
}

function hideError() {
  document.getElementById('error-banner').style.display = 'none';
}

function copyAPI() {
  navigator.clipboard.writeText(`${API_BASE}/predict`).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = '✓';
    setTimeout(() => btn.textContent = '⎘', 1500);
  });
}

// Enter to submit (Shift+Enter = newline)
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runScan(); }
});
