// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

let selectedFile = null;

// ── File input ───────────────────────────────────────────────────────────────
const emlInput = document.getElementById('eml-input');
const dropArea = document.getElementById('drop-area');

emlInput.addEventListener('change', () => {
  if (emlInput.files[0]) setFile(emlInput.files[0]);
});

// Drag and drop
dropArea.addEventListener('dragover', e => {
  e.preventDefault();
  dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
  dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', e => {
  e.preventDefault();
  dropArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.eml')) {
    setFile(file);
  } else {
    showError('Please drop a valid .eml file.');
  }
});

function setFile(file) {
  selectedFile = file;
  document.getElementById('file-name').textContent = file.name;
  dropArea.classList.add('has-file');
  dropArea.querySelector('.drop-icon').textContent = '✉️';
  dropArea.querySelector('.drop-primary').innerHTML = `<strong>${file.name}</strong>`;
  dropArea.querySelector('.drop-secondary').textContent = `${(file.size / 1024).toFixed(1)} KB — click to change`;
  hideError();
  hideResult();
}

// ── Scan ─────────────────────────────────────────────────────────────────────
async function runScan() {
  if (!selectedFile) {
    showError('Please select a .eml file first.');
    return;
  }

  setLoading(true);
  hideError();
  hideResult();

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const res = await fetch(`${API_BASE}/predict/eml`, {
      method: 'POST',
      body: formData,
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
  const isSpam   = data.is_spam;
  const isMaybe  = data.maybe_spam;
  const prob     = data.spam_probability;
  const pct      = Math.round(prob * 100);
  const upperPct = Math.round((data.maybe_spam_upper_threshold ?? 0.5) * 100);

  const card = document.getElementById('result-card');

  // Three-state class
  if (isSpam) {
    card.className = 'result-card spam';
  } else if (isMaybe) {
    card.className = 'result-card maybe';
  } else {
    card.className = 'result-card ham';
  }

  // Icon / headline / sub
  if (isSpam) {
    document.getElementById('verdict-icon').textContent = '🚨';
    document.getElementById('verdict-text').textContent = 'Spam detected';
    document.getElementById('verdict-sub').textContent  = 'Classified as spam by the ensemble';
  } else if (isMaybe) {
    document.getElementById('verdict-icon').textContent = '⚠️';
    document.getElementById('verdict-text').textContent = 'Maybe spam';
    document.getElementById('verdict-sub').textContent  = `The email has some spam signals but is not classified as spam by the ensemble`;
  } else {
    document.getElementById('verdict-icon').textContent = '✅';
    document.getElementById('verdict-text').textContent = 'Looks clean';
    document.getElementById('verdict-sub').textContent  = 'No spam signals found — classified as ham';
  }

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
    document.getElementById('roberta-prob').className = 'm-prob ' + modelColorClass(data.roberta.spam_probability, data.roberta.threshold);
    document.getElementById('roberta-verdict').textContent = modelVerdictLabel(data.roberta.spam_probability, data.roberta.threshold);
    document.getElementById('roberta-verdict').className = 'm-verdict ' + modelColorClass(data.roberta.spam_probability, data.roberta.threshold);
  }

  if (data.electra) {
    const ep = Math.round(data.electra.spam_probability * 100);
    document.getElementById('electra-prob').textContent = ep + '%';
    document.getElementById('electra-prob').className = 'm-prob ' + modelColorClass(data.electra.spam_probability, data.electra.threshold);
    document.getElementById('electra-verdict').textContent = modelVerdictLabel(data.electra.spam_probability, data.electra.threshold);
    document.getElementById('electra-verdict').className = 'm-verdict ' + modelColorClass(data.electra.spam_probability, data.electra.threshold);
  }

  // Animate in
  requestAnimationFrame(() => { card.classList.add('visible'); });
}

function modelColorClass(prob, threshold) {
  const upper = 0.5;
  if (prob >= upper)            return 'col-spam';
  if (prob >= threshold)        return 'col-maybe';
  return 'col-ham';
}

function modelVerdictLabel(prob, threshold) {
  const upper = 0.5;
  if (prob >= upper)     return '🚨 Spam';
  if (prob >= threshold) return '⚠️ Maybe';
  return '✅ Ham';
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('scan-btn').disabled = on;
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('btn-label').textContent  = on ? 'Scanning…' : 'Scan email';
}

function hideResult() {
  const card = document.getElementById('result-card');
  card.classList.remove('visible');
  document.getElementById('prob-bar').style.width = '0%';
}

function clearAll() {
  selectedFile = null;
  document.getElementById('eml-input').value = '';
  document.getElementById('file-name').textContent = 'No file selected';
  dropArea.classList.remove('has-file', 'dragover');
  dropArea.querySelector('.drop-icon').textContent = '📨';
  dropArea.querySelector('.drop-primary').innerHTML = 'Drop your <strong>.eml</strong> file here';
  dropArea.querySelector('.drop-secondary').textContent = 'or click to browse';
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
  navigator.clipboard.writeText(`${API_BASE}/predict/eml`).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.textContent = '✓';
    setTimeout(() => btn.textContent = '⎘', 1500);
  });
}