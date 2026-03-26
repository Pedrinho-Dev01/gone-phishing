// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = 'https://dpedrinho01-api-host.hf.space';

let selectedFile = null;

// Emotion metadata: icon + CSS variable name
const EMOTION_META = {
  admiration: { icon: '🤩', color: 'var(--emo-admiration)' },
  anger:      { icon: '😡', color: 'var(--emo-anger)' },
  caring:     { icon: '🤗', color: 'var(--emo-caring)' },
  confusion:  { icon: '😕', color: 'var(--emo-confusion)' },
  curiosity:  { icon: '🧐', color: 'var(--emo-curiosity)' },
  desire:     { icon: '🔥', color: 'var(--emo-desire)' },
  excitement: { icon: '🎉', color: 'var(--emo-excitement)' },
  fear:       { icon: '😨', color: 'var(--emo-fear)' },
  gratitude:  { icon: '🙏', color: 'var(--emo-gratitude)' },
  joy:        { icon: '😊', color: 'var(--emo-joy)' },
  neutral:    { icon: '😐', color: 'var(--emo-neutral)' },
  relief:     { icon: '😮‍💨', color: 'var(--emo-relief)' },
  sadness:    { icon: '😢', color: 'var(--emo-sadness)' },
  surprise:   { icon: '😲', color: 'var(--emo-surprise)' },
  unsure:     { icon: '🤷', color: 'var(--emo-unsure)' },
};

// ── File input ───────────────────────────────────────────────────────────────
const emlInput = document.getElementById('eml-input');
const dropArea = document.getElementById('drop-area');

emlInput.addEventListener('change', () => {
  if (emlInput.files[0]) setFile(emlInput.files[0]);
});

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
    const arrayBuffer = await selectedFile.arrayBuffer();
    const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

    const res = await fetch(`${API_BASE}/predict/eml`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: selectedFile.name, content: base64 }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Request failed');
    }

    const data = await res.json();

    // New API returns { spam: {...}, emotion: {...} }
    // Support old flat response too, for backwards compatibility
    if (data.spam && data.emotion) {
      renderSpamResult(data.spam);
      renderEmotionResult(data.emotion);
    } else {
      // Old flat response — spam only
      renderSpamResult(data);
    }
  } catch (e) {
    showError(e.message.includes('Failed to fetch')
      ? 'Cannot reach the API. Make sure the server is running at ' + API_BASE
      : e.message);
  } finally {
    setLoading(false);
  }
}

// ── Spam result renderer ─────────────────────────────────────────────────────
function renderSpamResult(data) {
  const isSpam  = data.is_spam;
  const isMaybe = data.maybe_spam;
  const prob    = data.spam_probability;
  const pct     = Math.round(prob * 100);

  const card = document.getElementById('result-card');

  if (isSpam)       card.className = 'result-card spam';
  else if (isMaybe) card.className = 'result-card maybe';
  else              card.className = 'result-card ham';

  if (isSpam) {
    document.getElementById('verdict-icon').textContent = '🚨';
    document.getElementById('verdict-text').textContent = 'Spam detected';
    document.getElementById('verdict-sub').textContent  = 'Classified as spam by the ensemble';
  } else if (isMaybe) {
    document.getElementById('verdict-icon').textContent = '⚠️';
    document.getElementById('verdict-text').textContent = 'Maybe spam';
    document.getElementById('verdict-sub').textContent  = 'The email has some spam signals but is not classified as spam by the ensemble';
  } else {
    document.getElementById('verdict-icon').textContent = '✅';
    document.getElementById('verdict-text').textContent = 'Looks clean';
    document.getElementById('verdict-sub').textContent  = 'No spam signals found — classified as ham';
  }

  document.getElementById('prob-big').textContent = pct + '%';
  document.getElementById('threshold-label').textContent =
    `Threshold: ${Math.round(data.ensemble_threshold * 100)}%`;

  requestAnimationFrame(() => {
    document.getElementById('prob-bar').style.width = pct + '%';
  });

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

  requestAnimationFrame(() => { card.classList.add('visible'); });
}

function modelColorClass(prob, threshold) {
  if (prob >= 0.5)      return 'col-spam';
  if (prob >= threshold) return 'col-maybe';
  return 'col-ham';
}

function modelVerdictLabel(prob, threshold) {
  if (prob >= 0.5)      return '🚨 Spam';
  if (prob >= threshold) return '⚠️ Maybe';
  return '✅ Ham';
}

// ── Emotion result renderer ───────────────────────────────────────────────────
function renderEmotionResult(data) {
  const card    = document.getElementById('emotion-card');
  const detected = data.detected_emotions || [];
  const scores   = data.all_scores || [];      // sorted by probability desc

  // Subtitle
  const subtitle = document.getElementById('emotion-subtitle');
  if (detected.length === 0) {
    subtitle.textContent = 'No strong emotions detected in this email.';
  } else {
    subtitle.textContent = `${detected.length} emotion${detected.length > 1 ? 's' : ''} detected by the ensemble.`;
  }

  // Detected chips
  const chipsWrap = document.getElementById('emotion-detected-wrap');
  chipsWrap.innerHTML = '';
  detected.forEach((emo, i) => {
    const meta = EMOTION_META[emo] || { icon: '•', color: 'var(--muted)' };
    const chip = document.createElement('span');
    chip.className = 'emo-chip';
    chip.style.cssText = `color:${meta.color};border-color:${meta.color}33;background:${meta.color}18;animation-delay:${i * 60}ms`;
    chip.innerHTML = `<span class="emo-chip-dot"></span>${meta.icon} ${emo}`;
    chipsWrap.appendChild(chip);
  });

  // Bar chart — show top 8 by probability
  const barsContainer = document.getElementById('emotion-bars');
  barsContainer.innerHTML = '';

  if (scores.length === 0) {
    barsContainer.innerHTML = '<div class="emotion-empty">No emotion scores returned.</div>';
  } else {
    const topN = scores.slice(0, 8);
    topN.forEach((score, i) => {
      const meta  = EMOTION_META[score.emotion] || { icon: '•', color: 'var(--muted)' };
      const pct   = Math.round(score.probability * 100);
      const thPct = Math.round(score.threshold * 100);

      const row = document.createElement('div');
      row.className = 'emo-bar-row';
      row.style.animationDelay = `${i * 40}ms`;
      row.innerHTML = `
        <div class="emo-bar-label">
          <span class="emo-bar-label-icon">${meta.icon}</span>
          <span>${score.emotion}</span>
        </div>
        <div class="emo-bar-track">
          <div class="emo-bar-fill" data-pct="${pct}" style="background:${meta.color}"></div>
          <div class="emo-threshold-tick" style="left:${thPct}%"></div>
        </div>
        <div class="emo-bar-pct ${score.detected ? '' : ''}" style="color:${score.detected ? meta.color : 'var(--muted)'}">${pct}%</div>
      `;
      barsContainer.appendChild(row);
    });

    // Animate bars after paint
    requestAnimationFrame(() => {
      barsContainer.querySelectorAll('.emo-bar-fill').forEach(fill => {
        fill.style.width = fill.dataset.pct + '%';
      });
    });
  }

  // Per-model breakdown
  renderModelEmotionChips('roberta-emotion-chips', data.roberta, 'var(--accent)');
  renderModelEmotionChips('electra-emotion-chips', data.electra, '#ff6ea8');

  // Reset toggle state
  document.getElementById('emotion-model-breakdown').classList.remove('open');
  document.getElementById('toggle-chevron').classList.remove('open');

  // Show card
  requestAnimationFrame(() => { card.classList.add('visible'); });
}

function renderModelEmotionChips(containerId, modelData, accentColor) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!modelData || !modelData.emotions) return;

  // Show top 8 by probability
  const top = modelData.emotions.slice(0, 8);
  top.forEach(score => {
    const meta = EMOTION_META[score.emotion] || { icon: '•', color: 'var(--muted)' };
    const pct  = Math.round(score.probability * 100);
    const row  = document.createElement('div');
    row.className = `em-chip-row${score.detected ? ' detected' : ''}`;
    row.innerHTML = `
      <div class="em-label">
        <span class="em-label-icon">${meta.icon}</span>
        <span>${score.emotion}</span>
      </div>
      <div class="em-bar-mini">
        <div class="em-bar-mini-fill" data-pct="${pct}" style="background:${meta.color}88"></div>
      </div>
      <div class="em-pct">${pct}%</div>
      <div class="em-detected-dot" title="${score.detected ? 'Detected' : 'Below threshold'}"></div>
    `;
    container.appendChild(row);
  });

  // Animate mini bars
  requestAnimationFrame(() => {
    container.querySelectorAll('.em-bar-mini-fill').forEach(fill => {
      fill.style.width = fill.dataset.pct + '%';
    });
  });
}

// ── Toggle model breakdown ───────────────────────────────────────────────────
function toggleEmotionBreakdown() {
  const panel   = document.getElementById('emotion-model-breakdown');
  const chevron = document.getElementById('toggle-chevron');
  panel.classList.toggle('open');
  chevron.classList.toggle('open');
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setLoading(on) {
  document.getElementById('scan-btn').disabled = on;
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('btn-label').textContent  = on ? 'Scanning…' : 'Scan email';
}

function hideResult() {
  const spamCard    = document.getElementById('result-card');
  const emotionCard = document.getElementById('emotion-card');
  spamCard.classList.remove('visible');
  emotionCard.classList.remove('visible');
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