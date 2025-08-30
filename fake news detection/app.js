const textEl = document.getElementById('text');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultEl = document.getElementById('result');
const labelPill = document.getElementById('labelPill');
const confidenceText = document.getElementById('confidenceText');
const barFill = document.getElementById('barFill');
const charCount = document.getElementById('charCount');
const themeToggle = document.getElementById('themeToggle');
const footerYear = document.getElementById('footerYear');

footerYear.textContent = new Date().getFullYear().toString();

/* Theme toggle with localStorage */
(function initTheme(){
  const saved = localStorage.getItem('fn_theme');
  if(saved){ document.documentElement.setAttribute('data-theme', saved); }
  themeToggle.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('fn_theme', next);
  });
})();

/* Character counter */
function updateCharCount(){
  const n = (textEl.value || '').length;
  charCount.textContent = `${n} chars`;
}
textEl.addEventListener('input', updateCharCount);
updateCharCount();

/* Auto-resize (nice UX) */
textEl.addEventListener('input', () => {
  textEl.style.height = 'auto';
  textEl.style.height = Math.min(textEl.scrollHeight, 360) + 'px';
});

/* Keyboard shortcut (Ctrl/Cmd + Enter) */
textEl.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    doPredict();
  }
});

/* Examples fill */
document.querySelectorAll('.example').forEach(btn => {
  btn.addEventListener('click', () => {
    textEl.value = btn.dataset.text || '';
    updateCharCount();
    textEl.dispatchEvent(new Event('input'));
    textEl.focus();
  });
});

clearBtn.addEventListener('click', () => {
  textEl.value = '';
  updateCharCount();
  resultEl.classList.add('hidden');
  barFill.style.width = '0%';
  labelPill.className = 'pill';
  labelPill.textContent = '—';
  confidenceText.textContent = 'Confidence —';
  document.getElementById('rawJson').textContent = '';
  textEl.focus();
});

predictBtn.addEventListener('click', doPredict);

async function doPredict(){
  const text = (textEl.value || '').trim();
  resultEl.classList.remove('hidden');

  if(!text){
    renderResult({ error: 'Please paste some text first.' });
    return;
  }

  setLoading(true);
  try{
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if(!res.ok){
      renderResult({ error: data.error || 'Prediction failed' });
      setLoading(false);
      return;
    }
    renderResult(data);
  }catch(e){
    console.error(e);
    renderResult({ error: 'Error contacting server.' });
  }finally{
    setLoading(false);
  }
}

function setLoading(isLoading){
  predictBtn.disabled = isLoading;
  predictBtn.innerHTML = isLoading ? 'Predicting… ⏳' : 'Predict';
}

function renderResult(payload){
  const raw = document.getElementById('rawJson');

  if(payload.error){
    labelPill.className = 'pill warn';
    labelPill.textContent = 'Error';
    confidenceText.textContent = payload.error;
    barFill.style.width = '0%';
    raw.textContent = JSON.stringify(payload, null, 2);
    return;
  }

  const label = (payload.label || '').toString().toUpperCase();
  const conf  = typeof payload.confidence === 'number' ? payload.confidence : null;
  const pct   = conf !== null ? Math.max(0, Math.min(1, conf)) * 100 : null;

  // Pill style
  if(label === 'FAKE'){
    labelPill.className = 'pill bad';
  }else if(label === 'REAL'){
    labelPill.className = 'pill ok';
  }else{
    labelPill.className = 'pill';
  }
  labelPill.textContent = label || '—';

  // Confidence text + bar
  if (pct !== null){
    confidenceText.textContent = `Confidence ${pct.toFixed(1)}%`;
    barFill.style.width = `${pct.toFixed(1)}%`;
  }else{
    confidenceText.textContent = 'Confidence —';
    barFill.style.width = '0%';
  }

  raw.textContent = JSON.stringify(payload, null, 2);
}
