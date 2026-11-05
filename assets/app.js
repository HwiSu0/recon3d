const $ = id => document.getElementById(id);
const clone = o => JSON.parse(JSON.stringify(o));

const state = {
  file: null,        // 고른 CSV
  columns: null,     // Map: Run 번호 -> Set(열 이름)
  cfg: null,         // 설정 (recon3d.py의 config JSON과 같은 모양)
  truth: null,       // 체크를 껐다 켜도 되돌릴 수 있게 보관
  worker: null,
  running: false,
  job: null,         // 지금 계산 중인 { run, title, csv }
  presets: [],
  result: null,      // 화면에 보이는 결과 { id, run, title, csv, time, html, json }
  results: [],       // 저장된 결과, 최신순
};

/* ---------- CSV 머리글 ---------- */
function parseHeader(line) {
  const cols = []; let cur = '', quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { cols.push(cur); cur = ''; }
    else cur += ch;
  }
  cols.push(cur);
  return cols;
}

async function readColumns(file) {
  let text = await file.slice(0, 8 << 20).text();
  const end = text.search(/\r?\n/);
  if (end >= 0) text = text.slice(0, end);
  const map = new Map();
  for (const col of parseHeader(text.replace(/^﻿/, ''))) {
    const m = col.match(/ Run #(\d+)$/);
    if (!m) continue;
    const run = Number(m[1]);
    if (!map.has(run)) map.set(run, new Set());
    map.get(run).add(col);
  }
  return map;
}

function runLabel(map) {
  const runs = [...map.keys()];
  return runs.length === 1 ? `Run #${runs[0]}` : `Run #${Math.min(...runs)}~#${Math.max(...runs)}`;
}

/* ---------- 측정 파일 ---------- */
async function setFile(file) {
  clearError();
  const columns = await readColumns(file);
  if (!columns.size) {
    state.file = null; state.columns = null;
    $('fileInfo').textContent = '파일 없음';
    showError(`${file.name}: Run 열 없음. Capstone에서 내보낸 CSV 필요`);
    fillRun(); validate();
    return;
  }
  state.file = file;
  state.columns = columns;
  $('fileInfo').textContent = `${file.name} (${runLabel(columns)})`;
  $('fileInfo').title = `${file.name}, ${(file.size / 1e6).toFixed(1)} MB`;
  if (!columns.has(state.cfg.run)) {
    const match = state.presets.find(p => columns.has(Number(p.id.slice(1))));
    if (match) await usePreset(match.id);
    else state.cfg.run = Math.max(...columns.keys());
  }
  fillRun(); sync();
  startWorker();
}

function wireFile() {
  const panel = document.querySelector('aside');
  $('openCsv').addEventListener('click', () => $('csvInput').click());
  $('csvInput').addEventListener('change', e => { if (e.target.files[0]) setFile(e.target.files[0]); e.target.value = ''; });
  for (const ev of ['dragenter', 'dragover']) document.addEventListener(ev, e => { e.preventDefault(); panel.classList.add('over'); });
  document.addEventListener('dragleave', e => { if (!e.relatedTarget) panel.classList.remove('over'); });
  document.addEventListener('drop', e => {
    e.preventDefault();
    panel.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
}

/* ---------- 설정 ---------- */
async function usePreset(id) {
  const res = await fetch(`configs/config_${id}.json`);
  if (!res.ok) { showError('설정 파일을 불러오지 못함'); return; }
  setConfig(await res.json());
  $('preset').value = id;
}

function normalize(cfg) {
  if (!cfg.hose_mm) cfg.hose_mm = { stem: 3, branches: cfg.outlets.map(() => 3) };
  if (!cfg.outlet_names) cfg.outlet_names = cfg.outlets.map((_, i) => `출구${i + 1}`);
  return cfg;
}

function keepTruth(cfg) {
  state.truth = cfg.truth || cfg.truth_points ? clone({ truth: cfg.truth, truth_points: cfg.truth_points }) : null;
}

function setConfig(cfg) {
  state.cfg = normalize(cfg);
  keepTruth(cfg);
  fillForm();
  $('cfgError').hidden = true;
  $('cfgText').value = JSON.stringify(cfg, null, 2);
  validate();
}

const val = v => (v === undefined || v === null ? '' : v);
const toNum = s => { const n = parseFloat(s); return Number.isFinite(n) ? n : null; };

function fillRun() {
  const select = $('fRunSel'), input = $('fRun'), run = state.cfg.run;
  if (state.columns) {
    const runs = [...state.columns.keys()].sort((a, b) => b - a);
    select.replaceChildren(...runs.map(r => new Option(`#${r}`, r)));
    if (!state.columns.has(run)) select.prepend(new Option(`#${run} (파일에 없음)`, run));
    select.value = run;
  }
  select.hidden = !state.columns;
  input.hidden = !!state.columns;
  input.value = val(run);
}

function fillForm() {
  const c = state.cfg;
  $('fFront').value = val(c.t_front_guess);
  $('fBase0').value = val(c.baseline && c.baseline[0]); $('fBase1').value = val(c.baseline && c.baseline[1]);
  $('fWin0').value = val(c.t_window && c.t_window[0]); $('fWin1').value = val(c.t_window && c.t_window[1]);
  $('fUp').value = val(c.upstream_len); $('fOut').value = val(c.outlet_len);
  $('fCurve').value = c.curve === false ? 'kink' : c.curve === 'free' ? 'free' : 'hole';
  $('fStem').value = val(c.hose_mm.stem);
  $('fBranches').value = (c.hose_mm.branches || []).join(', ');
  delete $('fBranches').dataset.bad;
  fillRun(); fillSensors(); fillPorts();
  $('fTruth').disabled = !state.truth;
  $('fTruth').checked = !!(c.truth || c.truth_points);
}

/* recon3d.py의 기본값과 같게 보여 준다 */
function fillStructure() {
  const c = state.cfg, nb = c.outlets.length;
  const topo = c.topology || (nb >= 2 ? 'y' : 'auto');
  $('fTopo').value = topo;
  const exit = $('fExit');
  exit.replaceChildren(new Option('자동 선택', 'auto'), ...c.outlet_names.map((name, i) => new Option(name, i)));
  exit.value = c.exit !== undefined ? c.exit : (nb >= 2 && topo !== 'y' ? 'auto' : 0);
  exit.disabled = nb < 2 || topo === 'y';
}

function maskText(ranges) { return (ranges || []).map(r => r.join('-')).join(', '); }
function parseMask(text) {
  return text.split(',').map(s => s.trim()).filter(Boolean)
    .map(s => s.match(/^(-?[\d.]+)\s*[-~]\s*(-?[\d.]+)$/))
    .map(m => (m ? [Number(m[1]), Number(m[2])] : null));
}

function cell(row, input) { const td = row.insertCell(); td.append(input); return td; }
function textInput(value, label, onInput) {
  const el = document.createElement('input');
  el.type = 'text'; el.value = value; el.setAttribute('aria-label', label); el.autocomplete = 'off';
  el.addEventListener('input', () => { onInput(el.value); sync(); });
  return el;
}
function numInput(value, label, onInput) {
  const el = document.createElement('input');
  el.type = 'number'; el.step = 'any'; el.value = val(value); el.setAttribute('aria-label', label);
  el.addEventListener('input', () => { onInput(toNum(el.value)); sync(); });
  return el;
}

function fillSensors() {
  const c = state.cfg, body = $('sensorTable').tBodies[0];
  body.replaceChildren();
  for (const key of Object.keys(c.sensors)) {
    const row = body.insertRow();
    row.insertCell().textContent = key;
    cell(row, textInput(val(c.ids && c.ids[key]), `${key} ID`, v => { c.ids = c.ids || {}; c.ids[key] = v.trim(); }));
    ['a', 'b', 'c'].forEach((axis, i) => cell(row, numInput(c.sensors[key][i], `${key} ${axis}`, v => { c.sensors[key][i] = v; })));
    cell(row, textInput(maskText(c.mask && c.mask[key]), `${key} 뺄 구간`, v => {
      c.mask = c.mask || {};
      const ranges = parseMask(v);
      if (ranges.includes(null)) { row.dataset.badMask = '1'; return; }
      delete row.dataset.badMask;
      if (ranges.length) c.mask[key] = ranges; else delete c.mask[key];
    }));
  }
}

function setFlow(k, v) {
  const c = state.cfg;
  const flows = c.outlets.map((_, i) => (c.flow_mL_s && c.flow_mL_s[i] !== undefined ? c.flow_mL_s[i] : null));
  flows[k] = v;
  if (flows.some(q => q !== null)) c.flow_mL_s = flows; else delete c.flow_mL_s;
}

function fillPorts() {
  const c = state.cfg, body = $('portTable').tBodies[0];
  body.replaceChildren();
  const inlet = body.insertRow();
  inlet.insertCell().textContent = '입구';
  ['a', 'b', 'c'].forEach((axis, i) => cell(inlet, numInput(c.inlet[i], `입구 ${axis}`, v => { c.inlet[i] = v; })));
  inlet.insertCell(); inlet.insertCell();
  c.outlets.forEach((port, k) => {
    const row = body.insertRow();
    cell(row, textInput(c.outlet_names[k], `출구 ${k + 1} 이름`, v => { c.outlet_names[k] = v; fillStructure(); }));
    ['a', 'b', 'c'].forEach((axis, i) => cell(row, numInput(port[i], `출구 ${k + 1} ${axis}`, v => { port[i] = v; })));
    const flow = numInput(c.flow_mL_s ? c.flow_mL_s[k] : null, `출구 ${k + 1} 유량`, v => setFlow(k, v));
    flow.min = '0';
    cell(row, flow);
    const td = row.insertCell();
    if (c.outlets.length > 1) {
      const del = document.createElement('button');
      del.type = 'button'; del.className = 'del'; del.textContent = '×';
      del.setAttribute('aria-label', `출구 ${k + 1} 삭제`);
      del.addEventListener('click', () => {
        c.outlets.splice(k, 1); c.outlet_names.splice(k, 1);
        if (c.flow_mL_s) c.flow_mL_s.splice(k, 1);
        if (c.exit !== undefined && c.exit !== 'auto') c.exit = 0;
        fillPorts(); sync();
      });
      td.append(del);
    }
  });
  fillStructure();
}

function wireFields() {
  const c = () => state.cfg;
  const bind = (id, apply) => $(id).addEventListener('input', e => { apply(e.target.value); sync(); });
  const optional = (key, v) => { const n = toNum(v); if (n === null) delete c()[key]; else c()[key] = n; };
  bind('fRun', v => { c().run = toNum(v); });
  $('fRunSel').addEventListener('change', e => { c().run = Number(e.target.value); fillRun(); sync(); });
  bind('fFront', v => optional('t_front_guess', v));
  bind('fBase0', v => { c().baseline = [toNum(v), c().baseline ? c().baseline[1] : null]; });
  bind('fBase1', v => { c().baseline = [c().baseline ? c().baseline[0] : null, toNum(v)]; });
  bind('fWin0', v => { c().t_window = [toNum(v), c().t_window ? c().t_window[1] : null]; });
  bind('fWin1', v => { c().t_window = [c().t_window ? c().t_window[0] : null, toNum(v)]; });
  bind('fUp', v => optional('upstream_len', v));
  bind('fOut', v => optional('outlet_len', v));
  bind('fStem', v => { c().hose_mm.stem = toNum(v); });
  bind('fBranches', v => {
    const list = v.split(',').map(s => s.trim()).filter(Boolean).map(Number);
    const ok = list.length > 0 && list.every(d => Number.isFinite(d) && d > 0);
    if (ok) { c().hose_mm.branches = list; delete $('fBranches').dataset.bad; } else $('fBranches').dataset.bad = '1';
  });
  $('fTopo').addEventListener('change', e => { c().topology = e.target.value; fillStructure(); sync(); });
  $('fExit').addEventListener('change', e => { c().exit = e.target.value === 'auto' ? 'auto' : Number(e.target.value); sync(); });
  $('fCurve').addEventListener('change', e => { c().curve = { hole: true, free: 'free', kink: false }[e.target.value]; sync(); });
  $('fTruth').addEventListener('change', e => {
    if (e.target.checked && state.truth) {
      if (state.truth.truth) c().truth = clone(state.truth.truth);
      if (state.truth.truth_points) c().truth_points = clone(state.truth.truth_points);
    } else {
      delete c().truth; delete c().truth_points;
    }
    sync();
  });
  $('addOutlet').addEventListener('click', () => {
    const cfg = c(), last = cfg.outlets[cfg.outlets.length - 1];
    cfg.outlets.push(last.slice());
    cfg.outlet_names.push(`출구${cfg.outlets.length}`);
    if (cfg.flow_mL_s) cfg.flow_mL_s.push(null);
    fillPorts(); sync();
  });
  $('preset').addEventListener('change', e => { if (e.target.value !== 'file') usePreset(e.target.value); });
  $('openCfg').addEventListener('click', () => $('cfgInput').click());
  $('cfgInput').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = '';
    try {
      setConfig(JSON.parse(await file.text()));
      let opt = $('preset').querySelector('option[value="file"]');
      if (!opt) { opt = new Option('', 'file'); $('preset').append(opt); }
      opt.textContent = file.name;
      $('preset').value = 'file';
      clearError();
    } catch (err) {
      showError(`${file.name}: JSON 형식 오류`);
    }
  });
  $('cfgText').addEventListener('input', e => {
    try {
      const cfg = JSON.parse(e.target.value);
      for (const key of ['sensors', 'inlet', 'outlets']) if (!cfg[key]) throw new Error(`"${key}" 항목 없음`);
      state.cfg = normalize(cfg);
      if (cfg.truth || cfg.truth_points) keepTruth(cfg);
      fillForm(); validate();
      $('cfgError').hidden = true;
    } catch (err) {
      $('cfgError').hidden = false;
      $('cfgError').textContent = err.message;
    }
  });
  $('saveCfg').addEventListener('click', () => {
    const cfg = clone(state.cfg);
    if (state.file) cfg.csv = state.file.name;
    save(`config_r${cfg.run}.json`, JSON.stringify(cfg, null, 2), 'application/json');
  });
}

function sync() {
  const box = $('cfgText');
  if (document.activeElement !== box) box.value = JSON.stringify(state.cfg, null, 2);
  validate();
}

/* ---------- 실행 전 점검 ---------- */
function checkConfig() {
  const c = state.cfg, out = [];
  const coord = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
  if (!Number.isInteger(c.run)) out.push('Run 번호 입력');
  const span = (r, name) => {
    if (!r || !Number.isFinite(r[0]) || !Number.isFinite(r[1])) out.push(`${name} 입력`);
    else if (r[0] >= r[1]) out.push(`${name}: 시작이 끝보다 작아야 함`);
  };
  span(c.baseline, '기준 구간'); span(c.t_window, '분석 구간');
  for (const [key, pos] of Object.entries(c.sensors)) {
    if (!coord(pos)) out.push(`${key} 위치 입력`);
    if (!c.ids || !c.ids[key]) out.push(`${key} ID 입력`);
  }
  if (!coord(c.inlet)) out.push('입구 좌표 입력');
  c.outlets.forEach((p, k) => { if (!coord(p)) out.push(`출구 ${k + 1} 좌표 입력`); });
  if (!(c.hose_mm.stem > 0) || $('fBranches').dataset.bad) out.push('호스 내경 형식: 5 또는 3, 3');
  if ((c.flow_mL_s || []).some(q => q !== null && !(q > 0))) out.push('유량은 0보다 커야 함');
  if (document.querySelector('[data-bad-mask]')) out.push('뺄 구간 형식: 69-999, 120-130');
  if (state.columns) {
    const cols = state.columns.get(c.run);
    if (!cols) out.push(`Run #${c.run} 없음 (파일: ${runLabel(state.columns)})`);
    else {
      if (!cols.has(`Time (s) Run #${c.run}`)) out.push(`Run #${c.run}: 시간 열 없음`);
      if (!cols.has(`Output Current (A) Run #${c.run}`)) out.push(`Run #${c.run}: 전류 열 없음`);
      for (const [key, id] of Object.entries(c.ids || {})) {
        if (id && !cols.has(`X Magnetic Field Strength, ${id} (G) Run #${c.run}`)) out.push(`Run #${c.run}: ${key} (${id}) 열 없음`);
      }
    }
  }
  return out;
}

function validate() {
  const box = $('problems'), list = checkConfig();
  box.replaceChildren(...list.map(t => { const p = document.createElement('p'); p.textContent = t; return p; }));
  $('runBtn').disabled = state.running || !state.file || list.length > 0;
}

/* ---------- 워커 ---------- */
function startWorker() {
  if (state.worker) return;
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => handle(data);
  worker.onerror = e => {
    finishRun();
    showError(`실행 오류: ${e.message || '브라우저가 작업을 멈춤'}`);
    dropWorker();
  };
  state.worker = worker;
  worker.postMessage({ type: 'boot' });
}

function dropWorker() {
  if (state.worker) state.worker.terminate();
  state.worker = null;
}

function handle(msg) {
  if (msg.type === 'stage' && state.running) setStage(null, msg.text);
  if (msg.type === 'log' && state.running) { addLog(msg.text); track(msg.text); }
  if (msg.type === 'failed') {
    if (!state.running) { dropWorker(); return; }
    finishRun();
    showError(explain(msg.message));
    if (msg.trace) addLog(msg.trace);
    $('progress').hidden = true;
    dropWorker();
  }
  if (msg.type === 'done') {
    const job = state.job;
    finishRun();
    setStage(1, '완료');
    addResult({ ...job, time: Date.now(), html: msg.html, json: msg.json });
    if (matchMedia('(max-width: 860px)').matches) $('result').scrollIntoView({ behavior: 'smooth' });
  }
}

/* recon3d.py의 진행 출력에서 단계를 읽는다 */
const cand = { n: 1, i: 1 };
function fitStage(done, text) {
  const prefix = cand.n > 1 ? `후보 ${cand.i}/${cand.n} ` : '';
  setStage(0.06 + 0.88 * (cand.i - 1 + done) / cand.n, prefix + text);
}
function track(line) {
  let m;
  if (/불러오는 중/.test(line)) setStage(0.04, '데이터 읽는 중');
  else if ((m = line.match(/^후보 (\d+)개/))) { cand.n = Number(m[1]); cand.i = 0; }
  else if (/^== 후보/.test(line)) { cand.i += 1; fitStage(0, '경로 맞춤'); }
  else if ((m = line.match(/1단계 (\d+)\/(\d+)/))) fitStage(0.8 * m[1] / m[2], `경로 맞춤 ${m[1]}/${m[2]}`);
  else if ((m = line.match(/2단계 (\d+)\/(\d+)/))) fitStage(0.8 + 0.2 * m[1] / m[2], `정밀 맞춤 ${m[1]}/${m[2]}`);
  else if (/굵기 추정 중/.test(line)) setStage(0.96, '내경 추정');
}

function explain(message) {
  if (/empty|finite|NaN|nan/i.test(message)) return `맞춤 실패: 기준 구간과 분석 구간이 측정 시간 안에 있는지 확인 (${message})`;
  if (/MemoryError|out of memory/i.test(message)) return '메모리 부족: 다른 탭을 닫거나 필요한 Run만 내보내기';
  return message;
}

/* ---------- 실행 ---------- */
let clockTimer = 0;
function setStage(p, text) {
  if (p !== null) $('bar').style.width = `${Math.round(p * 100)}%`;
  $('stageText').textContent = text;
}
function addLog(text) {
  const pre = $('log');
  pre.textContent += text + '\n';
  pre.scrollTop = pre.scrollHeight;
}
function showError(text) { $('runError').hidden = false; $('runError').textContent = text; }
function clearError() { $('runError').hidden = true; }

function beginRun() {
  clearError();
  state.running = true;
  $('log').textContent = '';
  $('progress').hidden = false;
  $('logBox').hidden = false;
  $('stopBtn').hidden = false;
  setStage(0.02, '실행 환경 준비');
  const started = performance.now();
  const tick = () => {
    const s = Math.floor((performance.now() - started) / 1000);
    $('clock').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };
  tick();
  clockTimer = setInterval(tick, 1000);
  validate();
}

function finishRun() {
  state.running = false;
  clearInterval(clockTimer);
  $('stopBtn').hidden = true;
  validate();
}

function run() {
  if (checkConfig().length || !state.file) return;
  state.job = { run: state.cfg.run, title: state.cfg.title || '', csv: state.file.name };
  beginRun();
  startWorker();
  state.worker.postMessage({ type: 'run', file: state.file, cfg: clone(state.cfg) });
}

function stop() {
  dropWorker();
  finishRun();
  $('progress').hidden = true;
  addLog('중단');
}

/* ---------- 결과 ---------- */
function showResult(result) {
  state.result = result;
  $('saveHtml').hidden = !result;
  $('saveJson').hidden = !result;
  $('empty').hidden = !!result;
  $('viewer').hidden = !result;
  if (result) $('viewer').srcdoc = result.html;
  else $('viewer').removeAttribute('srcdoc');
  renderList();
}

/* 저장된 결과: 이 브라우저의 IndexedDB에 둔다. 쓸 수 없으면 새로고침 전까지만 목록에 남는다 */
function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('recon3d', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('results', { keyPath: 'id', autoIncrement: true });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function store(mode, work) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('results', mode);
    const req = work(tx.objectStore('results'));
    tx.oncomplete = () => { db.close(); resolve(req.result); };
    tx.onerror = tx.onabort = () => { db.close(); reject(tx.error); };
  });
}

function rowSummary(entry) {
  try {
    const s = JSON.parse(entry.json);
    const kind = { single: '한 줄 관', step: '굵기 변화', loop: '합류 관', y: 'Y자' }[s.topology];
    const parts = [...(kind ? [kind] : []), `χ²/N ${s.chi2_per_N.toFixed(2)}`];
    const pieces = (s.thickness && s.thickness.pieces) || [];
    const sizes = pieces.filter(p => p.d_mm !== undefined).map(p => p.d_mm.toFixed(2));
    if (sizes.length) parts.push(`내경 ${sizes.join(', ')} mm`);
    else if (pieces.length && pieces.every(p => !p.resolved)) parts.push('내경 판정 불가');
    return parts.join(', ');
  } catch (e) {
    return '';
  }
}

function pad(n) { return String(n).padStart(2, '0'); }
function stamp(t) {
  const d = new Date(t);
  return `${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function renderList() {
  const current = state.result && state.result.id;
  $('resultCount').textContent = state.results.length || '';
  $('noResults').hidden = state.results.length > 0;
  $('results').replaceChildren(...state.results.map(entry => {
    const li = document.createElement('li');
    if (entry.id === current) li.className = 'on';
    const open = document.createElement('button');
    open.type = 'button'; open.className = 'item'; open.title = [entry.title, entry.csv].filter(Boolean).join('\n');
    const top = document.createElement('span'); top.className = 'top';
    const name = document.createElement('b'); name.textContent = entry.run ? `Run #${entry.run}` : (entry.title || entry.csv);
    const when = document.createElement('time'); when.textContent = stamp(entry.time);
    top.append(name, when);
    const sub = document.createElement('span'); sub.className = 'sub'; sub.textContent = rowSummary(entry);
    open.append(top, sub);
    open.addEventListener('click', () => showResult(entry));
    const del = document.createElement('button');
    del.type = 'button'; del.className = 'del'; del.textContent = '×';
    del.setAttribute('aria-label', `${name.textContent} ${stamp(entry.time)} 삭제`);
    del.addEventListener('click', () => removeResult(entry));
    li.append(open, del);
    return li;
  }));
}

async function addResult(entry) {
  try {
    if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
    entry.id = await store('readwrite', s => s.add(entry));
    $('listError').hidden = true;
  } catch (e) {
    entry.id = -Date.now();
    $('listError').hidden = false;
    $('listError').textContent = '브라우저 저장 실패: 새로고침하면 사라짐';
  }
  state.results.unshift(entry);
  showResult(entry);
}

async function removeResult(entry) {
  if (!confirm(`Run #${entry.run} ${stamp(entry.time)} 결과 삭제`)) return;
  if (entry.id > 0) {
    try { await store('readwrite', s => s.delete(entry.id)); } catch (e) { /* 목록에서는 뺀다 */ }
  }
  state.results = state.results.filter(r => r !== entry);
  if (state.result === entry) showResult(state.results[0] || null);
  else renderList();
}

async function loadResults() {
  try {
    state.results = (await store('readonly', s => s.getAll())).sort((a, b) => b.time - a.time);
  } catch (e) {
    state.results = [];
  }
  showResult(state.results[0] || null);
}

async function importResult(file) {
  const html = await file.text();
  const found = html.match(/const D = (\{[\s\S]*?\});\nconst css =/);
  try {
    const data = JSON.parse(found[1]);
    const run = (String(data.title).match(/Run #(\d+)/) || [])[1];
    await addResult({ run: run ? Number(run) : null, title: data.title || '', csv: file.name, time: file.lastModified || Date.now(), html, json: JSON.stringify(data.summary) });
    state.results.sort((a, b) => b.time - a.time);
    renderList();
  } catch (e) {
    $('listError').hidden = false;
    $('listError').textContent = `${file.name}: 복원 결과 HTML 아님`;
  }
}

function setTab(list) {
  $('paneForm').hidden = list;
  $('paneList').hidden = !list;
  $('tabForm').setAttribute('aria-selected', String(!list));
  $('tabList').setAttribute('aria-selected', String(list));
}

function save(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/* ---------- 시작 ---------- */
async function main() {
  wireFile(); wireFields();
  $('runBtn').addEventListener('click', run);
  $('stopBtn').addEventListener('click', stop);
  const fileName = () => `recon_run${state.result.run || 'result'}`;
  $('saveHtml').addEventListener('click', () => save(`${fileName()}.html`, state.result.html, 'text/html'));
  $('saveJson').addEventListener('click', () => save(`${fileName()}.json`, state.result.json, 'application/json'));
  $('tabForm').addEventListener('click', () => setTab(false));
  $('tabList').addEventListener('click', () => setTab(true));
  $('importBtn').addEventListener('click', () => $('importInput').click());
  $('importInput').addEventListener('change', e => { if (e.target.files[0]) importResult(e.target.files[0]); e.target.value = ''; });
  state.presets = await (await fetch('configs/index.json')).json();
  $('preset').replaceChildren(...state.presets.map(p => new Option(p.label, p.id)));
  await usePreset(state.presets[0].id);
  await loadResults();
}
main();
