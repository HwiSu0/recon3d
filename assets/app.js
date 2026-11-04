import { drawFilings } from './filings.js';

const $ = id => document.getElementById(id);
const clone = o => JSON.parse(JSON.stringify(o));

const state = {
  file: null,        // 고른 CSV
  columns: null,     // Map: Run 번호 -> Set(열 이름)
  cfg: null,         // 설정 (recon3d.py의 config JSON과 같은 모양)
  truth: null,       // 체크를 껐다 켜도 되돌릴 수 있게 보관
  worker: null,
  running: false,
  jobRun: null,      // 지금 계산 중인 Run 번호
  presets: [],
  result: null,      // { name, html, json }
};

/* ---------- CSV 머리글 읽기 ---------- */
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
  if (runs.length === 1) return `Run #${runs[0]}`;
  return `Run #${Math.min(...runs)}~#${Math.max(...runs)}`;
}

/* ---------- 측정 파일 ---------- */
async function setFile(file) {
  clearError();
  const columns = await readColumns(file);
  if (!columns.size) {
    state.file = null; state.columns = null;
    $('drop').classList.remove('loaded');
    $('dropText').textContent = 'Capstone에서 내보낸 CSV를 끌어다 놓거나 눌러서 고르세요';
    $('fileInfo').hidden = true;
    showError(`${file.name}에서 "Run #" 열을 찾지 못했어요. Capstone의 File > Export Data로 내보낸 CSV를 고르세요.`);
    fillRun(); validate();
    return;
  }
  state.file = file;
  state.columns = columns;
  $('drop').classList.add('loaded');
  $('dropText').textContent = file.name;
  $('fileInfo').hidden = false;
  $('fileInfo').textContent = `${(file.size / 1e6).toFixed(1)} MB, ${runLabel(columns)} 들어 있어요.`;
  if (!columns.has(state.cfg.run)) {
    const missing = state.cfg.run;
    const match = state.presets.find(p => columns.has(Number(p.id.slice(1))));
    if (match) await usePreset(match.id);
    else state.cfg.run = Math.max(...columns.keys());
    $('fileInfo').textContent += ` 파일에 없는 Run #${missing} 대신 Run #${state.cfg.run} 설정으로 바꿨어요. 실험 조건이 맞는지 확인하세요.`;
  }
  fillRun(); sync();
  startWorker();
}

function wireDrop() {
  const drop = $('drop');
  $('csvInput').addEventListener('change', e => { if (e.target.files[0]) setFile(e.target.files[0]); });
  for (const ev of ['dragenter', 'dragover']) drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); });
  for (const ev of ['dragleave', 'drop']) drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); });
  drop.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });
}

async function loadSample(id) {
  const res = await fetch(`samples/run${id.slice(1)}.csv`);
  if (!res.ok) { showError('예제 파일을 불러오지 못했어요.'); return; }
  const blob = await res.blob();
  await usePreset(id);
  await setFile(new File([blob], `run${id.slice(1)}.csv`, { type: 'text/csv' }));
}

/* ---------- 설정 불러오기 ---------- */
async function usePreset(id) {
  const res = await fetch(`configs/config_${id}.json`);
  if (!res.ok) { showError('설정 파일을 불러오지 못했어요.'); return; }
  setConfig(await res.json());
  $('preset').value = id;
}

function setConfig(cfg) {
  state.cfg = cfg;
  state.truth = cfg.truth ? clone(cfg.truth) : null;
  if (!cfg.hose_mm) cfg.hose_mm = { stem: 3, branches: cfg.outlets.map(() => 3) };
  if (!cfg.outlet_names) cfg.outlet_names = cfg.outlets.map((_, i) => `출구${i + 1}`);
  fillForm();
  $('cfgError').hidden = true;
  $('cfgText').value = JSON.stringify(cfg, null, 2);
  validate();
}

/* ---------- 설정 -> 화면 ---------- */
const val = v => (v === undefined || v === null ? '' : v);
const toNum = s => { const n = parseFloat(s); return Number.isFinite(n) ? n : null; };

function fillRun() {
  const select = $('fRunSel'), input = $('fRun'), run = state.cfg.run;
  if (state.columns) {
    const runs = [...state.columns.keys()].sort((a, b) => b - a);
    select.replaceChildren(...runs.map(r => new Option(`Run #${r}`, r)));
    if (!state.columns.has(run)) select.prepend(new Option(`Run #${run} (이 파일에 없음)`, run));
    select.value = run;
  }
  select.hidden = !state.columns;
  input.hidden = !!state.columns;
  input.value = val(run);
}

function fillForm() {
  const c = state.cfg;
  $('fTitle').value = val(c.title);
  $('fFront').value = val(c.t_front_guess);
  $('fBase0').value = val(c.baseline && c.baseline[0]); $('fBase1').value = val(c.baseline && c.baseline[1]);
  $('fWin0').value = val(c.t_window && c.t_window[0]); $('fWin1').value = val(c.t_window && c.t_window[1]);
  $('fUp').value = val(c.upstream_len); $('fOut').value = val(c.outlet_len);
  fillRun(); fillSensors(); fillPorts();
  const hasTruth = !!state.truth;
  $('fTruth').disabled = !hasTruth;
  $('fTruth').checked = !!c.truth;
  $('truthNote').textContent = hasTruth ? '블라인드 시험이면 끄세요' : '설정 JSON에 truth가 있을 때 쓸 수 있어요';
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

function fillPorts() {
  const c = state.cfg, body = $('portTable').tBodies[0];
  body.replaceChildren();
  const inlet = body.insertRow();
  inlet.insertCell().textContent = '입구';
  ['a', 'b', 'c'].forEach((axis, i) => cell(inlet, numInput(c.inlet[i], `입구 ${axis}`, v => { c.inlet[i] = v; })));
  cell(inlet, numInput(c.hose_mm.stem, '입구 쪽 호스 안지름', v => { c.hose_mm.stem = v; }));
  inlet.insertCell();
  c.outlets.forEach((port, k) => {
    const row = body.insertRow();
    cell(row, textInput(c.outlet_names[k], `출구 ${k + 1} 이름`, v => { c.outlet_names[k] = v; }));
    ['a', 'b', 'c'].forEach((axis, i) => cell(row, numInput(port[i], `출구 ${k + 1} ${axis}`, v => { port[i] = v; })));
    cell(row, numInput(c.hose_mm.branches[k], `출구 ${k + 1} 호스 안지름`, v => { c.hose_mm.branches[k] = v; }));
    const td = row.insertCell();
    if (c.outlets.length > 1) {
      const del = document.createElement('button');
      del.type = 'button'; del.className = 'del'; del.textContent = '×';
      del.setAttribute('aria-label', `출구 ${k + 1} 삭제`);
      del.addEventListener('click', () => {
        c.outlets.splice(k, 1); c.outlet_names.splice(k, 1); c.hose_mm.branches.splice(k, 1);
        fillPorts(); sync();
      });
      td.append(del);
    }
  });
}

function wireFields() {
  const c = () => state.cfg;
  const bind = (id, apply) => $(id).addEventListener('input', e => { apply(e.target.value); sync(); });
  bind('fTitle', v => { c().title = v; });
  bind('fRun', v => { c().run = toNum(v); });
  $('fRunSel').addEventListener('change', e => { c().run = Number(e.target.value); fillRun(); sync(); });
  bind('fFront', v => { const n = toNum(v); if (n === null) delete c().t_front_guess; else c().t_front_guess = n; });
  bind('fBase0', v => { c().baseline = [toNum(v), c().baseline ? c().baseline[1] : null]; });
  bind('fBase1', v => { c().baseline = [c().baseline ? c().baseline[0] : null, toNum(v)]; });
  bind('fWin0', v => { c().t_window = [toNum(v), c().t_window ? c().t_window[1] : null]; });
  bind('fWin1', v => { c().t_window = [c().t_window ? c().t_window[0] : null, toNum(v)]; });
  const optional = (key, v) => { const n = toNum(v); if (n === null) delete c()[key]; else c()[key] = n; };
  bind('fUp', v => optional('upstream_len', v));
  bind('fOut', v => optional('outlet_len', v));
  $('fTruth').addEventListener('change', e => {
    if (e.target.checked && state.truth) c().truth = clone(state.truth); else delete c().truth;
    sync();
  });
  $('addOutlet').addEventListener('click', () => {
    const cfg = c(), last = cfg.outlets[cfg.outlets.length - 1];
    cfg.outlets.push(last.slice());
    cfg.outlet_names.push(`출구${cfg.outlets.length}`);
    cfg.hose_mm.branches.push(cfg.hose_mm.stem);
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
      showError(`${file.name}을 읽지 못했어요. JSON 형식인지 확인하세요.`);
    }
  });
  $('cfgText').addEventListener('input', e => {
    try {
      const cfg = JSON.parse(e.target.value);
      for (const key of ['sensors', 'inlet', 'outlets']) if (!cfg[key]) throw new Error(`"${key}" 항목이 없어요`);
      state.cfg = cfg;
      if (cfg.truth) state.truth = clone(cfg.truth);
      if (!cfg.hose_mm) cfg.hose_mm = { stem: 3, branches: cfg.outlets.map(() => 3) };
      if (!cfg.outlet_names) cfg.outlet_names = cfg.outlets.map((_, i) => `출구${i + 1}`);
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
  if (!Number.isInteger(c.run)) out.push('Run 번호를 입력하세요.');
  const span = (r, name) => {
    if (!r || !Number.isFinite(r[0]) || !Number.isFinite(r[1])) out.push(`${name}을 입력하세요.`);
    else if (r[0] >= r[1]) out.push(`${name}은 시작이 끝보다 작아야 해요.`);
  };
  span(c.baseline, '기준 구간'); span(c.t_window, '분석 구간');
  for (const [key, pos] of Object.entries(c.sensors)) {
    if (!coord(pos)) out.push(`${key} 위치를 모두 입력하세요.`);
    if (!c.ids || !c.ids[key]) out.push(`${key} ID를 입력하세요.`);
  }
  if (!coord(c.inlet)) out.push('입구 좌표를 모두 입력하세요.');
  c.outlets.forEach((p, k) => { if (!coord(p)) out.push(`출구 ${k + 1} 좌표를 모두 입력하세요.`); });
  const hoses = [c.hose_mm.stem, ...c.hose_mm.branches.slice(0, c.outlets.length)];
  if (hoses.length !== c.outlets.length + 1 || !hoses.every(d => Number.isFinite(d) && d > 0)) out.push('호스 안지름을 모두 입력하세요.');
  if (document.querySelector('[data-bad-mask]')) out.push('뺄 구간은 69-999처럼 쓰고, 여러 개면 쉼표로 나누세요.');
  if (state.columns) {
    const cols = state.columns.get(c.run);
    if (!cols) out.push(`Run #${c.run}: 이 파일에 없는 번호예요. ${runLabel(state.columns)} 중에서 고르세요.`);
    else {
      if (!cols.has(`Time (s) Run #${c.run}`)) out.push(`Run #${c.run}에 시간 열이 없어요.`);
      if (!cols.has(`Output Current (A) Run #${c.run}`)) out.push(`Run #${c.run}에 전류(Output Current) 열이 없어요.`);
      for (const [key, id] of Object.entries(c.ids || {})) {
        if (id && !cols.has(`X Magnetic Field Strength, ${id} (G) Run #${c.run}`)) out.push(`Run #${c.run}에 ${key}(${id}) 값이 없어요. 센서 ID를 확인하세요.`);
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
    showError(`실행 중 문제가 생겼어요: ${e.message || '브라우저가 작업을 멈췄어요'}. 메모리가 부족했다면 다른 탭을 닫고 다시 시도하세요.`);
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
  if (msg.type === 'log') { if (state.running) { addLog(msg.text); track(msg.text); } }
  if (msg.type === 'failed') {
    if (!state.running) { dropWorker(); return; }
    finishRun();
    showError(explain(msg.message));
    if (msg.trace) addLog(msg.trace);
    $('logBox').open = true;
    $('progress').hidden = true;
    dropWorker();
  }
  if (msg.type === 'done') {
    const run = state.jobRun;
    finishRun();
    setStage(1, '복원을 마쳤어요');
    showResult({ name: `recon_run${run}`, html: msg.html, json: msg.json, label: `Run #${run} 복원 결과` });
    if (matchMedia('(max-width: 1040px)').matches) $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/* recon3d.py의 진행 출력에서 단계를 읽는다 */
function track(line) {
  let m;
  if (/불러오는 중/.test(line)) setStage(0.04, '측정값을 읽는 중');
  else if (/경로 맞추는 중/.test(line)) setStage(0.08, '시작점마다 경로를 맞추는 중');
  else if ((m = line.match(/1단계 (\d+)\/(\d+)/))) setStage(0.08 + 0.62 * m[1] / m[2], `시작점마다 경로를 맞추는 중 (${m[1]}/${m[2]})`);
  else if ((m = line.match(/2단계 (\d+)\/(\d+)/))) {
    const last = m[1] === m[2];
    setStage(0.7 + 0.25 * m[1] / m[2], last ? '결과를 정리하는 중' : `좋은 후보를 다듬는 중 (${m[1]}/${m[2]})`);
  }
}

function explain(message) {
  if (/empty|finite|NaN|nan/i.test(message)) return `맞춤을 시작하지 못했어요. 기준 구간과 분석 구간이 측정 시간 안에 있는지 확인하세요. (${message})`;
  if (/MemoryError|out of memory/i.test(message)) return '메모리가 부족해요. 다른 탭을 닫거나 CSV에서 필요한 Run만 내보내 다시 시도하세요.';
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
  $('logBox').open = false;
  $('runNote').hidden = true;
  $('stopBtn').hidden = false;
  setStage(0.02, '실행 환경을 준비하는 중');
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
  state.jobRun = state.cfg.run;
  beginRun();
  startWorker();
  state.worker.postMessage({ type: 'run', file: state.file, cfg: clone(state.cfg) });
}

function stop() {
  dropWorker();
  finishRun();
  $('progress').hidden = true;
  $('runNote').hidden = false;
  addLog('사용자가 중단했어요.');
}

/* ---------- 결과 ---------- */
const SANS = '"IBM Plex Sans KR","Pretendard","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif';
const EMBED_HEAD = `<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap"><style>
:root{--bg:#e7eef2;--panel:#f7fafb;--ink:#20262b;--muted:#566672;--line:#c6d2d9;--grid:#e3eaee;--scene:#dbe4e9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14191d;--panel:#1b2226;--ink:#e5ebee;--muted:#9aa9b3;--line:#333e45;--grid:#2a3339;--scene:#182025}}
html{overflow:hidden}body{font-family:${SANS};background:var(--bg);word-break:keep-all}
.wrap{max-width:none;padding:0}.card{border-radius:4px}#stage{border-radius:2px}button{border-radius:4px}h1{font-size:18px;font-weight:600}
</style>`;
const EMBED_TAIL = `<script>(function(){var send=function(){parent.postMessage({recon3dHeight:document.documentElement.scrollHeight},"*")};new ResizeObserver(send).observe(document.body);addEventListener("load",send);send()})();<\/script>`;

function embed(html) {
  const head = html.indexOf('</head>');
  const tail = html.lastIndexOf('</body>');
  if (head < 0 || tail < 0) return html;
  return html.slice(0, head) + EMBED_HEAD + html.slice(head, tail) + EMBED_TAIL + html.slice(tail);
}

function showResult(result) {
  state.result = result;
  $('resultTitle').textContent = result.label;
  $('actions').hidden = false;
  $('empty').hidden = true;
  const frame = $('viewer');
  frame.hidden = false;
  frame.srcdoc = embed(result.html);
}

function save(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function showExample() {
  try {
    const [html, json] = await Promise.all([fetch('samples/result_run40.html'), fetch('samples/result_run40.json')]);
    if (!html.ok || !json.ok) return;
    showResult({ name: 'recon_run40', html: await html.text(), json: await json.text(), label: '예제 결과: Run #40 곧은 호스' });
    $('samples').hidden = false;
  } catch (e) { /* 예제를 지운 저장소에서는 빈 화면으로 시작한다 */ }
}

/* 넓은 화면에서 설정 패널이 언제나 화면 안에 들어오게 한다 (복원 버튼이 가려지지 않도록) */
function fitRail() {
  const rail = document.querySelector('.rail');
  if (!matchMedia('(min-width: 1041px)').matches) { rail.style.maxHeight = ''; return; }
  const top = Math.max(0, rail.parentElement.getBoundingClientRect().top);
  rail.style.maxHeight = `${innerHeight - top}px`;
}

/* ---------- 시작 ---------- */
async function main() {
  drawFilings($('filings'));
  fitRail();
  addEventListener('scroll', fitRail, { passive: true });
  addEventListener('resize', fitRail);
  wireDrop(); wireFields();
  $('runBtn').addEventListener('click', run);
  $('stopBtn').addEventListener('click', stop);
  $('saveHtml').addEventListener('click', () => save(`${state.result.name}.html`, state.result.html, 'text/html'));
  $('saveJson').addEventListener('click', () => save(`${state.result.name}.json`, state.result.json, 'application/json'));
  document.querySelectorAll('[data-sample]').forEach(b => b.addEventListener('click', () => loadSample(b.dataset.sample)));
  addEventListener('message', e => {
    if (e.source === $('viewer').contentWindow && e.data && e.data.recon3dHeight) $('viewer').style.height = `${e.data.recon3dHeight}px`;
  });

  state.presets = await (await fetch('configs/index.json')).json();
  $('preset').replaceChildren(...state.presets.map(p => new Option(p.label, p.id)));
  await usePreset(state.presets[0].id);
  showExample();
}
main();
