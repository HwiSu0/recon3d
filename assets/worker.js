/* 브라우저 안에서 engine/recon3d.py를 그대로 돌리는 모듈 워커.
   Pyodide(웹어셈블리 파이썬)를 쓰며, 파일은 이 컴퓨터 밖으로 나가지 않는다. */
const PYODIDE_VERSION = '314.0.7';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let py = null, booting = null;
const say = msg => self.postMessage(msg);

const RUNNER = `
import sys, json, traceback
sys.path.insert(0, '/app')
import recon3d

def run_recon(config_path):
    sys.argv = ['recon3d.py', config_path]
    try:
        recon3d.main()
        return json.dumps({'ok': True})
    except SystemExit as e:
        return json.dumps({'ok': False, 'message': str(e.code)})
    except Exception as e:
        return json.dumps({'ok': False, 'message': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()})
`;

function boot() {
  if (!booting) booting = start();
  return booting;
}

async function start() {
  say({ type: 'stage', stage: 'boot', text: '실행 환경 불러오는 중 (처음 1회 약 40 MB)' });
  let loadPyodide;
  try {
    ({ loadPyodide } = await import(PYODIDE_URL + 'pyodide.mjs'));
  } catch (e) {
    say({ type: 'log', text: String(e) });
    throw new Error('실행 환경을 내려받지 못함. 인터넷 연결과 cdn.jsdelivr.net 차단 여부 확인');
  }
  const line = text => say({ type: 'log', text });
  py = await loadPyodide({ indexURL: PYODIDE_URL, stdout: line, stderr: line });
  await py.loadPackage(['numpy', 'scipy', 'pandas'], { messageCallback: () => {}, errorCallback: line });
  py.FS.mkdirTree('/app');
  for (const name of ['recon3d.py', 'thickness.py', 'template.html']) {
    const res = await fetch(new URL(`../engine/${name}`, self.location.href));
    if (!res.ok) throw new Error(`engine/${name} 파일을 불러오지 못함 (HTTP ${res.status})`);
    py.FS.writeFile(`/app/${name}`, new Uint8Array(await res.arrayBuffer()));
  }
  await py.runPythonAsync(RUNNER);
  say({ type: 'ready' });
}

function run(file, cfg) {
  const FS = py.FS;
  try { FS.mkdirTree('/in'); FS.mkdirTree('/out'); } catch (e) { /* 이미 있음 */ }
  FS.mount(FS.filesystems.WORKERFS, { blobs: [{ name: 'data.csv', data: file }] }, '/in');
  try {
    const job = Object.assign({}, cfg, { csv: '/in/data.csv', out: '/out/result' });
    FS.writeFile('/out/config.json', JSON.stringify(job));
    say({ type: 'stage', stage: 'run', text: '데이터 읽는 중' });
    const result = JSON.parse(py.globals.get('run_recon')('/out/config.json'));
    if (!result.ok) { say({ type: 'failed', message: result.message, trace: result.trace }); return; }
    say({
      type: 'done',
      html: FS.readFile('/out/result.html', { encoding: 'utf8' }),
      json: FS.readFile('/out/result.json', { encoding: 'utf8' }),
    });
  } finally {
    FS.unmount('/in');
  }
}

self.onmessage = async ({ data }) => {
  try {
    await boot();
    if (data.type === 'run') run(data.file, data.cfg);
  } catch (e) {
    say({ type: 'failed', message: e && e.message ? e.message : String(e) });
  }
};
