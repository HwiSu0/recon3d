/* 브라우저 안에서 engine/recon3d.py를 그대로 돌리는 모듈 워커.
   Pyodide(웹어셈블리 파이썬)를 쓰며, 파일은 이 컴퓨터 밖으로 나가지 않는다.
   lead 워커가 파일을 읽고 맞춤 순서를 정하며, 계산 조각은 모든 워커가 나눠 맡는다. */
import { readTable } from './table.js';

const PYODIDE_VERSION = '314.0.7';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

let py = null, booting = null;
const say = msg => self.postMessage(msg);

const RUNNER = `
import sys, json, traceback
sys.path.insert(0, '/app')
import recon3d

CTX = None; STEPS = None

def guarded(fn):
    def run(*args):
        try:
            return json.dumps({'ok': True, 'value': fn(*args)})
        except SystemExit as e:
            return json.dumps({'ok': False, 'message': str(e.code)})
        except Exception as e:
            return json.dumps({'ok': False, 'message': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()})
    return run

@guarded
def prepare(path):
    global CTX
    CTX = recon3d.Context(json.load(open(path, encoding='utf-8')))

@guarded
def begin():
    global STEPS
    STEPS = recon3d.plan(CTX)
    return next(STEPS)

@guarded
def advance(text):
    try:
        return STEPS.send(json.loads(text))
    except StopIteration:
        return None

@guarded
def work(text):
    return recon3d.do_job(CTX, json.loads(text))
`;

function boot() {
  if (!booting) booting = start();
  return booting;
}

async function start() {
  let loadPyodide;
  try {
    ({ loadPyodide } = await import(PYODIDE_URL + 'pyodide.mjs'));
  } catch (e) {
    say({ type: 'log', text: String(e) });
    throw new Error('실행 환경을 내려받지 못함. 인터넷 연결과 cdn.jsdelivr.net 차단 여부 확인');
  }
  const line = text => say({ type: 'log', text });
  py = await loadPyodide({ indexURL: PYODIDE_URL, stdout: line, stderr: line });
  await py.loadPackage(['numpy', 'scipy'], { messageCallback: () => {}, errorCallback: line });
  py.FS.mkdirTree('/app');
  for (const name of ['recon3d.py', 'grow.py', 'thickness.py', 'template.html']) {
    const res = await fetch(new URL(`../engine/${name}`, self.location.href), { cache: 'no-cache' });   // 새로 올린 엔진을 바로 쓴다
    if (!res.ok) throw new Error(`engine/${name} 파일을 불러오지 못함 (HTTP ${res.status})`);
    py.FS.writeFile(`/app/${name}`, new Uint8Array(await res.arrayBuffer()));
  }
  py.FS.mkdirTree('/in'); py.FS.mkdirTree('/out');
  await py.runPythonAsync(RUNNER);
  say({ type: 'ready' });
}

function call(name, arg) {
  const result = JSON.parse(arg === undefined ? py.globals.get(name)() : py.globals.get(name)(arg));
  if (!result.ok) throw Object.assign(new Error(result.message), { trace: result.trace });
  return result.value;
}

function prepare({ table, cfg }) {
  py.FS.writeFile('/in/table.bin', new Uint8Array(table.buffer));
  const job = { ...cfg, table: { path: '/in/table.bin', columns: table.columns }, out: '/out/result' };
  py.FS.writeFile('/in/config.json', JSON.stringify(job));
  call('prepare', '/in/config.json');
}

function finish() {
  say({
    type: 'done',
    html: py.FS.readFile('/out/result.html', { encoding: 'utf8' }),
    json: py.FS.readFile('/out/result.json', { encoding: 'utf8' }),
  });
}

self.onmessage = async ({ data }) => {
  if (data.type === 'read') {
    try {
      const table = await readTable(data.file, data.columns);
      self.postMessage({ type: 'table', table }, [table.buffer]);
    } catch (e) {
      say({ type: 'unreadable', message: e.message });
    }
    return;
  }
  try {
    await boot();
    if (data.type === 'prepare') {
      prepare(data);
      say({ type: 'prepared' });
      if (data.lead) say({ type: 'jobs', jobs: call('begin') });
    }
    if (data.type === 'job') say({ type: 'result', id: data.id, value: call('work', JSON.stringify(data.job)) });
    if (data.type === 'advance') {
      const jobs = call('advance', JSON.stringify(data.results));
      if (jobs === null) finish();
      else say({ type: 'jobs', jobs });
    }
  } catch (e) {
    say({ type: 'failed', message: e && e.message ? e.message : String(e), trace: e && e.trace });
  }
};
