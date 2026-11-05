"""
grow.py - 구조 키우기: 정해 둔 후보 구조 없이, 곧은 관에서 시작해 부품을 하나씩 더해 가며 구조를 찾는다.

부품: 휘어짐(곡선이 지나는 점 추가), 굵기 변화(관을 둘로 나누고 굵기 비를 따로), 갈라졌다 모임(관 일부를 두 갈래로),
      갈래 휘어짐(두 갈래 중 하나에 점 추가).
단계마다 모든 부품 후보를 성긴 모델로 맞춰 보고, 앞선 후보를 원래 해상도로 다듬은 뒤
recon3d.plan과 같은 보정 BIC(잔차의 이웃 상관과 크기로 chi2 차이를 줄임)로 확실히 나아질 때만 남긴다.
웹에서는 후보 맞춤 하나하나가 do_job 한 조각이라 워커들이 나눠 계산한다.
"""
import copy, math
import numpy as np
from scipy.optimize import least_squares
import recon3d as R


# ---------------- 모델 ----------------
class ModelGraph:
    """state = {'J': [구간 사이 점...],
               'sec': [{'t':'tube','v':[점..],'g':k} | {'t':'pair','v':[[..],[..]],'g':k,'U':[u,w]}],
               'grp': [{'lam':x,'U':u}, ...],      # 같은 굵기의 호스 묶음: 철 양 lambda와 앞머리 속도를 함께 쓴다
               'lam_up': x, 't0': x, 'U0': x, 'tau': x}   (점은 실험실 좌표 x=a, y=c, z=b)
    굵기 변화를 넣을 때만 새 묶음이 생긴다. 갈라진 두 갈래는 같은 굵기로 보고 갈래마다 속도만 따로 둔다
    (유량 보존으로 정한 속도는 실측 #54에서 맞지 않아 속도를 맞춤값으로 둔다)."""
    kind = 'graph'

    def __init__(self, geo, box, struct, oi=0, prior=None):
        self.g = geo; self.struct = struct; self.oi = oi; self.nb = 1
        self.prior = prior if prior is not None else {'straight_sigma_cm': 1.5, 'downhill': True}
        lo = R.abc2lab([box['a'][0], box['b'][0], box['c'][0]]); hi = R.abc2lab([box['a'][1], box['b'][1], box['c'][1]])
        self.plo = np.minimum(lo, hi); self.phi = np.maximum(lo, hi)
        if self.prior.get('downhill'):     # 호스는 내리막만: 중간 점이 입구보다 높거나 출구보다 낮을 수 없다
            tol = self.prior.get('flat_tol_cm', 0.3)
            self.plo[1] = max(self.plo[1], geo.outlets[oi][1] - tol); self.phi[1] = min(self.phi[1], geo.inlet[1] + tol)

    def pack(self, st):
        q = []; lo = []; hi = []
        P = lambda p: (q.extend(list(p)), lo.extend(list(self.plo)), hi.extend(list(self.phi)))
        S = lambda x, a, b: (q.append(x), lo.append(a), hi.append(b))
        for p in st['J']: P(p)
        for sc in st['sec']:
            if sc['t'] == 'tube':
                for p in sc['v']: P(p)
            else:
                for vv in sc['v']:
                    for p in vv: P(p)
                S(sc['U'][0], 0.1, 30); S(sc['U'][1], 0.1, 30)
        for gr in st['grp']: S(gr['lam'], 0.0, 6.0); S(gr['U'], 0.1, 30)
        S(st['lam_up'], 0.0, 3.0); S(st['t0'], -60, 250); S(st['U0'], 0.3, 30); S(st['tau'], 0.1, 20)
        lo = np.array(lo, float); hi = np.array(hi, float)
        return np.clip(np.array(q, float), lo + 1e-6, hi - 1e-6), lo, hi

    def unpack(self, q):
        i = 0; st = {'J': [], 'sec': [], 'grp': []}
        for _ in range(len(self.struct['sec']) - 1): st['J'].append(np.array(q[i:i + 3], float)); i += 3
        for sd in self.struct['sec']:
            if sd['t'] == 'tube':
                v = []
                for _ in range(sd['nv']): v.append(np.array(q[i:i + 3], float)); i += 3
                st['sec'].append({'t': 'tube', 'v': v, 'g': sd['g']})
            else:
                vv = []
                for n in sd['nv']:
                    v = []
                    for _ in range(n): v.append(np.array(q[i:i + 3], float)); i += 3
                    vv.append(v)
                st['sec'].append({'t': 'pair', 'v': vv, 'g': sd['g'], 'U': [float(q[i]), float(q[i + 1])]}); i += 2
        for _ in range(self.struct['ng']): st['grp'].append({'lam': float(q[i]), 'U': float(q[i + 1])}); i += 2
        st['lam_up'], st['t0'], st['U0'], st['tau'] = (float(x) for x in q[i:i + 4])
        return st

    def ends(self, st, s):
        g = self.g
        A = g.inlet if s == 0 else st['J'][s - 1]
        B = g.outlets[self.oi] if s == len(st['sec']) - 1 else st['J'][s]
        return A, B

    def chains(self, st):
        out = []
        for s, sc in enumerate(st['sec']):
            A, B = self.ends(st, s)
            if sc['t'] == 'tube': out.append([A] + sc['v'] + [B])
            else: out += [[A] + v + [B] for v in sc['v']]
        return out

    def lam(self, st, s): return st['grp'][st['sec'][s]['g']]['lam']
    def speeds(self, st):
        return [st['grp'][sc['g']]['U'] if sc['t'] == 'tube' else list(sc['U']) for sc in st['sec']]

    def predict(self, q, tt):
        st = self.unpack(q); g = self.g; tau = st['tau']; out = np.zeros((len(tt), g.nc))
        lu = max(st['lam_up'], 1e-6); U0 = st['U0']; sp = self.speeds(st)
        R.front_sum(out, tt, tau, g.up_G, st['t0'] + g.up_s / U0, lu)
        tc = st['t0']; lastU, lastLam = U0, lu
        for s, sc in enumerate(st['sec']):
            A, B = self.ends(st, s); lam = self.lam(st, s)
            if sc['t'] == 'tube':
                G, ss, L = g.piece([A] + sc['v'] + [B]); U = sp[s]
                R.front_sum(out, tt, tau, G, tc + ss / U, lam)
                tc += L / U; lastU, lastLam = U, lam
            else:
                arr = []
                for k, v in enumerate(sc['v']):
                    G, ss, L = g.piece([A] + v + [B]); U = sp[s][k]
                    R.front_sum(out, tt, tau, G, tc + ss / U, lam); arr.append(tc + L / U)
                tc = min(arr); lastU, lastLam = st['grp'][sc['g']]['U'], lam      # 먼저 도착한 갈래가 합류점을 지나 나간다
        R.front_sum(out, tt, tau, g.out_G[self.oi], tc + g.out_pts[self.oi][1] / lastU, lastLam)
        return g.split(out)

    def nodes(self, st):
        return [[R.lab2abc(x) for x in self.g.curve_pts(ch)] for ch in self.chains(st)]

    def prior_res(self, q):
        """자연스러운 관: (1) 중간 점이 앞뒤 점을 이은 직선에서 벗어난 거리 / straight_sigma_cm  (2) 오르막 벌점
        (3) 입구면에서 출구면 쪽(a)으로 가다가 되돌아가면 벌점  (4) 두 갈래는 같은 자리를 지날 수 없음
        (5) 굵기가 바뀌는 점에서 관 방향이 갑자기 꺾이지 않음 (한 호스를 이어 붙인 것)"""
        st = self.unpack(q); r = []; P = self.prior
        sig = P.get('straight_sigma_cm', 1.5); dh = P.get('downhill', True)
        tol = P.get('flat_tol_cm', 0.3); dsig = P.get('downhill_sigma_cm', 0.2)
        fw = P.get('forward_sigma_cm', 0.3); sep = P.get('pair_min_sep_cm', 0.6); js = P.get('joint_sigma_deg', 20)
        secs = st['sec']
        for s, sc in enumerate(secs):
            chs = [[*self.ends(st, s)[:1]] + sc['v'] + [self.ends(st, s)[1]]] if sc['t'] == 'tube' else [[self.ends(st, s)[0]] + v + [self.ends(st, s)[1]] for v in sc['v']]
            for ch in chs:
                if sc['t'] == 'tube':     # 두 갈래는 서로 벌어져야 하므로 곧게 하라는 조건은 한 줄 구간에만
                    for i in range(1, len(ch) - 1):
                        A, C, B = ch[i - 1], ch[i], ch[i + 1]; d = B - A; L = np.linalg.norm(d)
                        if L > 1e-6 and sig:
                            v = C - A; dn = d / L; r += list((v - (v @ dn) * dn) / sig)
                for a, b in zip(ch[:-1], ch[1:]):
                    if dh: r.append(max(0.0, b[1] - a[1] - tol) / dsig)
                    if fw: r.append(max(0.0, a[0] - b[0]) / fw)
        for s in range(len(secs) - 1):
            if js and secs[s]['t'] == 'tube' and secs[s + 1]['t'] == 'tube':
                A, J = self.ends(st, s); _, B = self.ends(st, s + 1)
                a = J - (secs[s]['v'][-1] if secs[s]['v'] else A); b = (secs[s + 1]['v'][0] if secs[s + 1]['v'] else B) - J
                na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-6 and nb_ > 1e-6: r += list(np.cross(a / na, b / nb_) / math.sin(math.radians(js)))
        if sep:
            for s, sc in enumerate(secs):
                if sc['t'] != 'pair': continue
                A, B = self.ends(st, s)
                c0 = [A] + sc['v'][0] + [B]; c1 = [A] + sc['v'][1] + [B]
                for fr in (0.3, 0.5, 0.7):
                    d = np.linalg.norm(point_along(c0, fr) - point_along(c1, fr))
                    r.append(max(0.0, sep - d) / 0.15)
        return np.array(r)


def point_along(nodes, frac):
    return split_at(nodes, frac)[0]

def split_at(nodes, frac):
    """점 목록을 길이 비 frac에서 나눈 (그 점, 앞쪽에 남는 중간 점 수)"""
    nodes = [np.asarray(n, float) for n in nodes]; seg = [np.linalg.norm(b - a) for a, b in zip(nodes[:-1], nodes[1:])]
    L = sum(seg); target = frac * L; acc = 0.0
    for k, ((a, b), l) in enumerate(zip(zip(nodes[:-1], nodes[1:]), seg)):
        if acc + l >= target and l > 0: return a + (b - a) * (target - acc) / l, k
        acc += l
    return nodes[-1], len(seg) - 1

def struct_of(st):
    return {'sec': [{'t': 'tube', 'nv': len(sc['v']), 'g': sc['g']} if sc['t'] == 'tube' else {'t': 'pair', 'nv': [len(v) for v in sc['v']], 'g': sc['g']}
                    for sc in st['sec']], 'ng': len(st['grp'])}

MOVE_NAME = {'bend': '휘어짐', 'step': '굵기 변화', 'pair': '갈라졌다 모임', 'branch_bend': '갈래 휘어짐'}

def moves(model, st):
    """지금 구조에 부품 하나를 더한 후보들: (부품 종류, 구간 번호, 새 상태)"""
    out = []
    for s, sc in enumerate(st['sec']):
        A, B = model.ends(st, s); grp = st['grp'][sc['g']]
        if sc['t'] == 'tube':
            nodes = [A] + sc['v'] + [B]
            segl = [np.linalg.norm(nodes[k + 1] - nodes[k]) for k in range(len(nodes) - 1)]; k = int(np.argmax(segl))
            if len(sc['v']) < 3:
                for dz in ([0, -0.8, 0], [0, 0, 1.0], [0, 0, -1.0]):
                    st2 = copy.deepcopy(st); st2['sec'][s]['v'] = sc['v'][:k] + [(nodes[k] + nodes[k + 1]) / 2 + np.array(dz)] + sc['v'][k:]
                    out.append(('bend', s, st2))
            for frac in (0.35, 0.65):
                P, k = split_at(nodes, frac)
                for ratio in (2.8, 0.36):       # 3 mm <-> 5 mm 의 단면적 비
                    st2 = copy.deepcopy(st); ng = len(st2['grp'])
                    st2['grp'].append({'lam': grp['lam'] * ratio, 'U': grp['U'] / ratio})
                    after = {'t': 'tube', 'v': sc['v'][k:], 'g': ng}
                    st2['sec'][s:s + 1] = [{'t': 'tube', 'v': sc['v'][:k], 'g': sc['g']}, after]
                    for later in st2['sec'][s + 2:]:       # 뒤쪽 구간들도 같은 굵기로 따라간다
                        if later['g'] == sc['g']: later['g'] = ng
                    st2['J'][s:s] = [P]; out.append(('step', s, st2))
            for f0, f1 in ((0.0, 0.6), (0.25, 0.75), (0.4, 1.0)):
                if (f0 == 0 and s != 0) or (f1 == 1 and s != len(st['sec']) - 1): continue
                Pa, ka = split_at(nodes, f0) if f0 > 0 else (None, 0)
                Pb, kb = split_at(nodes, f1) if f1 < 1 else (None, len(nodes) - 1)
                a_ = A if Pa is None else Pa; b_ = B if Pb is None else Pb; mid = (a_ + b_) / 2
                for off in ([0, 0, 1.5], [0, 1.2, 0]):
                    o = np.array(off)
                    pair = {'t': 'pair', 'v': [[mid + o], [mid - o]], 'g': sc['g'], 'U': [grp['U'] / 2, grp['U'] / 2]}
                    new = ([{'t': 'tube', 'v': sc['v'][:ka], 'g': sc['g']}] if Pa is not None else []) + [pair] + \
                          ([{'t': 'tube', 'v': sc['v'][kb:], 'g': sc['g']}] if Pb is not None else [])
                    st2 = copy.deepcopy(st); st2['sec'][s:s + 1] = new
                    st2['J'][s:s] = [p for p in (Pa, Pb) if p is not None]
                    out.append(('pair', s, st2))
        else:
            for k in range(2):
                if len(sc['v'][k]) >= 2: continue
                nodes = [A] + sc['v'][k] + [B]; segl = [np.linalg.norm(nodes[j + 1] - nodes[j]) for j in range(len(nodes) - 1)]
                j = int(np.argmax(segl))
                st2 = copy.deepcopy(st); st2['sec'][s]['v'][k] = sc['v'][k][:j] + [(nodes[j] + nodes[j + 1]) / 2 + np.array([0, -0.5, 0])] + sc['v'][k][j:]
                out.append(('branch_bend', s, st2))
    return out


# ---------------- 숫자만으로 주고받기 (웹 워커) ----------------
def to_json(st):
    L = lambda p: [float(x) for x in p]
    sec = []
    for sc in st['sec']:
        if sc['t'] == 'tube': sec.append({'t': 'tube', 'v': [L(p) for p in sc['v']], 'g': int(sc['g'])})
        else: sec.append({'t': 'pair', 'v': [[L(p) for p in v] for v in sc['v']], 'g': int(sc['g']), 'U': [float(x) for x in sc['U']]})
    return {'J': [L(p) for p in st['J']], 'sec': sec, 'grp': [{'lam': float(g['lam']), 'U': float(g['U'])} for g in st['grp']],
            **{k: float(st[k]) for k in ('lam_up', 't0', 'U0', 'tau')}}

def from_json(d):
    A = lambda p: np.array(p, float)
    sec = []
    for sc in d['sec']:
        if sc['t'] == 'tube': sec.append({'t': 'tube', 'v': [A(p) for p in sc['v']], 'g': sc['g']})
        else: sec.append({'t': 'pair', 'v': [[A(p) for p in v] for v in sc['v']], 'g': sc['g'], 'U': list(sc['U'])})
    return {'J': [A(p) for p in d['J']], 'sec': sec, 'grp': [dict(g) for g in d['grp']], **{k: d[k] for k in ('lam_up', 't0', 'U0', 'tau')}}


# ---------------- 맞춤 ----------------
def level(ctx, exit_i, name):
    """출구 하나, 해상도 하나에 대한 (설정, 시간, 측정, 기하) 묶음을 만들어 둔다."""
    cache = ctx.__dict__.setdefault('grow_levels', {})
    key = (exit_i, name)
    if key not in cache:
        cfg = ctx.cfg; c = dict(cfg); c['outlets'] = [cfg['outlets'][exit_i]]
        if name == 'fine': tt, data, bands = ctx.tw, ctx.dw, R.BANDS_FINE
        else:
            c['ds'] = 2 * cfg.get('ds', 0.25); bands = R.BANDS_COARSE
            tt, data = R.rebin(ctx.tw, ctx.dw, cfg.get('coarse_bin', 2))
        cache[key] = (c, tt, data, R.Geometry(c, ctx.I, bands))
    return cache[key]

def build(ctx, exit_i, name, st):
    c, tt, data, geo = level(ctx, exit_i, name)
    model = ModelGraph(geo, c['box'], struct_of(st), 0, grow_prior(ctx.cfg))
    b0a, b0b = c.get('baseline', [2, 12])
    f, Y, U = R.residuals(model, data, tt, drift=c.get('drift', False), tb=(b0a + b0b) / 2, knocks=c.get('free_knocks', ()))
    ft = lambda q: np.concatenate([f(q), model.prior_res(q)])
    return model, f, ft, Y, U, tt, data

def grow_prior(cfg):
    p = dict(cfg.get('prior') or {'straight_sigma_cm': 1.5, 'downhill': True})
    p.update(cfg.get('grow_prior') or {})
    return p

def grow_job(ctx, job):
    """후보 구조 하나를 맞춘다 (웹 워커 한 조각)."""
    try:
        st = from_json(job['st'])
        model, f, ft, *_ = build(ctx, job['exit'], job['level'], st)
        q0, lo, hi = model.pack(st)
        s = least_squares(ft, q0, bounds=(lo, hi), max_nfev=job.get('nfev', 60), ftol=job.get('ftol', 1e-3), xtol=1e-4)
        rd = f(s.x)
        return {'st': to_json(model.unpack(s.x)), 'cost': float(s.cost), 'chi2': float(rd @ rd), 'N': int(len(rd)), 'k': int(len(s.x)),
                'move': job.get('move'), 'sec': job.get('sec')}
    except Exception as e:
        print(f'  구조 키우기 한 조각 실패: {e}', flush=True)
        return None

def grow_evidence(ctx, job):
    """recon3d.evidence_penalty와 같은 라플라스 근사 벌점 (구조 키우기 답)."""
    try:
        st = from_json(job['st']); model, f, ft, *_ = build(ctx, job['exit'], 'fine', st)
        x, lo, hi = model.pack(st); wide = hi - lo; J = []
        for i in range(len(x)):
            a = x.copy(); b = x.copy(); h = 2e-3 * wide[i]
            a[i] = max(lo[i], x[i] - h); b[i] = min(hi[i], x[i] + h)
            J.append((ft(b) - ft(a)) / ((b[i] - a[i]) / wide[i]))
        J = np.array(J).T
        return {'penalty': float(np.sum(np.log(np.maximum(np.linalg.eigvalsh(J.T @ J / job['scale']) / (2 * np.pi), 1.0))))}
    except Exception as e:
        print(f'  구조 키우기 점수 계산 실패: {e}', flush=True)
        return None

def gain(ctx, cur, new):
    """recon3d.plan의 보정 BIC와 같은 식의 개선량. 클수록 새 부품이 확실히 필요하다."""
    ns, nt = len(ctx.dw), len(ctx.tw)
    corr = R.correlation(cur['fun'], ns, nt); scale = corr * max(1.0, cur['chi2'] / cur['N'])
    return (cur['chi2'] - new['chi2']) / scale - (new['k'] - cur['k']) * math.log(cur['N'] / corr)

def finalize(ctx, exit_i, st_json):
    st = from_json(st_json); model, f, ft, Y, U, tt, data = build(ctx, exit_i, 'fine', st)
    x = model.pack(st)[0]; rd = f(x)
    return {'model': model, 'f': f, 'ft': ft, 'Y': Y, 'U': U, 'x': x, 'fun': rd, 'chi2': float(rd @ rd), 'N': len(rd), 'k': len(x), 'st': st_json}

def start_state(model, x):
    """구조 후보 '한 줄 관'으로 맞춘 답을 출발 상태로 바꾼다 (중간점 C는 곡선이 지나는 점 그대로)."""
    p = model.unpack(np.asarray(x, float))
    return {'J': [], 'sec': [{'t': 'tube', 'v': [np.array(p['C'][0], float)], 'g': 0}], 'grp': [{'lam': max(float(p['lam'][0]), 1e-3), 'U': float(p['U'][0])}],
            'lam_up': max(float(p['lam0']), 1e-3), 't0': float(p['t0']), 'U0': float(p['U0']), 'tau': float(p['tau'])}

def grow_plan(ctx, results):
    """한 줄 관 답에서 출발해 출구마다 구조를 키운다. 할 일 목록을 내보내고 결과를 받는 생성기."""
    cfg = ctx.cfg; steps = cfg.get('grow_steps', 3); min_gain = cfg.get('grow_min_gain', 10.0)
    keep = cfg.get('grow_keep', 3); nf1, nf2 = cfg.get('grow_nfev', (40, 120))
    singles = [r for r in results if r['kind'] == 'single']
    if not singles: return []
    low = min(r['chi2'] for r in singles)
    singles = [r for r in singles if r['chi2'] <= cfg.get('exit_near', 1.3) * low]      # 잘 맞은 출구만 키운다
    cur = {}
    for r in singles:
        e = r['idx'][0]; st = to_json(start_state(r['model'], r['best'].x))
        cur[e] = {**finalize(ctx, e, st), 'hist': [{'step': 0, 'added': '한 줄 관', 'chi2_per_N': round(r['chi2_per_N'], 3)}], 'label': r['label']}
    alive = set(cur)
    for step in range(1, steps + 1):
        if not alive: break
        jobs = []
        for e in sorted(alive):
            m = cur[e]['model']; st = from_json(cur[e]['st'])
            for move, s, st2 in moves(m, st):
                jobs.append({'stage': 'grow', 'exit': e, 'level': 'coarse', 'st': to_json(st2), 'nfev': nf1, 'move': move, 'sec': s})
        print(f'구조 키우기 {step}단계: 후보 {len(jobs)}개', flush=True)
        got = yield jobs
        polish = []
        for e in sorted(alive):
            mine = sorted([g for j, g in zip(jobs, got) if g and j['exit'] == e], key=lambda g: g['cost'])[:keep]
            polish += [{'stage': 'grow', 'exit': e, 'level': 'fine', 'st': g['st'], 'nfev': nf2, 'ftol': 1e-5, 'move': g['move'], 'sec': g['sec']} for g in mine]
        got = yield polish
        for e in sorted(alive):
            mine = [g for j, g in zip(polish, got) if g and j['exit'] == e]
            if not mine: alive.discard(e); continue
            g = min(mine, key=lambda g: g['cost']); new = finalize(ctx, e, g['st'])
            gn = gain(ctx, cur[e], new)
            print(f"   {cur[e]['label']}: + {MOVE_NAME[g['move']]} → chi2/N {new['chi2'] / new['N']:.3f}, 보정 개선량 {gn:.1f}", flush=True)
            if gn < min_gain: alive.discard(e); continue
            cur[e] = {**new, 'hist': cur[e]['hist'] + [{'step': step, 'added': MOVE_NAME[g['move']], 'chi2_per_N': round(new['chi2'] / new['N'], 3), 'gain': round(gn, 1)}],
                      'label': cur[e]['label']}
    out = []
    names = ctx.all_names
    for e, c in cur.items():
        if len(c['hist']) == 1: continue          # 아무것도 더하지 않았으면 '한 줄 관'과 같은 답
        st = from_json(c['st']); kinds = describe(st)
        nstep = sum(1 for i in range(len(st['sec']) - 1) if st['sec'][i]['t'] == 'tube' and st['sec'][i + 1]['t'] == 'tube')
        npair = sum(1 for sc in st['sec'] if sc['t'] == 'pair'); nbend = sum(len(sc['v']) if sc['t'] == 'tube' else max(len(v) for v in sc['v']) - 1 for sc in st['sec'])
        same_as = 'step' if (nstep, npair) == (1, 0) and nbend <= 1 else 'loop' if (nstep, npair) == (0, 1) and nbend <= 0 else None
        cc = dict(ctx.cfg); cc['outlets'] = [ctx.cfg['outlets'][e]]; cc['outlet_names'] = [names[e]]
        label = f"구조 키우기: {kinds}" + (f' · {names[e]}' if len(ctx.cfg['outlets']) >= 2 else '')
        r0 = c['ft'](c['x']); chi2 = float(r0 @ r0)      # 다른 후보처럼 사전 조건까지 넣은 chi2
        out.append({'kind': 'graph', 'idx': [e], 'cfg': cc, 'label': label, 'c': None, 'search': None, 'model': c['model'], 'Y': c['Y'], 'U': c['U'],
                    'f': c['ft'], 'best': R.Sol(c['x'], 0.5 * chi2, r0), 'rough': False,
                    'chi2_per_N': chi2 / len(r0), 'chi2': chi2, 'N': len(r0), 'npar': c['k'], 'grow_history': c['hist'], 'same_as': same_as,
                    'st_json': c['st']})
    return out

def describe(st):
    parts = []
    for s in range(len(st['sec']) - 1):
        a, b = st['sec'][s]['t'], st['sec'][s + 1]['t']
        if a == 'tube' and b == 'tube': parts.append('굵기 변화')
        elif b == 'pair': parts.append('갈라졌다 모임')
    if st['sec'][0]['t'] == 'pair' and 'pair' not in [x for x in parts]: parts.insert(0, '갈라졌다 모임')
    return ', '.join(dict.fromkeys(parts)) or '휘어진 한 줄 관'


# ---------------- 결과 정리 ----------------
def features(model, st):
    """찾은 부품을 관을 따라 순서대로. 갈라짐/합쳐짐/굵기 변화와 곡선이 지나는 점."""
    out = []; secs = st['sec']
    for s, sc in enumerate(secs):
        if sc['t'] == 'tube': out += [{'type': '곡선이 지나는 점', 'p': R.lab2abc(v), 'key': ('v', s, 0, j)} for j, v in enumerate(sc['v'])]
        else:
            for k, vv in enumerate(sc['v']):
                out += [{'type': f'곡선이 지나는 점 (갈래{k + 1})', 'p': R.lab2abc(v), 'key': ('v', s, k, j)} for j, v in enumerate(vv)]
        if s < len(secs) - 1:
            a, b = secs[s], secs[s + 1]; J = R.lab2abc(st['J'][s])
            if a['t'] == 'tube' and b['t'] == 'tube':
                la, lb = st['grp'][a['g']]['lam'], st['grp'][b['g']]['lam']
                out.append({'type': '굵기 변화', 'p': J, 'ratio': float(math.sqrt(max(lb, 0) / max(la, 1e-9))), 'key': ('J', s)})
            elif b['t'] == 'pair': out.append({'type': '갈라짐', 'p': J, 'key': ('J', s)})
            elif a['t'] == 'pair': out.append({'type': '합쳐짐', 'p': J, 'key': ('J', s)})
    if secs and secs[0]['t'] == 'pair': out.insert(0, {'type': '갈라짐', 'p': R.lab2abc(model.g.inlet), 'key': None, 'at_inlet': True})
    if secs and secs[-1]['t'] == 'pair': out.append({'type': '합쳐짐', 'p': R.lab2abc(model.g.outlets[model.oi]), 'key': None, 'at_outlet': True})
    return out

def point_index(st):
    """pack() 순서에서 각 점이 몇 번째 숫자부터인지"""
    idx = {}; i = 0
    for s in range(len(st['J'])): idx[('J', s)] = i; i += 3
    for s, sc in enumerate(st['sec']):
        if sc['t'] == 'tube':
            for j in range(len(sc['v'])): idx[('v', s, 0, j)] = i; i += 3
        else:
            for k, v in enumerate(sc['v']):
                for j in range(len(v)): idx[('v', s, k, j)] = i; i += 3
            i += 2
    return idx

def pieces_payload(model, st, d_up_mm=3.0):
    """3D 그림용: 구간마다 (곡선 점들, 추정 굵기, 역할)"""
    out = []; base = max(st['grp'][st['sec'][0]['g']]['lam'], 1e-9)      # 입구 쪽 구간을 d_up_mm로 보고 철 양 비로 굵기를 낸다
    for s, sc in enumerate(st['sec']):
        A, B = model.ends(st, s); d = d_up_mm * math.sqrt(max(model.lam(st, s), 0) / base)
        if sc['t'] == 'tube':
            out.append({'nodes': [R.lab2abc(x) for x in model.g.curve_pts([A] + sc['v'] + [B])], 'd': d, 'role': 'tube', 'sec': s})
        else:
            for k, v in enumerate(sc['v']):
                out.append({'nodes': [R.lab2abc(x) for x in model.g.curve_pts([A] + v + [B])], 'd': d, 'role': 'branch', 'sec': s, 'k': k})
    return out

def summarize_graph(model, x, N, chi2, d_up_mm=3.0):
    st = model.unpack(np.asarray(x, float)); sp = model.speeds(st); feats = features(model, st)
    res = {'chi2_per_N': float(chi2 / N), 't0': st['t0'], 'U0': st['U0'], 'tau': st['tau'], 'lam0': st['lam_up'], 'topology': 'graph',
           'U': [float(sp[-1]) if not isinstance(sp[-1], list) else float(min(sp[-1]))],
           'lam': [float(model.lam(st, len(st['sec']) - 1))],
           'C': [f['p'] for f in feats if f['type'].startswith('곡선')], 'features': [{k: v for k, v in f.items() if k != 'key'} for f in feats],
           'sections': [{'t': sc['t'], 'U': sp[s], 'lam': model.lam(st, s)} for s, sc in enumerate(st['sec'])],
           'branch_len_cm': [], 'stem_len_cm': 0.0}
    for key, typ in (('J', '갈라짐'), ('J2', '합쳐짐'), ('P', '굵기 변화')):
        f = next((f for f in feats if f['type'] == typ), None)
        if f: res[key] = f['p']
    if 'P' in res: res['diam_ratio'] = next(f['ratio'] for f in feats if f['type'] == '굵기 변화')
    return res
