#!/usr/bin/env python3
"""
recon3d.py - 자기장 측정으로 숨은 통로의 3D 경로를 복원하는 프로그램

사용법:  python recon3d.py config.json
결과:    <out>.json (숫자 결과), <out>.html (돌려 볼 수 있는 3D 그림 + 측정/모델 비교)

모델
- 코일 자기장 B0는 두 원형 코일(200회, 500회)의 비오-사바르 계산으로 구한다.
- 철가루가 찬 호스 조각 하나하나를 자기 쌍극자로 본다. 쌍극자 모멘트 = lambda * B0(위치) * 길이
  (lambda = 실효 자화율 x 단면적, 단위 cm^2).
- 철가루 앞머리가 입구에서 속도 U0로 들어와 분기점에서 갈래마다 속도 Uk로 나뉘어 흐른다.
  앞머리는 시간 폭 tau로 부드럽게 올라간다.
- 센서마다 코일 자기장 방향(한 축)은 측정값으로 정하고, 그 축을 중심으로 한 회전 psi는
  매 계산마다 가장 잘 맞는 값으로 정한다(센서 방향을 다 몰라도 되게).
- 미지수: 분기점 위치, 갈래마다 꺾이는 점 1개, 앞머리 도착 시각, 속도, lambda, tau.
  여러 시작점에서 최소제곱으로 맞추고, 가장 잘 맞는 답과 비슷하게 맞는 답들을 함께 보여 준다.
좌표: 입력은 (a, b, c) cm = (입구면에서 출구 쪽, 앞면에서 안쪽, 탁자에서 높이).
"""
import sys, os, json, math, time
import numpy as np, pandas as pd
from scipy.optimize import least_squares

MU0 = 4e-7 * math.pi
# ---------------- coil model (lab frame: x=a, y=c, z=b; cm) ----------------
R200 = (0.1006 + 0.1058) / 2; R500 = (0.1006 + 0.1137) / 2; DCOIL = 0.12
AXIS_Y, AXIS_Z, XC = 16.5, 4.0, 5.0
_th = np.linspace(0, 2 * np.pi, 360, endpoint=False)

def _loop(P, zc, R):
    pts = np.stack([R * np.cos(_th), R * np.sin(_th), np.full_like(_th, zc)], 1)
    dl = np.stack([-R * np.sin(_th), R * np.cos(_th), np.zeros_like(_th)], 1) * (2 * np.pi / len(_th))
    r = P[:, None, :] - pts[None]; rn = np.linalg.norm(r, axis=2)[..., None]
    return MU0 / (4 * np.pi) * np.sum(np.cross(dl[None], r) / rn ** 3, 1)

def B0lab(P, I):
    P = np.atleast_2d(np.asarray(P, float))
    pm = np.stack([(P[:, 1] - AXIS_Y) / 100, (P[:, 2] - AXIS_Z) / 100, (P[:, 0] - XC) / 100], 1)
    b = 500 * _loop(pm, DCOIL / 2, R500) + 200 * _loop(pm, -DCOIL / 2, R200)
    return np.stack([b[:, 2], b[:, 0], b[:, 1]], 1) * I * 1e6   # uT

def abc2lab(p): return np.array([p[0], p[2], p[1]], float)
def lab2abc(p): return [float(p[0]), float(p[2]), float(p[1])]

def rot_align(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = a @ b
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)

def rot_axis(u, psi):
    u = u / np.linalg.norm(u); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(psi) * K + (1 - np.cos(psi)) * K @ K

def best_psi(x, y, u, w=None):
    xw = x if w is None else x * w[:, None]; yw = y if w is None else y * w[:, None]
    xp = xw - np.outer(xw @ u, u); yp = yw - np.outer(yw @ u, u)
    psi = np.arctan2(np.sum(np.cross(xp, yp) @ u), np.sum(xp * yp))
    return x @ rot_axis(u, psi).T, psi

# ---------------- data ----------------
def load(cfg):
    run = cfg['run']; ids = cfg['ids']
    df = pd.read_csv(cfg['csv'], encoding='utf-8-sig', low_memory=False, usecols=lambda c: c.endswith(f'Run #{run}'))
    if f'Time (s) Run #{run}' not in df.columns:
        cols = pd.read_csv(cfg['csv'], encoding='utf-8-sig', nrows=0).columns
        runs = sorted({int(c.split('Run #')[1]) for c in cols if 'Run #' in c})
        sys.exit(f"[오류] CSV에 Run #{run} 데이터가 없어요. 이 파일에 있는 Run 번호: {runs}")
    t = df[f'Time (s) Run #{run}'].dropna().values
    I = df[f'Output Current (A) Run #{run}'].dropna().values
    n = min(len(t), len(I)); t = t[:n]; I = I[:n]
    b0a, b0b = cfg.get('baseline', [2, 12])
    m0 = (t > b0a) & (t < b0b); Iref = I[m0].mean()
    t0, t1 = cfg['t_window']; dt = cfg.get('bin', 0.5)
    t0 = max(t0, float(t[0])); t1 = min(t1, float(t[-1]))      # 데이터 길이를 넘으면 자동으로 줄임
    edges = np.arange(t0, t1 + 1e-9, dt); tc = (edges[:-1] + edges[1:]) / 2
    data = {}
    for k, s in ids.items():
        B = np.stack([df[f'{c} Magnetic Field Strength, {s} (G) Run #{run}'].dropna().values for c in 'XYZ'], 1)[:n] * 100
        b0 = B[m0].mean(0); u = b0 / np.linalg.norm(b0); g = np.linalg.norm(b0) / Iref
        dB = B - np.outer(g * (I - Iref), u) - b0
        V = np.array([dB[(t >= a) & (t < a + dt)].mean(0) for a in edges[:-1]])
        # 잡음: 기준 구간의 0.5초 평균값들이 (직선 추세를 뺀 뒤) 흔들리는 정도
        e0 = np.arange(b0a, b0b - dt + 1e-9, dt)
        Vb = np.array([dB[(t >= a) & (t < a + dt)].mean(0) for a in e0])
        if len(Vb) >= 4:
            A = np.stack([np.ones(len(e0)), e0 - e0.mean()], 1)
            rr = Vb - A @ np.linalg.lstsq(A, Vb, rcond=None)[0]
            noise = float(np.sqrt((rr ** 2).sum(0) / (len(e0) - 2)).mean())
        else:
            noise = float(dB[m0].std(0).mean() / math.sqrt(max(1, dt * 50 / 5)))
        w = np.ones(len(tc))
        for ma, mb in cfg.get('mask', {}).get(k, []):      # 고장/외부 방해 구간 제외
            w[(tc >= ma) & (tc < mb)] = 0
        V[w == 0] = 0
        data[k] = {'b0': b0, 'dB': V, 'noise': max(noise, cfg.get('noise_floor', 0.05)), 'w': w}
    return tc, data, Iref

# ---------------- geometry ----------------
class Geometry:
    def __init__(self, cfg, I):
        self.I = I; self.ds = cfg.get('ds', 0.25)
        self.inlet = abc2lab(cfg['inlet'])
        self.outlets = [abc2lab(o) for o in cfg['outlets']]
        self.up_dir = np.array(cfg.get('upstream_dir_abc', [-1, 0, 0]), float)[[0, 2, 1]]
        self.out_dir = np.array(cfg.get('outlet_dir_abc', [1, 0, 0]), float)[[0, 2, 1]]
        self.L_up = cfg.get('upstream_len', 40.0); self.L_out = cfg.get('outlet_len', 20.0)
        self.sensors = {k: abc2lab(v) for k, v in cfg['sensors'].items()}
        # fixed parts (precomputed): upstream hose and outside outlet hoses
        s = np.arange(-self.L_up, 0, self.ds) + self.ds / 2
        self.up_pts = self.inlet[None] + np.outer(-s, self.up_dir / np.linalg.norm(self.up_dir))
        self.up_s = s; self.up_G = self.unit_field(self.up_pts)
        self.out_pts = []; self.out_G = []
        for o in self.outlets:
            so = np.arange(0, self.L_out, self.ds) + self.ds / 2
            p = o[None] + np.outer(so, self.out_dir / np.linalg.norm(self.out_dir))
            self.out_pts.append((p, so)); self.out_G.append(self.unit_field(p))

    def unit_field(self, P):
        B0 = B0lab(P, self.I); G = {}
        for k, S in self.sensors.items():
            r = S[None] - P; rn = np.linalg.norm(r, axis=1); rh = r / rn[:, None]
            G[k] = (self.ds / (4 * np.pi * rn ** 3))[:, None] * (3 * np.sum(B0 * rh, 1)[:, None] * rh - B0)
        return G

    def polyline(self, nodes):
        pts = []; ss = []; s0 = 0.0
        for a, b in zip(nodes[:-1], nodes[1:]):
            L = np.linalg.norm(b - a)
            if L < 1e-6: continue
            n = max(1, int(round(L / self.ds))); u = (np.arange(n) + 0.5) / n
            pts.append(a[None] + np.outer(u, b - a)); ss.append(s0 + u * L); s0 += L
        if not pts: return np.zeros((0, 3)), np.zeros(0), 0.0
        return np.concatenate(pts), np.concatenate(ss), s0

# ---------------- model ----------------
class Model:
    """nb = number of branches (= number of outlets). nb=1 : inlet -> C -> outlet.
    nb>=2 : inlet -> J (junction) -> C_k -> outlet_k."""
    def __init__(self, geo, nb, bounds_abc, phys=None):
        # phys = {'d_stem': 3, 'd_br': [3, 5], 'split': [0.75, 0.25] or None}
        #  -> 같은 농도라고 보고 lambda_k = lambda0 x (d_k/d_stem)^2,
        #     유량 보존: U_k = U0 x (A_stem/A_k) x f_k  (f_k = 갈래 k로 가는 유량 비율)
        self.g = geo; self.nb = nb; self.phys = phys
        lo = abc2lab([bounds_abc['a'][0], bounds_abc['b'][0], bounds_abc['c'][0]])
        hi = abc2lab([bounds_abc['a'][1], bounds_abc['b'][1], bounds_abc['c'][1]])
        self.plo = np.minimum(lo, hi); self.phi = np.maximum(lo, hi)

    # parameter vector layout
    def unpack(self, q):
        i = 0; out = {}
        if self.nb >= 2: out['J'] = q[i:i + 3]; i += 3
        out['C'] = [q[i + 3 * k:i + 3 * k + 3] for k in range(self.nb)]; i += 3 * self.nb
        out['t0'], out['U0'] = q[i], q[i + 1]; i += 2
        if self.phys:
            ph = self.phys; A0 = ph['d_stem'] ** 2; A = [d ** 2 for d in ph['d_br']]
            if ph.get('split'): f = list(ph['split'])
            else:
                f = list(q[i:i + self.nb - 1]); i += self.nb - 1
                f = f + [max(0.01, 1 - sum(f))]
            out['f'] = f
            out['U'] = [out['U0'] * A0 / A[k] * f[k] for k in range(self.nb)]
            out['lam0'] = q[i]; i += 1
            out['lam'] = [out['lam0'] * A[k] / A0 for k in range(self.nb)]
        else:
            out['U'] = list(q[i:i + self.nb]); i += self.nb
            out['lam0'] = q[i]; i += 1
            out['lam'] = list(q[i:i + self.nb]); i += self.nb
        out['tau'] = q[i]
        return out

    def tail_guess(self, cfg, tt, rng, tg=None):
        if tg is None: tg = cfg.get('t_front_guess', float(tt[0]) + 5)
        tg = tg + rng.uniform(-3, 4)
        Ug = float(np.exp(rng.uniform(np.log(0.7), np.log(7.0)))) if cfg.get('U_guess') is None else cfg['U_guess'] * rng.uniform(0.5, 2.0)
        q = [tg, Ug]
        if self.phys:
            if not self.phys.get('split'): q += [1.0 / self.nb] * (self.nb - 1)
            q += [0.1]
        else:
            q += [Ug * 0.8] * self.nb + [0.1] + [0.1] * self.nb
        return q + [2.0]

    def bounds(self):
        lo = []; hi = []
        if self.nb >= 2: lo += list(self.plo); hi += list(self.phi)
        for _ in range(self.nb): lo += list(self.plo); hi += list(self.phi)
        lo += [-60, 0.3]; hi += [200, 30]
        if self.phys:
            if not self.phys.get('split'): lo += [0.02] * (self.nb - 1); hi += [0.98] * (self.nb - 1)
            lo += [0.0]; hi += [3.0]
        else:
            lo += [0.1] * self.nb; hi += [30] * self.nb
            lo += [0.0]; hi += [3.0]
            lo += [0.0] * self.nb; hi += [3.0] * self.nb
        lo += [0.1]; hi += [20]
        return np.array(lo), np.array(hi)

    def paths(self, p):
        g = self.g
        J = p['J'] if self.nb >= 2 else None
        if self.nb == 1:
            pts, s, L = g.polyline([g.inlet, p['C'][0], g.outlets[0]])
            return {'stem': (np.zeros((0, 3)), np.zeros(0), 0.0), 'br': [(pts, s, L)]}
        stem = g.polyline([g.inlet, J])
        br = [g.polyline([J, p['C'][k], g.outlets[k]]) for k in range(self.nb)]
        return {'stem': stem, 'br': br}

    def predict(self, q, tt):
        p = self.unpack(q); g = self.g; P = self.paths(p)
        tau = p['tau']; sig = lambda x: 0.5 * (1 + np.tanh(x / (2 * tau)))
        out = {k: np.zeros((len(tt), 3)) for k in g.sensors}
        def add(G, tarr, lam):
            L = lam * sig(tt[:, None] - tarr[None])           # (Nt, Np)
            for k in out: out[k] += L @ G[k]
        # upstream
        add(g.up_G, p['t0'] + g.up_s / p['U0'], p['lam0'])
        stem_pts, stem_s, LJ = P['stem']
        if len(stem_pts): add(g.unit_field(stem_pts), p['t0'] + stem_s / p['U0'], p['lam0'])
        tJ = p['t0'] + LJ / p['U0']
        for k, (pts, s, Lb) in enumerate(P['br']):
            if len(pts): add(g.unit_field(pts), tJ + s / p['U'][k], p['lam'][k])
            op, so = g.out_pts[k]
            add(g.out_G[k], tJ + (Lb + so) / p['U'][k], p['lam'][k])
        return out

def residuals(model, data, tt, rotfree=True, drift=False, tb=0.0):
    U = {k: B0lab(model.g.sensors[k], model.g.I)[0] for k in data}
    U = {k: v / np.linalg.norm(v) for k, v in U.items()}
    RA = {k: rot_align(data[k]['b0'], U[k] * np.linalg.norm(data[k]['b0'])) for k in data}
    Y = {k: data[k]['dB'] @ RA[k].T for k in data}
    def f(q):
        pr = model.predict(q, tt); r = []
        for k in data:
            x = pr[k]; w = data[k]['w']; y = Y[k]
            if drift:   # 천천히 변하는 센서 표류(기준 구간에서 0, 시간에 비례)를 같이 빼 줌
                x = fit_one(x, y, U[k], w, rotfree, tt - tb)
            elif rotfree: x, _ = best_psi(x, y, U[k], w)
            r.append((((x - y) * w[:, None]) / data[k]['noise']).ravel())
        return np.concatenate(r)
    return f, Y, U

def fit_one(x, y, u, w, rotfree, dt):
    """rotate model x about u (psi) and add a linear drift d*dt (per component) to best match y."""
    d = np.zeros(3); xr = x
    for _ in range(2):
        if rotfree: xr, _ = best_psi(x, y - np.outer(dt, d), u, w)
        d = ((w * dt) @ (y - xr)) / max((w * dt) @ dt, 1e-9)
    return xr + np.outer(dt, d)

def auto_t_on(tt, data):
    """신호가 처음 뚜렷하게 커지는 시각(모든 센서의 |dB|/잡음 합이 최대의 30%를 넘는 때)"""
    s = sum(np.linalg.norm(d['dB'] * d['w'][:, None], axis=1) / d['noise'] for d in data.values())
    n = max(1, int(round(2.0 / (tt[1] - tt[0]))))
    s = np.convolve(s, np.ones(n) / n, mode='same')
    thr = s.min() + 0.3 * (s.max() - s.min())
    return float(tt[int(np.argmax(s > thr))])

def fit(cfg, tt, data, geo, nb):
    """여러 시작점에서 짧게 맞춰 본 뒤(1단계), 잘 맞은 몇 개만 끝까지 맞춤(2단계)."""
    model = Model(geo, nb, cfg['box'], cfg.get('phys'))
    b0a, b0b = cfg.get('baseline', [2, 12])
    f, Y, U = residuals(model, data, tt, drift=cfg.get('drift', False), tb=(b0a + b0b) / 2)
    lo, hi = model.bounds()
    rng = np.random.default_rng(cfg.get('seed', 0))
    t_on = auto_t_on(tt, data)
    tg_list = [t_on] + ([cfg['t_front_guess']] if cfg.get('t_front_guess') is not None else [])
    print(f'  신호가 커지기 시작한 시각(자동): {t_on:.1f} s', flush=True)
    span = model.phi - model.plo
    a_only = np.array([1.0, 0, 0]); not_a = np.array([0, 1.0, 1.0])
    starts = []
    nstart = cfg.get('starts', 16)
    for i in range(nstart):
        q = []
        J = geo.inlet
        if nb >= 2:
            if i % 2 == 0:   # 입구 근처
                J = geo.inlet + rng.uniform(0, 0.4) * span * a_only + rng.normal(0, 1.0, 3) * not_a
            else:            # 상자 앞쪽 60% 아무 데나
                J = model.plo + rng.uniform(0, 1, 3) * span * np.array([0.6, 1, 1])
            J = np.clip(J, model.plo, model.phi); q += list(J)
        for k in range(nb):
            if i % 3 == 2: C = model.plo + rng.uniform(0, 1, 3) * span
            else: C = (J + geo.outlets[k]) / 2 + rng.normal(0, 1.5, 3)
            q += list(np.clip(C, model.plo, model.phi))
        q += model.tail_guess(cfg, tt, rng, tg_list[i % len(tg_list)])
        starts.append(np.clip(np.array(q, float), lo + 1e-6, hi - 1e-6))
    sols1 = []
    t1 = time.time()
    for i, q0 in enumerate(starts):
        try:
            sols1.append(least_squares(f, q0, bounds=(lo, hi), max_nfev=cfg.get('nfev1', 60)))
        except Exception as e:
            print(f'  시작점 {i + 1} 실패:', e, flush=True)
        if (i + 1) % 4 == 0 or i + 1 == len(starts):
            print(f'  1단계 {i + 1}/{len(starts)} ({time.time() - t1:.0f}초), 지금까지 최소 chi2/N = {min(2 * s.cost / len(s.fun) for s in sols1):.3f}', flush=True)
    sols1.sort(key=lambda s: s.cost)
    m = cfg.get('refine', 4); sols = []
    for i, s in enumerate(sols1[:m]):
        s2 = least_squares(f, s.x, bounds=(lo, hi), max_nfev=cfg.get('max_nfev', 400))
        sols.append(s2 if s2.cost <= s.cost else s)
        print(f'  2단계 {i + 1}/{min(m, len(sols1))}: chi2/N = {2 * sols[-1].cost / len(s.fun):.3f}', flush=True)
    sols += sols1[m:]
    sols.sort(key=lambda s: s.cost)
    return model, sols, Y, U, f

def geom_from_truth(model, truth):
    """truth polylines -> geometry part of the parameter vector (None if the shape does not fit the model)."""
    g = []
    if model.nb == 1:
        pl = truth[0]
        if len(pl) == 2: c = (np.array(pl[0]) + np.array(pl[1])) / 2
        elif len(pl) == 3: c = np.array(pl[1])
        else: return None
        return np.array(abc2lab(c))
    if len(truth) != model.nb + 1: return None
    g += list(abc2lab(truth[0][-1]))
    for pl in truth[1:]:
        if len(pl) == 2: g += list(abc2lab((np.array(pl[0]) + np.array(pl[1])) / 2))
        elif len(pl) == 3: g += list(abc2lab(pl[1]))
        else: return None
    return np.array(g)

def fit_truth(model, f, sol, truth):
    """경로 모양을 실제 경로로 고정하고 나머지(시각, 속도, lambda, tau)만 맞춤 -> 비교용 chi2"""
    g = geom_from_truth(model, truth)
    if g is None: return None
    ng = len(g); lo, hi = model.bounds()
    glo = np.minimum(lo[:ng], g - 1e-3); ghi = np.maximum(hi[:ng], g + 1e-3)
    model_lo = lo.copy(); model_lo[:ng] = glo; model_hi = hi.copy(); model_hi[:ng] = ghi
    best = None
    for q0 in [sol.x[ng:], np.clip(np.r_[sol.x[ng:ng + 2], np.full(len(sol.x) - ng - 2, 0.3)], lo[ng:] + 1e-6, hi[ng:] - 1e-6)]:
        try:
            s = least_squares(lambda r: f(np.r_[g, r]), q0, bounds=(lo[ng:], hi[ng:]), max_nfev=300)
        except Exception as e:
            print('truth fit failed:', e); continue
        if best is None or s.cost < best.cost: best = s
    if best is None: return None
    best.x = np.r_[g, best.x]
    return best

def summarize(model, s, N):
    p = model.unpack(s.x); P = model.paths(p)
    res = {'chi2_per_N': float(2 * s.cost / N),
           't0': float(p['t0']), 'U0': float(p['U0']), 'U': [float(u) for u in p['U']],
           'lam0': float(p['lam0']), 'lam': [float(l) for l in p['lam']], 'tau': float(p['tau']),
           'C': [lab2abc(c) for c in p['C']]}
    if model.nb >= 2: res['J'] = lab2abc(p['J'])
    if 'f' in p: res['flow_split'] = [float(x) for x in p['f']]
    res['branch_len_cm'] = [float(b[2]) for b in P['br']]
    res['stem_len_cm'] = float(P['stem'][2])
    if model.nb >= 2 and min(p['lam']) > 0:
        res['area_ratio_from_lambda'] = [float(l / p['lam'][0]) for l in p['lam']]
    return res

def _dense(nodes, step=0.1):
    nodes = [np.asarray(n, float) for n in nodes]; out = [nodes[0]]
    for a, b in zip(nodes[:-1], nodes[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step)); out += [a + (b - a) * (i + 1) / n for i in range(n)]
    return np.array(out)

def path_error(model, s, truth):
    """truth: list of polylines in (a,b,c), same order as the drawn best path (nb>=2: stem, branch1, branch2...).
    Returns, per piece, mean and max distance (cm) between the fitted and true curves (both directions)."""
    p = model.unpack(s.x); g = model.g
    if model.nb == 1: best = [[lab2abc(g.inlet), lab2abc(p['C'][0]), lab2abc(g.outlets[0])]]
    else: best = [[lab2abc(g.inlet), lab2abc(p['J'])]] + [[lab2abc(p['J']), lab2abc(p['C'][k]), lab2abc(g.outlets[k])] for k in range(model.nb)]
    out = []
    for b, t in zip(best, truth):
        A = _dense(b); B = _dense(t)
        d = np.linalg.norm(A[:, None] - B[None], axis=2)
        dd = np.concatenate([d.min(1), d.min(0)])
        out.append({'mean': round(float(dd.mean()), 2), 'max': round(float(dd.max()), 2)})
    return out

def geom_payload(cfg, nb):
    """3D 그림용: 실제 상자 크기, 벽 두께, 구멍, 호스 굵기, 코일 위치"""
    bx = cfg['box']; w = cfg.get('wall', 0.5)
    outer = cfg.get('box_outer') or {'a': [bx['a'][0], bx['a'][1]], 'b': [bx['b'][0] - w, bx['b'][1] + w], 'c': [bx['c'][0] - w, bx['c'][1] + w]}
    hose = cfg.get('hose_mm') or {'stem': 3, 'branches': [3] * nb}
    return {'box_outer': outer, 'wall': w, 'hole_mm': cfg.get('hole_mm', 6.5), 'hose_mm': hose,
            'hose_wall_mm': cfg.get('hose_wall_mm', 1.0), 'outlet_len': cfg.get('outlet_len', 20.0),
            'up_dir': cfg.get('upstream_dir_abc', [-1, 0, 0]), 'out_dir': cfg.get('outlet_dir_abc', [1, 0, 0]),
            'coils': [{'a': XC - DCOIL * 50, 'R': R200 * 100, 'name': '200회 코일'}, {'a': XC + DCOIL * 50, 'R': R500 * 100, 'name': '500회 코일'}],
            'coil_axis_bc': [AXIS_Z, AXIS_Y]}

# ---------------- html ----------------
def html(cfg, model, sols, Y, U, tt, N, data, summ):
    best = sols[0]; p = model.unpack(best.x)
    def path_nodes(pp):
        g = model.g
        if model.nb == 1: return [[lab2abc(g.inlet), lab2abc(pp['C'][0]), lab2abc(g.outlets[0])]]
        return [[lab2abc(g.inlet), lab2abc(pp['J'])]] + [[lab2abc(pp['J']), lab2abc(pp['C'][k]), lab2abc(g.outlets[k])] for k in range(model.nb)]
    alt = []
    for s in sols[1:6]:
        if 2 * (s.cost - best.cost) < cfg.get('alt_dchi2', 50): alt.append(path_nodes(model.unpack(s.x)))
    pr = model.predict(best.x, tt)
    traces = {}
    for k in Y:
        w = data[k]['w']
        if cfg.get('drift'):
            b0a, b0b = cfg.get('baseline', [2, 12]); x = fit_one(pr[k], Y[k], U[k], w, True, tt - (b0a + b0b) / 2)
        else: x, _ = best_psi(pr[k], Y[k], U[k], w)
        e1 = np.cross(U[k], [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(U[k], e1)
        comp = lambda A: [[(round(float(v @ e), 3) if wi > 0 else None) for v, wi in zip(A, w)] for e in (U[k], e1, e2)]
        traces[k] = {'meas': comp(Y[k]), 'model': comp(x), 'noise': round(float(data[k]['noise']), 3)}
    payload = {'title': cfg.get('title', f"Run #{cfg['run']} 3D 복원"), 'box': cfg['box'],
               'sensors': cfg['sensors'], 'inlet': cfg['inlet'], 'outlets': cfg['outlets'],
               'best': path_nodes(p), 'alt': alt, 'truth': cfg.get('truth'),
               'summary': summ, 't': [round(float(x), 2) for x in tt], 'traces': traces,
               'outlet_names': cfg.get('outlet_names', [f'출구{i+1}' for i in range(len(cfg['outlets']))]),
               'geom': geom_payload(cfg, model.nb)}
    tpl = open(__file__.replace('recon3d.py', 'template.html'), encoding='utf-8').read()
    return tpl.replace('__DATA__', json.dumps(payload, ensure_ascii=False))

def main():
    if len(sys.argv) < 2:
        sys.exit('사용법: python recon3d.py 설정파일.json')
    cfg = json.load(open(sys.argv[1], encoding='utf-8'))
    # CSV 경로: 지금 폴더 기준으로 없으면 설정 파일이 있는 폴더 기준으로 찾음
    if not os.path.isabs(cfg['csv']) and not os.path.exists(cfg['csv']):
        alt = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), cfg['csv'])
        if os.path.exists(alt): cfg['csv'] = alt
    if not os.path.exists(cfg['csv']):
        sys.exit(f"[오류] CSV 파일을 찾을 수 없어요: {cfg['csv']}  (recon3d.py와 같은 폴더에 넣었는지 확인)")
    print(f"Run #{cfg['run']} 불러오는 중: {cfg['csv']}", flush=True)
    tt, data, I = load(cfg)
    print(f"  시간 {tt[0]:.1f}~{tt[-1]:.1f}초, 전류 {I:.3f} A, 잡음(μT) " + ', '.join(f'{k} {d["noise"]:.2f}' for k, d in data.items()), flush=True)
    print(f"경로 맞추는 중 (시작점 {cfg.get('starts', 10)}개, 보통 1분 안쪽)...", flush=True)
    geo = Geometry(cfg, I)
    nb = len(cfg['outlets'])
    model, sols, Y, U, f = fit(cfg, tt, data, geo, nb)
    N = len(sols[0].fun)
    z = float(np.sum(np.concatenate([(Y[k] * data[k]['w'][:, None] / data[k]['noise']).ravel() for k in Y]) ** 2) / N)
    summ = summarize(model, sols[0], N); summ['chi2_per_N_no_iron_model'] = z
    summ['noise_uT'] = {k: round(float(d['noise']), 3) for k, d in data.items()}
    summ['alternatives'] = [summarize(model, s, N) for s in sols[1:4]]
    if cfg.get('truth'):
        summ['truth_error_cm'] = path_error(model, sols[0], cfg['truth'])
        st = fit_truth(model, f, sols[0], cfg['truth'])
        if st is not None:
            summ['truth_fit'] = summarize(model, st, N)
            summ['chi2_per_N_truth_path'] = summ['truth_fit']['chi2_per_N']
    if cfg.get('thickness', True):      # 호스 안지름 추정 (실패해도 복원 결과는 그대로 저장)
        print('호스 굵기 추정 중...', flush=True)
        try:
            import thickness
            best_p = model.unpack(sols[0].x)
            summ['thickness'] = thickness.estimate(cfg, tt, data, I, model, {k: best_p[k] for k in ('J', 'C') if k in best_p})
            for pc in summ['thickness']['pieces']:
                print(f"  {pc['name']}: 앞머리 속도 {pc['U']:.2f} cm/s, 안지름 {pc.get('d_mm', '-')} mm" if pc['resolved'] else f"  {pc['name']}: 판정 불가", flush=True)
        except Exception as e:
            print('  굵기 추정 실패:', e, flush=True)
            summ['thickness'] = {'error': str(e)}
    out = cfg.get('out', f"recon_run{cfg['run']}")
    if os.path.dirname(out): os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summ, open(out + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(out + '.html', 'w', encoding='utf-8').write(html(cfg, model, sols, Y, U, tt, N, data, summ))
    print(json.dumps({k: summ[k] for k in ['chi2_per_N', 'chi2_per_N_truth_path', 'chi2_per_N_no_iron_model', 'J', 'C', 'U0', 'U', 'flow_split', 'lam', 'branch_len_cm', 'truth_error_cm'] if k in summ}, ensure_ascii=False, indent=1))
    print('saved', out + '.json', out + '.html')

if __name__ == '__main__':
    main()
