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
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ellipk, ellipe
from contextlib import contextmanager

MU0 = 4e-7 * math.pi
# ---------------- coil model (lab frame: x=a, y=c, z=b; cm) ----------------
R200 = (0.1006 + 0.1058) / 2; R500 = (0.1006 + 0.1137) / 2; DCOIL = 0.12
AXIS_Y, AXIS_Z, XC = 16.5, 4.0, 5.0

def _loop(P, zc, R):
    """반지름 R 원형 전류(1회, 1 A)의 자기장. 타원 적분 공식이라 원 둘레를 나눠 더할 때보다 훨씬 빠르다."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2] - zc
    rho2 = x * x + y * y; rho = np.sqrt(rho2); z2 = z * z
    a2 = (R + rho) ** 2 + z2; b2 = (R - rho) ** 2 + z2
    m = 4 * R * rho / a2; K = ellipk(m); E = ellipe(m)
    c = MU0 / (2 * np.pi * np.sqrt(a2))
    Bz = c * (K + (R * R - rho2 - z2) / b2 * E)
    k = c * z * (-K + (R * R + rho2 + z2) / b2 * E)
    safe = rho2 > 1e-18
    kr = np.divide(k, rho2, out=np.zeros_like(k), where=safe)
    return np.stack([kr * x, kr * y, Bz], 1)

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
def read_table(cfg):
    """Run의 열을 {열 이름: 값}으로 읽는다. 웹은 필요한 열만 숫자로 잘라 넘겨 주므로 pandas를 쓰지 않는다."""
    run = cfg['run']; tb = cfg.get('table')
    if tb:
        v = np.fromfile(tb['path'], dtype=np.float64).reshape(-1, len(tb['columns']))
        return {c: v[:, i] for i, c in enumerate(tb['columns'])}
    import pandas as pd
    df = pd.read_csv(cfg['csv'], encoding='utf-8-sig', low_memory=False, usecols=lambda c: c.endswith(f'Run #{run}'))
    if f'Time (s) Run #{run}' not in df.columns:
        cols = pd.read_csv(cfg['csv'], encoding='utf-8-sig', nrows=0).columns
        runs = sorted({int(c.split('Run #')[1]) for c in cols if 'Run #' in c})
        sys.exit(f"[오류] CSV에 Run #{run} 데이터가 없어요. 이 파일에 있는 Run 번호: {runs}")
    return {c: df[c].values.astype(float) for c in df.columns}

def bin_mean(t, v, a, dt):
    """시각이 [a, a + dt)인 값들의 평균. t는 커지는 순서."""
    i0 = np.searchsorted(t, a, 'left'); i1 = np.searchsorted(t, a + dt, 'left')
    C = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, 0)])
    with np.errstate(invalid='ignore', divide='ignore'):
        return (C[i1] - C[i0]) / (i1 - i0)[:, None]

def load(cfg):
    run = cfg['run']; ids = cfg['ids']
    tab = read_table(cfg)
    col = lambda name: tab[name][~np.isnan(tab[name])]
    t = col(f'Time (s) Run #{run}')
    I = col(f'Output Current (A) Run #{run}')
    n = min(len(t), len(I)); t = t[:n]; I = I[:n]
    b0a, b0b = cfg.get('baseline', [2, 12])
    m0 = (t > b0a) & (t < b0b); Iref = I[m0].mean()
    t0, t1 = cfg['t_window']; dt = cfg.get('bin', 0.5)
    t0 = max(t0, float(t[0])); t1 = min(t1, float(t[-1]))      # 데이터 길이를 넘으면 자동으로 줄임
    edges = np.arange(t0, t1 + 1e-9, dt); tc = (edges[:-1] + edges[1:]) / 2
    data = {}
    for k, s in ids.items():
        B = np.stack([col(f'{c} Magnetic Field Strength, {s} (G) Run #{run}')[:n] for c in 'XYZ'], 1) * 100
        b0 = B[m0].mean(0); u = b0 / np.linalg.norm(b0); g = np.linalg.norm(b0) / Iref
        dB = B - np.outer(g * (I - Iref), u) - b0
        V = bin_mean(t, dB, edges[:-1], dt)
        # 잡음: 기준 구간의 0.5초 평균값들이 (직선 추세를 뺀 뒤) 흔들리는 정도
        e0 = np.arange(b0a, b0b - dt + 1e-9, dt)
        Vb = bin_mean(t, dB, e0, dt)
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
# 상자에서 멀수록 호스 조각을 크게 묶는다: (입구나 출구에서의 거리 cm, 묶는 길이 cm), None이면 나누는 간격 그대로
BANDS_FINE = ((6, None), (20, 1.0), (1e9, 5.0))
BANDS_COARSE = ((4, None), (15, 1.5), (1e9, 5.0))
CURVE_SUB = 16; HERMITE = {}

class Geometry:
    def __init__(self, cfg, I, bands=BANDS_FINE):
        self.I = I; self.ds = cfg.get('ds', 0.25); self.curve = cfg.get('curve', True); self.bands = bands
        self.inlet = abc2lab(cfg['inlet'])
        self.outlets = [abc2lab(o) for o in cfg['outlets']]
        self.up_dir = np.array(cfg.get('upstream_dir_abc', [-1, 0, 0]), float)[[0, 2, 1]]
        self.out_dir = np.array(cfg.get('outlet_dir_abc', [1, 0, 0]), float)[[0, 2, 1]]
        self.L_up = cfg.get('upstream_len', 40.0); self.L_out = cfg.get('outlet_len', 20.0)
        self.sensors = {k: abc2lab(v) for k, v in cfg['sensors'].items()}
        self.S = np.array(list(self.sensors.values())); self.nc = 3 * len(self.S); self.pieces = {}
        # fixed parts (precomputed): upstream hose and outside outlet hoses
        s = np.arange(-self.L_up, 0, self.ds) + self.ds / 2
        self.up_pts = self.inlet[None] + np.outer(-s, self.up_dir / np.linalg.norm(self.up_dir))
        self.up_G, self.up_s = self.lump(self.unit_field(self.up_pts), s, -s)
        self.out_pts = []; self.out_G = []
        for o in self.outlets:
            so = np.arange(0, self.L_out, self.ds) + self.ds / 2
            p = o[None] + np.outer(so, self.out_dir / np.linalg.norm(self.out_dir))
            G, sl = self.lump(self.unit_field(p), so, so)
            self.out_pts.append((p, sl)); self.out_G.append(G)

    def lump(self, G, s, dist):
        """상자에서 먼 호스는 신호가 작고 천천히 변하므로 조각을 묶어 계산량을 줄인다."""
        order = np.argsort(dist); d = dist[order]; groups = []; lo = -1.0
        for up, width in self.bands:
            m = order[(d > lo) & (d <= up)]; lo = up
            n = max(1, int(round((width or self.ds) / self.ds)))
            groups += [m[i:i + n] for i in range(0, len(m), n)]
        return np.array([G[g].sum(0) for g in groups]), np.array([s[g].mean() for g in groups])

    def unit_field(self, P):
        """호스 조각(Np개)이 센서에 만드는 신호 계수. 행은 조각, 열은 센서별 x, y, z."""
        B0 = B0lab(P, self.I)
        r = self.S[:, None] - P[None]; r2 = np.sum(r * r, 2)
        G = (self.ds / (4 * np.pi * r2 * np.sqrt(r2)))[:, :, None] * (3 * (np.sum(B0[None] * r, 2) / r2)[:, :, None] * r - B0[None])
        return G.transpose(1, 0, 2).reshape(len(P), self.nc)

    def split(self, out):
        return {k: out[:, 3 * i:3 * i + 3] for i, k in enumerate(self.sensors)}

    def piece(self, nodes):
        """호스 한 토막의 (계수, 위치, 길이). 끝점이 그대로면 다시 계산하지 않는다."""
        key = (self.curve, b''.join(n.tobytes() for n in nodes))
        v = self.pieces.get(key)
        if v is None:
            pts, s, L, dl = self.polyline(nodes)
            # 점 수는 정수라서 호스 길이를 실제 간격으로 재야 경로가 조금 바뀔 때 신호가 튀지 않는다
            v = (self.unit_field(pts) * (dl / self.ds)[:, None] if len(pts) else None, s, L)
            if len(self.pieces) >= 64: self.pieces.pop(next(iter(self.pieces)))
            self.pieces[key] = v
        return v

    def curve_pts(self, nodes, sub=CURVE_SUB):
        """점들을 지나는 매끈한 곡선(3차 에르미트). 상자 구멍에서는 호스가 벽에 수직으로 드나든다."""
        nodes = [np.asarray(n, float) for n in nodes]
        if not self.curve: return np.array(nodes)
        nodes = [n for i, n in enumerate(nodes) if i == 0 or np.linalg.norm(n - nodes[i - 1]) > 1e-6]
        if len(nodes) < 2: return np.array(nodes)
        unit = lambda v: v / max(np.linalg.norm(v), 1e-9)
        tang = [unit(nodes[min(i + 1, len(nodes) - 1)] - nodes[max(i - 1, 0)]) for i in range(len(nodes))]
        if self.curve != 'free':
            if np.array_equal(nodes[0], self.inlet): tang[0] = -unit(self.up_dir)
            if any(np.array_equal(nodes[-1], o) for o in self.outlets): tang[-1] = unit(self.out_dir)
        h = HERMITE.get(sub)
        if h is None:
            u = (np.arange(sub) / sub)[:, None]
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h = HERMITE[sub] = (h00, u ** 3 - 2 * u ** 2 + u, 1 - h00, u ** 3 - u ** 2)
        out = []
        for a, b, ta, tb in zip(nodes[:-1], nodes[1:], tang[:-1], tang[1:]):
            L = np.linalg.norm(b - a)
            out.append(h[0] * a + h[1] * (L * ta) + h[2] * b + h[3] * (L * tb))
        out.append(nodes[-1][None])
        return np.concatenate(out)

    def polyline(self, nodes):
        """경로를 일정한 간격으로 나눈 점, 호스를 따라 잰 위치, 전체 길이, 점마다의 호스 길이"""
        Q = self.curve_pts(nodes)
        if len(Q) < 2: return np.zeros((0, 3)), np.zeros(0), 0.0, np.zeros(0)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; L = cum[-1]
        if L < 1e-6: return np.zeros((0, 3)), np.zeros(0), 0.0, np.zeros(0)
        if self.curve:
            n = max(1, int(round(L / self.ds))); s = (np.arange(n) + 0.5) * L / n
            return np.stack([np.interp(s, cum, Q[:, k]) for k in range(3)], 1), s, L, np.full(n, L / n)
        ss = []; dl = []
        for a, b in zip(cum[:-1], cum[1:]):
            if b - a < 1e-6: continue
            n = max(1, int(round((b - a) / self.ds)))
            ss.append(a + (np.arange(n) + 0.5) * (b - a) / n); dl.append(np.full(n, (b - a) / n))
        s = np.concatenate(ss)
        return np.stack([np.interp(s, cum, Q[:, k]) for k in range(3)], 1), s, L, np.concatenate(dl)

# ---------------- model ----------------
EMPTY = (None, np.zeros(0), 0.0)

def front_sum(out, tt, tau, G, tarr, lam):
    """앞머리가 시각 tarr에 지나가는 호스 조각들의 신호를 더한다. lam은 수 하나 또는 조각별 배열."""
    if G is None: return
    out += (0.5 * lam * (1 + np.tanh((tt[:, None] - tarr[None]) / (2 * tau)))) @ G

class Model:
    """nb = number of branches (= number of outlets). nb=1 : inlet -> C -> outlet.
    nb>=2 : inlet -> J (junction) -> C_k -> outlet_k."""
    def __init__(self, geo, nb, bounds_abc, phys=None):
        # phys = {'d_stem': 3, 'd_br': [3, 5], 'split': [0.75, 0.25] or None}
        #  -> 같은 농도라고 보고 lambda_k = lambda0 x (d_k/d_stem)^2,
        #     유량 보존: U_k = U0 x (A_stem/A_k) x f_k  (f_k = 갈래 k로 가는 유량 비율)
        self.g = geo; self.nb = nb; self.phys = phys; self.kind = 'y' if nb >= 2 else 'single'
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

    def nodes(self, p):
        g = self.g; ch = lambda n: [lab2abc(x) for x in g.curve_pts(n)]
        if self.nb == 1: return [ch([g.inlet, p['C'][0], g.outlets[0]])]
        return [ch([g.inlet, p['J']])] + [ch([p['J'], p['C'][k], g.outlets[k]]) for k in range(self.nb)]

    def init_geom(self, rng, i):
        g = self.g; span = self.phi - self.plo; a_only = np.array([1.0, 0, 0]); not_a = np.array([0, 1.0, 1.0])
        q = []; J = g.inlet
        if self.nb >= 2:
            if i % 2 == 0: J = g.inlet + rng.uniform(0, 0.4) * span * a_only + rng.normal(0, 1.0, 3) * not_a
            else: J = self.plo + rng.uniform(0, 1, 3) * span * np.array([0.6, 1, 1])
            J = np.clip(J, self.plo, self.phi); q += list(J)
        for k in range(self.nb):
            if i % 3 == 2: C = self.plo + rng.uniform(0, 1, 3) * span
            else: C = (J + g.outlets[k]) / 2 + rng.normal(0, 1.5, 3)
            q += list(np.clip(C, self.plo, self.phi))
        return q

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
        if self.nb == 1:
            return {'stem': EMPTY, 'br': [g.piece([g.inlet, p['C'][0], g.outlets[0]])]}
        J = p['J']
        return {'stem': g.piece([g.inlet, J]), 'br': [g.piece([J, p['C'][k], g.outlets[k]]) for k in range(self.nb)]}

    def predict(self, q, tt):
        p = self.unpack(q); g = self.g; P = self.paths(p); tau = p['tau']; t0 = p['t0']; U0 = p['U0']
        out = np.zeros((len(tt), g.nc))
        front_sum(out, tt, tau, g.up_G, t0 + g.up_s / U0, p['lam0'])
        G, s, LJ = P['stem']
        front_sum(out, tt, tau, G, t0 + s / U0, p['lam0'])
        tJ = t0 + LJ / U0
        for k, (G, s, Lb) in enumerate(P['br']):
            Uk = p['U'][k]; lk = p['lam'][k]
            front_sum(out, tt, tau, G, tJ + s / Uk, lk)
            front_sum(out, tt, tau, g.out_G[k], tJ + (Lb + g.out_pts[k][1]) / Uk, lk)
        return g.split(out)

class ModelLoop:
    """갈라졌다가 다시 모이는 관: 입구 -> J1(분기) -> 갈래 k (꺾이는 점 C_k) -> J2(합류) -> 출구 1개.
    합류 뒤 관은 입구 관과 같은 굵기·속도·철 양이라고 봄. 합류 뒤 앞머리는 먼저 도착한 갈래에서 시작."""
    def __init__(self, geo, bounds_abc, out_index=0, tie=False):
        self.g = geo; self.nb = 2; self.kind = 'loop'; self.oi = out_index; self.phys = None; self.tie = tie
        lo = abc2lab([bounds_abc['a'][0], bounds_abc['b'][0], bounds_abc['c'][0]])
        hi = abc2lab([bounds_abc['a'][1], bounds_abc['b'][1], bounds_abc['c'][1]])
        self.plo = np.minimum(lo, hi); self.phi = np.maximum(lo, hi)
    def unpack(self, q):
        if self.tie:   # 두 갈래 굵기·농도가 같음: 철 양 하나
            return {'J': q[0:3], 'J2': q[3:6], 'C': [q[6:9], q[9:12]], 't0': q[12], 'U0': q[13], 'U': [q[14], q[15]],
                    'lam0': q[16], 'lam': [q[16], q[16]], 'tau': q[17]}
        return {'J': q[0:3], 'J2': q[3:6], 'C': [q[6:9], q[9:12]], 't0': q[12], 'U0': q[13], 'U': [q[14], q[15]],
                'lam0': q[16], 'lam': [q[17], q[18]], 'tau': q[19]}
    def bounds(self):
        if self.tie:
            lo = list(self.plo) * 4 + [-60, 0.3, 0.1, 0.1, 0.0, 0.1]; hi = list(self.phi) * 4 + [200, 30, 30, 30, 3.0, 20]
        else:
            lo = list(self.plo) * 4 + [-60, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1]
            hi = list(self.phi) * 4 + [200, 30, 30, 30, 3.0, 3.0, 3.0, 20]
        return np.array(lo, float), np.array(hi, float)
    def paths(self, p):
        g = self.g; O = g.outlets[self.oi]
        return {'stem': g.piece([g.inlet, p['J']]), 'br': [g.piece([p['J'], p['C'][k], p['J2']]) for k in range(2)],
                'merge': g.piece([p['J2'], O])}
    def nodes(self, p):
        g = self.g; O = g.outlets[self.oi]; ch = lambda n: [lab2abc(x) for x in g.curve_pts(n)]
        return [ch([g.inlet, p['J']]), ch([p['J'], p['C'][0], p['J2']]), ch([p['J'], p['C'][1], p['J2']]), ch([p['J2'], O])]
    def init_geom(self, rng, i):
        g = self.g; span = self.phi - self.plo; O = g.outlets[self.oi]
        a_only = np.array([1.0, 0, 0]); not_a = np.array([0, 1.0, 1.0])
        if i % 2 == 0: J1 = g.inlet + rng.uniform(0, 0.35) * span * a_only + rng.normal(0, 1.0, 3) * not_a
        else: J1 = self.plo + rng.uniform(0, 1, 3) * span * np.array([0.5, 1, 1])
        if i % 2 == 0: J2 = O - rng.uniform(0, 0.35) * span * a_only + rng.normal(0, 1.0, 3) * not_a
        else: J2 = self.plo + span * (np.array([0.5, 0, 0]) + rng.uniform(0, 1, 3) * np.array([0.5, 1, 1]))
        J1 = np.clip(J1, self.plo, self.phi); J2 = np.clip(J2, self.plo, self.phi)
        mid = (J1 + J2) / 2; q = list(J1) + list(J2)
        for k, sgn in enumerate([-1, 1]):
            if i % 3 == 2: C = self.plo + rng.uniform(0, 1, 3) * span
            else: C = mid + np.array([0, 0, sgn * rng.uniform(1.0, 3.5)]) + rng.normal(0, 1.0, 3)
            q += list(np.clip(C, self.plo, self.phi))
        return q
    def tail_guess(self, cfg, tt, rng, tg=None):
        if tg is None: tg = cfg.get('t_front_guess', float(tt[0]) + 5)
        tg = tg + rng.uniform(-3, 4)
        Ug = float(np.exp(rng.uniform(np.log(0.7), np.log(7.0)))) if cfg.get('U_guess') is None else cfg['U_guess'] * rng.uniform(0.5, 2.0)
        return [tg, Ug, Ug * 0.6, Ug * 0.6, 0.1, 2.0] if self.tie else [tg, Ug, Ug * 0.6, Ug * 0.6, 0.1, 0.1, 0.1, 2.0]
    def predict(self, q, tt):
        p = self.unpack(q); g = self.g; P = self.paths(p); tau = p['tau']; t0 = p['t0']; U0 = p['U0']; l0 = p['lam0']
        out = np.zeros((len(tt), g.nc))
        front_sum(out, tt, tau, g.up_G, t0 + g.up_s / U0, l0)
        G, s, LJ = P['stem']
        front_sum(out, tt, tau, G, t0 + s / U0, l0)
        tJ = t0 + LJ / U0; arr = []
        for k, (G, s, Lb) in enumerate(P['br']):
            front_sum(out, tt, tau, G, tJ + s / p['U'][k], p['lam'][k])
            arr.append(tJ + Lb / p['U'][k])
        tJ2 = min(arr)
        G, s, LM = P['merge']
        front_sum(out, tt, tau, G, tJ2 + s / U0, l0)
        front_sum(out, tt, tau, g.out_G[self.oi], tJ2 + (LM + g.out_pts[self.oi][1]) / U0, l0)
        return g.split(out)

class ModelStep:
    """중간에 굵어지는(또는 가늘어지는) 한 줄 관: 입구 -> (꺾이는 점 C) -> 출구, 관을 따라 sf 비율 지점 P에서 굵기가 바뀜.
    같은 현탁액이므로 철 양은 단면적에 비례(lam0 -> lam1), 속도는 단면적에 반비례(U1 = U0 * lam0 / lam1)."""
    def __init__(self, geo, bounds_abc, out_index=0, tie_speed=True, outside_after=True):
        self.g = geo; self.nb = 1; self.kind = 'step'; self.oi = out_index; self.phys = None
        self.tie = tie_speed; self.oa = outside_after
        lo = abc2lab([bounds_abc['a'][0], bounds_abc['b'][0], bounds_abc['c'][0]])
        hi = abc2lab([bounds_abc['a'][1], bounds_abc['b'][1], bounds_abc['c'][1]])
        self.plo = np.minimum(lo, hi); self.phi = np.maximum(lo, hi)
    def unpack(self, q):
        p = {'C': [q[0:3]], 'sf': q[3], 't0': q[4], 'U0': q[5], 'lam0': q[6], 'lam': [q[7]], 'tau': q[8]}
        p['U'] = [p['U0'] * p['lam0'] / max(p['lam'][0], 1e-6)] if self.tie else [q[9]]
        if self.tie: p['U'][0] = min(p['U'][0], 60.0)
        return p
    def bounds(self):
        lo = list(self.plo) + [0.05, -60, 0.3, 0.0, 0.0, 0.1]; hi = list(self.phi) + [0.95, 200, 30, 3.0, 6.0, 20]
        if not self.tie: lo += [0.1]; hi += [30]
        return np.array(lo, float), np.array(hi, float)
    def _line(self, p):
        g = self.g; return [g.inlet, p['C'][0], g.outlets[self.oi]]
    def point_at(self, p):
        Q = self.g.curve_pts(self._line(p))
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; L = cum[-1]; target = p['sf'] * L
        return np.array([np.interp(target, cum, Q[:, k]) for k in range(3)]), L, target
    def paths(self, p):
        return {'stem': EMPTY, 'br': [self.g.piece(self._line(p))]}
    def nodes(self, p):
        Q = self.g.curve_pts(self._line(p)); P, L, sP = self.point_at(p)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        first = [lab2abc(n) for n, sv in zip(Q, cum) if sv < sP - 1e-9] + [lab2abc(P)]
        second = [lab2abc(P)] + [lab2abc(n) for n, sv in zip(Q, cum) if sv > sP + 1e-9]
        return [first, second]
    def init_geom(self, rng, i):
        g = self.g; span = self.phi - self.plo; O = g.outlets[self.oi]
        C = (g.inlet + O) / 2 + rng.normal(0, 1.5, 3) if i % 3 != 2 else self.plo + rng.uniform(0, 1, 3) * span
        return list(np.clip(C, self.plo, self.phi)) + [rng.uniform(0.2, 0.8)]
    def tail_guess(self, cfg, tt, rng, tg=None):
        if tg is None: tg = cfg.get('t_front_guess', float(tt[0]) + 5)
        tg = tg + rng.uniform(-3, 4)
        Ug = float(np.exp(rng.uniform(np.log(0.7), np.log(7.0)))) if cfg.get('U_guess') is None else cfg['U_guess'] * rng.uniform(0.5, 2.0)
        r = rng.uniform(0.5, 3.5)
        q = [tg, Ug, 0.05, 0.05 * r, 2.0]
        return q if self.tie else q + [Ug / r]
    def predict(self, q, tt):
        p = self.unpack(q); g = self.g; tau = p['tau']; t0 = p['t0']
        U0, U1, l0, l1 = p['U0'], p['U'][0], p['lam0'], p['lam'][0]
        out = np.zeros((len(tt), g.nc))
        front_sum(out, tt, tau, g.up_G, t0 + g.up_s / U0, l0)
        G, s, L = g.piece(self._line(p)); sP = p['sf'] * L; tP = t0 + sP / U0
        if G is not None:
            # 굵기 바뀌는 점이 걸친 조각은 앞뒤 값을 나눠 가져서 P가 조금 움직일 때 신호가 튀지 않는다
            mix = np.clip((s - sP) * len(s) / L + 0.5, 0, 1)
            front_sum(out, tt, tau, G, np.where(s < sP, t0 + s / U0, tP + (s - sP) / U1), l0 + (l1 - l0) * mix)
        front_sum(out, tt, tau, g.out_G[self.oi], tP + (L - sP + g.out_pts[self.oi][1]) / U1, l1 if self.oa else l0)
        return g.split(out)

def node_chains(model, p):
    """흐르는 순서대로 이은 점들 (lab 좌표). 사전 조건(곧게, 내리막)을 걸 때 씀."""
    g = model.g
    if model.kind == 'loop':
        O = g.outlets[model.oi]
        return [[g.inlet, p['J']], [p['J'], p['C'][0], p['J2']], [p['J'], p['C'][1], p['J2']], [p['J2'], O]]
    if model.kind == 'step':
        return [[g.inlet, p['C'][0], g.outlets[model.oi]]]
    if model.nb == 1: return [[g.inlet, p['C'][0], g.outlets[0]]]
    return [[g.inlet, p['J']]] + [[p['J'], p['C'][k], g.outlets[k]] for k in range(model.nb)]

def prior_res(model, q, prior):
    """사전 조건: (1) 꺾이는 점이 양 끝을 이은 직선에서 벗어나는 거리 / sigma  (2) 오르막이면 벌점"""
    p = model.unpack(q); r = []
    sig = prior.get('straight_sigma_cm'); dh = prior.get('downhill', False)
    for ch in node_chains(model, p):
        ch = [np.asarray(x, float) for x in ch]
        if sig and len(ch) == 3:
            A, C, B = ch; d = B - A; L = np.linalg.norm(d)
            if L > 1e-6:
                v = C - A; dn = d / L; perp = v - (v @ dn) * dn; r += list(perp / sig)
        if dh:
            for a, b in zip(ch[:-1], ch[1:]):
                r.append(max(0.0, b[1] - a[1] - prior.get('flat_tol_cm', 0.3)) / prior.get('downhill_sigma_cm', 0.2))
    return np.array(r)

def make_model(kind, geo, cfg):
    if kind == 'step':
        m = ModelStep(geo, cfg['box'], cfg.get('loop_outlet', 0), cfg.get('step_tie_speed', False), cfg.get('step_outside_after', True))
    elif kind == 'loop':
        hm = cfg.get('hose_mm') or {}
        tie = cfg.get('tie_lambda', bool(hm) and all(abs(d - hm.get('stem', 3)) < 1e-6 for d in hm.get('branches', [3, 3])))
        m = ModelLoop(geo, cfg['box'], cfg.get('loop_outlet', 0), tie)
    else:
        m = Model(geo, 1 if kind == 'single' else len(cfg['outlets']), cfg['box'], cfg.get('phys'))
    pr = cfg.get('prior') or {}
    if pr.get('downhill'):      # 호스는 내리막만 있다: 중간 점이 입구보다 높거나 출구보다 낮을 수 없다
        tol = pr.get('flat_tol_cm', 0.3); ends = [o[1] for o in geo.outlets]
        m.plo[1] = max(m.plo[1], min(ends) - tol); m.phi[1] = min(m.phi[1], geo.inlet[1] + tol)
    return m

def aligner(Y, U, W, noise, dt, drift):
    """best_psi, fit_one과 같은 식을 센서 전체에 한 번에 계산한다. Y는 (센서, 시간, 3), U는 (센서, 3)."""
    n = len(U)
    Kt = np.array([[[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]] for u in U]).transpose(0, 2, 1)
    Pu = U[:, :, None] * U[:, None, :]
    w2 = W * W; wy = w2[:, :, None] * Y; wd = (W * dt)[:, None]; wd2 = (w2 * dt)[:, None]
    den = np.maximum(np.sum(W * dt * dt, 1), 1e-9)[:, None]; yv = (wd @ Y)[:, 0]; sc = W[:, :, None] / noise[:, None, None]
    rowt = dt[None, :, None]
    def run(x):
        par = x @ Pu; per = x - par; cro = x @ Kt
        A = np.sum(per * wy, (1, 2)); B = np.sum(cro * wy, (1, 2))
        if not drift:
            psi = np.arctan2(B, A)
            return ((par + np.cos(psi)[:, None, None] * per + np.sin(psi)[:, None, None] * cro - Y) * sc).ravel()
        vp = (wd2 @ per)[:, 0]; vc = (wd2 @ cro)[:, 0]; ap = (wd @ par)[:, 0]; bp = (wd @ per)[:, 0]; cp = (wd @ cro)[:, 0]
        d = np.zeros((n, 3))
        for _ in range(2):
            psi = np.arctan2(B - np.sum(vc * d, 1), A - np.sum(vp * d, 1)); c = np.cos(psi)[:, None]; s = np.sin(psi)[:, None]
            d = (yv - ap - c * bp - s * cp) / den
        return ((par + c[:, :, None] * per + s[:, :, None] * cro + rowt * d[:, None] - Y) * sc).ravel()
    return run

def residuals(model, data, tt, drift=False, tb=0.0, prior=None):
    U = {k: B0lab(model.g.sensors[k], model.g.I)[0] for k in data}
    U = {k: v / np.linalg.norm(v) for k, v in U.items()}
    RA = {k: rot_align(data[k]['b0'], U[k] * np.linalg.norm(data[k]['b0'])) for k in data}
    Y = {k: data[k]['dB'] @ RA[k].T for k in data}
    keys = list(data)
    align = aligner(np.stack([Y[k] for k in keys]), np.stack([U[k] for k in keys]), np.stack([data[k]['w'] for k in keys]),
                    np.array([data[k]['noise'] for k in keys]), tt - tb, drift)
    def f(q):
        pr = model.predict(q, tt)
        r = align(np.stack([pr[k] for k in keys]))
        return np.concatenate([r, prior_res(model, q, prior)]) if prior else r
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

def rebin(tt, data, k):
    """이웃한 시간 구간 k개를 하나로 묶는다. 잡음은 sqrt(k)만큼 줄어든다."""
    n = len(tt) // k; out = {}
    for key, d in data.items():
        w = d['w'][:n * k].reshape(n, k); V = d['dB'][:n * k].reshape(n, k, 3); sw = w.sum(1)
        out[key] = {'b0': d['b0'], 'dB': (V * w[:, :, None]).sum(1) / np.maximum(sw, 1)[:, None],
                    'noise': d['noise'] / math.sqrt(k), 'w': (sw > 0).astype(float)}
    return tt[:n * k].reshape(n, k).mean(1), out

def fit_range(cfg, tt, data, I):
    """맞출 시간 구간. 앞머리가 지나간 뒤에는 뒤따르는 철과 호스에 남은 철이 섞여 앞머리 모델로 설명되지 않는다."""
    fw = cfg.get('fit_window', 'auto')
    first, last = float(tt[0]), float(tt[-1])
    if fw is False: return first, last
    if fw != 'auto': return max(first, float(fw[0])), min(last, float(fw[1]))
    import thickness as T
    t_on = T.onset(tt, data)
    guess = cfg.get('t_front_guess')       # 신호가 약하면 앞머리를 늦게 잡으므로, 예상보다 한참 늦으면 예상 시각을 믿는다
    if guess is not None and t_on > guess + 8: t_on = float(guess)
    a = max(first, t_on - 12)
    tw, dw = rebin(*T.window_data(tt, data, a, min(t_on + 12, last)), 2)
    span = 8.0
    for o in cfg['outlets']:
        c = dict(cfg); c['outlets'] = [o]; c['curve'] = True      # 구간은 경로 모양 설정과 상관없이 같게 정한다
        g = Geometry(c, I, BANDS_COARSE)
        front = T.FrontModel(Model(g, 1, cfg['box']), {'C': [(g.inlet + g.outlets[0]) / 2]})
        ft = T.Fitter(front, c, tw, dw, t_on)
        cost = [ft.fit(ft.start(u)[1], {1: u}, nfev=8)[1] for u in T.SPEED_GRID]
        span = max(span, 1.5 * front.reach[0] / float(T.SPEED_GRID[int(np.argmin(cost))]) + 6)
    return a, min(t_on + min(span, 40.0), last)

def cut(tt, data, a, b):
    m = (tt >= a) & (tt <= b)
    return tt[m], {k: {**d, 'dB': d['dB'][m], 'w': d['w'][m]} for k, d in data.items()}

class Sol:
    def __init__(self, x, cost, fun): self.x = x; self.cost = cost; self.fun = fun

class Search:
    """구조 후보 하나를 맞춘다. 성긴 모델(coarse)로 답이 있는 곳을 찾고 원래 해상도(fine)로 마무리한다."""
    def __init__(self, cfg, tt, data, I, kind):
        self.cfg = cfg; self.tt = tt; self.data = data; self.I = I; self.kind = kind; self.lv = {}; self.q0 = None
    def level(self, name):
        if name not in self.lv:
            c = self.cfg; b0a, b0b = c.get('baseline', [2, 12])
            if name == 'fine': cc, tt, data, bands = c, self.tt, self.data, BANDS_FINE
            else:
                cc = dict(c); cc['ds'] = 2 * c.get('ds', 0.25); bands = BANDS_COARSE
                tt, data = rebin(self.tt, self.data, c.get('coarse_bin', 2))
            model = make_model(self.kind, Geometry(cc, self.I, bands), cc)
            f, Y, U = residuals(model, data, tt, drift=c.get('drift', False), tb=(b0a + b0b) / 2, prior=c.get('prior'))
            self.lv[name] = (model, f, Y, U)
        return self.lv[name]
    def starts(self):
        if self.q0 is None:
            cfg = self.cfg; model = self.level('fine')[0]; lo, hi = model.bounds()
            rng = np.random.default_rng(cfg.get('seed', 0))
            t_on = auto_t_on(self.tt, self.data)
            tg = [t_on] + ([cfg['t_front_guess']] if cfg.get('t_front_guess') is not None else [])
            self.q0 = []
            for i in range(cfg.get('starts', 6) * (2 if model.nb >= 2 else 1)):
                q = model.init_geom(rng, i) + model.tail_guess(cfg, self.tt, rng, tg[i % len(tg)])
                self.q0.append(np.clip(np.array(q, float), lo + 1e-6, hi - 1e-6))
        return self.q0
    def solve(self, level, x, nfev, ftol, xtol=1e-3):
        model, f = self.level(level)[:2]; lo, hi = model.bounds()
        return least_squares(f, np.clip(x, lo, hi), bounds=(lo, hi), max_nfev=nfev, ftol=ftol, xtol=xtol)

class Context:
    """측정값과 후보 목록. 웹에서는 계산을 나눠 맡는 워커마다 하나씩 만든다."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.tt, self.data, self.I = load(cfg)
        self.win = fit_range(cfg, self.tt, self.data, self.I)
        self.tw, self.dw = cut(self.tt, self.data, *self.win)
        nb = len(cfg['outlets'])
        names = cfg.get('outlet_names', [f'출구{i+1}' for i in range(nb)])
        topo = cfg.get('topology', 'y' if nb >= 2 else 'auto')
        # 출구 고르기: "exit": "auto"(출구마다 다 맞춰 보고 고름) 또는 출구 번호(0부터)
        exit_sel = cfg.get('exit', 'auto' if (nb >= 2 and topo != 'y') else 0)
        one = cfg.get('candidates', ['single', 'step', 'loop']) if topo == 'auto' else ([topo] if topo != 'y' else [])
        hm = cfg.get('hose_mm') or {}
        if topo == 'auto' and hm.get('branches') and all(abs(d - hm.get('stem', 3)) < 1e-6 for d in hm['branches']):
            one = [k for k in one if k != 'step']      # 호스 굵기가 모두 같으면 굵기 변화 관은 후보가 아니다
        exits = list(range(nb)) if exit_sel == 'auto' else [int(exit_sel)]
        cands = [(k, [i]) for i in exits for k in one]
        if topo == 'y' or (topo == 'auto' and nb >= 2 and cfg.get('try_y', True)): cands.append(('y', list(range(nb))))
        self.all_names = names; self.cands = []
        for kind, idx in cands:
            c = dict(cfg); c['outlets'] = [cfg['outlets'][i] for i in idx]; c['outlet_names'] = [names[i] for i in idx]
            label = KIND_NAME[kind] + (f' · {names[idx[0]]}' if (nb >= 2 and kind != 'y') else '')
            self.cands.append({'kind': kind, 'idx': idx, 'cfg': c, 'label': label})
        self.found = {}
    def search(self, c):
        if c not in self.found:
            d = self.cands[c]; self.found[c] = Search(d['cfg'], self.tw, self.dw, self.I, d['kind'])
        return self.found[c]

def correlation(r, ns, nt):
    """잔차의 이웃한 시간끼리 닮은 정도 rho. 모델이 설명 못 하는 신호가 남으면 커진다. (1 + rho) / (1 - rho)를 돌려준다."""
    T = r[:3 * ns * nt].reshape(ns, nt, 3); a = T[:, :-1].ravel(); b = T[:, 1:].ravel()
    d = float(a @ a); rho = min(max(float(a @ b) / d if d > 0 else 0.0, 0.0), 0.8)
    return (1 + rho) / (1 - rho)

def do_job(ctx, job):
    """계산 한 조각. 입력과 출력이 숫자뿐이라 다른 워커에 맡길 수 있고, 맡긴 순서와 상관없이 답이 같다."""
    s = ctx.search(job['c']); cfg = s.cfg; st = job['stage']
    try:
        if st == 'score':
            r = s.level('fine')[1](np.array(job['x'], float))
            return {'x': job['x'], 'cost': 0.5 * float(r @ r)}
        if st == 'scout': r = s.solve('coarse', s.starts()[job['i']], cfg.get('nfev1', 8), 1e-2)
        elif st == 'settle': r = s.solve('coarse', np.array(job['x'], float), cfg.get('nfev2', 25), 1e-3)
        else: r = s.solve('fine', np.array(job['x'], float), cfg.get('nfev3', 60), cfg.get('ftol3', 1e-6), 1e-5)
        return {'x': [float(v) for v in r.x], 'cost': float(r.cost)}
    except Exception as e:
        print(f'  계산 한 조각 실패: {e}', flush=True)
        return None

def climb(ctx, group, part, parts):
    """후보 묶음을 세 단계로 맞춘다: 성긴 모델로 찾기, 잘 맞은 답 다듬기, 앞선 답을 원래 해상도로 마무리."""
    cfg = ctx.cfg; keep = cfg.get('keep', 3); top = cfg.get('polish', 2)
    def rank(jobs, got):
        out = {c: [] for c in group}
        for j, r in zip(jobs, got):
            if r: out[j['c']].append(r)
        for c in out: out[c].sort(key=lambda r: r['cost'])
        return out
    jobs = [{'stage': 'scout', 'c': c, 'i': i} for c in group for i in range(len(ctx.search(c).starts()))]
    print(f'1단계 ({part}/{parts}): 시작점 {len(jobs)}개', flush=True)
    got = yield jobs
    best = rank(jobs, got)
    jobs = [{'stage': 'settle', 'c': c, 'x': r['x']} for c in best for r in best[c][:keep]]
    print(f'2단계 ({part}/{parts}): 답 {len(jobs)}개', flush=True)
    got = yield jobs
    best = rank(jobs, got)
    # 2단계에서 크게 뒤처진 후보는 마무리하지 않고 원래 해상도로 점수만 매긴다.
    # 매개변수가 더 적은 후보는 남긴다: 실측에서는 복잡한 구조가 잔차를 흡수해 낮은 chi2를 내기 쉽다
    live = [c for c in best if best[c]]
    if not live: return best, {c: [] for c in group}, set(group)
    lead = min(live, key=lambda c: best[c][0]['cost']); low = best[lead][0]['cost']
    size = lambda c: len(best[c][0]['x'])
    near = [c for c in live if best[c][0]['cost'] <= cfg.get('near', 1.5) * low or size(c) < size(lead)]
    jobs = [{'stage': 'polish', 'c': c, 'x': r['x']} for c in near for r in best[c][:top]]
    jobs += [{'stage': 'score', 'c': c, 'x': best[c][0]['x']} for c in live if c not in near]
    print(f'3단계 ({part}/{parts}): 답 {len(jobs)}개', flush=True)
    got = yield jobs
    return best, rank(jobs, got), set(group) - set(near)

def worth_trying(ctx, rich, done):
    """단순한 구조의 결과로 복잡한 구조를 맞출 필요가 있는지 정한다."""
    fits = {c: r[0]['cost'] * 2 for c, r in done.items() if r}
    if not fits: return rich
    N = 3 * len(ctx.dw) * len(ctx.tw); low = min(fits.values())
    low_k = len(ctx.search(min(fits, key=fits.get)).starts()[0])
    keep = []
    # 복잡한 구조가 잡음까지 완벽하게 맞춰도 BIC가 더 나빠지면 맞춰 볼 필요가 없다
    if low - N < min(len(ctx.search(c).starts()[0]) - low_k for c in rich) * math.log(N): return []
    # 합류 관은 한 줄 관이 잘 맞은 출구만 시험한다
    by_exit = {}
    for c, v in fits.items():
        i = ctx.cands[c]['idx'][0]; by_exit[i] = min(by_exit.get(i, v), v)
    for c in rich:
        d = ctx.cands[c]
        if d['kind'] != 'loop' or by_exit.get(d['idx'][0], low) <= ctx.cfg.get('exit_near', 1.3) * low: keep.append(c)
    return keep

def plan(ctx):
    """맞춤의 순서. 할 일 목록을 내보내고 결과를 받아 다음으로 넘어간다. 단순한 구조부터 맞춘다."""
    cfg = ctx.cfg; cands = ctx.cands; top = cfg.get('polish', 2)
    print(f'후보 {len(cands)}개', flush=True)
    simple = [c for c, d in enumerate(cands) if d['kind'] in ('single', 'step')]
    rich = [c for c in range(len(cands)) if c not in simple]
    parts = 2 if simple and rich else 1
    best = {}; done = {}; rough = set()
    if simple:
        b, d, r = yield from climb(ctx, simple, 1, parts)
        best.update(b); done.update(d); rough |= r
    if simple and rich:
        rich = worth_trying(ctx, rich, done)
        if not rich: print('단순한 구조가 잡음 수준까지 맞아 복잡한 구조는 맞추지 않음', flush=True)
    if rich:
        b, d, r = yield from climb(ctx, rich, parts, parts)
        best.update(b); done.update(d); rough |= r
    results = []
    for c, d in enumerate(cands):
        if not done.get(c): continue
        s = ctx.search(c); model, f, Y, U = s.level('fine')
        x = np.array(done[c][0]['x']); r0 = f(x); Nk = len(r0); chi2 = float(r0 @ r0); npar = len(x)
        results.append({**d, 'c': c, 'search': s, 'model': model, 'Y': Y, 'U': U, 'f': f, 'best': Sol(x, 0.5 * chi2, r0), 'rough': c in rough,
                        'chi2_per_N': chi2 / Nk, 'chi2': chi2, 'N': Nk, 'npar': npar})
    if not results: sys.exit('[오류] 어떤 시작점에서도 맞춤이 끝나지 않았어요.')
    # BIC는 잔차가 서로 독립인 잡음이라고 본다. 실측처럼 그렇지 않으면 chi2 차이를 그만큼 줄이고 독립인 자료 수를 쓴다
    flex = min(results, key=lambda r: r['chi2'])
    corr = correlation(flex['best'].fun, len(ctx.dw), len(ctx.tw))
    scale = corr * max(1.0, flex['chi2'] / flex['N'])
    for r in results:
        r['bic'] = r['chi2'] / scale + r['npar'] * np.log(r['N'] / corr); r['scale'] = scale
        print(f"   {r['label']}: chi2/N = {r['chi2_per_N']:.3f}, BIC = {r['bic']:.0f}", flush=True)
    # 호스는 3 mm와 5 mm뿐이므로 굵기 비가 1에 가까운 답은 굵기 변화가 아니라 곧은 호스의 다른 표현이다
    for r in results:
        r['flat'] = False
        if r['kind'] == 'step':
            p = r['model'].unpack(r['best'].x); k = math.sqrt(max(p['lam'][0], 1e-9) / max(p['lam0'], 1e-9))
            r['flat'] = max(k, 1 / k) < cfg.get('step_min_ratio', 1.3)
    results.sort(key=lambda r: (r['rough'], r['flat'], r['bic']))
    r0 = results[0]
    # 고른 후보의 다른 답은 원래 해상도로 점수만 다시 매긴다
    jobs = [{'stage': 'score', 'c': r0['c'], 'x': r['x']} for r in done[r0['c']][1:] + best[r0['c']][top:]]
    got = yield jobs
    rest = [Sol(np.array(r['x']), r['cost'], r0['best'].fun) for r in got if r]
    rest.sort(key=lambda s: s.cost)
    finish(ctx, results, [r0['best']] + rest)

@contextmanager
def as_written(g):
    """실제 경로는 적힌 점을 잇는 매끈한 곡선으로 쓴다. 구멍 방향을 강제하지 않으므로 두 점이면 직선이다."""
    keep = g.curve; g.curve = 'free'
    try: yield
    finally: g.curve = keep

def geom_from_truth(model, truth):
    """truth polylines -> geometry part of the parameter vector (None if the shape does not fit the model)."""
    g = []
    mid = lambda pl: abc2lab((np.array(pl[0]) + np.array(pl[-1])) / 2) if len(pl) == 2 else abc2lab(pl[len(pl) // 2])
    if model.kind == 'step': return None
    if model.kind == 'loop':
        if len(truth) != 4: return None
        return np.array(list(abc2lab(truth[0][-1])) + list(abc2lab(truth[3][0])) + list(mid(truth[1])) + list(mid(truth[2])))
    if model.nb == 1:
        if len(truth) != 1: return None
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
    with as_written(model.g):
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
    res['topology'] = model.kind
    if model.kind == 'step':
        P_, L_, sP_ = model.point_at(p)
        res['P'] = lab2abc(P_); res['step_s_cm'] = float(sP_); res['lam_ratio'] = float(p['lam'][0] / max(p['lam0'], 1e-9))
        res['diam_ratio'] = float(np.sqrt(max(res['lam_ratio'], 0)))
    if model.kind == 'loop':
        res['J2'] = lab2abc(p['J2']); res['merge_len_cm'] = float(P['merge'][2])
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
    p = model.unpack(s.x); best = model.nodes(p)
    def err(b, t):
        A = _dense(b); B = _dense(t)
        d = np.linalg.norm(A[:, None] - B[None], axis=2)
        dd = np.concatenate([d.min(1), d.min(0)])
        return {'mean': round(float(dd.mean()), 2), 'max': round(float(dd.max()), 2)}
    if len(best) != len(truth):          # 구조가 다르면: 전체 관끼리 비교
        allb = [q for pl in best for q in _dense(pl)]; allt = [q for pl in truth for q in _dense(pl)]
        return [err(allb, allt)]
    out = [err(b, t) for b, t in zip(best, truth)]
    if model.kind == 'loop':             # 두 갈래는 이름이 바뀌어도 같은 모양
        sw = [err(best[1], truth[2]), err(best[2], truth[1])]
        if sw[0]['mean'] + sw[1]['mean'] < out[1]['mean'] + out[2]['mean']: out[1], out[2] = sw[0], sw[1]
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
def html(cfg, model, sols, ctx, summ):
    best = sols[0]; p = model.unpack(best.x); tt = ctx.tt; data = ctx.data
    path_nodes = model.nodes
    alt = []
    for s in sols[1:6]:
        if 2 * (s.cost - best.cost) < cfg.get('alt_dchi2', 50): alt.append(path_nodes(model.unpack(s.x)))
    _, Y, U = residuals(model, data, tt)
    pr = model.predict(best.x, tt)
    b0a, b0b = cfg.get('baseline', [2, 12]); inside = (tt >= ctx.win[0]) & (tt <= ctx.win[1])
    traces = {}
    for k in Y:
        w = data[k]['w']; wf = w * inside      # 그래프는 전체 시간을 그리고, 센서 회전과 표류는 맞춘 구간에서 정한다
        if cfg.get('drift'): x = fit_one(pr[k], Y[k], U[k], wf, True, tt - (b0a + b0b) / 2)
        else: x, _ = best_psi(pr[k], Y[k], U[k], wf)
        e1 = np.cross(U[k], [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(U[k], e1)
        comp = lambda A: [[(round(float(v @ e), 3) if wi > 0 else None) for v, wi in zip(A, w)] for e in (U[k], e1, e2)]
        traces[k] = {'meas': comp(Y[k]), 'model': comp(x), 'noise': round(float(data[k]['noise']), 3)}
    payload = {'title': cfg.get('title', f"Run #{cfg['run']} 3D 복원"), 'box': cfg['box'],
               'sensors': cfg['sensors'], 'inlet': cfg['inlet'], 'outlets': cfg['outlets'],
               'best': path_nodes(p), 'alt': alt, 'truth': cfg.get('truth'),
               'summary': summ, 't': [round(float(x), 2) for x in tt], 'traces': traces,
               'outlet_names': cfg.get('outlet_names', [f'출구{i+1}' for i in range(len(cfg['outlets']))]),
               'geom': geom_payload(cfg, model.nb), 'topology': model.kind,
               'all_outlets': cfg.get('_all_outlets', cfg['outlets']), 'all_outlet_names': cfg.get('_all_names'),
               'loop_outlet': getattr(model, 'oi', None)}
    tpl = open(__file__.replace('recon3d.py', 'template.html'), encoding='utf-8').read()
    return tpl.replace('__DATA__', json.dumps(payload, ensure_ascii=False))

KIND_NAME = {'single': '한 줄 관', 'y': 'Y자 (갈래마다 다른 출구)', 'loop': '갈라졌다 다시 모이는 관', 'step': '중간에 굵기가 바뀌는 한 줄 관'}

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
    ctx = Context(cfg)
    print(f"  시간 {ctx.tt[0]:.1f}~{ctx.tt[-1]:.1f}초, 전류 {ctx.I:.3f} A, 잡음(μT) " + ', '.join(f'{k} {d["noise"]:.2f}' for k, d in ctx.data.items()), flush=True)
    steps = plan(ctx); jobs = next(steps)
    try:
        while True: jobs = steps.send([do_job(ctx, j) for j in jobs])
    except StopIteration:
        pass

def swapped_sensors(data, sensors, I):
    """측정한 코일 자기장 크기가 설정 위치보다 다른 센서 위치에 맞으면 두 센서를 돌려준다. 코일 자기장은 위치마다 크기가 달라서 구별된다."""
    keys = list(data); out = []
    m = {k: math.log(np.linalg.norm(data[k]['b0'])) for k in keys}
    c = {k: math.log(np.linalg.norm(B0lab(sensors[k], I)[0])) for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            now = abs(m[a] - c[a]) + abs(m[b] - c[b]); cross = abs(m[a] - c[b]) + abs(m[b] - c[a])
            if now > 0.15 and cross < now - 0.1: out.append([a, b])
    return out

def finish(ctx, results, sols):
    """고른 답으로 표, 내경, 결과 파일을 만든다."""
    cfg = ctx.cfg; tt = ctx.tw; data = ctx.dw; I = ctx.I
    r0 = results[0]; model, Y, U, f = r0['model'], r0['Y'], r0['U'], r0['f']
    used_cfg = r0['cfg']; used_cfg['_all_outlets'] = cfg['outlets']; used_cfg['_all_names'] = ctx.all_names
    N = len(sols[0].fun)
    z = float(np.sum(np.concatenate([(Y[k] * data[k]['w'][:, None] / data[k]['noise']).ravel() for k in Y]) ** 2) / N)
    summ = summarize(model, sols[0], N); summ['chi2_per_N_no_iron_model'] = z
    summ['noise_uT'] = {k: round(float(d['noise']), 3) for k, d in data.items()}
    summ['fit_window'] = [round(float(v), 1) for v in ctx.win]
    summ['sensors_swapped'] = swapped_sensors(data, model.g.sensors, I)
    pb = model.unpack(sols[0].x)
    summ['at_limit'] = bool(pb['U0'] > 29 or pb['tau'] < 0.12)      # 앞머리가 지나가는 신호로 보기 어려운 답
    summ['alternatives'] = [summarize(model, s, N) for s in sols[1:4]]
    if len(results) > 1:
        summ['topology_scale'] = round(float(results[0]['scale']), 2)
        summ['topology_compare'] = [{'kind': r['kind'], 'name': r['label'], 'chi2_per_N': round(float(r['chi2_per_N']), 3),
                                     'bic': round(float(r['bic']), 1), 'npar': r['npar'], 'rough': r['rough'], 'flat': r['flat']} for r in results]
    summ['exit_used'] = used_cfg['outlet_names']
    if cfg.get('truth_points'):          # 실제 경로의 일부 점만 알 때 (예: 분기점, 합류점)
        tp = cfg['truth_points']; tpe = {}
        for key, val in tp.items():
            if key in summ: tpe[key] = round(float(np.linalg.norm(np.array(summ[key]) - np.array(val))), 2)
        summ['truth_point_error_cm'] = tpe
    if cfg.get('truth'):
        summ['truth_error_cm'] = path_error(model, sols[0], cfg['truth'])
        st = fit_truth(model, f, sols[0], cfg['truth'])
        if st is not None:
            with as_written(model.g):
                summ['truth_fit'] = summarize(model, st, N)
            summ['chi2_per_N_truth_path'] = summ['truth_fit']['chi2_per_N']
    if cfg.get('thickness', True):      # 호스 내경 추정 (실패해도 복원 결과는 그대로 저장)
        print('호스 굵기 추정 중...', flush=True)
        try:
            import thickness
            tcfg = dict(used_cfg)
            # 적어 둔 실제 경로가 복원 경로보다 측정을 훨씬 못 맞추면 그 경로로 속도를 재지 않는다
            if summ.get('chi2_per_N_truth_path', 0) > 1.25 * summ['chi2_per_N']: tcfg.pop('truth', None)
            flows = cfg.get('flow_mL_s') or []
            tcfg['flow_mL_s'] = [flows[i] if i < len(flows) else None for i in r0['idx']]
            best_p = model.unpack(sols[0].x)
            summ['thickness'] = thickness.estimate(tcfg, ctx.tt, ctx.data, I, model, {k: best_p[k] for k in ('J', 'J2', 'C', 'sf') if k in best_p})
            for pc in summ['thickness']['pieces']:
                print(f"  {pc['name']}: 앞머리 속도 {pc['U']:.2f} cm/s, 내경 {pc.get('d_mm', '-')} mm" if pc['resolved'] else f"  {pc['name']}: 판정 불가", flush=True)
        except Exception as e:
            print('  굵기 추정 실패:', e, flush=True)
            summ['thickness'] = {'error': str(e)}
    out = cfg.get('out', f"recon_run{cfg['run']}")
    if os.path.dirname(out): os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summ, open(out + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(out + '.html', 'w', encoding='utf-8').write(html(used_cfg, model, sols, ctx, summ))
    print(json.dumps({k: summ[k] for k in ['topology', 'exit_used', 'topology_compare', 'P', 'diam_ratio', 'chi2_per_N', 'chi2_per_N_truth_path', 'chi2_per_N_no_iron_model', 'J', 'J2', 'C', 'U0', 'U', 'flow_split', 'lam', 'branch_len_cm', 'truth_error_cm', 'truth_point_error_cm'] if k in summ}, ensure_ascii=False, indent=1))
    print('saved', out + '.json', out + '.html')

if __name__ == '__main__':
    main()
