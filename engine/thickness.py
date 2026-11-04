#!/usr/bin/env python3
"""
thickness.py - 앞머리 속도와 유량으로 호스 안지름을 추정한다.

자기장만으로는 lambda = (실효 자화율 x 단면적)만 정해지고, 호스가 센서에서 충분히 멀어
굵기 자체는 보이지 않는다. 그래서 두 방법을 쓴다.

1) 유량법: d = 2 sqrt(Q / (pi U))
   U는 Capstone 측정값에서 구한 앞머리 속도, Q는 출구에서 모은 양 / 걸린 시간.
2) 신호 크기 비교 (갈래가 있을 때, 농도가 같다고 가정): d_k = d_stem sqrt(lam_k / lam_stem)

속도는 첫 앞머리가 상자를 지나는 구간만 맞춘다. 그 뒤의 신호에는 뒤따르는 덩어리와
호스에 남은 철이 섞여 있어 단일 앞머리 모델로 설명되지 않기 때문이다.
속도를 바꿔 가며 chi2가 어떻게 변하는지(프로파일) 보고, 최솟값이 뚜렷하지 않으면
숫자를 내지 않고 판정 불가로 둔다.
"""
import math
import numpy as np
from scipy.optimize import least_squares
import recon3d as R

SPEED_GRID = np.exp(np.linspace(np.log(0.3), np.log(30.0), 13))   # cm/s
RESOLVE_SIGMA = 3.0      # U/2, 2U에서 이만큼 나빠져야 속도가 정해졌다고 본다
FLOW_REL_ERR = 0.05      # 유량 측정의 상대 불확도(눈금 읽기와 시간 재기 가정)


class FrontModel:
    """경로를 고정한 앞머리 모델. recon3d.Model.predict와 같은 식이고 쌍극자 계수만 미리 계산한다.
    매개변수: nb=1 [t0, U, lam, tau] / nb>=2 [t0, U0, U1.., lam0, lam1.., tau]"""

    def __init__(self, model, nodes):
        self.g = g = model.g
        self.nb = model.nb
        P = model.paths(nodes)
        pts, self.stem_s, self.LJ = P['stem']
        self.stem_G = g.unit_field(pts) if len(pts) else None
        self.br = [(g.unit_field(p) if len(p) else None, s, L) for p, s, L in P['br']]
        self.lengths = [self.LJ] + [b[2] for b in self.br]
        self.n_speed = 1 if self.nb == 1 else self.nb + 1

    def split(self, q):
        if self.nb == 1:
            t0, U, lam, tau = q
            return t0, U, [U], lam, [lam], tau
        nb = self.nb
        return q[0], q[1], list(q[2:2 + nb]), q[2 + nb], list(q[3 + nb:3 + 2 * nb]), q[3 + 2 * nb]

    def predict(self, q, tt):
        t0, U0, Us, lam0, lams, tau = self.split(q)
        g = self.g
        out = {k: np.zeros((len(tt), 3)) for k in g.sensors}
        sig = lambda x: 0.5 * (1 + np.tanh(x / (2 * tau)))

        def add(G, tarr, lam):
            L = lam * sig(tt[:, None] - tarr[None])
            for k in out:
                out[k] += L @ G[k]
        add(g.up_G, t0 + g.up_s / U0, lam0)
        if self.stem_G is not None:
            add(self.stem_G, t0 + self.stem_s / U0, lam0)
        tJ = t0 + self.LJ / U0
        for k, (G, s, Lb) in enumerate(self.br):
            if G is not None:
                add(G, tJ + s / Us[k], lams[k])
            add(g.out_G[k], tJ + (Lb + g.out_pts[k][1]) / Us[k], lams[k])
        return out

    def bounds(self):
        ns = self.n_speed
        nl = 1 if self.nb == 1 else self.nb + 1
        lo = [-60.0] + [SPEED_GRID[0] * 0.5] * ns + [0.0] * nl + [0.1]
        hi = [400.0] + [SPEED_GRID[-1] * 2] * ns + [5.0] * nl + [15.0]
        return np.array(lo), np.array(hi)


def onset(tt, data):
    """신호가 기준 구간 잡음을 뚜렷하게 벗어나는 첫 시각.
    앞머리는 모든 센서에 나타나므로 센서들의 중앙값을 써서 한 센서의 튀는 값에 끌려가지 않게 한다."""
    snr = np.stack([np.where(d['w'] > 0, np.linalg.norm(d['dB'], axis=1) / d['noise'], np.nan) for d in data.values()])
    s = np.nanmedian(snr, axis=0)
    n = max(1, int(round(1.5 / (tt[1] - tt[0]))))
    s = np.convolve(s, np.ones(n) / n, mode='same')
    head = max(4, len(s) // 8)
    level = np.median(s[:head]); spread = max(np.std(s[:head]), 1e-6)
    above = s > level + 0.35 * (np.percentile(s, 99) - level) + 2 * spread
    run = max(1, int(round(1.0 / (tt[1] - tt[0]))))
    for i in range(len(s) - run):
        if above[i:i + run].all():
            return float(tt[i])
    return float(tt[int(np.argmax(s))])


def window_data(tt, data, a, b):
    m = (tt >= a) & (tt <= b)
    sub = {k: {'b0': d['b0'], 'dB': d['dB'][m], 'w': d['w'][m], 'noise': d['noise']} for k, d in data.items()}
    return tt[m], sub


def residual_inflation(r, n_sensor_axes):
    """잔차가 chi2/N 1을 넘거나 이웃 구간끼리 닮은 만큼 불확도를 키운다."""
    r = np.asarray(r)
    rows = r.reshape(n_sensor_axes, -1)
    a = rows[:, :-1].ravel(); b = rows[:, 1:].ravel()
    denom = float(a @ a)
    rho = float(a @ b) / denom if denom > 0 else 0.0
    rho = min(max(rho, 0.0), 0.8)
    return max(1.0, float(r @ r) / max(1, r.size)) * (1 + rho) / (1 - rho)


class Fitter:
    def __init__(self, front, cfg, tt, data, t_on):
        self.front = front
        self.tt = tt; self.t_on = t_on
        b0a, b0b = cfg.get('baseline', [2, 12])
        self.f, _, _ = R.residuals(front, data, tt, drift=cfg.get('drift', False), tb=(b0a + b0b) / 2)
        self.lo, self.hi = front.bounds()
        self.axes = 3 * len(data)

    def fit(self, x0, fixed=None, nfev=40):
        """fixed = {색인: 값}. 나머지 매개변수만 맞춘다."""
        fixed = fixed or {}
        free = [i for i in range(len(x0)) if i not in fixed]
        x = np.array(x0, float)
        for i, v in fixed.items():
            x[i] = v

        def g(y):
            z = x.copy(); z[free] = y
            return self.f(z)
        y0 = np.clip(x[free], self.lo[free] + 1e-6, self.hi[free] - 1e-6)
        s = least_squares(g, y0, bounds=(self.lo[free], self.hi[free]), max_nfev=nfev)
        z = x.copy(); z[free] = s.x
        return z, 2 * s.cost, s

    def start(self, u):
        fr = self.front
        tail = [u] * fr.n_speed + [0.1] * (1 if fr.nb == 1 else fr.nb + 1) + [1.0]
        return [[t] + tail for t in (self.t_on - 6, self.t_on - 2, self.t_on + 1)]

    def best_at_common_speed(self, u):
        best = None
        fixed = {1 + i: u for i in range(self.front.n_speed)}
        for x0 in self.start(u):
            x, c, _ = self.fit(x0, fixed)
            if best is None or c < best[1]:
                best = (x, c)
        return best

    def best_overall(self):
        grid = [self.best_at_common_speed(u) for u in SPEED_GRID]
        x, c, u = min(((g[0], g[1], u) for g, u in zip(grid, SPEED_GRID)), key=lambda z: z[1])
        z, c2, s = self.fit(x, nfev=120)
        return (z, c2, s) if c2 <= c else (x, c, s)

    def profile(self, x, idx):
        """속도 하나를 격자로 고정하고 나머지를 다시 맞춘 chi2. 가까운 격자의 답에서 이어 맞춘다."""
        order = np.argsort(np.abs(np.log(SPEED_GRID / x[idx])))
        cur = {}
        for j in order:
            u = SPEED_GRID[j]
            near = min(cur, key=lambda k: abs(k - j)) if cur else None
            x0 = cur[near][0] if near is not None else x
            cur[j] = self.fit(x0, {idx: u})[:2]
        return np.array([cur[j][1] for j in range(len(SPEED_GRID))])


def refine(fitter, x, c, idx):
    """chi2 최소 근처를 촘촘히 재서 포물선으로 속도와 곡률을 구한다. 곡률은 log U 기준."""
    u0 = x[idx]
    us = u0 * np.exp(np.array([-0.3, -0.15, 0.0, 0.15, 0.3]))
    cs = np.array([fitter.fit(x, {idx: u})[1] for u in us])
    a, b, _ = np.polyfit(np.log(us / u0), cs, 2)
    if a <= 0:
        return float(u0), None
    shift = float(np.clip(-b / (2 * a), -0.4, 0.4))
    return float(u0 * math.exp(shift)), float(a)


def fit_window(cfg, front, tt, data, t_on, a, b):
    tw, dw = window_data(tt, data, a, b)
    fitter = Fitter(front, cfg, tw, dw, t_on)
    return fitter, fitter.best_overall()


def piece_names(cfg, nb):
    names = cfg.get('outlet_names') or [f'출구{i + 1}' for i in range(nb)]
    return ['통로'] if nb == 1 else ['입구 쪽'] + list(names)


def nodes_from(cfg, model, best_nodes):
    """실제 경로를 알면 그것을, 아니면 복원한 경로를 쓴다."""
    truth = cfg.get('truth')
    if truth:
        g = R.geom_from_truth(model, truth)
        if g is not None:
            if model.nb == 1:
                return {'C': [g[:3]]}, '실제 경로'
            return {'J': g[:3], 'C': [g[3 + 3 * k:6 + 3 * k] for k in range(model.nb)]}, '실제 경로'
    return best_nodes, '복원 경로'


def estimate(cfg, tt, data, I, model, best_nodes):
    nb = model.nb
    nodes, path_kind = nodes_from(cfg, model, best_nodes)
    front = FrontModel(model, nodes)
    t_on = onset(tt, data)
    start = max(float(tt[0]), t_on - 12)
    box_len = sum(front.lengths[:1]) + max(front.lengths[1:])

    # 1) 넓은 창으로 속도의 크기를 잡고, 앞머리가 상자를 지나는 시간만큼 창을 정한다
    fitter, (x, c, _) = fit_window(cfg, front, tt, data, t_on, start, t_on + 12)
    speeds = x[1:1 + front.n_speed]
    typical = float(np.median(speeds))
    span = float(np.clip(1.2 * box_len / typical + 3, 6, 40))
    span = min(span, float(tt[-1]) - t_on)
    end = t_on + span

    # 2) 그 창에서 속도마다 프로파일을 구한다
    fitter, (x, c, s) = fit_window(cfg, front, tt, data, t_on, start, end)
    infl = residual_inflation(s.fun, fitter.axes)
    n_lam = 1 if nb == 1 else nb + 1
    pieces = []
    names = piece_names(cfg, nb)
    flows = cfg.get('flow_mL_s') or []
    flows = [float(q) if q not in (None, '') else None for q in flows] + [None] * nb
    hose = cfg.get('hose_mm') or {'stem': 3, 'branches': [3] * nb}
    if nb == 1:
        Qs = [flows[0]]; dset = [hose['branches'][0] if hose.get('branches') else hose['stem']]
    else:
        q_all = flows[:nb]
        Qs = [sum(q_all) if all(q is not None for q in q_all) else None] + q_all
        dset = [hose['stem']] + list(hose['branches'])[:nb]

    last = float(tt[-1])
    windows = {}

    def at_window(length):
        """끝 시각이 t_on + length인 창에서 지금 답을 이어 맞춘다 (다른 해로 넘어가지 않게)."""
        key = round(length, 1)
        if key not in windows:
            tw, dw = window_data(tt, data, start, min(t_on + length, last))
            ft = Fitter(front, cfg, tw, dw, t_on)
            xw, cw, sw = ft.fit(x, nfev=60)
            windows[key] = (ft, xw, cw, residual_inflation(sw.fun, ft.axes))
        return windows[key]

    for i in range(front.n_speed):
        idx = 1 + i
        # 3) 구간마다 앞머리가 지나는 데 걸리는 시간만큼 창을 잡는다 (느린 갈래는 더 길게)
        reach = sum(front.lengths[:1]) + (front.lengths[i] if i else 0) if nb > 1 else front.lengths[1]
        own = float(np.clip(1.2 * reach / x[idx] + 3, 6, 40))
        own = min(own, last - t_on)
        ft, xi, ci, inf_i = (fitter, x, c, infl) if own <= span * 1.3 else at_window(own)
        used = span if ft is fitter else own

        prof = ft.profile(xi, idx)
        dchi = (prof - prof.min()) / inf_i
        U, curv = refine(ft, xi, ci, idx)
        logu = np.log(SPEED_GRID)
        half = float(np.interp(math.log(U / 2), logu, dchi)); dbl = float(np.interp(math.log(U * 2), logu, dchi))
        resolved = bool(curv and min(half, dbl) >= RESOLVE_SIGMA ** 2 and SPEED_GRID[0] * 1.2 < U < SPEED_GRID[-1] / 1.2)
        stat = math.sqrt(inf_i / curv) if curv else None
        # 창을 줄이고 늘려도 속도가 유지되는지 본다
        spread = 0.0
        if resolved:
            for k in (0.75, 1.3):
                if used * k > 3 and t_on + used * k <= last:
                    spread = max(spread, abs(math.log(at_window(used * k)[1][idx] / U)))
        rel = math.hypot(stat, spread) if stat is not None else None
        piece = {'name': names[i], 'U': round(U, 3), 'resolved': resolved, 'L_cm': round(float(front.lengths[i] if nb > 1 else front.lengths[1]), 2),
                 'profile': {'U': [round(float(u), 3) for u in SPEED_GRID], 'dchi2': [round(float(v), 2) for v in dchi]},
                 'd_set_mm': dset[i], 'Q_mL_s': Qs[i]}
        if resolved:
            piece['U_err'] = round(U * rel, 3)
            Q = Qs[i]
            if Q:
                d = 20 * math.sqrt(Q / (math.pi * U))
                piece['d_mm'] = round(d, 2)
                piece['d_err_mm'] = round(d * 0.5 * math.hypot(rel, FLOW_REL_ERR), 2)
        pieces.append(piece)

    # 4) 갈래가 있으면 신호 크기 비로도 굵기를 낸다
    lam = x[1 + front.n_speed:1 + front.n_speed + n_lam]
    lam_err = None
    try:
        J = s.jac
        cov = np.linalg.inv(J.T @ J) * infl
        lam_err = np.sqrt(np.diag(cov))[1 + front.n_speed:1 + front.n_speed + n_lam]
    except (np.linalg.LinAlgError, ValueError, IndexError):
        pass
    ratio = []
    if nb >= 2 and lam_err is not None:
        for k in range(nb):
            rel = math.hypot(lam_err[k + 1] / max(lam[k + 1], 1e-9), lam_err[0] / max(lam[0], 1e-9))
            item = {'name': names[k + 1], 'resolved': bool(rel < 0.4)}
            if item['resolved']:
                d = hose['stem'] * math.sqrt(lam[k + 1] / lam[0])
                item['d_mm'] = round(float(d), 2); item['d_err_mm'] = round(float(d * rel / 2), 2)
            ratio.append(item)

    return {'path': path_kind, 't_on': round(t_on, 1), 'window': [round(start, 1), round(end, 1)],
            'chi2_per_N': round(float(s.fun @ s.fun) / len(s.fun), 3), 'inflation': round(infl, 2),
            'flow_rel_err': FLOW_REL_ERR, 'lam': [round(float(v), 4) for v in lam], 'pieces': pieces, 'from_signal_ratio': ratio}
