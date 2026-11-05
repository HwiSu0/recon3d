#!/usr/bin/env python3
"""
thickness.py - 앞머리 속도와 유량으로 호스 내경을 추정한다.

자기장만으로는 lambda = (실효 자화율 x 단면적)만 정해지고, 호스가 센서에서 충분히 멀어
굵기 자체는 보이지 않는다. 그래서 두 방법을 쓴다.

1) 유량법: d = 2 sqrt(Q / (pi U))
   U는 Capstone 측정값에서 구한 앞머리 속도, Q는 출구에서 모은 양 / 걸린 시간.
2) 신호 크기 비교 (농도가 같다고 가정): d_k = d_입구 sqrt(lam_k / lam_입구)

복원에서 고른 구조(한 줄 관, 굵기 변화, Y자, 합류 관)의 경로를 고정하고 구간마다 속도를 구한다.
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
MAX_REL_ERR = 0.5
FLOW_REL_ERR = 0.05      # 유량 측정의 상대 불확도(눈금 읽기와 시간 재기 가정)


class FrontModel:
    """경로를 고정한 앞머리 모델. recon3d의 Model, ModelStep, ModelLoop.predict와 같은 식이고
    쌍극자 계수만 미리 계산한다. 매개변수: [t0, U 구간별.., lam 구간별.., tau]"""

    def __init__(self, model, nodes):
        self.g = g = model.g
        self.kind = model.kind
        self.oi = getattr(model, 'oi', 0)
        self.oa = getattr(model, 'oa', True)
        P = model.paths(nodes)
        self.stem_G, self.stem_s, self.LJ = P['stem']
        self.br = P['br']
        if self.kind == 'step':
            G, s, L = self.br[0]
            self.sP = float(nodes['sf'] * L)
            self.L = L
            self.mix = np.clip((s - self.sP) * len(s) / L + 0.5, 0, 1)
            self.lengths = [self.sP, L - self.sP]
            self.reach = [self.sP, L]
        elif self.kind == 'single':
            self.lengths = [self.br[0][2]]
            self.reach = [self.br[0][2]]
        else:
            self.lengths = [self.LJ] + [b[2] for b in self.br]
            self.reach = [self.LJ] + [self.LJ + b[2] for b in self.br]
        if self.kind == 'loop':
            self.merge_G, self.merge_s, self.LM = P['merge']
        self.n = len(self.lengths)

    def split(self, q):
        n = self.n
        return q[0], list(q[1:1 + n]), list(q[1 + n:1 + 2 * n]), q[1 + 2 * n]

    def predict(self, q, tt):
        t0, Us, lams, tau = self.split(q)
        g = self.g
        res = np.zeros((len(tt), g.nc))
        add = lambda G, tarr, lam: R.front_sum(res, tt, tau, G, tarr, lam)
        U0, lam0 = Us[0], lams[0]
        add(g.up_G, t0 + g.up_s / U0, lam0)
        so = g.out_pts[self.oi][1]
        if self.kind == 'single':
            G, s, L = self.br[0]
            add(G, t0 + s / U0, lam0)
            add(g.out_G[self.oi], t0 + (L + so) / U0, lam0)
        elif self.kind == 'step':
            tP = t0 + self.sP / U0
            G, s, _ = self.br[0]
            add(G, np.where(s < self.sP, t0 + s / U0, tP + (s - self.sP) / Us[1]), lam0 + (lams[1] - lam0) * self.mix)
            add(g.out_G[self.oi], tP + (self.L - self.sP + so) / Us[1], lams[1] if self.oa else lam0)
        else:
            add(self.stem_G, t0 + self.stem_s / U0, lam0)
            tJ = t0 + self.LJ / U0
            if self.kind == 'loop':
                arrive = []
                for k, (G, s, Lb) in enumerate(self.br):
                    add(G, tJ + s / Us[k + 1], lams[k + 1])
                    arrive.append(tJ + Lb / Us[k + 1])
                tJ2 = min(arrive)
                add(self.merge_G, tJ2 + self.merge_s / U0, lam0)
                add(g.out_G[self.oi], tJ2 + (self.LM + so) / U0, lam0)
            else:
                for k, (G, s, Lb) in enumerate(self.br):
                    add(G, tJ + s / Us[k + 1], lams[k + 1])
                    add(g.out_G[k], tJ + (Lb + g.out_pts[k][1]) / Us[k + 1], lams[k + 1])
        return g.split(res)

    def bounds(self):
        n = self.n
        lo = [-60.0] + [SPEED_GRID[0] * 0.5] * n + [0.0] * n + [0.1]
        hi = [400.0] + [SPEED_GRID[-1] * 2] * n + [5.0] * n + [15.0]
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
        s = least_squares(g, y0, bounds=(self.lo[free], self.hi[free]), max_nfev=nfev, ftol=1e-6, xtol=1e-5)
        z = x.copy(); z[free] = s.x
        return z, 2 * s.cost, s

    def start(self, u):
        n = self.front.n
        tail = [u] * n + [0.1] * n + [1.0]
        return [[t] + tail for t in (self.t_on - 6, self.t_on - 2, self.t_on + 1)]

    def at_speed(self, u, x0):
        return self.fit(x0, {1 + i: u for i in range(self.front.n)})[:2]

    def best_overall(self):
        """속도 격자마다 도착 시각 하나로 맞춘 뒤, 잘 맞은 속도만 다른 도착 시각에서도 맞춘다."""
        first = [self.at_speed(u, self.start(u)[1]) for u in SPEED_GRID]
        top = np.argsort([g[1] for g in first])[:3]
        best = min(first, key=lambda g: g[1])
        for j in top:
            for x0 in self.start(SPEED_GRID[j])[::2]:
                g = self.at_speed(SPEED_GRID[j], x0)
                if g[1] < best[1]: best = g
        z, c2, s = self.fit(best[0], nfev=120)
        return (z, c2, s) if c2 <= best[1] else (best[0], best[1], s)

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


def refine(fitter, x, idx):
    """chi2 최소 근처를 촘촘히 재서 포물선으로 속도와 곡률을 구한다. 곡률은 log U 기준."""
    u0 = x[idx]
    us = u0 * np.exp(np.array([-0.3, -0.15, 0.0, 0.15, 0.3]))
    cs = np.array([fitter.fit(x, {idx: u})[1] for u in us])
    a, b, _ = np.polyfit(np.log(us / u0), cs, 2)
    if a <= 0:
        return float(u0), None
    shift = float(np.clip(-b / (2 * a), -0.4, 0.4))
    return float(u0 * math.exp(shift)), float(a)


def fit_window(cfg, front, tt, data, t_on, a, b, x0=None):
    tw, dw = window_data(tt, data, a, b)
    fitter = Fitter(front, cfg, tw, dw, t_on)
    return fitter, (fitter.best_overall() if x0 is None else fitter.fit(x0, nfev=120))


def nodes_from(cfg, model, best_nodes):
    """실제 경로를 알면 그것을, 아니면 복원한 경로를 쓴다."""
    truth = cfg.get('truth')
    if truth:
        g = R.geom_from_truth(model, truth)
        if g is not None:
            if model.kind == 'single':
                return {'C': [g[:3]]}, '실제 경로'
            if model.kind == 'loop':
                return {'J': g[:3], 'J2': g[3:6], 'C': [g[6:9], g[9:12]]}, '실제 경로'
            return {'J': g[:3], 'C': [g[3 + 3 * k:6 + 3 * k] for k in range(model.nb)]}, '실제 경로'
    return best_nodes, '복원 경로'


def piece_info(cfg, model, n):
    """구간 이름, 구간마다 흐르는 유량, 설정의 내경"""
    nb = len(cfg['outlets'])
    hose = cfg.get('hose_mm') or {'stem': 3, 'branches': []}
    stem = hose['stem']
    br = list(hose.get('branches') or [])
    br += [stem] * max(0, n - len(br))
    flows = [float(q) if q not in (None, '') else None for q in (cfg.get('flow_mL_s') or [])] + [None] * nb
    if model.kind == 'single':
        return ['통로'], [flows[0]], [stem]
    if model.kind == 'step':
        return ['굵기 바뀌기 전', '굵기 바뀐 뒤'], [flows[0], flows[0]], [stem, br[0]]
    if model.kind == 'loop':
        return ['입구 쪽', '갈래1', '갈래2'], [flows[0], None, None], [stem, br[0], br[1]]
    names = list(cfg.get('outlet_names') or [f'출구{i + 1}' for i in range(nb)])
    each = flows[:nb]
    total = sum(each) if all(q is not None for q in each) else None
    return ['입구 쪽'] + names, [total] + each, [stem] + br[:nb]


def estimate(cfg, tt, data, I, model, best_nodes):
    nodes, path_kind = nodes_from(cfg, model, best_nodes)
    if path_kind == '실제 경로':
        with R.as_written(model.g):
            return measure(cfg, tt, data, I, model, nodes, path_kind)
    return measure(cfg, tt, data, I, model, nodes, path_kind)


def measure(cfg, tt, data, I, model, nodes, path_kind):
    front = FrontModel(model, nodes)
    t_on = onset(tt, data)
    start = max(float(tt[0]), t_on - 12)
    last = float(tt[-1])
    n = front.n
    box_len = max(front.reach)

    # 1) 넓은 창으로 속도의 크기를 잡고, 앞머리가 상자를 지나는 시간만큼 창을 정한다
    fitter, (x, c, _) = fit_window(cfg, front, tt, data, t_on, start, t_on + 12)
    typical = float(np.median(x[1:1 + n]))
    span = min(float(np.clip(1.2 * box_len / typical + 3, 6, 40)), last - t_on)
    end = t_on + span

    # 2) 그 창에서 다시 맞춘다
    fitter, (x, c, s) = fit_window(cfg, front, tt, data, t_on, start, end, x)
    infl = residual_inflation(s.fun, fitter.axes)
    names, Qs, dset = piece_info(cfg, model, n)
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

    pieces = []
    for i in range(n):
        idx = 1 + i
        # 3) 구간마다 앞머리가 지나는 데 걸리는 시간만큼 창을 잡는다 (느린 구간은 더 길게)
        own = float(np.clip(1.2 * front.reach[i] / x[idx] + 3, 6, 40))
        own = min(own, last - t_on)
        ft, xi, ci, inf_i = (fitter, x, c, infl) if own <= span * 1.3 else at_window(own)
        used = span if ft is fitter else own

        prof = ft.profile(xi, idx)
        dchi = (prof - prof.min()) / inf_i
        U, curv = refine(ft, xi, idx)
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
        if rel is None or rel >= MAX_REL_ERR:      # 불확도가 이렇게 크면 값을 낼 수 없다
            resolved = False
        piece = {'name': names[i], 'U': round(U, 3), 'resolved': resolved, 'L_cm': round(float(front.lengths[i]), 2),
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

    # 4) 구간이 나뉘면 신호 크기 비로도 굵기를 낸다
    lam = x[1 + n:1 + 2 * n]
    lam_err = None
    try:
        free = list(range(len(x)))
        J = s.jac
        cov = np.linalg.inv(J.T @ J) * infl
        lam_err = np.sqrt(np.diag(cov))[1 + n:1 + 2 * n]
    except (np.linalg.LinAlgError, ValueError, IndexError):
        pass
    ratio = []
    if n >= 2 and lam_err is not None:
        for k in range(1, n):
            rel = math.hypot(lam_err[k] / max(lam[k], 1e-9), lam_err[0] / max(lam[0], 1e-9))
            item = {'name': names[k], 'resolved': bool(rel < 0.4)}
            if item['resolved']:
                d = dset[0] * math.sqrt(lam[k] / lam[0])
                item['d_mm'] = round(float(d), 2); item['d_err_mm'] = round(float(d * rel / 2), 2)
            ratio.append(item)

    return {'path': path_kind, 't_on': round(t_on, 1), 'window': [round(start, 1), round(end, 1)],
            'chi2_per_N': round(float(s.fun @ s.fun) / len(s.fun), 3), 'inflation': round(infl, 2),
            'flow_rel_err': FLOW_REL_ERR, 'lam': [round(float(v), 4) for v in lam], 'pieces': pieces, 'from_signal_ratio': ratio}
