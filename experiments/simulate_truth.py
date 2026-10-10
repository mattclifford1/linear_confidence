'''
E1 of the extra-experiments plan: do the bounds hold at their stated level?

One class's scores are drawn from a known law (mean 0, s.d. 1; the class sits
below the boundary, so its error at b is P(Z > b)), so every true error is exact.
At a fixed delta, each tolerance limit puts the boundary at b(delta), and every
bound's certificate U is checked against the exact error there.

    uv run python simulate_truth.py                 # all shapes, N, 20k reps
    uv run python simulate_truth.py --reps 2000     # quick

Writes results/simulate_truth.csv: one row per (shape, N, delta, bound, at):
    at             where e is measured: the bound's own tolerance limit, the
                   furthest training point, or (certificates) the 2-eps limit
                   and the class mean + 1 s.d.
    mean_err       E[e]
    promised       1/(N+1) (+ delta for the tolerance limits)
    p_exceed       P(e > promised), or P(e > U) for the certificates: at most
                   delta for a high-probability bound
    mean_U         E[U] for the certificates
    p95_err        95th percentile of e
    mean_err_good  E[e | G], G = {|sample mean - mu| <= eps}
    mean_q_good    E[q | G], q = P(Z > max sample), the published proof's quantity
'''
import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from deltas.bounds import (ClopperPearson, DKW, GaussianConfidence,
                           PublishedToleranceLimit)
from deltas.core import ClassSample

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'simulate_truth.csv')


class Shape:
    '''a score law standardised to mean 0, s.d. 1: draw(rng, n), sf(b), range'''

    def __init__(self, name, dist, score_range=None):
        self.name = name
        self.dist = dist
        self.mu, self.sd = float(dist.mean()), float(dist.std())
        self.score_range = None if score_range is None else tuple(
            (np.asarray(score_range) - self.mu) / self.sd)

    def draw(self, rng, n):
        return (self.dist.rvs(n, random_state=rng) - self.mu) / self.sd

    def sf(self, b):
        return self.dist.sf(np.asarray(b) * self.sd + self.mu)


class Mixture(Shape):
    '''two equal normal subgroups, 3 s.d.s apart before standardising'''

    def __init__(self, name, gap=3.0):
        self.name, self.gap = name, gap
        self.mu, self.sd = 0.0, float(np.sqrt(1.0 + gap ** 2 / 4.0))
        self.score_range = None

    def draw(self, rng, n):
        side = rng.random(n) < 0.5
        z = rng.standard_normal(n) + np.where(side, self.gap / 2, -self.gap / 2)
        return z / self.sd

    def sf(self, b):
        x = np.asarray(b) * self.sd
        return 0.5 * (stats.norm.sf(x - self.gap / 2) + stats.norm.sf(x + self.gap / 2))


SHAPES = {
    'normal': Shape('normal', stats.norm()),
    # long right tail, towards the boundary
    'skewed': Shape('skewed', stats.gamma(2.0)),
    'heavy': Shape('heavy', stats.t(3)),
    'two subgroups': Mixture('two subgroups'),
    'bounded': Shape('bounded', stats.beta(2.0, 5.0), score_range=(0.0, 1.0)),
}


def _tolerance_b(bound, delta):
    '''b where the fitted tolerance limit's delta equals `delta`'''
    q = np.sqrt(2.0 * np.log(1.0 / delta))
    unit = bound.factor * bound.R_conc / np.sqrt(bound.N)
    return bound.sample.mean + bound.R + unit * (q + 2.0)


def cell(shape_name, N, delta, reps, seed):
    shape = SHAPES[shape_name]
    rng = np.random.default_rng(seed)
    limits = {'published (eps)': dict(factor=1.0),
              'published (2 eps)': dict(factor=2.0)}
    if shape.score_range is not None:
        limits['published (2 eps, known range)'] = dict(
            factor=2.0, score_range=shape.score_range)
    certs = {'Clopper-Pearson': ClopperPearson(), 'DKW': DKW(),
             'Gaussian band': GaussianConfidence()}
    rec = {name: {'e': [], 'good': []} for name in limits}
    q_all = []
    cert = {(name, at): {'e': [], 'U': []} for name in certs
            for at in ('2 eps limit', 'mean + 1 s.d.')}

    for _ in range(reps):
        s = ClassSample(shape.draw(rng, N), 0, 'low')
        q_all.append(float(shape.sf(s.z[-1])))
        for name, kw in limits.items():
            bound = PublishedToleranceLimit(**kw).fit(s)
            eps = bound.R_conc / np.sqrt(N) * (2.0 + np.sqrt(2.0 * np.log(1 / delta)))
            rec[name]['e'].append(float(shape.sf(_tolerance_b(bound, delta))))
            rec[name]['good'].append(abs(s.mean) <= eps)
        at = {'2 eps limit': _tolerance_b(
                  PublishedToleranceLimit(factor=2.0).fit(s), delta),
              'mean + 1 s.d.': s.mean + s.std}
        for name, c in certs.items():
            c.fit(s)
            for where, b in at.items():
                cert[(name, where)]['e'].append(float(shape.sf(b)))
                cert[(name, where)]['U'].append(
                    float(np.asarray(c.curve(np.array([b]), delta)).ravel()[0]))

    base = {'shape': shape_name, 'N': N, 'delta': delta, 'reps': reps}
    q = np.array(q_all)
    # the published proof's quantity: the tail beyond the furthest point
    rows = [{**base, 'bound': 'furthest point', 'at': 'max sample',
             'mean_err': q.mean(), 'promised': 1.0 / (N + 1),
             'p_exceed': float((q > 1.0 / (N + 1)).mean()),
             'p95_err': float(np.quantile(q, 0.95))}]
    promised = 1.0 / (N + 1) + delta
    for name, r in rec.items():
        e, good = np.array(r['e']), np.array(r['good'])
        rows.append({**base, 'bound': name, 'at': 'own limit',
                     'mean_err': e.mean(), 'promised': promised,
                     'p_exceed': float((e > promised).mean()),
                     'p95_err': float(np.quantile(e, 0.95)),
                     'p_good': float(good.mean()),
                     'mean_err_good': float(e[good].mean()) if good.any() else np.nan,
                     'mean_q_good': float(q[good].mean()) if good.any() else np.nan})
    for (name, where), r in cert.items():
        e, U = np.array(r['e']), np.array(r['U'])
        rows.append({**base, 'bound': name, 'at': where, 'mean_err': e.mean(),
                     'mean_U': U.mean(), 'p_exceed': float((e > U).mean()),
                     'p95_err': float(np.quantile(e, 0.95))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shapes', nargs='*', default=list(SHAPES))
    ap.add_argument('--N', nargs='*', type=int,
                    default=[5, 10, 25, 50, 100, 250, 1000])
    ap.add_argument('--delta', nargs='*', type=float,
                    default=[0.01, 0.05, 0.2, 0.5])
    ap.add_argument('--reps', type=int, default=20_000)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    cells = [(s, n, d) for s in args.shapes for n in args.N for d in args.delta]
    t0 = time.time()
    chunks = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(cell)(s, n, d, args.reps, seed)
        for seed, (s, n, d) in enumerate(cells))
    df = pd.DataFrame([r for c in chunks for r in c])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'wrote {args.out} ({len(df)} rows, {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
