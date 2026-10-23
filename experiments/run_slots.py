'''
E6 and E10 of the extra-experiments plan (../papers/ecai-extra-experiments): swap one
DeltasEstimator slot at a time on the exported projections, split certificates only.

    uv run python run_slots.py --jobs 16
    uv run python run_slots.py --only E6

E10's calibration fractions other than the exported 0.35 are exported on first use to
projections-cal/. Writes results/slots.csv (run_wide.py's columns plus experiment, slot
and variant) and results/margins.csv (F4: how far the published limit sits from the
minority mean as the minority is subsampled).
'''
import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import export_projections as ep
import run_wide as rw
from deltas.bounds import (GaussianConfidence, KthPointToleranceLimit,
                           PublishedToleranceLimit)
from deltas.confidence import FixedDelta
from deltas.core import ClassSample, DeltasEstimator, ProjectedData
from deltas.rules import Risk, Sum
from deltas.transforms import YeoJohnson

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.join(HERE, 'projections-extra')
CAL = os.path.join(HERE, 'projections-cal')

#: score ranges fixed by the model: tree vote margins and sigmoid - 1/2
SCORE_RANGE = {'DecisionTree': (-1.0, 1.0), 'RandomForest': (-1.0, 1.0),
               'ConvNet-prob': (-0.5, 0.5), 'ResNet-prob': (-0.5, 0.5)}

E6_TASKS = ['MNIST: 4 vs 9', 'CIFAR-10: cat vs dog', 'DermaMNIST: melanoma vs rest',
            'Truncated Normal']
E6_MODELS = ['ConvNet-prob', 'ConvNet-logit', 'ResNet-prob', 'ResNet-logit',
             'DecisionTree', 'RandomForest', 'SVM-rbf', 'HistGradientBoosting', 'Linear']
E10_TASKS = ['Diabetes Pima Indian', 'Breast Cancer', 'Heart Disease', 'Hepatitis',
             'Framingham CHD', 'MIMIC-III Mortality', 'Spambase', 'MNIST: 4 vs 9',
             'DermaMNIST: melanoma vs rest', 'CIFAR-10: cat vs dog [resnet18]']
E10_MODELS = ['Linear', 'HistGradientBoosting']
CAL_FRACS = (0.2, 0.5)
MARGIN_N = (5, 10, 25, 50, 100, 250, 1000)
DELTA = 0.05


class KthPointR(KthPointToleranceLimit):
    '''
    the k-th point limit with its concentration R swapped: 'one_sided' or a known
    (lo, hi); here, not in deltas/, so run_wide.py's cache fingerprint is unchanged
    '''

    def __init__(self, factor=2.0, concentration_R='two_sided'):
        super().__init__(factor, aggregate='min', loss_form='expected')
        self.concentration_R = concentration_R

    def fit(self, sample):
        super().fit(sample)
        c = self.concentration_R
        if c == 'one_sided':
            far = sample.z[-1] - sample.mean if sample.side == 'low' \
                else sample.mean - sample.z[0]
            self.R = float(max(far, 0.0))
        elif c != 'two_sided':
            self.R = 0.5 * (c[1] - c[0])
        self.min_conc = self.factor * self.R / np.sqrt(self.N)
        return self


def _kth(concentration_R='two_sided', factor=2.0):
    def build(clf, model, a, p):
        if concentration_R == 'known' and model not in SCORE_RANGE:
            return None
        c = SCORE_RANGE[model] if concentration_R == 'known' else concentration_R
        return DeltasEstimator(clf, bound=KthPointR(factor, c), rule=Sum())
    return build


def _published(**kw):
    return lambda clf, model, a, p: DeltasEstimator(
        clf, bound=PublishedToleranceLimit(**kw), rule=Sum())


def _known_range(clf, model, a, p):
    if model not in SCORE_RANGE:
        return None
    return DeltasEstimator(clf, bound=PublishedToleranceLimit(
        2.0, score_range=SCORE_RANGE[model]), rule=Sum())


def _gaussian(rule, transform=False):
    def build(clf, model, a, p):
        t = YeoJohnson().fit(a[p + 'z_fit']) if transform else None
        return DeltasEstimator(clf, bound=GaussianConfidence(),
                               confidence=FixedDelta(DELTA), rule=rule, transform=t)
    return build


def _cp(**kw):
    return lambda clf, model, a, p: DeltasEstimator(clf, **kw)


#: (experiment, slot, variant, calibration fraction, build); every E6 variant uses the
#: sum rule, so only the bound changes. The published limit has no slacks, so it solves
#: only separable cells; the k-th point limit carries the R and factor swaps.
VARIANTS = [
    ('E6', 'R', 'published, R-bar two-sided', None, _published(factor=2.0)),
    ('E6', 'R', 'published, R-bar one-sided', None,
     _published(factor=2.0, radius='one_sided')),
    ('E6', 'R', 'published, known range', None, _known_range),
    ('E6', 'R', 'k-th point, R-bar two-sided', None, _kth('two_sided')),
    ('E6', 'R', 'k-th point, R-bar one-sided', None, _kth('one_sided')),
    ('E6', 'R', 'k-th point, known range', None, _kth('known')),
    ('E6', 'R', 'counts (CP)', None, _cp(rule=Sum())),
    ('E6', 'R', 'shape (Gaussian)', None, _gaussian(Sum())),
    ('E10', 'rule', 'minimax', None, _cp()),
    ('E10', 'rule', 'sum', None, _cp(rule=Sum())),
    ('E10', 'rule', 'risk', None, _cp(rule=Risk())),
    ('E10', 'delta', 'optimised', None, _cp()),
    ('E10', 'delta', f'fixed {DELTA}', None, _cp(confidence=FixedDelta(DELTA))),
    ('E10', 'calibration', f'{ep.CAL_FRAC:g}', None, _cp()),
    *[('E10', 'calibration', f'{f:g}', f, _cp()) for f in CAL_FRACS],
    ('E10', 'transform', 'identity', None, _gaussian('minimax')),
    ('E10', 'transform', 'Yeo-Johnson', None, _gaussian('minimax', transform=True)),
    ('E10', 'margin factor', '2', None, _kth(factor=2.0)),
    ('E10', 'margin factor', '1', None, _kth(factor=1.0)),
]


def _path(dataset, model, frac=None):
    folder = PROJ if frac is None else os.path.join(CAL, f'cal{frac:g}')
    return os.path.join(folder, f'{ep._slug(dataset)}__{model}.npz')


def export_cal(dataset, model, seeds, frac):
    '''the pair's export at another calibration fraction'''
    ep.CAL_FRAC = frac                  # one_case reads the module's value
    settings = {**ep.export_settings(dataset, model), 'cal_frac': frac}
    return ep.run_pair(dataset, model, seeds, out_dir=os.path.dirname(_path(
        dataset, model, frac)), settings=settings)


def _fit_row(build, clf, base, a, p, model, timeout):
    y_cert, y_test = a[p + 'y_cert'].astype(int), a[p + 'y_test'].astype(int)
    t0 = time.time()
    try:
        est = build(clf, model, a, p)
        if est is None:
            return None
        with rw.time_budget(timeout):
            est.fit(a[p + 'z_cert'][:, None], y_cert)
        if est.is_fit:
            preds = np.asarray(est.predict(a[p + 'z_test'][:, None])).astype(int)
            rec = rw._row(base, y_test, '', preds=preds, fitted=est)
        else:
            rec = rw._row(base, y_test, '', ok=False)
    except rw._Timeout:
        rec = rw._row(base, y_test, '', ok=False, reason='timeout')
    except Exception:
        rec = rw._row(base, y_test, '', ok=False, reason='error')
    return {**rec, 'seconds': round(time.time() - t0, 3)}


def run_file(path, variants, seeds, timeout):
    '''one row per (seed, variant) of the file's split cells'''
    rows = []
    with np.load(path, allow_pickle=False) as f:
        for meta, seed, mode, p, a, _ in rw._cells(f, seeds):
            if mode != 'split':
                continue
            base, clf = rw._cell(meta, seed, mode, a, p)
            for experiment, slot, variant, _, build in variants:
                rec = _fit_row(build, clf, base, a, p, meta['model'], timeout)
                if rec is not None:
                    rows.append({**rec, 'method': f'{slot}: {variant}',
                                 'experiment': experiment, 'slot': slot,
                                 'variant': variant})
    return rows


def margins_file(path, seeds):
    '''the published limit's distance from the minority mean, over subsampled N'''
    rows = []
    with np.load(path, allow_pickle=False) as f:
        for meta, seed, mode, p, a, _ in rw._cells(f, seeds):
            if mode != 'split':
                continue
            lo, hi = SCORE_RANGE[meta['model']]
            z, y = a[p + 'z_cert'], a[p + 'y_cert'].astype(int)
            side = ProjectedData(z, y).by_label(1).side
            rng = np.random.default_rng(seed)
            for n in MARGIN_N:
                if n > (y == 1).sum():
                    continue
                s = ClassSample(rng.choice(z[y == 1], n, replace=False), 1, side)
                for variant, kw in (('R-bar, two-sided', {}),
                                    ('R-bar, one-sided', {'radius': 'one_sided'}),
                                    ('known range', {'score_range': (lo, hi)})):
                    b = PublishedToleranceLimit(2.0, **kw).fit(s)
                    conc = b.factor * b.R_conc / np.sqrt(n) * (
                        2 + np.sqrt(2 * np.log(1 / DELTA)))
                    rows.append({'dataset': meta['dataset'], 'model': meta['model'],
                                 'seed': seed, 'N': n, 'variant': variant,
                                 'R': b.R, 'concentration': conc,
                                 'distance': (b.R + conc) / (hi - lo)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['E6', 'E10'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=300)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    want = [args.only] if args.only else ['E6', 'E10']
    t0 = time.time()

    jobs = []           # (path, variants)
    if 'E6' in want:
        e6 = [v for v in VARIANTS if v[0] == 'E6']
        jobs += [(_path(d, m), e6) for d in E6_TASKS for m in E6_MODELS
                 if os.path.exists(_path(d, m))]
    if 'E10' in want:
        pairs = [(d, m) for d in E10_TASKS for m in E10_MODELS]
        exported = Parallel(n_jobs=args.jobs)(
            delayed(export_cal)(d, m, seeds, f) for d, m in pairs for f in CAL_FRACS)
        for (path, status, err) in exported:
            if status in ('failed', 'stale') or err:
                print(f'  {os.path.basename(path)}: {status} {err}', flush=True)
        for frac in (None, *CAL_FRACS):
            e10 = [v for v in VARIANTS if v[0] == 'E10' and v[3] == frac]
            jobs += [(_path(d, m, frac), e10) for d, m in pairs
                     if os.path.exists(_path(d, m, frac))]
    print(f'{len(jobs)} files, {len(seeds)} seeds ({time.time() - t0:.0f}s)', flush=True)

    chunks = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_file)(path, variants, seeds, args.timeout) for path, variants in jobs)
    df = pd.DataFrame([r for c in chunks for r in c])
    out = os.path.join(rw.RESULTS, 'slots.csv')
    if args.only and os.path.exists(out):
        old = pd.read_csv(out, low_memory=False)
        df = pd.concat([old[old.experiment != args.only], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f'wrote {out} ({len(df)} rows, {time.time() - t0:.0f}s)', flush=True)

    if 'E6' in want:
        bounded = [_path(d, m) for d in E6_TASKS for m in SCORE_RANGE
                   if os.path.exists(_path(d, m))]
        m = pd.DataFrame([r for f in bounded for r in margins_file(f, seeds)])
        m.to_csv(os.path.join(rw.RESULTS, 'margins.csv'), index=False)
        print(f'wrote margins.csv ({len(m)} rows)', flush=True)
    print(df.groupby(['experiment', 'slot', 'variant']).agg(
        solved=('fit', 'mean'), gmean=('G-Mean', 'mean')).round(3).to_string())


if __name__ == '__main__':
    main()
