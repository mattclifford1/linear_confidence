'''
Paired significance tests over the per-seed raw results.

The old runners threw the per-seed numbers away, so no test was possible after
the fact; `run_experiments.py` keeps them in results/raw/*.csv.

Two views:
  * per dataset, a paired Wilcoxon signed-rank test of one method against a
    reference (default Thresholding, the strongest post-hoc baseline)
  * across datasets, average ranks (the input to a Friedman/Nemenyi analysis)

Only seeds where BOTH methods produced a solution are paired, and the count is
reported - a method that solves 4/10 is not silently compared on 4 points as if
it were 10.
'''
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'results', 'raw')


def load():
    frames = {}
    for path in sorted(glob.glob(os.path.join(RAW, '*.csv'))):
        name = os.path.basename(path)[:-4]
        frames[name] = pd.read_csv(path)
    return frames


def pairwise(metric='G-Mean', reference='Threshold'):
    rows = []
    for name, df in load().items():
        piv = df.pivot_table(index='seed', columns='method', values=metric)
        if reference not in piv:
            continue
        for method in piv.columns:
            if method == reference:
                continue
            ok = piv[method].notna() & piv[reference].notna()
            n = int(ok.sum())
            if n < 6:
                rows.append(dict(dataset=name, method=method, n=n,
                                 mean=piv[method][ok].mean(),
                                 ref=piv[reference][ok].mean(), p=np.nan))
                continue
            diff = piv[method][ok] - piv[reference][ok]
            if np.allclose(diff, 0):
                p = 1.0
            else:
                p = wilcoxon(piv[method][ok], piv[reference][ok]).pvalue
            rows.append(dict(dataset=name, method=method, n=n,
                             mean=piv[method][ok].mean(),
                             ref=piv[reference][ok].mean(), p=p))
    out = pd.DataFrame(rows)
    out['delta'] = out['mean'] - out['ref']
    out['sig'] = np.where(out.p.isna(), '', np.where(
        out.p < 0.01, '**', np.where(out.p < 0.05, '*', '')))
    return out


def ranks(metric='G-Mean'):
    '''average rank per method across datasets (NaN -> worst rank)'''
    per = {}
    for name, df in load().items():
        agg = df.groupby('method')[metric].mean()
        per[name] = agg
    tab = pd.DataFrame(per)
    r = tab.rank(ascending=False, na_option='bottom')
    out = pd.DataFrame({'avg rank': r.mean(axis=1),
                        f'mean {metric}': tab.mean(axis=1)})
    return out.sort_values('avg rank'), tab


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric', default='G-Mean')
    ap.add_argument('--reference', default='Threshold')
    args = ap.parse_args()

    pw = pairwise(args.metric, args.reference)
    print(f'=== paired Wilcoxon vs {args.reference} ({args.metric}) ===')
    print(pw[['dataset', 'method', 'n', 'mean', 'ref', 'delta', 'p', 'sig']]
          .round(4).to_string(index=False))

    rk, tab = ranks(args.metric)
    print(f'\n=== average rank across datasets ({args.metric}) ===')
    print(rk.round(3).to_string())

    clean = tab.dropna()
    if len(clean.columns) >= 3 and len(clean) >= 3:
        stat, p = friedmanchisquare(*[clean.loc[m] for m in clean.index])
        print(f'\nFriedman over {len(clean)} methods x {len(clean.columns)} '
              f'datasets (methods solving everywhere): chi2={stat:.2f} p={p:.4f}')
