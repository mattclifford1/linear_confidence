'''
Analyse the wide grid produced by run_wide.py.

    python analyse_wide.py              # everything, to stdout + results/

Produces
  * coverage of the reported certificate, naive vs calibration split
  * coverage broken down by classifier optimism (the proposed mechanism)
  * solve rates per method (the "never fails" claim)
  * average ranks by G-Mean, and the cost of calibration in accuracy terms
  * LaTeX fragments for the draft
'''
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

#: methods that report a certificate, so coverage is defined for them
CERT_METHODS = ['CP Sum', 'CP Minimax', 'DKW Sum', 'DKW Minimax']
#: order used in the tables
METHOD_ORDER = ['Baseline', 'SMOTE', 'Balanced Weights', 'Threshold',
                'Slacks Deltas', 'Min Deltas', 'F Deltas',
                'CP Sum', 'CP Minimax', 'DKW Sum', 'DKW Minimax']


def test_sizes():
    '''
    minority test-set size per dataset, read back from the exported metadata

    Coverage is *measured* against the test set, so a cell with 10 test points
    per class can only resolve the error to 0.1 and its coverage number is
    mostly noise. This lets the analysis be restricted to cells where the
    measurement is meaningful.
    '''
    import glob
    import json
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, 'projections', '*.npz'))):
        try:
            with np.load(p, allow_pickle=False) as f:
                m = json.loads(str(f['meta'][0]))
            rows.append({'dataset': m['dataset'],
                         'test_per_class': min(m['test_counts'])})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=['dataset', 'test_per_class'])
    return (pd.DataFrame(rows).drop_duplicates('dataset')
            .reset_index(drop=True))


def load(path=None):
    df = pd.read_csv(path or os.path.join(RESULTS, 'wide.csv'))
    for col in ('covered_0', 'covered_1'):
        df[col] = df[col].map({True: 1.0, False: 0.0, 'True': 1.0,
                               'False': 0.0}).astype(float)
    sizes = test_sizes()
    if len(sizes):
        df = df.merge(sizes, on='dataset', how='left')
    return df


# ------------------------------------------------------------- coverage ---
def coverage_long(df):
    '''one row per (…, class): did the certificate hold?'''
    cert = df[df['method'].isin(CERT_METHODS) & df['fit']].copy()
    carry = ['dataset', 'model', 'seed', 'mode', 'method', 'optimism',
             'n_cert', 'n_cert_minority']
    if 'test_per_class' in cert.columns:
        carry.append('test_per_class')
    parts = []
    for cls, cov, ub, dl in ((0, 'covered_0', 'U_0', 'delta_0'),
                             (1, 'covered_1', 'U_1', 'delta_1')):
        p = cert[carry].copy()
        p['cls'] = cls
        p['covered'] = cert[cov].values
        p['U'] = cert[ub].values
        p['delta'] = cert[dl].values
        parts.append(p)
    out = pd.concat(parts, ignore_index=True).dropna(subset=['covered'])
    out['nominal'] = 1.0 - out['delta']
    return out


def coverage_summary(cov):
    g = cov.groupby('mode').agg(
        coverage=('covered', 'mean'), nominal=('nominal', 'mean'),
        mean_U=('U', 'mean'), n=('covered', 'size'))
    return g


def coverage_by_optimism(cov, bins=(-1, 0.02, 0.05, 0.10, 0.20, 1.0)):
    '''the mechanism check: does the shortfall track classifier optimism?'''
    c = cov.copy()
    c['bucket'] = pd.cut(c['optimism'], bins=list(bins))
    g = c.groupby(['bucket', 'mode'], observed=True).agg(
        coverage=('covered', 'mean'), nominal=('nominal', 'mean'),
        n=('covered', 'size'))
    return g.unstack('mode')


def coverage_by(cov, key):
    g = cov.groupby([key, 'mode']).agg(coverage=('covered', 'mean'))
    return g.unstack('mode')['coverage']


# ------------------------------------------------------- solve / ranking ---
def solve_rates(df):
    g = df.groupby(['method', 'mode'])['fit'].mean().unstack('mode')
    order = [m for m in METHOD_ORDER if m in g.index]
    return g.loc[order]


def ranks(df, metric='G-Mean', mode='naive'):
    '''average rank across (dataset, model) cells; unsolved ranks last'''
    sub = df[df['mode'] == mode]
    cell = sub.groupby(['dataset', 'model', 'method'])[metric].mean()
    tab = cell.unstack(['dataset', 'model'])
    r = tab.rank(ascending=False, na_option='bottom')
    out = pd.DataFrame({'avg rank': r.mean(axis=1),
                        f'mean {metric}': tab.mean(axis=1),
                        'cells solved': tab.notna().sum(axis=1),
                        'cells': tab.shape[1]})
    return out.sort_values('avg rank')


def calibration_cost(df, metric='G-Mean'):
    '''what does the split cost in performance, per method?'''
    piv = df[df['fit']].groupby(['method', 'mode'])[metric].mean().unstack()
    if 'split' not in piv or 'naive' not in piv:
        return piv
    piv['delta'] = piv['split'] - piv['naive']
    order = [m for m in METHOD_ORDER if m in piv.index]
    return piv.loc[order]


# ----------------------------------------------------------------- latex ---
def _num(x, places=3):
    '''.812 rather than 0.812, the convention used in the draft's tables'''
    return f'{x:.{places}f}'.lstrip('0')


def latex_coverage(cov):
    s = coverage_summary(cov)
    label = {'naive': 'the training data',
             'split': 'a held-out calibration split'}
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Certificate computed on & coverage & nominal & '
             r'mean $U_i$ & $n$ \\',
             r'\midrule']
    for mode in ('naive', 'split'):
        if mode not in s.index:
            continue
        r = s.loc[mode]
        cov_s = _num(r['coverage'])
        if r['coverage'] < r['nominal'] - 0.01:      # a real violation
            cov_s = f'\\mathbf{{{cov_s}}}'
        lines.append(f"{label[mode]} & ${cov_s}$ & ${_num(r['nominal'])}$ & "
                     f"${_num(r['mean_U'])}$ & {int(r['n'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def main():
    df = load()
    cov = coverage_long(df)

    print('=' * 72)
    print(f'wide grid: {df["dataset"].nunique()} datasets x '
          f'{df["model"].nunique()} models x {df["seed"].nunique()} seeds')
    print(f'{len(df)} rows, {len(cov)} certificate observations')

    print('\n=== certificate coverage ===')
    print(coverage_summary(cov).round(3).to_string())

    print('\n=== coverage by classifier optimism ===')
    print(coverage_by_optimism(cov).round(3).to_string())

    if 'test_per_class' in cov.columns:
        big = cov[cov['test_per_class'] >= 50]
        print(f'\n=== coverage, cells with >=50 test points per class '
              f'({big["dataset"].nunique()} of {cov["dataset"].nunique()} '
              f'datasets) ===')
        print(coverage_summary(big).round(3).to_string())

    print('\n=== coverage by model ===')
    print(coverage_by(cov, 'model').round(3).to_string())

    print('\n=== worst 12 datasets by naive coverage ===')
    by_ds = coverage_by(cov, 'dataset')
    if 'naive' in by_ds:
        print(by_ds.sort_values('naive').head(12).round(3).to_string())

    print('\n=== solve rate ===')
    print(solve_rates(df).round(3).to_string())

    print('\n=== average rank (G-Mean, naive) ===')
    print(ranks(df).round(3).to_string())

    print('\n=== average rank (G-Mean, split) ===')
    print(ranks(df, mode='split').round(3).to_string())

    print('\n=== cost of calibration (G-Mean) ===')
    print(calibration_cost(df).round(3).to_string())

    os.makedirs(RESULTS, exist_ok=True)
    cov.to_csv(os.path.join(RESULTS, 'wide_coverage.csv'), index=False)
    with open(os.path.join(RESULTS, 'wide_coverage.txt'), 'w') as f:
        f.write(latex_coverage(cov))
    ranks(df).to_csv(os.path.join(RESULTS, 'wide_ranks.csv'))
    solve_rates(df).to_csv(os.path.join(RESULTS, 'wide_solve.csv'))
    print(f'\nwrote {RESULTS}/wide_coverage.csv, wide_ranks.csv, wide_solve.csv')


if __name__ == '__main__':
    main()
