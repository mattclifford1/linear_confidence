'''
Tables and figures for the extra-experiments paper (../papers/ecai-extra-experiments),
from the results/ CSVs of run_wide.py, run_costs.py, simulate_truth.py and
run_slots.py; F2 also reads the synthetic exports for the test-set oracle.

    uv run python make_extra_figures.py

Writes tables/*.tex and figs/*.pdf into the paper (override with EXTRA_PAPER_OUT).
Every mean is over the cells the method solved; split certificates unless a table
says otherwise.
'''
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extra_grid as eg
import run_slots as rs
import run_wide as rw
from deltas.methods import METHODS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
OUT = os.environ.get('EXTRA_PAPER_OUT', os.path.normpath(os.path.join(
    HERE, '..', '..', 'papers', 'ecai-extra-experiments')))

GROUPS = {'tabular': 'Tabular', 'image': 'Pixels', 'frozen': 'Frozen', 'deep': 'Deep',
          'nsweep': '$N$-sweep', 'gauss': 'Gauss.', 'synth': 'Synth.'}
CERT = ['CP Sum', 'CP Minimax', 'DKW Sum', 'DKW Minimax', 'Gaussian Predictive',
        'Gaussian Band', 'Logistic Band']
PUBLISHED = ['Slacks Deltas', 'Min Deltas', 'F Deltas']
BASE = ['Baseline', 'Threshold', 'SMOTE', 'Balanced Weights', 'Focal', 'LDAM']
DEEP = ['ConvNet-prob', 'ConvNet-logit', 'ResNet-prob', 'ResNet-logit']

C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'violet': '#4a3aa7'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2.0, 'legend.frameon': False,
})


def group(dataset, model):
    if model in DEEP:
        return 'deep'
    if '[' in dataset:
        return 'frozen'
    if dataset.startswith('Gaussian sep'):
        return 'gauss'
    if ' @n' in dataset and dataset.split(' @')[0] in eg.SYNTHETIC_SWEEP:
        return 'synth'
    if ' @n' in dataset:
        return 'nsweep'
    return 'image' if dataset in eg.IMAGE else 'tabular'


def load():
    d = pd.read_csv(os.path.join(RESULTS, 'extra.csv'), low_memory=False)
    d['group'] = [group(a, b) for a, b in zip(d.dataset, d.model)]
    d['ok'] = d.fit.astype(str) == 'True'
    cov = [pd.to_numeric(d[f'covered_{i}'], errors='coerce') for i in (0, 1)]
    d['covered'] = cov[0] * cov[1]
    d['nominal'] = (1 - d.delta_0) * (1 - d.delta_1)
    d['N1'] = d.dataset.str.extract(r'@n(\d+)', expand=False).astype(float)
    return d


def f2(x):
    return '--' if pd.isna(x) else f'{x:.2f}'


def pct(x):
    return '--' if pd.isna(x) else f'{100 * x:.0f}'


def tabular(cols, header, rows, mids=()):
    '''booktabs table; `mids` are row indices that start a new block'''
    lines = [rf'\begin{{tabular}}{{@{{}}{cols}@{{}}}}', r'\toprule', header + r' \\',
             r'\midrule']
    for i, r in enumerate(rows):
        if i in mids:
            lines.append(r'\midrule')
        lines.append(' & '.join(r) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def write(name, text):
    os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
    with open(os.path.join(OUT, 'tables', name + '.tex'), 'w') as fh:
        fh.write(text)


def bold_best(values, best):
    '''bold every value that rounds to `best`'''
    return [rf'\textbf{{{f2(v)}}}' if f2(v) == f2(best) else f2(v) for v in values]


def table_gmean(split):
    '''T2: mean G-Mean per method and group, and how often CP Minimax beats the baselines'''
    ok = split[split.ok]
    means = ok.pivot_table(index='method', columns='group', values='G-Mean')
    cols = [g for g in GROUPS if g in means]
    rows = []
    for g_methods in (CERT, BASE):
        block = means.reindex(g_methods)[cols]
        for m in g_methods:
            rows.append([m] + [f2(block.loc[m, g]) for g in cols])
    best = means.reindex(CERT + BASE)[cols].max()
    for r in rows:
        r[1:] = [rf'\textbf{{{v}}}' if v == f2(best[g]) else v
                 for v, g in zip(r[1:], cols)]
    cell = ok.pivot_table(index=['group', 'dataset', 'model', 'seed'], columns='method',
                          values='G-Mean')
    top = cell[[b for b in BASE if b in cell]].max(axis=1)
    wins = (cell['CP Minimax'] >= top - 1e-12).groupby('group').mean()
    rows.append([r'CP Minimax $\geq$ best, \% cells']
                + [pct(wins.get(g)) for g in cols])
    write('t2_gmean', tabular('l' + 'r' * len(cols),
                              'G-Mean & ' + ' & '.join(GROUPS[g] for g in cols),
                              rows, mids=(len(CERT), len(CERT) + len(BASE))))


def table_coverage(d):
    '''T3: both class bounds held, split vs naive, with the nominal level'''
    ok = d[d.ok & d.method.isin(CERT)]
    rows = []
    for mode in ('split', 'naive'):
        x = ok[ok['mode'] == mode]
        cov = x.pivot_table(index='method', columns='group', values='covered')
        nom = x.groupby('method').nominal.mean()
        cols = [g for g in GROUPS if g in cov]
        for m in CERT:
            rows.append([f'{m}, {mode}'] + [f2(cov.loc[m, g]) for g in cols]
                        + [f2(nom.get(m))])
    write('t3_coverage', tabular('l' + 'r' * (len(cols) + 1),
                                 'Coverage & ' + ' & '.join(GROUPS[g] for g in cols)
                                 + ' & nominal', rows, mids=(len(CERT),)))


def table_published(split):
    '''T4: how often the published and non-separable methods solve, and what they give'''
    rate = split.pivot_table(index='method', columns='group', values='ok')
    cols = [g for g in GROUPS if g in rate]
    cell = split[split.ok].pivot_table(index=['dataset', 'model', 'seed'],
                                       columns='method', values='G-Mean')
    secs = split[split.ok].groupby('method').seconds.median()
    rows = []
    for m in PUBLISHED + ['CP Minimax']:
        both = cell[list({m, 'CP Minimax'})].dropna()
        rows.append([m] + [pct(rate.loc[m, g]) for g in cols]
                    + [f2(both[m].mean()), f2(both['CP Minimax'].mean()),
                       f'{secs.get(m, np.nan):.2f}'])
    write('t4_solved', tabular('l' + 'r' * (len(cols) + 3),
                               r'Solved, \% & ' + ' & '.join(GROUPS[g] for g in cols)
                               + r' & G-Mean & CP Min. & med.\ s', rows,
                               mids=(len(PUBLISHED),)))


def table_simulation():
    '''T5: the furthest-point bound on average and with high probability; certificate validity'''
    s = pd.read_csv(os.path.join(RESULTS, 'simulate_truth.csv'))
    shapes = ['normal', 'skewed', 'heavy', 'two subgroups', 'bounded']
    fp = s[s.bound == 'furthest point'].assign(r=lambda x: x.mean_err * (x.N + 1))
    rows = [[r'Furthest point: $E[e]\,(N+1)$']
            + [f'{fp[fp["shape"] == sh].r.min():.2f}--{fp[fp["shape"] == sh].r.max():.2f}'
               for sh in shapes],
            [r'Furthest point: $P(e > 1/(N+1))$']
            + [f'{fp[fp["shape"] == sh].p_exceed.min():.2f}--'
               f'{fp[fp["shape"] == sh].p_exceed.max():.2f}' for sh in shapes]]
    pub = s[s.bound.isin(['published (eps)', 'published (2 eps)'])]
    for b, label in (('published (eps)', r'$\epsilon$'), ('published (2 eps)', r'$2\epsilon$')):
        x = pub[pub.bound == b]
        rows.append([rf'Published limit at {label}: max $P(e > 1/(N+1)+\delta)$']
                    + [f'{x[x["shape"] == sh].p_exceed.max():.3f}' for sh in shapes])
    cert = s[s.bound.isin(['Clopper-Pearson', 'DKW', 'Gaussian band'])].copy()
    tol = 2 * np.sqrt(cert.delta * (1 - cert.delta) / cert.reps)
    cert['valid'] = cert.p_exceed <= cert.delta + tol
    for b in ['Clopper-Pearson', 'DKW', 'Gaussian band']:
        x = cert[cert.bound == b]
        rows.append([rf'{b}: $P(e > U) \leq \delta$, \% of settings']
                    + [pct(x[x['shape'] == sh].valid.mean()) for sh in shapes])
    write('t5_simulation', tabular('l' + 'r' * len(shapes),
                                   ' & ' + ' & '.join(shapes), rows, mids=(2, 4)))


def table_gaussian(split):
    '''E2: the Gaussian grid by minority size'''
    x = split[(split.group == 'gauss') & split.ok]
    g = x.pivot_table(index='N1', columns='method', values='G-Mean')
    c = x.pivot_table(index='N1', columns='method', values='covered')
    rows = [[f'{int(n)}'] + [f2(g.loc[n, m]) for m in ('CP Minimax', 'Threshold', 'Baseline')]
            + [f2(c.loc[n, m]) for m in ('CP Minimax', 'Gaussian Predictive')]
            for n in g.index]
    write('e2_gaussian', tabular('r' * 6, r' & \multicolumn{3}{c}{G-Mean} & '
                                 r'\multicolumn{2}{c}{coverage} \\ \cmidrule(lr){2-4}'
                                 r'\cmidrule(l){5-6} $N_1$ & CP Minimax & Threshold & '
                                 r'Baseline & CP Minimax & Gaussian Pred.', rows))


def table_deep(d):
    '''T7: trained nets in both score spaces, and the frozen-feature heads'''
    x = d[d.group.isin(['deep', 'frozen']) & d.ok].copy()
    x['row'] = np.where(x.group == 'deep', x.model,
                        np.where(x.dataset.str.contains(r'\[resnet18\]'),
                                 'ResNet-18 features', 'ConvNeXt-T features'))
    sp = x[x['mode'] == 'split']
    g = sp.pivot_table(index='row', columns='method', values='G-Mean')
    cs = sp.pivot_table(index='row', columns='method', values='covered')
    cn = x[x['mode'] == 'naive'].pivot_table(index='row', columns='method', values='covered')
    order = DEEP + ['ResNet-18 features', 'ConvNeXt-T features']
    show = ['CP Minimax', 'Baseline', 'Threshold', 'Balanced Weights', 'Focal', 'LDAM']
    rows = [[r] + bold_best(g.loc[r].reindex(show),
                            g.loc[r].reindex(show).max())
            + [f2(cs.loc[r, 'CP Minimax']), f2(cn.loc[r, 'CP Minimax'])] for r in order]
    write('t7_deep', tabular('l' + 'r' * 8, r' & \multicolumn{6}{c}{G-Mean} & '
                             r'\multicolumn{2}{c}{CP Minimax coverage} \\ '
                             r'\cmidrule(lr){2-7}\cmidrule(l){8-9} & CP Minimax & Baseline '
                             r'& Threshold & BW & Focal & LDAM '
                             r'& split & naive', rows, mids=(len(DEEP),)))


def table_limits(split):
    '''T8: each stress beside its control'''
    x = split[split.ok & (split.group != 'deep')]
    mixed = ['MNIST: 3+5 vs rest', 'Fashion-MNIST: Sandal+Sneaker+Ankle boot vs rest',
             'CIFAR-100: fish vs rest', 'CIFAR-100: vehicles 1 vs rest',
             'CIFAR-100: large carnivores vs large omnivores and herbivores']
    base = x.dataset.str.replace(r' \[[^\]]+\]', '', regex=True)
    healthy = [v[:-len(' (disease-positive)')] for v in eg.DISEASE_POSITIVE]
    pairs = ['cat vs dog', 'automobile vs truck']
    sets = [
        ('Mixed minority', base.isin(mixed)),
        ('Single-class image tasks', (x.group.isin(['image', 'frozen']) & ~base.isin(mixed)
                                      & ~base.str.contains('@test|except|CIFAR-10N'))),
        ('Label noise, aggregate', base.str.startswith('CIFAR-10N aggre')),
        ('Label noise, worst', base.str.startswith('CIFAR-10N worse')),
        ('Clean labels, same pairs', base.isin([f'CIFAR-10: {p}' for p in pairs])),
        ('Shifted test minority', base.str.contains('@test')),
        ('Same tasks, no shift', base.str.contains('except') & ~base.str.contains('@test')),
        ('Disease-positive minority', base.isin(eg.DISEASE_POSITIVE)),
        ('Healthy minority (default)', base.isin(healthy)),
    ]
    rows = []
    for label, mask in sets:
        y = x[mask]
        cell = y.pivot_table(index=['dataset', 'model', 'seed'], columns='method',
                             values='G-Mean')
        top = cell[[b for b in BASE if b in cell]].max(axis=1).mean()
        cov = y.groupby('method').covered.mean()
        rows.append([label, f2(cov.get('CP Minimax')), f2(cov.get('Gaussian Predictive')),
                     f2(cell['CP Minimax'].mean()), f2(top)])
    write('t8_limits', tabular('lrrrr', r' & \multicolumn{2}{c}{coverage} & '
                               r'\multicolumn{2}{c}{G-Mean} \\ \cmidrule(lr){2-3}'
                               r'\cmidrule(l){4-5} & CP Minimax & Gaussian Pred. '
                               r'& CP Minimax & best baseline', rows, mids=(2, 5, 7)))


def table_costs():
    '''T9: total test cost relative to the test-set oracle'''
    c = pd.read_csv(os.path.join(RESULTS, 'costs-extra.csv'))
    c = c[c['mode'] == 'split']
    oracle = c[c.method == 'Oracle (test)'].set_index(['dataset', 'model', 'seed']).cost
    c = c[c.method != 'Oracle (test)'].join(oracle.rename('oracle'),
                                            on=['dataset', 'model', 'seed'])
    c['ratio'] = c.cost / c.oracle
    t = c.pivot_table(index='method', columns='dataset', values='ratio', aggfunc='median')
    names = {'Costcla Credit Scoring Kaggle 2011': 'Kaggle 2011',
             'Costcla Credit Scoring PAKDD 2009': 'PAKDD 2009',
             'Costcla Direct Marketing': 'Direct Marketing', 'German Credit': 'German Credit'}
    cols = list(names)
    order = ['Baseline', 'Threshold (cost)'] + [
        f'{b} {r} ({k})' for k in ('uncosted', 'costed', 'prior-costed')
        for b in ('CP', 'DKW') for r in ('sum', 'minimax')]
    best = t.reindex(order)[cols].min()
    rows = [[m] + [bold_best([t.loc[m, k]], best[k])[0] for k in cols] for m in order]
    write('t9_costs', tabular('l' + 'r' * len(cols),
                              'Cost / oracle cost & ' + ' & '.join(names[k] for k in cols),
                              rows, mids=(2, 6, 10)))


def fig_nsweep(d):
    '''F3: certified minority bound and coverage against the minority training size'''
    x = d[(d.group == 'nsweep') & d.ok]
    sp = x[x['mode'] == 'split']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.7))
    for m, col in (('CP Minimax', C['blue']), ('DKW Minimax', C['aqua']),
                   ('Gaussian Predictive', C['orange'])):
        y = sp[sp.method == m].groupby('N1')
        a.plot(y.U_1.mean().index, y.U_1.mean(), color=col, label=m)
    true = sp[sp.method == 'CP Minimax'].groupby('N1').err_1.mean()
    a.plot(true.index, true, color=INK2, ls='--', lw=1.2, label='true error (CP Minimax)')
    a.set(xscale='log', xlabel='$N_1$ (minority training points)',
          ylabel='minority error bound $U_1$', ylim=(0, None))
    a.legend()
    for mode, ls in (('split', '-'), ('naive', ':')):
        for m, col in (('CP Minimax', C['blue']), ('Gaussian Predictive', C['orange'])):
            y = x[(x['mode'] == mode) & (x.method == m)].groupby('N1').covered.mean()
            b.plot(y.index, y, color=col, ls=ls, label=f'{m}, {mode}')
    b.set(xscale='log', xlabel='$N_1$', ylabel='coverage (both classes)', ylim=(0, 1.02))
    b.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    _savefig(fig, 'f3_nsweep')


def _savefig(fig, name):
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
    fig.savefig(os.path.join(OUT, 'figs', name + '.pdf'))
    plt.close(fig)


def fig_simulation():
    """F1: the true error at each limit against what the published loss promises (delta = 0.05)"""
    s = pd.read_csv(os.path.join(RESULTS, 'simulate_truth.csv'))
    s = s[s.delta == 0.05]
    shapes = ['normal', 'skewed', 'heavy', 'two subgroups', 'bounded']
    fig, axes = plt.subplots(1, len(shapes), figsize=(7.6, 2.3), sharey=True)
    for ax, sh in zip(axes, shapes):
        x = s[s['shape'] == sh]
        fp_ = x[x.bound == 'furthest point'].sort_values('N')
        ax.plot(fp_.N, fp_.mean_err, color=C['blue'], label='furthest point, mean')
        ax.plot(fp_.N, fp_.p95_err, color=C['blue'], ls='--', lw=1.2,
                label='furthest point, 95th pct.')
        for b, col in (('published (eps)', C['orange']), ('published (2 eps)', C['aqua'])):
            y = x[x.bound == b].sort_values('N')
            ax.plot(y.N, y.p95_err, color=col, lw=1.2,
                    label=b.replace('eps', r'$\epsilon$') + ', 95th pct.')
        ax.plot(fp_.N, 1 / (fp_.N + 1) + 0.05, color=INK, lw=1, ls=':',
                label=r'promised $1/(N+1)+\delta$')
        ax.set(xscale='log', yscale='log', title=sh, xlabel='$N$')
    axes[0].set_ylabel('true error $e$')
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    _savefig(fig, 'f1_simulation')


def _oracle(dataset, model):
    """balanced error of the best test-set threshold, per split seed"""
    path = os.path.join(rs.PROJ, f'{rs.ep._slug(dataset)}__{model}.npz')
    out = {}
    with np.load(path, allow_pickle=False) as f:
        for _, seed, mode, p, a, _ in rw._cells(f, range(10)):
            if mode == 'split':
                z, y = a[p + 'z_test'], a[p + 'y_test'].astype(int)
                t = rw._best_balanced_threshold(z, y)
                out[seed] = 0.5 * ((z[y == 0] > t).mean() + (z[y == 1] <= t).mean())
    return out


def _excess(split, mask):
    """rows of `split` under `mask` with their balanced error above the test-set oracle"""
    x = split[mask & split.ok].copy()
    oracle = {(d, m, s): v for d, m in x[['dataset', 'model']].drop_duplicates().values
              for s, v in _oracle(d, m).items()}
    x['excess'] = 0.5 * (x.err_0 + x.err_1) - [oracle.get(k, np.nan) for k in
                                               zip(x.dataset, x.model, x.seed)]
    return x


def fig_synthetic(split):
    """F2: excess balanced error over the best test-set threshold on the Gaussian grid"""
    x = _excess(split, split.group == 'gauss')
    x['sep'] = x.dataset.str.extract(r'sep(\d+)', expand=False).astype(int)
    seps = sorted(x.sep.unique())
    fig, axes = plt.subplots(1, len(seps), figsize=(7.6, 2.4), sharey=True)
    for ax, sep in zip(axes, seps):
        y = x[x.sep == sep]
        for m, col, ls in (('Baseline', INK2, ':'), ('Threshold', INK2, '-'),
                           ('SMOTE', C['violet'], '-'), ('CP Minimax', C['blue'], '-'),
                           ('Gaussian Predictive', C['orange'], '-')):
            e = y[y.method == m].groupby('N1').excess.mean()
            ax.plot(e.index, e, color=col, ls=ls, label=m)
        ax.set(xscale='log', title=f'separation {sep}', xlabel='$N_1$')
    axes[0].set_ylabel('excess balanced error')
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    _savefig(fig, 'f2_gaussian')


def table_synthetic(split):
    """E2: the other synthetic sets at N1 = 10 and 100"""
    names = {d: d.replace(' (20k)', '') for d in eg.SYNTHETIC_SWEEP}
    base = split.dataset.str.replace(r' @n\d+$', '', regex=True)
    x = _excess(split, base.isin(list(names)))
    x['base'] = x.dataset.str.replace(r' @n\d+$', '', regex=True)
    e = x.pivot_table(index=['base', 'N1'], columns='method', values='excess')
    c = x.pivot_table(index=['base', 'N1'], columns='method', values='covered')
    show = ['CP Minimax', 'Threshold', 'Gaussian Predictive']
    rows = [[names[b]] + [f2(e.loc[(b, n), m]) for n in (10, 100) for m in show]
            + [f2(c.loc[(b, n), 'CP Minimax']) for n in (10, 100)]
            for b in names if (b, 10) in e.index]
    write('e2_synthetic', tabular('l' + 'r' * 8,
                                  r' & \multicolumn{3}{c}{excess error, $N_1=10$} & '
                                  r'\multicolumn{3}{c}{excess error, $N_1=100$} & '
                                  r'\multicolumn{2}{c}{CP coverage} \\ '
                                  r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9} '
                                  r'& CP Min. & Thresh. & G. Pred. & CP Min. & Thresh. '
                                  r'& G. Pred. & 10 & 100', rows))


def _slots(experiment):
    s = pd.read_csv(os.path.join(RESULTS, 'slots.csv'), low_memory=False)
    s = s[s.experiment == experiment].copy()
    s['ok'] = s.fit.astype(str) == 'True'
    s['covered'] = (pd.to_numeric(s.covered_0, errors='coerce')
                    * pd.to_numeric(s.covered_1, errors='coerce'))
    return s


def table_r(split):
    """T6: the choice of R, bounded against unbounded scores, beside the published method"""
    s = _slots('E6')
    files = s[['dataset', 'model']].drop_duplicates()
    main = split.merge(files, on=['dataset', 'model'])
    main = main[main.method == 'Slacks Deltas'].assign(
        variant='published with slacks (main run)')
    x = pd.concat([s, main], ignore_index=True)
    x['bounded'] = x.model.isin(list(rs.SCORE_RANGE))
    cp = x[(x.variant == 'counts (CP)') & x.ok].set_index(
        ['dataset', 'model', 'seed'])['G-Mean']
    order = [v for _, slot, v, _, _ in rs.VARIANTS if slot == 'R'] + [
        'published with slacks (main run)']
    rows = []
    for v in order:
        y = x[x.variant == v]
        solved = y.groupby('bounded').ok.mean()
        ok = y[y.ok]
        same = cp.reindex(list(zip(ok.dataset, ok.model, ok.seed)))
        rows.append([v, pct(solved.get(True)), pct(solved.get(False)),
                     f2(ok.covered.mean()), f2(ok['G-Mean'].mean()), f2(same.mean())])
    write('t6_r', tabular('lrrrrr', r' & \multicolumn{2}{c}{solved, \%} & & '
                          r'\multicolumn{2}{c}{G-Mean} \\ \cmidrule(lr){2-3}'
                          r'\cmidrule(l){5-6} $R$ & bounded & unbounded & coverage & '
                          r'own & CP, same cells', rows, mids=(3, 6, 8)))


def fig_margins():
    """F4: the published limit's distance from the minority mean, as a share of the score range"""
    m = pd.read_csv(os.path.join(RESULTS, 'margins.csv'))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for v, col in (('R-bar, two-sided', C['blue']), ('R-bar, one-sided', C['aqua']),
                   ('known range', C['orange'])):
        y = m[m.variant == v].groupby('N').distance
        ax.plot(y.median().index, y.median(), color=col, label=v)
        ax.fill_between(y.median().index, y.quantile(0.25), y.quantile(0.75),
                        color=col, alpha=0.15, lw=0)
    ax.axhline(1, color=INK2, lw=1, ls=':')
    ax.set(xscale='log', ylim=(0, None), xlabel='$N_1$ (minority points subsampled)',
           ylabel='limit distance / score range')
    ax.legend()
    _savefig(fig, 'f4_margins')


def table_ablations():
    """T10: one slot swapped at a time; metrics over the cells every value of the slot solved"""
    s = _slots('E10')
    key = ['slot', 'dataset', 'model', 'seed']
    n_values = s.groupby('slot').variant.nunique()

    def common(x):
        n = x.groupby(key).variant.nunique().reset_index()
        return x.merge(n[n.variant == n.slot.map(n_values)][key], on=key)
    s = common(s)
    solved = common(s[s.ok])
    rows, mids, last = [], [], None
    for _, slot, v, _, _ in [r for r in rs.VARIANTS if r[0] == 'E10']:
        if slot != last:
            mids.append(len(rows))
            last = slot
        y = s[(s.slot == slot) & (s.variant == v)]
        ok = solved[(solved.slot == slot) & (solved.variant == v)]
        g = ok.groupby('model')['G-Mean'].mean()
        rows.append([f'{slot}: {v}', pct(y.ok.mean()), f2(g.get('Linear')),
                     f2(g.get('HistGradientBoosting')), f2(ok.covered.mean()),
                     f2(ok.U_1.mean()), f2(ok.err_1.mean())])
    write('t10_ablations', tabular('lrrrrrr', r' & & \multicolumn{2}{c}{G-Mean} & & '
                                   r'\multicolumn{2}{c}{minority} \\ \cmidrule(lr){3-4}'
                                   r'\cmidrule(l){6-7} Slot: value & solved, \% & Linear & '
                                   r'HGB & coverage & $U_1$ & error', rows,
                                   mids=tuple(mids[1:])))


def fig_mixed():
    """F5: a mixed minority's scores, with the Gaussian fit and the two boundaries (log density)"""
    dataset, model, seed = 'MNIST: 3+5 vs rest [resnet18]', 'Linear', 0
    with np.load(os.path.join(rs.PROJ, f'{rs.ep._slug(dataset)}__{model}.npz')) as f:
        p = f's{seed}_split_'
        zc, yc = f[p + 'z_cert'], f[p + 'y_cert'].astype(int)
        zt, yt = f[p + 'z_test'], f[p + 'y_test'].astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bins = np.linspace(min(zt.min(), zc.min()), max(zt.max(), zc.max()), 80)
    ax.hist(zt[yt == 0], bins, density=True, color=INK2, alpha=0.35, label='majority (test)')
    ax.hist(zt[yt == 1], bins, density=True, color=C['blue'], alpha=0.45,
            label='minority 3+5 (test)')
    z1 = zc[yc == 1]
    grid = np.linspace(bins[0], bins[-1], 400)
    pdf = np.exp(-0.5 * ((grid - z1.mean()) / z1.std()) ** 2) / (z1.std() * np.sqrt(2 * np.pi))
    ax.plot(grid, pdf, color=C['blue'], lw=1.2, label='Gaussian fit (certificate set)')
    for m, col in (('CP Minimax', INK), ('Gaussian Predictive', C['orange'])):
        est = METHODS[m](rw.FrozenProjection(0.0), zc[:, None], yc)
        ax.axvline(est.boundary, color=col, ls='--', lw=1.2, label=f'{m} boundary')
    ax.set(xlabel='projected score', ylabel='density', yscale='log', ylim=(1e-3, None))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    _savefig(fig, 'f5_mixed')


def main():
    d = load()
    split = d[d['mode'] == 'split']
    table_simulation()
    table_gaussian(split)
    table_gmean(split)
    table_coverage(d)
    table_published(split)
    table_deep(d)
    table_limits(split)
    table_costs()
    fig_nsweep(d)
    fig_simulation()
    fig_synthetic(split)
    table_synthetic(split)
    fig_mixed()
    if os.path.exists(os.path.join(RESULTS, 'slots.csv')):
        table_r(split)
        fig_margins()
        table_ablations()
    print(f'wrote tables and figures to {OUT}')


if __name__ == '__main__':
    main()
