'''
F1 (deltas:fig:f1) and F4 (deltas:fig:f4) of experiments/make_extra_figures.py in the thesis style.

Reads experiments/results/simulate_truth.csv (simulate_truth.py) and margins.csv (run_slots.py).
'''
import os

import matplotlib.pyplot as plt
import pandas as pd

import style

RESULTS = os.path.join(style.HERE, '..', 'experiments', 'results')
NAMES = {'simulation': 'deltas/follow-up/f1_simulation', 'margins': 'deltas/follow-up/f4_margins'}
C = style.C


def simulation(delta=0.05):
    s = pd.read_csv(os.path.join(RESULTS, 'simulate_truth.csv'))
    s = s[s.delta == delta]
    shapes = ['normal', 'skewed', 'heavy', 'two subgroups', 'bounded']
    fig, axes = plt.subplots(1, len(shapes), figsize=style.figsize(1.0, 0.36), sharey=True,
                             layout='constrained')
    for ax, sh in zip(axes, shapes):
        x = s[s['shape'] == sh]
        fp_ = x[x.bound == 'furthest point'].sort_values('N')
        ax.plot(fp_.N, fp_.mean_err, color=C['blue'], label='furthest point, mean')
        ax.plot(fp_.N, fp_.p95_err, color=C['blue'], ls='--', lw=1.0,
                label='furthest point, 95th pct.')
        for b, col in (('published (eps)', C['orange']), ('published (2 eps)', C['aqua'])):
            y = x[x.bound == b].sort_values('N')
            ax.plot(y.N, y.p95_err, color=col, lw=1.0,
                    label=b.replace('eps', r'$\epsilon$') + ', 95th pct.')
        ax.plot(fp_.N, 1 / (fp_.N + 1) + delta, color=style.INK, lw=0.8, ls=':',
                label=r'promised $1/(N+1)+\delta$')
        ax.set(xscale='log', yscale='log', title=sh, xlabel='$N$')
        ax.grid(True, which='major')
    axes[0].set_ylabel('true error $e$')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
    style.save(fig, NAMES['simulation'])


def margins():
    m = pd.read_csv(os.path.join(RESULTS, 'margins.csv'))
    fig, ax = plt.subplots(figsize=style.figsize(0.6, 0.62))
    for v, col in (('R-bar, two-sided', C['blue']), ('R-bar, one-sided', C['aqua']),
                   ('known range', C['orange'])):
        y = m[m.variant == v].groupby('N').distance
        ax.plot(y.median().index, y.median(), color=col, label=v)
        ax.fill_between(y.median().index, y.quantile(0.25), y.quantile(0.75),
                        color=col, alpha=0.15, lw=0)
    ax.axhline(1, color=style.INK2, lw=0.8, ls=':')
    ax.set(xscale='log', ylim=(0, None), xlabel='$N_1$ (minority points subsampled)',
           ylabel='limit distance / score range')
    ax.grid(True)
    ax.legend()
    style.save(fig, NAMES['margins'])


def make():
    simulation()
    margins()


if __name__ == '__main__':
    make()
