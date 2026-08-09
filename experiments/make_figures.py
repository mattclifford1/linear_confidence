'''
Figures for the ground-up overlap section of the non-separable draft.

fig 1  certified loss landscape on the synthetic Gaussian - shows why the
       "minimise the sum of certified errors" rule degenerates when the
       minority class is scarce, and why minimax does not
fig 2  overlap sweep - class separation from separable to heavy overlap,
       showing where the slack-based method stops finding a solution at all
       while the overlap-native methods degrade smoothly

Writes PNGs to the Overleaf non-separable draft directory.
'''
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import deltas.classifiers.models as models
from deltas.pipeline import data as pdata
from deltas.model import overlap, downsample
from deltas.costcla_local.models import Thresholding

OUT = os.environ.get(
    'DELTAS_FIG_OUT',
    '/home/matt/Repos/Overleaf/deltas/deltas-non-separable')

# categorical slots, fixed order, from the validated reference palette
# (line charts use the adjacent pairlist, for which this order validates)
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a',
     'yellow': '#eda100', 'magenta': '#e87ba4', 'violet': '#4a3aa7'}
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#d8d7d2'

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


def _balanced_err(z, y, b, low_label):
    '''test balanced error of a threshold, given which label sits below b'''
    high_label = 1 - low_label
    e_low = (z[y == low_label] > b).mean()
    e_high = (z[y == high_label] <= b).mean()
    return 0.5 * (e_low + e_high)


# ------------------------------------------------------------------ fig 1 ---
def fig_loss_landscape(N1=1000, N2=10, seed=0):
    dc = pdata.get_data(m1=[-1, -1], m2=[1, 1], N1=N1, N2=N2,
                        scale=False, seed=seed)
    X, y = dc['data']['X'], dc['data']['y']
    clf = models.linear().fit(X, y)
    zt = clf.get_projection(dc['data_test']['X']).squeeze()
    yt = dc['data_test']['y']

    m_sum = overlap.binomial_deltas(clf, objective='sum').fit(X, y)
    m_mmx = overlap.binomial_deltas(clf, objective='minimax').fit(X, y)

    b = m_sum.candidates
    L_low, L_high = m_sum._L_low, m_sum._L_high          # majority, minority
    cands = np.linspace(b.min(), b.max(), 1500)
    best_b = cands[int(np.argmin([_balanced_err(zt, yt, c, 0) for c in cands]))]

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)

    ax = axs[0]
    ax.plot(b, L_low, color=C['blue'], label=f'majority  $N_1$={N1}')
    ax.plot(b, L_high, color=C['orange'], label=f'minority  $N_2$={N2}')
    ax.set_ylabel('certified class loss  $L_i(b)$')
    ax.set_xlabel('boundary  $b$')
    ax.set_title('Per-class certificates', loc='left')
    flat = float(L_high.min())
    ax.annotate('flat wherever $m_2=0$:\nthe certificate cannot\nresolve $b$ here',
                xy=(0.0, flat), xytext=(-4.3, flat - 0.30),
                fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle='->', color=INK2, linewidth=0.9))
    ax.legend(loc='lower right')

    ax = axs[1]
    ax.plot(b, L_low + L_high, color=C['aqua'], label='sum rule')
    ax.plot(b, np.maximum(L_low, L_high), color=C['violet'],
            label='minimax rule')
    for xb, col, lab in [(m_sum.boundary, C['aqua'], 'sum'),
                         (m_mmx.boundary, C['violet'], 'minimax'),
                         (best_b, INK2, 'test optimum')]:
        ax.axvline(xb, color=col, linestyle=(0, (4, 3)), linewidth=1.4)
        ax.annotate(lab, xy=(xb, ax.get_ylim()[1]), xytext=(2, -9),
                    textcoords='offset points', fontsize=7.5, color=col,
                    rotation=90, va='top')
    ax.set_ylabel('objective')
    ax.set_xlabel('boundary  $b$')
    ax.set_title('Decision rules', loc='left')
    ax.legend(loc='upper left')

    fig.tight_layout()
    path = os.path.join(OUT, 'groundup_loss_landscape.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    stats = {
        'sum_b': m_sum.boundary, 'minimax_b': m_mmx.boundary,
        'test_b': best_b,
        'sum_err': _balanced_err(zt, yt, m_sum.boundary, 0),
        'minimax_err': _balanced_err(zt, yt, m_mmx.boundary, 0),
        'test_err': _balanced_err(zt, yt, best_b, 0),
        'L_min_flat': float(L_high.min()),
        'L_maj_range': (float(L_low.min()), float(L_low.max())),
    }
    return path, stats


# ------------------------------------------------------------------ fig 2 ---
def _sweep_data(mus, seeds, N1, N2):
    '''run the sweep, memoised on disk (it is the slow part)'''
    import joblib
    store = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'results', 'overlap_sweep.joblib')
    key = (tuple(mus), seeds, N1, N2)
    if os.path.exists(store):
        cached_key, payload = joblib.load(store)
        if cached_key == key:
            print('  reusing cached sweep')
            return payload
    payload = _run_sweep(mus, seeds, N1, N2)
    os.makedirs(os.path.dirname(store), exist_ok=True)
    joblib.dump((key, payload), store)
    return payload


def _run_sweep(mus, seeds, N1, N2):
    methods = ['Baseline', 'Threshold', 'Slacks Deltas',
               'CP Minimax', 'DKW Minimax']
    err = {m: np.full((len(mus), seeds), np.nan) for m in methods}
    solved = np.zeros((len(mus), seeds))

    for i, mu in enumerate(mus):
        for s in range(seeds):
            dc = pdata.get_data(m1=[-mu, -mu], m2=[mu, mu], N1=N1, N2=N2,
                                scale=False, seed=s)
            X, y = dc['data']['X'], dc['data']['y']
            Xt, yt = dc['data_test']['X'], dc['data_test']['y']
            clf = models.linear().fit(X, y)

            def berr(preds):
                preds = np.asarray(preds).squeeze()
                return 0.5 * ((preds[yt == 0] != 0).mean() +
                              (preds[yt == 1] != 1).mean())

            err['Baseline'][i, s] = berr(clf.predict(Xt))
            try:
                th = Thresholding(clf, calibration=False).fit(X, y)
                err['Threshold'][i, s] = berr(th.predict(Xt))
            except Exception:
                pass
            sl = downsample.downsample_deltas(clf).fit(
                X, y, max_trials=10000, parallel=False)
            if sl.is_fit:
                err['Slacks Deltas'][i, s] = berr(sl.predict(Xt))
                solved[i, s] = 1
            err['CP Minimax'][i, s] = berr(
                overlap.binomial_deltas(clf, objective='minimax')
                .fit(X, y).predict(Xt))
            err['DKW Minimax'][i, s] = berr(
                overlap.dkw_deltas(clf, objective='minimax')
                .fit(X, y).predict(Xt))
        print(f'  mu={mu}: slacks solved {int(solved[i].sum())}/{seeds}',
              flush=True)
    return {'mus': list(mus), 'methods': methods, 'err': err,
            'solved': solved}


def fig_overlap_sweep(mus=None, seeds=5, N1=500, N2=25):
    '''sweep class separation from separable to heavily overlapping'''
    if mus is None:
        mus = [0.15, 0.25, 0.4, 0.55, 0.75, 1.0, 1.4, 2.0]
    d = _sweep_data(mus, seeds, N1, N2)
    mus, methods, err, solved = d['mus'], d['methods'], d['err'], d['solved']
    colours = [C['blue'], C['orange'], C['aqua'], C['violet'], C['magenta']]
    solved_frac = solved.mean(axis=1)
    # largest mu at which the slack method never finds a solution
    never = [mu for mu, s in zip(mus, solved_frac) if s == 0]

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axs[0]
    if never:
        edge = max(never) + 0.5 * (
            min(m for m in mus if m > max(never)) - max(never))
        ax.axvspan(min(mus) - 0.1, edge, color=C['aqua'], alpha=0.07, lw=0)
        ax.annotate('slacks: no\nsolution here', xy=(min(mus) + 0.02, 0.02),
                    fontsize=7.5, color=INK2)
    for m, col in zip(methods, colours):
        ax.plot(mus, np.nanmean(err[m], axis=1), color=col, marker='o',
                markersize=4, label=m)
    ax.set_xlabel(r'class separation  $\mu$   (overlapping $\leftarrow$)')
    ax.set_ylabel('test balanced error')
    ax.set_title('Performance vs overlap', loc='left')
    ax.legend(loc='upper right')

    ax = axs[1]
    ax.plot(mus, solved_frac * 100, color=C['aqua'], marker='o',
            markersize=4, label='Slacks Deltas')
    ax.axhline(100, color=C['violet'], linestyle=(0, (4, 3)), linewidth=1.6)
    ax.annotate('CP / DKW: always solvable by construction',
                xy=(mus[0], 100), xytext=(0, -12), textcoords='offset points',
                fontsize=7.5, color=C['violet'])
    knee = int(np.argmax(solved_frac > 0))       # first mu with any solution
    ax.annotate('Slacks Deltas', xy=(mus[knee], solved_frac[knee] * 100),
                xytext=(6, -4), textcoords='offset points',
                fontsize=7.5, color=C['aqua'], ha='left')
    ax.set_ylim(-5, 112)
    ax.set_xlabel(r'class separation  $\mu$')
    ax.set_ylabel('seeds with a solution (%)')
    ax.set_title('Feasibility vs overlap', loc='left')

    fig.tight_layout()
    path = os.path.join(OUT, 'groundup_overlap_sweep.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return path, {'mus': mus,
                  'solved': solved_frac.tolist(),
                  'err': {m: np.nanmean(err[m], axis=1).tolist()
                          for m in methods}}


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', '1'):
        p, s = fig_loss_landscape()
        print('fig1 ->', p)
        for k, v in s.items():
            print(f'   {k}: {v}')
    if which in ('all', '2'):
        p, s = fig_overlap_sweep()
        print('fig2 ->', p)
        print('   solved:', np.round(s['solved'], 2))
        for m, v in s['err'].items():
            print(f'   {m:15s}', np.round(v, 3))
