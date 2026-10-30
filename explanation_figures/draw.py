'''Drawing helpers shared by the figures, so a style change made here reaches every figure.'''
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

import style

SCORE_LABEL = r'$\langle \phi(x), w \rangle$'


def feature_axes(ax):
    ax.set(xlabel='Feature 1', ylabel='Feature 2', aspect='equal')


def classes(ax, X, y, size=14, alpha=0.5):
    '''Majority then minority, so the minority sits on top.'''
    for label, col in ((0, style.MAJ), (1, style.MIN)):
        x = X[y == label]
        ax.scatter(x[:, 0], x[:, 1], s=size, color=col, alpha=alpha,
                   edgecolors=style.INK, linewidths=0.3, zorder=2)


def regions(ax, predict, X, pad=1.0, res=300, boundary=True):
    '''Shade the predicted-class regions of `predict` over the extent of X.'''
    lo, hi = X.min(axis=0) - pad, X.max(axis=0) + pad
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], res), np.linspace(lo[1], hi[1], res))
    zz = np.asarray(predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    cmap = ListedColormap([to_rgba(style.MAJ, style.REGION_ALPHA),
                           to_rgba(style.MIN, style.REGION_ALPHA)])
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=cmap, zorder=0)
    if boundary:
        ax.contour(xx, yy, zz, levels=[0.5], colors=style.INK, linewidths=1.0, zorder=3)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))


def line(ax, a, c, xlim, **kw):
    '''The line a . x = c across xlim, for a in R^2.'''
    xs = np.array(xlim)
    ax.plot(xs, (c - a[0] * xs) / a[1], **kw)


def strip_axes(ax):
    '''A 1-D score axis: no y axis, only the bottom spine.'''
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(SCORE_LABEL)


def scores(ax, x1, x2, y=0.0, size=10):
    '''Projected scores of S_1 and S_2 on one row.'''
    for x, col in ((x1, style.MAJ), (x2, style.MIN)):
        ax.scatter(x, np.full_like(x, y), s=size, color=col, linewidths=0, zorder=2)


def mean(ax, x, label, y=0.0, dy=0.12):
    ax.scatter([x], [y], marker='x', s=60, color=style.INK, linewidths=1.2, zorder=3)
    ax.text(x, y + dy, label, ha='center', va='bottom')


def bar(ax, x0, x1, y, label, colour=style.INK, dashed=False, dy=0.05):
    '''A capped interval from x0 to x1 with a label centred above it.'''
    ax.plot([x0, x1], [y, y], color=colour, ls='--' if dashed else '-', lw=1.2,
            marker='|', markersize=7, markeredgewidth=1.2)
    ax.text((x0 + x1) / 2, y + dy, label, ha='center', va='bottom')


MEAN_1 = r'$\langle\bar{\phi}_{S_1}, w\rangle$'
MEAN_2 = r'$\langle\bar{\phi}_{S_2}, w\rangle$'
