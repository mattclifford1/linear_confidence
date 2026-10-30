'''The ML pipeline with the class imbalance techniques that act at each stage (literature review).'''
import functools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.linear_model import LogisticRegression

import draw
import style

NAME = '02-literature-review/imbalance_pipeline'
WIDTH = 1.0
# (title, when, techniques); a technique ending in '*' is drawn highlighted as this thesis's
STAGES = [
    ('Data', 'before training',
     ['Undersampling', 'Oversampling', 'SMOTE', 'ADASYN']),
    ('Model training', 'during training',
     ['Class weights', 'Class-balanced loss', 'LDAM loss', 'Cost-sensitive SVMs']),
    ('Model adjustment', 'after training',
     ['Thresholding', 'Bayes minimum risk', 'Bias adjustment*']),
    ('Evaluation', 'on held-out data',
     ['Balanced accuracy', 'G-mean', 'F1 score', 'Precision, recall']),
]
GAP = 0.32          # inches between columns, where the arrows go
HEAD_H = 0.48       # stage box height
THUMB_H = 1.0       # thumbnail height
ROW_H = 0.2         # technique row height
SEP = 0.08          # vertical space between the three blocks of a column
PAD = 0.1
N_MAJ, N_MIN, N_SMOTE, SEED = 60, 6, 8, 3


@functools.cache
def toy():
    '''One small imbalanced data set shared by the thumbnails.'''
    rng = np.random.default_rng(SEED)
    X = np.vstack([rng.normal(-0.8, 0.7, (N_MAJ, 2)), rng.normal(1.0, 0.45, (N_MIN, 2))])
    y = np.r_[np.zeros(N_MAJ), np.ones(N_MIN)]
    Xm = X[y == 1]
    i, j = rng.integers(0, N_MIN, (2, N_SMOTE))
    smote = Xm[i] + rng.uniform(0, 1, (N_SMOTE, 1)) * (Xm[j] - Xm[i])
    plain = LogisticRegression().fit(X, y)
    weighted = LogisticRegression(class_weight='balanced').fit(X, y)
    return X, y, smote, plain, weighted


def _scatter_lims(ax):
    X, _, smote, *_ = toy()
    pts = np.vstack([X, smote])
    ax.set(xlim=(pts[:, 0].min() - 0.3, pts[:, 0].max() + 0.3),
           ylim=(pts[:, 1].min() - 0.3, pts[:, 1].max() + 0.3))


def _thumb_axes(ax):
    ax.set(xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set(visible=True, color=style.GRID)


def data_thumb(ax):
    X, y, smote, *_ = toy()
    draw.classes(ax, X, y, size=8)
    ax.scatter(smote[:, 0], smote[:, 1], s=8, facecolors='none', edgecolors=style.MIN,
               linewidths=0.8, zorder=3)
    _scatter_lims(ax)
    _thumb_axes(ax)


def training_thumb(ax):
    X, y, _, plain, weighted = toy()
    draw.classes(ax, X, y, size=8)
    _scatter_lims(ax)
    lim = ax.get_xlim()
    for m, ls in ((plain, '--'), (weighted, '-')):
        draw.line(ax, m.coef_[0], -m.intercept_[0], lim, color=style.INK, lw=1.0, ls=ls)
    _scatter_lims(ax)
    _thumb_axes(ax)


def adjustment_thumb(ax):
    X, y, _, plain, _ = toy()
    z = plain.decision_function(X)
    draw.scores(ax, z[y == 0], z[y == 1], size=6)
    # b' at the midpoint of the class score means, as a minimum-distance rule would put it
    b, b_new = 0.0, (z[y == 0].mean() + z[y == 1].mean()) / 2
    for t, ls, lab in ((b, '--', '$b$'), (b_new, '-', "$b'$")):
        ax.plot([t, t], [-0.35, 0.35], color=style.INK, lw=1.0, ls=ls)
        ax.text(t, 0.42, lab, ha='center', va='bottom', fontsize=8)
    ax.annotate('', xy=(b_new, -0.5), xytext=(b, -0.5),
                arrowprops=dict(arrowstyle='-|>', color=style.INK, lw=0.8, mutation_scale=7))
    ax.set_ylim(-0.8, 0.8)
    _thumb_axes(ax)


def evaluation_thumb(ax):
    '''A confusion matrix with the minority as the positive class.'''
    cells = [['TP', 'FN'], ['FP', 'TN']]
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((c, 1 - r), 1, 1, fc=style.PANEL if r == c else 'white',
                                   ec=style.INK2, lw=0.6))
            ax.text(c + 0.5, 1.5 - r, cells[r][c], ha='center', va='center', fontsize=8)
    for k, lab in enumerate(('+', '−')):
        ax.text(k + 0.5, 2.1, lab, ha='center', va='bottom', fontsize=7)
        ax.text(-0.1, 1.5 - k, lab, ha='right', va='center', fontsize=7)
    ax.text(1, 2.35, 'predicted', ha='center', va='bottom', fontsize=7, color=style.INK2)
    ax.text(-0.35, 1, 'true', ha='right', va='center', fontsize=7, color=style.INK2,
            rotation=90)
    ax.set(xlim=(-0.9, 2.3), ylim=(-0.1, 2.75), aspect='equal')
    ax.axis('off')


THUMBS = [data_thumb, training_thumb, adjustment_thumb, evaluation_thumb]


def make():
    w, _ = style.figsize(WIDTH)
    n_rows = max(len(t) for *_, t in STAGES)
    list_h = n_rows * ROW_H + 0.14
    h = PAD + HEAD_H + SEP + THUMB_H + SEP + list_h + PAD
    col = (w - GAP * (len(STAGES) - 1)) / len(STAGES)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis('off')

    top = h - PAD
    thumb_top = top - HEAD_H - SEP
    list_top = thumb_top - THUMB_H - SEP
    for i, ((title, when, techniques), thumb) in enumerate(zip(STAGES, THUMBS)):
        x = i * (col + GAP)
        ax.add_patch(FancyBboxPatch((x, top - HEAD_H), col, HEAD_H,
                                    boxstyle='round,pad=0,rounding_size=0.06',
                                    fc=style.PANEL, ec=style.INK2, lw=0.8))
        ax.text(x + col / 2, top - 0.17, title, ha='center', va='center', weight='bold')
        ax.text(x + col / 2, top - 0.36, when, ha='center', va='center', fontsize=8,
                color=style.INK2, style='italic')

        thumb(fig.add_axes([x / w, (thumb_top - THUMB_H) / h, col / w, THUMB_H / h]))

        ax.add_patch(FancyBboxPatch((x, PAD), col, list_top - PAD,
                                    boxstyle='round,pad=0,rounding_size=0.06',
                                    fc='none', ec=style.GRID, lw=0.8))
        for j, t in enumerate(techniques):
            ours = t.endswith('*')
            ax.text(x + col / 2, list_top - 0.07 - (j + 0.5) * ROW_H, t.rstrip('*'),
                    ha='center', va='center', fontsize=8,
                    color=style.C['blue'] if ours else style.INK,
                    weight='bold' if ours else 'normal')
        if i:
            y = top - HEAD_H / 2
            ax.add_patch(FancyArrowPatch((x - GAP + 0.05, y), (x - 0.05, y),
                                         arrowstyle='-|>', mutation_scale=10,
                                         color=style.INK, lw=1.0))
    style.save(fig, NAME)


if __name__ == '__main__':
    make()
