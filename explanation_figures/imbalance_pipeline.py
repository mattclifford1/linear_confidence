'''The ML pipeline with the class imbalance techniques that act at each stage (literature review).'''
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
ROW_H = 0.2         # technique row height
PAD = 0.1


def make():
    w, _ = style.figsize(WIDTH)
    n_rows = max(len(t) for *_, t in STAGES)
    h = HEAD_H + 0.15 + n_rows * ROW_H + 2 * PAD
    col = (w - GAP * (len(STAGES) - 1)) / len(STAGES)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis('off')

    top = h - PAD
    for i, (title, when, techniques) in enumerate(STAGES):
        x = i * (col + GAP)
        ax.add_patch(FancyBboxPatch((x, top - HEAD_H), col, HEAD_H,
                                    boxstyle='round,pad=0,rounding_size=0.06',
                                    fc=style.PANEL, ec=style.INK2, lw=0.8))
        ax.text(x + col / 2, top - 0.17, title, ha='center', va='center', weight='bold')
        ax.text(x + col / 2, top - 0.36, when, ha='center', va='center', fontsize=8,
                color=style.INK2, style='italic')
        list_top = top - HEAD_H - 0.08
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
