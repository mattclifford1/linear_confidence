'''Each method's decision regions on the overlapping Gaussians' test set (deltas:fig:sythetic-test).'''
import matplotlib.pyplot as plt

import draw
import style
import synthetic

NAME = 'deltas/synthetic/Gaussian_eval'
WIDTH = 1.0
METHODS = ['Baseline', 'SMOTE', 'Balanced Weights', 'BMR', 'Threshold', 'Our Method']


def make():
    s = synthetic.overlapping()
    X, y = s['test']['X'], s['test']['y']
    fig, axes = plt.subplots(3, 2, figsize=style.figsize(WIDTH, 1.45), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), METHODS):
        draw.regions(ax, s['clfs'][name].predict, X)
        draw.classes(ax, X, y, size=6, alpha=0.35)
        # Bayes optimal for equal priors and covariances at means -/+[1, 1]
        draw.line(ax, (1, 1), 0, ax.get_xlim(), **style.LINES['optimal'])
        draw.feature_axes(ax)
        ax.set_title(name)
    for ax in axes[:, 1]:
        ax.set_ylabel('')
    for ax in axes[:-1].ravel():
        ax.set_xlabel('')
    style.save(fig, NAME)


if __name__ == '__main__':
    make()
