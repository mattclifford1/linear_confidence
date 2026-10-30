'''Logistic regression trained on the overlapping Gaussians, as is and on SMOTE data (deltas:fig:sythetic-LR-clfs).'''
import matplotlib.pyplot as plt

import draw
import style
import synthetic

NAME = 'deltas/synthetic/Gaussian_data'
WIDTH = 1.0


def make():
    s = synthetic.overlapping()
    fig, axes = plt.subplots(1, 2, figsize=style.figsize(WIDTH, 0.5), sharex=True, sharey=True)
    for ax, name, d in ((axes[0], 'Baseline', {'X': s['X'], 'y': s['y']}),
                        (axes[1], 'SMOTE', s['smote'])):
        draw.regions(ax, s['clfs'][name].predict, s['X'])
        draw.classes(ax, d['X'], d['y'])
        draw.feature_axes(ax)
        ax.set_title(name)
    axes[1].set_ylabel('')
    style.save(fig, NAME)


if __name__ == '__main__':
    make()
