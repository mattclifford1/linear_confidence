'''The overlapping Gaussians in projected space, before (deltas:fig:sythetic-deltas-data) and after (deltas:fig:sythetic-deltas-optimised) optimising the deltas.'''
import matplotlib.pyplot as plt

import draw
import style
import synthetic

NAMES = {'training': 'deltas/synthetic/Gaussian-training',
         'optimised': 'deltas/synthetic/Gaussian-optimised'}
WIDTH = 1.0
ROW_R = -0.4


def training():
    info = synthetic.overlapping()['info']
    fig, ax = plt.subplots(figsize=style.figsize(WIDTH, 0.36))
    x1, x2, m1, m2 = synthetic.class_scores(info)
    draw.scores(ax, x1, x2)
    draw.mean(ax, m1, draw.MEAN_1)
    draw.mean(ax, m2, draw.MEAN_2)
    draw.bar(ax, m1, m1 + info['empirical R1'], ROW_R - 0.35, r'$\hat{\bar{R}}_1$', style.MAJ)
    draw.bar(ax, m2 - info['empirical R2'], m2, ROW_R, r'$\hat{\bar{R}}_2$', style.MIN)
    ax.set_ylim(ROW_R - 0.55, 0.5)
    draw.strip_axes(ax)
    style.save(fig, NAMES['training'])


def optimised():
    s = synthetic.overlapping()
    m, info = s['deltas'], s['deltas'].data_info
    fig, ax = plt.subplots(figsize=style.figsize(WIDTH, 0.3))
    x1, x2, m1, m2 = synthetic.class_scores(info)
    draw.scores(ax, x1, x2)
    draw.mean(ax, m1, draw.MEAN_1)
    draw.mean(ax, m2, draw.MEAN_2)
    R1 = synthetic.R_hat(info['empirical R1'], info, info['N1'], m.delta1)
    R2 = synthetic.R_hat(info['empirical R2'], info, info['N2'], m.delta2)
    draw.bar(ax, m1, m1 + R1, ROW_R, r'$\hat{R}_1$', style.MAJ)
    draw.bar(ax, m2 - R2, m2, ROW_R, r'$\hat{R}_2$', style.MIN)
    ax.axvline(-m.get_bias(), **style.LINES['adjusted'] | {'lw': 0.8}, zorder=1)
    ax.set_ylim(ROW_R - 0.2, 0.5)
    draw.strip_axes(ax)
    style.save(fig, NAMES['optimised'])


def make():
    training()
    optimised()


if __name__ == '__main__':
    make()
