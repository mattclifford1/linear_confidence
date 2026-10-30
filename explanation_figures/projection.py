'''Training data and its projection with the distances the method uses (deltas:fig:projection-diagram).'''
import matplotlib.pyplot as plt

import draw
import style
import synthetic

NAME = 'deltas/diagrams/projection'
WIDTH = 1.0
ROW_R_BAR, ROW_R_HAT, ROW_D = -0.4, 0.45, 0.85


def strip(ax, info, delta=synthetic.DELTA_DOT_NINE, show_D=True, show_R_hat=True):
    '''Projected scores with the class means, R-bar_i, and optionally R-hat_i(delta) and D-hat.'''
    x1, x2, m1, m2 = synthetic.class_scores(info)
    draw.scores(ax, x1, x2)
    draw.mean(ax, m1, draw.MEAN_1)
    draw.mean(ax, m2, draw.MEAN_2)
    R1, R2 = info['empirical R1'], info['empirical R2']
    draw.bar(ax, m1, m1 + R1, ROW_R_BAR, r'$\hat{\bar{R}}_1$', style.MAJ)
    draw.bar(ax, m2 - R2, m2, ROW_R_BAR, r'$\hat{\bar{R}}_2$', style.MIN)
    if show_R_hat:
        R1h = synthetic.R_hat(R1, info, info['N1'], delta)
        R2h = synthetic.R_hat(R2, info, info['N2'], delta)
        draw.bar(ax, m1, m1 + R1h, ROW_R_HAT, r'$\hat{R}_1(\delta_1 = 0.\dot{9})$', style.MAJ, dashed=True)
        draw.bar(ax, m2 - R2h, m2, ROW_R_HAT, r'$\hat{R}_2(\delta_2 = 0.\dot{9})$', style.MIN, dashed=True)
    if show_D:
        draw.bar(ax, m1, m2, ROW_D, r'$\hat{D}$')
    ax.set_ylim(ROW_R_BAR - 0.2, ROW_D + 0.3)
    draw.strip_axes(ax)


def make():
    s = synthetic.separated()
    fig, (a, mid, b) = plt.subplots(1, 3, figsize=style.figsize(WIDTH, 0.33),
                                    gridspec_kw={'width_ratios': [1, 0.3, 2.3]})
    draw.classes(a, s['X'], s['y'])
    draw.feature_axes(a)
    a.text(-1.5, -7.0, '$S_1$', fontsize=11)
    a.text(1.0, 5.0, '$S_2$', fontsize=11)

    mid.axis('off')
    mid.annotate('', xy=(1, 0.5), xytext=(0, 0.5), xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='-|>', color=style.INK, lw=1.2, mutation_scale=10))
    mid.text(0.5, 0.42, draw.SCORE_LABEL, ha='center', va='top', transform=mid.transAxes)

    strip(b, s['info'])
    style.save(fig, NAME)


if __name__ == '__main__':
    make()
