'''Moving the bias: a classifier's boundary shifted away from a small minority class (deltas:fig:moving-bias-diagram).'''
import matplotlib.pyplot as plt

import draw
import style
import synthetic

NAME = 'deltas/diagrams/diagrams_data_original'
WIDTH = 0.55
# schematic boundaries x_1 + x_2 = c, not the fitted SVM's
C_B, C_B_PRIME = 4, -2
# label position along each line from its midpoint, and offset off it
LABEL_ALONG, LABEL_OFF = 2.2, 0.45


def make():
    s = synthetic.separated()
    fig, ax = plt.subplots(figsize=style.figsize(WIDTH, 1.0))
    draw.classes(ax, s['X'], s['y'])
    draw.line(ax, (1, 1), C_B, (C_B - 6, 6), color=style.INK, lw=2)
    draw.line(ax, (1, 1), C_B_PRIME, (-6, C_B_PRIME + 6), color=style.INK, lw=2, ls='--')
    mid, mid_p = C_B / 2, C_B_PRIME / 2
    ax.annotate('', xy=(mid_p + 0.4, mid_p + 0.4), xytext=(mid - 0.4, mid - 0.4),
                arrowprops=dict(arrowstyle='-|>', color=style.INK, lw=1.5, mutation_scale=12))
    for c, lab in ((C_B, r'$\langle \phi(x), w \rangle + b = 0$'),
                   (C_B_PRIME, r"$\langle \phi(x), w \rangle + b' = 0$")):
        ax.text(c / 2 + LABEL_ALONG + LABEL_OFF, c / 2 - LABEL_ALONG + LABEL_OFF, lab,
                rotation=-45, rotation_mode='anchor', ha='center', va='bottom',
                transform_rotates_text=True)
    draw.feature_axes(ax)
    style.save(fig, NAME)


if __name__ == '__main__':
    make()
