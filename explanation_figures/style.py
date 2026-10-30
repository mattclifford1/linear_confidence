'''
Shared look for the thesis figures: colours, fonts, sizes and saving.

Figures are drawn at their printed size, so the font sizes here are the sizes on the page.
Set `width` in `figsize` to the fraction of \\textwidth used in \\includegraphics.
'''
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
# mirrors the thesis's chapters/<chapter>/figures/ as out/<chapter>/
OUT = os.path.join(HERE, 'out')

# thesis type block is 448.13pt wide (preamble.tex)
TEXTWIDTH = 448.13 / 72.27

# majority (S_1) and minority (S_2); captions call them blue and red
MAJ, MIN = '#2a78d6', '#d0342c'
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'violet': '#4a3aa7'}
INK, INK2, GRID, PANEL = '#0b0b0b', '#52514e', '#d8d7d2', '#f1f0ec'
REGION_ALPHA = 0.18

# the thesis body font is New Century Schoolbook (fouriernc); TeX Gyre Schola is its clone
FONT_DIRS = ['/usr/share/texmf/fonts/opentype/public/tex-gyre',
             '/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre']
FONT = 'TeX Gyre Schola'


def _register_font():
    for d in FONT_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith('texgyreschola'):
                    font_manager.fontManager.addfont(os.path.join(d, f))
    return any(f.name == FONT for f in font_manager.fontManager.ttflist)


def apply():
    serif = [FONT] if _register_font() else ['DejaVu Serif']
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': serif,
        'mathtext.fontset': 'custom', 'mathtext.rm': serif[0],
        'mathtext.it': f'{serif[0]}:italic', 'mathtext.bf': f'{serif[0]}:bold',
        'mathtext.fallback': 'cm',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.5, 'legend.frameon': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })


def figsize(width=1.0, aspect=0.4):
    '''(w, h) in inches for a figure `width` x \\textwidth wide and `aspect` x as tall.'''
    w = width * TEXTWIDTH
    return w, w * aspect


def save(fig, name):
    '''Write `name`.pdf under OUT, where `name` is `<chapter>/<path in its figures folder>`.'''
    path = os.path.join(OUT, name + '.pdf')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # no creation date, so an unchanged figure is byte-identical and sync skips it
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)
    print(f'wrote {os.path.relpath(path)}')


apply()
