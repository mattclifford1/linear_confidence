'''
The Deltas thesis figures' style: the shared thesis style (~/Repos/lib/thesis_figures) saving into out/.

Set `width` in `figsize` to the fraction of \\textwidth used in \\includegraphics.
'''
import os

from thesis_figures.roles import *  # noqa: F401,F403
from thesis_figures.roles import CLASS, MAJORITY, MINORITY
from thesis_figures.style import TEXTWIDTH, figsize  # noqa: F401
from thesis_figures import style as _shared

HERE = os.path.dirname(os.path.abspath(__file__))
# mirrors the thesis's chapters/<chapter>/figures/ as out/<chapter>/
OUT = os.path.join(HERE, 'out')

# majority (S_1) and minority (S_2); captions call them blue and red
MAJ, MIN = CLASS[MAJORITY], CLASS[MINORITY]


def save(fig, name):
    '''Write `name`.pdf under OUT, where `name` is `<chapter>/<path in its figures folder>`.'''
    _shared.save(fig, name, OUT)
