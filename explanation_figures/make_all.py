'''
Regenerate the thesis figures into out/, and with --thesis copy the changed ones into the thesis.

    uv run python explanation_figures/make_all.py                # all
    uv run python explanation_figures/make_all.py projection     # by module name
    uv run python explanation_figures/make_all.py --thesis       # all, then sync_thesis.py
'''
import argparse
import importlib

import sync_thesis

FIGURES = ['imbalance_pipeline', 'moving_bias', 'projection', 'gaussian_classifiers',
           'gaussian_projection', 'gaussian_eval', 'follow_up']


def main(names, thesis=False):
    for name in names or FIGURES:
        importlib.import_module(name).make()
    if thesis:
        sync_thesis.sync()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='*', help=f'any of: {", ".join(FIGURES)}')
    p.add_argument('--thesis', action='store_true', help='copy changed figures into the thesis')
    a = p.parse_args()
    unknown = set(a.names) - set(FIGURES)
    if unknown:
        p.error(f'unknown figure(s): {", ".join(sorted(unknown))}')
    main(a.names, a.thesis)
