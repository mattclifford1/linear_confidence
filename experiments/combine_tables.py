'''
Stitch the per-dataset LaTeX fragments into the multi-row table used in the
draft. Unlike the old notebooks-*/run_all*.py version this does not index into
pandas' to_latex output by line number - it reads the aggregated CSVs.
'''
import os

import numpy as np
import pandas as pd

from run_experiments import EXPERIMENTS, SHORT_NAMES, METRICS, RESULTS
import deltas.misc.use_two as use_two_cfg

# rows to show, in order, and how to label them in the paper
PAPER_METHODS = [
    ('Baseline', 'Baseline'),
    ('SMOTE', 'SMOTE \\cite{Chawla_2002_JAIR}'),
    ('Balanced Weights', 'BW'),
    ('BMR', 'BMR \\cite{Bahnsen_2014_SIAM}'),
    ('Threshold', 'Thresh \\cite{Sheng_2006_AAAI}'),
    ('Slacks Deltas', 'Slacks Deltas \\cite{Clifford_2024_ECAI}'),
    ('F Deltas', 'F Deltas'),
    ('Min Deltas', 'Min Deltas'),
    ('CP Sum', 'CP sum'),
    ('CP Minimax', 'CP minimax'),
    ('DKW Sum', 'DKW sum'),
    ('DKW Minimax', 'DKW minimax'),
]


def fmt(mu, sd):
    if np.isnan(mu):
        return '--'
    return f"${f'{mu:.3f}'.lstrip('0')} \\pm {f'{sd:.2f}'.lstrip('0')}$"


def build(datasets=(1, 2, 3, 4, 5), show_solved=True):
    lines = []
    ncol = len(METRICS) + (1 if show_solved else 0)
    lines.append('\\begin{tabular}{@{}ll' + 'c' * ncol + '@{}}')
    lines.append('\\toprule')
    head = '& Methods & ' + ' & '.join(METRICS)
    if show_solved:
        head += ' & solved'
    lines.append(head + ' \\\\')

    for idx in datasets:
        dataset = EXPERIMENTS[idx]['dataset']
        model = EXPERIMENTS[idx]['model']
        path = os.path.join(RESULTS, f'agg-{idx}-{dataset}.csv')
        if not os.path.exists(path):
            print(f'  skipping {dataset} (no results yet)')
            continue
        agg = pd.read_csv(path, index_col=0)
        present = [(k, lab) for k, lab in PAPER_METHODS if k in agg.index]

        lines.append('\\midrule')
        lines.append(f'% {dataset} - {model} - '
                     f'USE_TWO={use_two_cfg.USE_TWO}')
        n_rows = len(present)
        lines.append('\\multirow{' + str(n_rows) + '}{*}{\\rotatebox{90}{'
                     + SHORT_NAMES[dataset] + '}}')

        best = {}
        for m in METRICS:
            vals = [agg.loc[k, f'{m} mean'] for k, _ in present]
            best[m] = int(np.nanargmax(vals)) if not np.all(
                np.isnan(vals)) else -1
        for i, (key, label) in enumerate(present):
            cells = []
            for m in METRICS:
                s = fmt(agg.loc[key, f'{m} mean'], agg.loc[key, f'{m} std'])
                if i == best[m] and s != '--':
                    s = s.replace('$', '', 1)
                    s = '$\\textbf{' + s.split(' \\pm ')[0] + '} \\pm ' + \
                        s.split(' \\pm ')[1]
                cells.append(s)
            row = f'& {label} & ' + ' & '.join(cells)
            if show_solved:
                row += (f" & {int(agg.loc[key, 'n'])}"
                        f"/{int(agg.loc[key, 'total'])}")
            lines.append(row + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines)


if __name__ == '__main__':
    table = build()
    out = os.path.join(RESULTS, 'combined_table-groundup.txt')
    with open(out, 'w') as f:
        f.write(table)
    print(table)
    print('\nwrote', out)
