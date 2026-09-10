'''
Cost-sensitive deltas: does supplying real misclassification costs move the
certified boundary the right way?

WHY THIS EXPERIMENT EXISTS
--------------------------
`overlap.base_overlap_deltas.fit` takes `costs=(c1, c2)` and multiplies each
class's certified loss by it, so the objective is

    sum rule      c1 L1(b) + c2 L2(b)
    minimax rule  max(c1 L1(b), c2 L2(b))

which is an upper bound on the *cost-weighted* risk rather than the balanced
one. That argument has been in the code since the published paper and has
never been exercised: every experiment in this repo, and both papers, run with
`costs = (1, 1)`.

The three Costcla datasets ship genuine per-sample cost matrices, so we can
supply real costs and measure real cost. This is the natural home for a
comparison against BMR, which is itself a cost-sensitive method.

COSTS
-----
The matrices are `(n, 4)` columns `[FP, FN, TP, TN]`. What matters to a
decision is the *net* cost of getting a sample wrong rather than right:

    net cost of an error on class 0   =  FP - TN
    net cost of an error on class 1   =  FN - TP

Class costs are the means of those over the TRAINING split only - using the
test matrix would leak. Evaluation then charges the actual per-sample cost of
every decision made on the test set, which is the number these datasets exist
to measure.

    uv run python run_costs.py
    uv run python run_costs.py --datasets "Costcla Direct Marketing"
'''
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from data_loaders import get_dataset                       # noqa: E402
from data_loaders.utils import proportional_split          # noqa: E402
from sklearn.model_selection import train_test_split       # noqa: E402

import projection_models as pm                             # noqa: E402
from deltas.classifiers.frozen import FrozenProjection     # noqa: E402
from deltas.model import overlap                           # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

DATASETS = ['Costcla Direct Marketing',
            'Costcla Credit Scoring Kaggle 2011',
            'Costcla Credit Scoring PAKDD 2009']

MODELS = {
    'Linear': lambda: pm.Linear(max_iter=2000),
    'LDA': lambda: pm.LDA(),
    'RandomForest': lambda: pm.RandomForest(n_estimators=200),
    'GradientBoosting': lambda: pm.GradientBoosting(),
}

CAL_FRAC = 0.35
TARGET_RATIO = 10.0


# ------------------------------------------------------------------ costs ---
def net_costs(cost_matrix):
    '''
    (c1, c2): mean net cost of an error on class 0 and on class 1

    columns are [FP, FN, TP, TN]; an error on class 0 costs FP where the right
    call would have cost TN, and an error on class 1 costs FN against TP.
    '''
    cm = np.asarray(cost_matrix, dtype=float)
    c1 = float(np.mean(cm[:, 0] - cm[:, 3]))
    c2 = float(np.mean(cm[:, 1] - cm[:, 2]))
    return max(c1, 1e-9), max(c2, 1e-9)


def total_cost(y, preds, cost_matrix):
    '''charge every decision its own per-sample cost'''
    cm = np.asarray(cost_matrix, dtype=float)
    y = np.asarray(y).astype(int)
    p = np.asarray(preds).astype(int)
    cost = np.where(
        y == 0,
        np.where(p == 1, cm[:, 0], cm[:, 3]),      # FP else TN
        np.where(p == 0, cm[:, 1], cm[:, 2]))      # FN else TP
    return float(cost.sum())


def best_threshold_by_cost(z, y, cost_matrix):
    '''
    the cost-minimising threshold on these projections

    Used two ways: on the training projections it is the cost-sensitive
    threshold-moving baseline; on the test projections it is an oracle, the
    best any bias-shifting method could have done.
    '''
    order = np.argsort(z)
    zs = z[order]
    cands = np.unique(zs)
    if len(cands) < 2:
        return float(cands[0]) if len(cands) else 0.0
    mids = (cands[:-1] + cands[1:]) / 2.0
    # evaluate on a bounded grid so this stays cheap on the larger datasets
    if len(mids) > 4000:
        mids = np.quantile(mids, np.linspace(0, 1, 4000))
    costs = [total_cost(y, (z > m).astype(int), cost_matrix) for m in mids]
    return float(mids[int(np.argmin(costs))])


# ------------------------------------------------------------------- data ---
def load(name, seed):
    loader = get_dataset(name)
    dd = loader.get_data_dict()
    X = np.asarray(dd['X'], dtype=float)
    y = np.asarray(dd['y']).squeeze().astype(int)
    cm = np.asarray(dd['cost_matrix'], dtype=float)
    if np.bincount(y, minlength=2)[1] > np.bincount(y, minlength=2)[0]:
        y = 1 - y
        cm = cm[:, [1, 0, 3, 2]]        # swapping labels swaps the columns

    n_maj_train = int(np.bincount(y, minlength=2)[0] * 0.5)
    n_min = int(np.bincount(y, minlength=2)[1])
    n_min_train = int(np.clip(int(n_maj_train / TARGET_RATIO), 8, n_min // 2))

    train, test = proportional_split(
        {'X': X, 'y': y, 'cost_matrix': cm}, train_size=0.5, seed=seed,
        minority_reduce_scaler=n_maj_train / n_min_train, equal_test=False)
    # the population base rate, before the training minority is thinned. A
    # practitioner knows this; the thinned training prior is an artefact of the
    # experimental protocol and would be the wrong thing to correct costs by.
    prior_1 = float((y == 1).mean())
    return train, test, prior_1


# ------------------------------------------------------------------- runs ---
def run_cell(name, model_name, seed):
    train, test, prior_1 = load(name, seed)
    Xtr, ytr, cmtr = train['X'], train['y'], train['cost_matrix']
    Xte, yte, cmte = test['X'], test['y'], test['cost_matrix']
    c1, c2 = net_costs(cmtr)

    rows = []
    for mode in ('naive', 'split'):
        if mode == 'naive':
            X_fit, y_fit, cm_fit = Xtr, ytr, cmtr
            X_cert, y_cert, cm_cert = Xtr, ytr, cmtr
        else:
            # the cost matrix is per-sample, so it has to travel with the
            # split - slicing it by length would attach each sample's cost to
            # whichever row happened to land in that position
            (X_fit, X_cert, y_fit, y_cert,
             cm_fit, cm_cert) = train_test_split(
                Xtr, ytr, cmtr, test_size=CAL_FRAC, stratify=ytr,
                random_state=seed)

        clf = MODELS[model_name]().fit(X_fit, y_fit)
        z_fit = np.asarray(clf.get_projection(X_fit)).reshape(-1)
        z_cert = np.asarray(clf.get_projection(X_cert)).reshape(-1)
        z_test = np.asarray(clf.get_projection(Xte)).reshape(-1)
        thr = float(np.asarray(clf.get_threshold()).reshape(-1)[0])

        base = {'dataset': name, 'model': model_name, 'seed': seed,
                'mode': mode, 'c1': c1, 'c2': c2, 'cost_ratio': c2 / c1,
                'n_test': int(len(yte))}

        def add(method, preds):
            rows.append({**base, 'method': method,
                         'cost': total_cost(yte, preds, cmte),
                         'err_0': float((preds[yte == 0] != 0).mean()),
                         'err_1': float((preds[yte == 1] != 1).mean())})

        add('Baseline', (z_test > thr).astype(int))
        # cost-sensitive threshold moving, chosen on whatever data the deltas
        # methods are allowed to see in this mode, so the comparison is fair
        t_cost = best_threshold_by_cost(z_cert, y_cert, cm_cert)
        add('Threshold (cost)', (z_test > t_cost).astype(int))

        frozen = FrozenProjection(thr)
        Xc, Xt = z_cert[:, None], z_test[:, None]
        # Both rules, because they mean different things here. Only the SUM
        # rule bounds the cost-weighted risk, c1 L1 + c2 L2; minimax equalises
        # the two certified terms, which is a robustness criterion and has no
        # reason to minimise expected cost. Reporting only minimax would have
        # made the costs argument look useless.
        # Prior-corrected costs. deltas' `costs` multiply the certified
        # class-conditional error *rates*, but what a decision maker minimises
        # is the risk pi_0 c1 e_1 + pi_1 c2 e_2. Passing raw costs therefore
        # over-weights the minority by exactly the imbalance factor, which is
        # the regime this method exists for. Estimate the priors from the
        # population base rate, never from test.
        pi1, pi0 = prior_1, 1.0 - prior_1
        for label, costs in (('uncosted', (1, 1)),
                             ('costed', (c1, c2)),
                             ('prior-costed', (pi0 * c1, pi1 * c2))):
            for rule in ('sum', 'minimax'):
                for mname, M in (('CP', overlap.binomial_deltas),
                                 ('DKW', overlap.dkw_deltas)):
                    m = M(frozen, objective=rule).fit(Xc, y_cert, costs=costs)
                    add(f'{mname} {rule} ({label})',
                        np.asarray(m.predict(Xt)).squeeze().astype(int))

        # oracle: the best a bias shift could possibly do, chosen on test
        add('Oracle (test)',
            (z_test > best_threshold_by_cost(z_test, yte, cmte)).astype(int))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=DATASETS)
    ap.add_argument('--models', nargs='*', default=list(MODELS))
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        for model_name in args.models:
            for seed in range(args.seeds):
                try:
                    rows += run_cell(name, model_name, seed)
                except Exception as exc:
                    print(f'  ! {name} / {model_name} / {seed}: '
                          f'{type(exc).__name__}: {exc}', flush=True)
            print(f'  done {name} / {model_name}', flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, 'costs.csv'), index=False)

    # cost relative to the oracle, per (dataset, model, seed, mode)
    key = ['dataset', 'model', 'seed', 'mode']
    oracle = (df[df.method == 'Oracle (test)']
              .set_index(key)['cost'].rename('oracle'))
    d = df.join(oracle, on=key)
    d['rel'] = d['cost'] / d['oracle']

    print('\n=== cost relative to the test-set oracle (1.00 = optimal) ===')
    piv = d.pivot_table(index='method', columns='mode', values='rel',
                        aggfunc='mean')
    print(piv.round(3).to_string())

    print('\n=== by dataset, naive, mean relative cost ===')
    piv2 = d[d['mode'] == 'naive'].pivot_table(
        index='method', columns='dataset', values='rel', aggfunc='mean')
    piv2.columns = [c.replace('Costcla ', '') for c in piv2.columns]
    print(piv2.round(3).to_string())

    print(f'\nwrote {os.path.join(RESULTS, "costs.csv")} ({len(df)} rows)')
    with open(os.path.join(RESULTS, 'costs_config.json'), 'w') as f:
        json.dump({'datasets': args.datasets, 'models': args.models,
                   'seeds': args.seeds, 'cal_frac': CAL_FRAC}, f, indent=2)


if __name__ == '__main__':
    main()
