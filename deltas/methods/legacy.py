'''
The legacy methods, with the options the experiment runners use. The code is
frozen (deltas/legacy/); these entries only name it and fix its options.
'''
from deltas.methods.base import Method, lazy

ECAI = 'Clifford et al., ECAI 2024'
NON_SEP = 'deltas-non-separable draft, Dec 2024'
_DOWNSAMPLE = ('deltas.model.downsample', 'downsample_deltas')
_NON_SEP = ('deltas.model.non_sep', 'deltas')

METHODS = [
    Method('Slacks Deltas', lazy(*_DOWNSAMPLE),
           dict(max_trials=10000, parallel=True), family='published',
           reference=ECAI, target=':'.join(_DOWNSAMPLE),
           description='the published method: fences meet, binary slacks'),
    Method('Slacks Deltas (continuous)', lazy(*_DOWNSAMPLE),
           dict(max_trials=10000, parallel=True, continuous_slacks=True),
           family='published', reference=ECAI + ', App. B',
           target=':'.join(_DOWNSAMPLE),
           description='the published method with continuous slacks'),
    Method('Min Deltas', lazy(*_NON_SEP), dict(loss_type='min'),
           family='non_separable', reference=NON_SEP,
           target=':'.join(_NON_SEP),
           description='k-th point fences, minimum over k'),
    Method('Max Deltas', lazy(*_NON_SEP), dict(loss_type='max'),
           family='non_separable', reference=NON_SEP,
           target=':'.join(_NON_SEP),
           description='k-th point fences, maximum over k'),
    Method('Avg Deltas', lazy(*_NON_SEP), dict(loss_type='mean'),
           family='non_separable', reference=NON_SEP,
           target=':'.join(_NON_SEP),
           description='k-th point fences, mean over k'),
    Method('F Deltas', lazy(*_NON_SEP), dict(only_furtherest_k=True),
           family='non_separable', reference=NON_SEP,
           target=':'.join(_NON_SEP),
           description='k-th point fences, furthest point only'),
]
