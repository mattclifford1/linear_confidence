'''
The overlap-native methods (Clopper-Pearson / DKW), as the 34-dataset grid
ran them. They go through deltas.model.overlap - a shim over DeltasEstimator,
bit-for-bit the original (tests/modular/test_overlap_equivalence.py).
'''
from deltas.methods.base import Method

REF = 'deltas-non-separable draft, Sec. "Overlap from the ground up"'


def _overlap(cls, objective):
    def build(clf):
        from deltas.model import overlap
        return getattr(overlap, cls)(clf, objective=objective)
    return build


METHODS = [
    Method('CP Sum', _overlap('binomial_deltas', 'sum'), family='overlap',
           reference=REF, description='Clopper-Pearson, sum rule'),
    Method('CP Minimax', _overlap('binomial_deltas', 'minimax'),
           family='overlap', reference=REF,
           description='Clopper-Pearson, minimax rule'),
    Method('DKW Sum', _overlap('dkw_deltas', 'sum'), family='overlap',
           reference=REF, description='DKW, sum rule'),
    Method('DKW Minimax', _overlap('dkw_deltas', 'minimax'), family='overlap',
           reference=REF, description='DKW, minimax rule'),
]
