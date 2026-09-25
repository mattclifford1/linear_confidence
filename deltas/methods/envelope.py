'''
The shape-aware methods of the other-concentration working notes.

Each reports two certificates at delta = 0.05 - a model-based one (tight if
the shape assumption holds) and Clopper-Pearson (valid regardless) - so the
reader never has to trust the model to get *some* valid statement.

A data-dependent transform (Yeo-Johnson) has to be fitted on the classifier's
fit split first; pass it with METHODS[name].with_params(transform=fitted).
'''
from deltas.bounds import (GaussianConfidence, LocationScaleConfidence,
                           StudentTPredictive)
from deltas.confidence import FixedDelta
from deltas.core import DeltasEstimator
from deltas.methods.base import Method
from deltas.rules import Minimax, NeymanPearson

REF = 'deltas-other-concentration working notes, Sec. 5-6'
CERTIFY = [GaussianConfidence(), 'clopper_pearson']


def _est(**kw):
    def build(clf):
        return DeltasEstimator(clf, **kw)
    return build


METHODS = [
    Method('Gaussian Predictive',
           _est(bound=StudentTPredictive(), rule=Minimax(), certify=CERTIFY),
           family='envelope', reference=REF,
           description='decide with the Student-t predictive errors, '
                       'certify with the Gaussian envelope and CP'),
    Method('Gaussian Envelope',
           _est(bound=GaussianConfidence(), confidence=FixedDelta(0.05),
                rule=Minimax(), certify=CERTIFY),
           family='envelope', reference=REF,
           description='minimax of the 95% Gaussian confidence envelopes'),
    Method('Gaussian Envelope (MC band)',
           _est(bound=GaussianConfidence(band='mc'),
                confidence=FixedDelta(0.05), rule=Minimax(),
                certify=[GaussianConfidence(band='mc'), 'clopper_pearson']),
           family='envelope', reference=REF,
           description='as Gaussian Envelope, simultaneous Monte Carlo band'),
    Method('Logistic Envelope',
           _est(bound=LocationScaleConfidence('logistic'),
                confidence=FixedDelta(0.05), rule=Minimax(),
                certify=[LocationScaleConfidence('logistic'), 'clopper_pearson']),
           family='envelope', reference=REF,
           description='heavier-tailed hedge: logistic confidence envelopes'),
    Method('Gaussian Neyman-Pearson',
           _est(bound=GaussianConfidence(), confidence=FixedDelta(0.05),
                rule=NeymanPearson(alpha=0.1), certify=CERTIFY),
           family='envelope', reference=REF,
           description='miss at most 10% of the minority (95% confidence)'),
    Method('Saw-Yang-Mo Minimax',
           _est(bound='saw_yang_mo', rule=Minimax(),
                certify=['clopper_pearson']),
           family='envelope', reference=REF,
           description='moments without a shape (a comparator)'),
]
