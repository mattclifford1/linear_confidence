'''
The abstract machinery of a deltas method: the estimator, the shared component
base class, the per-class samples and the component registry. Each slot's own
base class lives next to its implementations, in the base.py of its package
(deltas.bounds.base.Bound, deltas.rules.base.DecisionRule, ...). Nothing here
knows about any particular inequality.
'''
from deltas.core.component import Component
from deltas.core.estimator import DeltasEstimator
from deltas.core.sample import ClassSample, ProjectedData

__all__ = ['ClassSample', 'Component', 'DeltasEstimator', 'ProjectedData']
