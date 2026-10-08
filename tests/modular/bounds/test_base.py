'''
Properties every moment-based bound shares.
'''
import numpy as np

from deltas.bounds import GaussianConfidence, SawYangMo, StudentTPredictive
from deltas.core import ClassSample


def test_degenerate_samples_carry_no_information():
    for z in ([1.0], [2.0, 2.0, 2.0]):
        s = ClassSample(z, 0, 'low')
        for B in (GaussianConfidence(), StudentTPredictive(), SawYangMo()):
            assert np.all(B.fit(s).curve(np.array([5.0, 50.0]), 0.05) == 1.0)
