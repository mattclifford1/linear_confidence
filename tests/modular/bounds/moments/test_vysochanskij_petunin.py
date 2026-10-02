'''
Vysochanskij-Petunin: tighter than Cantelli in the tail, continuous where it switches regime.
'''
import numpy as np
import pytest

from deltas.bounds import Cantelli, VysochanskijPetunin


def test_unimodal_tail_is_tighter_than_cantelli_in_the_tail():
    k = np.array([1.5, 2.0, 4.0])
    assert np.all(VysochanskijPetunin(score_range=(0, 1)).tail(k) <
                  Cantelli(score_range=(0, 1)).tail(k))
    # and both tails are continuous where VP switches regime
    kk = np.sqrt(5 / 3)
    vp = VysochanskijPetunin(score_range=(0, 1))
    assert vp.tail(kk - 1e-9) == pytest.approx(vp.tail(kk + 1e-9), abs=1e-7)
