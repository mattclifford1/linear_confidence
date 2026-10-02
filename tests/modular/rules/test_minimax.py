'''
Minimax: plateau midpoint in value, and the root-finding refinement.
'''
import numpy as np
import pytest

from deltas.rules import Minimax


def test_minimax_takes_the_plateau_midpoint_in_value():
    c = np.array([0.0, 1.0, 3.0, 10.0])
    L_low = np.array([0.9, 0.2, 0.2, 0.2])
    L_high = np.array([0.1, 0.2, 0.2, 0.9])
    b, losses = Minimax().choose(c, L_low, L_high)
    assert b == 2.0                       # middle of [1, 3] in value
    assert np.array_equal(losses, np.maximum(L_low, L_high))
    assert Minimax().combine(0.3, 0.4) == 0.4


def test_minimax_refine_finds_the_crossing():
    low = lambda b: {'L': np.asarray(b, dtype=float)}           # rises
    high = lambda b: {'L': 1.0 - np.asarray(b, dtype=float)}    # falls
    assert Minimax().refine(low, high, (1.0, 1.0), (0.0, 1.0)) == pytest.approx(0.5)
    assert Minimax().refine(low, high, (1.0, 3.0), (0.0, 1.0)) == pytest.approx(0.75)
    assert Minimax().refine(low, high, (1.0, 1.0), (0.6, 1.0)) is None
