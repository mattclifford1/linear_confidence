'''
FixedDelta validates its level.
'''
import pytest

from deltas.confidence import FixedDelta


def test_fixed_delta_validates():
    with pytest.raises(ValueError):
        FixedDelta(0.0)
