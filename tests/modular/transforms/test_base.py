'''
Every transform is strictly increasing and invertible.
'''
import numpy as np
import pytest

from deltas.transforms import Identity, Logit, Standardise, YeoJohnson


@pytest.mark.parametrize('t', [Identity(), Logit(), Standardise(), YeoJohnson()])
def test_transforms_are_increasing_and_invertible(t):
    rng = np.random.default_rng(1)
    z = rng.uniform(0.001, 0.999, 400) if isinstance(t, Logit) else \
        rng.gamma(2.0, 1.5, 400) - 1.0
    if t.requires_fit:
        assert not t.is_fitted
        t.fit(z)
        assert t.is_fitted
    s = np.sort(z)
    ts = t(s)
    assert np.all(np.diff(ts) > 0)
    assert np.allclose(t.inverse(ts), s, rtol=1e-9, atol=1e-9)
