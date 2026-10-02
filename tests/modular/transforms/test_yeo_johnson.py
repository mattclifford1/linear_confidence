'''
YeoJohnson: the inverse across lambda regimes.
'''
import numpy as np
import pytest

from deltas.transforms import YeoJohnson


@pytest.mark.parametrize('lam_data', [
    np.random.default_rng(2).normal(0, 1, 300),                  # lambda ~ 1
    np.random.default_rng(3).lognormal(0, 1, 300),               # lambda < 1
    -np.random.default_rng(4).lognormal(0, 1, 300) + 5.0,        # lambda > 1
])
def test_yeo_johnson_inverse_across_lambdas(lam_data):
    t = YeoJohnson().fit(lam_data)
    grid = np.linspace(lam_data.min(), lam_data.max(), 50)
    assert np.allclose(t.inverse(t(grid)), grid, atol=1e-8)
