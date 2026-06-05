import numpy as np

from sacsma.metrics import kge, nse, pbias


def test_perfect_scores():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert nse(obs, obs) == 1.0
    assert abs(kge(obs, obs) - 1.0) < 1e-12
    assert abs(pbias(obs, obs)) < 1e-12


def test_nan_handling():
    sim = np.array([1.0, np.nan, 3.0])
    obs = np.array([1.0, 2.0, 3.0])
    # NaN pair dropped; remaining two match perfectly.
    assert nse(sim, obs) == 1.0


def test_bias():
    obs = np.array([1.0, 1.0, 1.0, 1.0])
    sim = obs * 1.1
    assert abs(pbias(sim, obs) - 10.0) < 1e-9
