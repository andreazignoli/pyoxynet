"""The small helpers in pyoxynet.utilities."""
import numpy as np
import pytest

from pyoxynet import utilities


@pytest.mark.parametrize('text, seconds', [
    ('00:30', 30), ('01:00', 60), ('10:30', 630), ('00:00', 0), ('0:0', 0),
    ('00:00:30', 30), ('01:00:00', 3600), ('01:30:45', 5445),
])
def test_get_sec(text, seconds):
    assert utilities.get_sec(text) == seconds


def test_optimal_filter_smooths_and_keeps_the_length():
    rng = np.random.default_rng(0)
    t = np.arange(300, dtype=float)
    y = np.sin(t / 30) + rng.normal(0, 0.3, t.size)
    smooth = utilities.optimal_filter(t, y, 100)
    assert smooth.shape == y.shape
    assert np.std(np.diff(smooth)) < np.std(np.diff(y)) / 10
    assert np.max(np.abs(smooth - np.sin(t / 30))) < 0.5
