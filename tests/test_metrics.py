"""M8: prognostic metric functions."""

import numpy as np
import pytest

from evaluation import rmse, mae, coverage, alpha_lambda_accuracy, prognostic_horizon


def test_rmse_and_mae():
    assert rmse([1, 2, 3], [1, 2, 4]) == pytest.approx((1/3) ** 0.5)
    assert mae([1, 2, 3], [1, 2, 4]) == pytest.approx(1/3)

def test_coverage():
    assert coverage([1, 5, 9], [0, 0, 0], [4, 4, 4]) == pytest.approx(1/3)  # only first inside

def test_alpha_lambda_accuracy():
    assert alpha_lambda_accuracy([10, 20], [10, 20]) == 1.0       # perfect
    assert alpha_lambda_accuracy([20, 40], [10, 20]) == 0.0       # 2x off, outside +/-20%
    assert alpha_lambda_accuracy([11, 19], [10, 20], 0.2) == 1.0  # within +/-20%

def test_prognostic_horizon_locks_on():
    cuts = [1, 2, 3, 4, 5, 6, 7]
    true = np.array([70, 60, 50, 40, 30, 20, 10])
    pred = np.array([200, 150, 90, 55, 33, 22, 11])   # in-band from cut 5 on
    assert prognostic_horizon(cuts, pred, true, 0.2) == 5

def test_prognostic_horizon_never_locks():
    cuts = [1, 2, 3]
    true = np.array([30, 20, 10])
    pred = np.array([100, 100, 100])                  # never in band
    assert prognostic_horizon(cuts, pred, true, 0.2) is None
