import numpy as np
import sys
import os

# Add the project root to sys.path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compute_navigation_error, compute_improvement_pct

def test_navigation_error_zero():
    y = np.random.rand(10, 3)
    metrics = compute_navigation_error(y, y)
    assert metrics['steer_mae'] == 0.0
    assert metrics['throttle_mae'] == 0.0
    assert metrics['brake_mae'] == 0.0
    assert metrics['overall_mae'] == 0.0

def test_navigation_error_shape():
    y_true = np.random.rand(50, 3)
    y_pred = np.random.rand(50, 3)
    metrics = compute_navigation_error(y_true, y_pred)
    expected_keys = {"steer_mae", "throttle_mae", "brake_mae", "overall_mae", "steer_rmse"}
    assert set(metrics.keys()) == expected_keys

def test_improvement_pct():
    # baseline 0.5, improved 0.35 -> (0.5-0.35)/0.5 = 0.15/0.5 = 0.3 = 30%
    improvement = compute_improvement_pct(0.5, 0.35)
    assert np.isclose(improvement, 30.0)
