"""
Navigation error metrics for behavioral cloning evaluation.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_navigation_error(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true and y_pred are shape (N, 3) arrays: [steer, throttle, brake]
    Returns: {"steer_mae": float, "throttle_mae": float, "brake_mae": float, "overall_mae": float, "steer_rmse": float}
    """
    steer_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    throttle_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    brake_mae = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
    
    overall_mae = mean_absolute_error(y_true, y_pred)
    
    steer_rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    
    return {
        "steer_mae": float(steer_mae),
        "throttle_mae": float(throttle_mae),
        "brake_mae": float(brake_mae),
        "overall_mae": float(overall_mae),
        "steer_rmse": float(steer_rmse)
    }

def compute_improvement_pct(baseline_error: float, improved_error: float) -> float:
    """
    Returns percentage improvement
    """
    if baseline_error == 0:
        return 0.0
    return ((baseline_error - improved_error) / baseline_error) * 100.0
