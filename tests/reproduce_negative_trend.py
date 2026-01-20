
import numpy as np

def test_scale_decay():
    est_scale = 0.6578
    est_trend = -0.051384
    
    # Logic from prediction_engine.py
    est_scale = np.clip(est_scale, 0.5, 3.0)
    est_trend = np.clip(est_trend, -0.05, 0.05)
    
    print(f"Clipped Scale: {est_scale}")
    print(f"Clipped Trend: {est_trend}")
    
    decay_lambda = np.log(2) / 12.0
    print(f"Decay Lambda: {decay_lambda}")
    
    # Simulate future deltas (e.g., 100 hours into future)
    future_deltas = np.linspace(0, 100, 100)
    
    trend_impact = est_trend * (1.0 - np.exp(-decay_lambda * future_deltas)) / decay_lambda
    scale_curve = est_scale + trend_impact
    
    print(f"Min Scale: {np.min(scale_curve)}")
    
    if np.min(scale_curve) < 0:
        print("ISSUE REPRODUCED: Scale curve goes negative!")
    else:
        print("Issue not reproduced.")

if __name__ == "__main__":
    test_scale_decay()
