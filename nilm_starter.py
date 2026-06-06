"""
NILM (Non-Intrusive Load Monitoring) Disaggregation Portfolio
=============================================================
This script provides an energy disaggregation pipeline using real low-frequency
data from the REDD dataset (House 3). It takes the aggregate power signal of the
monitored appliances and extracts the power consumption of the Refrigerator.

Physics-informed features (first-order power differences and rolling window
statistics) are engineered to capture load transitions (ON/OFF step changes)
and standby baselines.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_real_redd_data(file_path):
    """Loads and prepares real REDD House 3 aligned data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    print(f"Loading real REDD data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert UNIX timestamps to datetime index for proper time-series handling
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('timestamp').drop(columns=['time'])
    
    # In REDD, the total monitored aggregate is the sum of sub-metered channels
    # (Since we only have these 5 channels in the subset, their sum is our aggregate load)
    device_cols = ['lighting2', 'lighting5', 'lighting4', 'refrigerator', 'microwave']
    df['aggregate'] = df[device_cols].sum(axis=1)
    
    return df

def create_physics_informed_features(series, window_size=5):
    """Creates sliding window features and physics-informed delta features.
       For a given time T, features include:
       - Raw aggregate power from T-W to T+W (sliding window).
       - First-order differences (dP = P_t - P_t-1) to detect step transitions.
       - Rolling statistics (mean, std, min, max, range) over the centered window.
    """
    df = pd.DataFrame(index=series.index)
    df['aggregate'] = series
    
    # 1. Temporal sliding window (past and future values)
    for i in range(1, window_size + 1):
        df[f't-{i}'] = series.shift(i)
        df[f't+{i}'] = series.shift(-i)
        
    # 2. Physics-informed Delta (first-order difference) to detect step changes (ON/OFF transitions)
    df['diff_t'] = series.diff()
    for i in range(1, window_size + 1):
        df[f'diff_t-{i}'] = df[f't-{i}'].diff()
        
    # 3. Rolling window statistics
    rolling = series.rolling(window=2 * window_size + 1, center=True)
    df['rolling_mean'] = rolling.mean()
    df['rolling_std'] = rolling.std()
    df['rolling_min'] = rolling.min()
    df['rolling_max'] = rolling.max()
    df['rolling_range'] = df['rolling_max'] - df['rolling_min']
    
    return df.dropna()

def main():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "house3_5devices_train.csv")
    df = load_real_redd_data(data_path)
    
    TARGET_APPLIANCE = 'refrigerator'
    print(f"Dataset shape: {df.shape}")
    print(f"Targeting appliance: {TARGET_APPLIANCE}")
    
    # 2. Feature Engineering (Sliding Window & Physics Features on Aggregate)
    print("Engineering physics-informed features from aggregate mains signal...")
    features = create_physics_informed_features(df['aggregate'], window_size=5)
    
    # Align target with features (drops NaN rows caused by window shifting)
    target = df[TARGET_APPLIANCE].loc[features.index]
    
    # 3. Chronological Train/Test Split (80/20) to prevent temporal leakage
    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = target.iloc[split_idx:]
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # 4. Model Training
    print("Training RandomForest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluation
    y_pred = model.predict(X_test)
    
    # Post-process: power consumption cannot be negative
    y_pred = np.clip(y_pred, 0, None)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Relative Error in Energy (RE): standard NILM metric for total energy estimation
    total_energy_actual = y_test.sum()
    total_energy_pred = y_pred.sum()
    relative_energy_error = abs(total_energy_pred - total_energy_actual) / total_energy_actual
    
    print("\n--- Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} Watts")
    print(f"R2 Score: {r2:.3f}")
    print(f"Relative Energy Error (RE): {relative_energy_error * 100:.2f}%")
    
    print("\nHonest Engineering Assessment (Real REDD Data):")
    print("1. R2 Score represents realistic performance on low-frequency active power signals.")
    print("2. First-order delta features (dP) capture the step transients of the refrigerator cycle.")
    print("3. Rolling min represents baseline standby power, which helps isolate independent appliance loads.")

if __name__ == "__main__":
    main()
