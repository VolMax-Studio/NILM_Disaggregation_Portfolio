"""
NILM Starter -- Non-Intrusive Load Monitoring (energy disaggregation)
=====================================================================
Estimate per-appliance power from the whole-house aggregate meter signal.

Runs OUT OF THE BOX on the provided real REDD House 3 data subset if found.
Otherwise, falls back to a synthetic REDD-format demo.

Honest baseline:
  - per-appliance RandomForest regressor on windowed aggregate features
  - TIME-based train/test split (NILM must not shuffle -> avoids leakage)
  - reports per-appliance MAE, energy-fraction error, and ON/OFF F1

This is the honest starting line. State-of-the-art NILM (seq2point CNNs,
HMMs) does better -- but this baseline is correct, transparent, and shows you
understand the data, not just a black-box toolkit. Swap in real data and the
numbers drop to realistic (NILM is genuinely hard) -- that's the point.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score

REAL_DATA_FILE = "house3_5devices_train.csv"
N = 30000                # synthetic sample count (used only if real data is missing)
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_real_redd(file_path):
    """Load pre-processed real REDD House 3 aligned data."""
    print(f"Loading real REDD House 3 data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Convert UNIX timestamps to datetime index
    idx = pd.to_datetime(df["time"], unit="s")
    
    # Target appliances
    appliance_names = ["lighting2", "lighting5", "lighting4", "refrigerator", "microwave"]
    appliances = {name: pd.Series(df[name].values, index=idx) for name in appliance_names}
    
    # Sum monitored sub-meters to get the aggregate mains signal
    mains = pd.Series(df[appliance_names].sum(axis=1).values, index=idx)
    
    return mains, appliances


def make_synthetic():
    """Synthetic REDD-like data: aggregate = sum of appliances + base + noise."""
    print("Running synthetic REDD-format demo.")
    idx = pd.date_range("2024-01-01", periods=N, freq="3s")
    t = np.arange(N)

    def duty(period, on_len, power, jitter=0.0):
        s = np.zeros(N)
        i = 0
        while i < N:
            start = i + int(RNG.integers(0, period))
            end = min(N, start + on_len + int(RNG.integers(-jitter, jitter + 1)))
            s[start:end] = power * (1 + 0.05 * RNG.standard_normal(max(0, end - start)))
            i = start + period
        return np.clip(s, 0, None)

    appliances = {
        "fridge":  duty(period=300, on_len=120, power=150, jitter=20),
        "washer":  duty(period=4000, on_len=400, power=500, jitter=60),
        "kettle":  duty(period=6000, on_len=40,  power=2000, jitter=8),
        "lighting": (60 + 40 * (np.sin(t / 2000) > 0)).astype(float),
    }
    base = 30.0
    agg = base + sum(appliances.values()) + 8 * RNG.standard_normal(N)
    mains = pd.Series(np.clip(agg, 0, None), index=idx)
    appliances = {k: pd.Series(v, index=idx) for k, v in appliances.items()}
    return mains, appliances


# --------------------------------------------------------------------------
# Features + evaluation
# --------------------------------------------------------------------------
def build_features(mains, w):
    m = mains.astype(float)
    return pd.DataFrame({
        "p": m,
        "roll_mean": m.rolling(w, min_periods=1).mean(),
        "roll_std": m.rolling(w, min_periods=1).std().fillna(0),
        "diff": m.diff().fillna(0),
        "roll_max": m.rolling(w, min_periods=1).max(),
        "roll_min": m.rolling(w, min_periods=1).min(),
    }, index=m.index)


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    real_data_path = os.path.join(dir_path, REAL_DATA_FILE)
    
    if os.path.exists(real_data_path):
        mains, appliances = load_real_redd(real_data_path)
        is_real = True
        w = 5  # Smaller window for lower frequency real data (~17s samples)
    else:
        mains, appliances = make_synthetic()
        is_real = False
        w = 30  # Larger window for high frequency synthetic data (3s samples)

    # align everything to the mains index
    df = build_features(mains, w)
    for name, s in appliances.items():
        df[f"appl_{name}"] = s.reindex(df.index).interpolate().fillna(0)
    df = df.dropna()

    feat_cols = ["p", "roll_mean", "roll_std", "diff", "roll_max", "roll_min"]
    appl_cols = [c for c in df.columns if c.startswith("appl_")]

    # TIME-based split (first 70% train, last 30% test)
    split = int(len(df) * 0.7)
    tr, te = df.iloc[:split], df.iloc[split:]

    print(f"Samples: {len(df)} | train {len(tr)} / test {len(te)} | "
          f"appliances: {[c[5:] for c in appl_cols]}\n")
    print(f"{'appliance':<14}{'MAE (W)':>10}{'energy err':>12}{'on/off F1':>11}")
    print("-" * 49)

    for col in appl_cols:
        reg = RandomForestRegressor(n_estimators=100, max_depth=12,
                                    random_state=0, n_jobs=-1)
        reg.fit(tr[feat_cols], tr[col])
        pred = np.clip(reg.predict(te[feat_cols]), 0, None)
        true = te[col].values

        mae = np.mean(np.abs(pred - true))
        energy_err = abs(pred.sum() - true.sum()) / (true.sum() + 1e-9)
        thr = max(10, 0.1 * true.max())
        f1 = f1_score(true > thr, pred > thr, zero_division=0)
        print(f"{col[5:]:<14}{mae:>10.1f}{energy_err:>12.1%}{f1:>11.2f}")

    print("\nHow to read this:")
    print("  MAE        = avg watts off, per appliance (lower better)")
    print("  energy err = % error in total energy assigned (lower better)")
    print("  on/off F1  = how well on/off cycles are caught (1.0 = perfect)")
    print("\nOn synthetic data these look good; on real REDD they get")
    print("harder and more honest. That difficulty IS the NILM problem.")


if __name__ == "__main__":
    main()
