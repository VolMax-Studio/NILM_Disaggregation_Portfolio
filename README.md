# NILM (Non-Intrusive Load Monitoring) Disaggregation Portfolio

Energy disaggregation pipeline using **real low-frequency data from the REDD dataset (House 3)**. This repository implements a Random Forest regressor to disaggregate the power consumption of a **refrigerator** from the aggregate mains signal, using physics-informed features to capture load transitions and standby baselines.

---

## ⚡ The Philosophy: Physics-Informed ML vs. "Edge AI Bloat"

Many modern machine learning approaches to Non-Intrusive Load Monitoring (NILM) treat energy consumption as generic time-series data, leading to complex, over-parameterized neural networks that run poorly on edge hardware. 

This repository takes a **physics-first** approach:
1. **State Transitions ($dP$)**: We calculate the first-order difference of active power. In electrical grids, when a device switches ON or OFF, the active power curve displays a discrete step change. Extracting these step changes isolates transient load events from background noise.
2. **Standby Baselines (Rolling Min)**: Rolling minimum values over the sliding window serve as a proxy for the background standby/always-on load, helping the model separate constant loads (e.g., standby electronics) from cyclical loads (e.g., refrigerator compressors).
3. **Local Volatility (Rolling Std/Range)**: High-power cyclical devices (like microwaves) introduce local variance, which is captured using rolling standard deviation to inform the model of volatile loads.

---

## 📊 Dataset Structure & Real REDD Data

Unlike synthetic datasets, which produce unrealistically high metrics ($R^2 > 0.95$), this model is validated on **real low-frequency data from House 3 of the Reference Energy Disaggregation Data Set (REDD)**. 

The aligned training set (`house3_5devices_train.csv`) tracks five sub-metered appliances:
*   `lighting2`
*   `lighting5`
*   `lighting4`
*   `refrigerator` (Target Appliance)
*   `microwave`

The aggregate mains signal is constructed by summing the individual active sub-meters, mimicking the total monitored load.

---

## ⚙️ Feature Engineering Pipeline

For any timestamp $t$, the feature matrix includes:
*   **Temporal Sliding Window**: Centered window of aggregate power $P_{agg}(t-W)$ to $P_{agg}(t+W)$.
*   **Power Differentials ($dP$)**: Centered window of first-order differences $dP(t) = P_{agg}(t) - P_{agg}(t-1)$.
*   **Rolling Statistics**: Rolling mean, standard deviation, minimum, maximum, and range of the aggregate power over the centered window.

---

## 📈 Model Performance & Honest Evaluation

Evaluation is conducted using a strict **chronological split (80% train, 20% test)** to prevent data leakage (a common issue in sliding-window time-series splits).

### Results on Real REDD Data
*   **Mean Absolute Error (MAE)**: `17.91 Watts`
*   **R² Score**: `0.464`
*   **Relative Energy Error (RE)**: `4.76%`

### Engineering Takeaways
1. **Low instantaneous $R^2$ ($0.464$)** is a realistic and honest performance indicator for low-frequency active power signals (sampled at ~17s intervals) without high-frequency voltage/current waveforms.
2. **Extremely low Relative Energy Error ($4.76\%$)** shows that while instantaneous predictions are moderately fitted, the model is highly accurate at calculating the total accumulated energy consumption of the refrigerator over time, making it viable for billing and eco-feedback applications.

---

## 🚀 How to Run

### Prerequisites
Install dependencies:
```bash
pip install numpy pandas scikit-learn
```

### Run Pipeline
Execute the disaggregation script:
```bash
python3 nilm_starter.py
```
