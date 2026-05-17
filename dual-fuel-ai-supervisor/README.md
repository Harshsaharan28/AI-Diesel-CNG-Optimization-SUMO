# AI-Driven Dual-Fuel Engine Optimization

> XGBoost-based supervisory controller for real-time diesel ↔ CNG fuel-mode switching in urban vehicles, evaluated across two Indian road networks using SUMO microscopic traffic simulation.

**Paper:** AI-Driven Dual-Fuel Engine Optimization for Sustainable Mobility and Improved Efficiency  
**Conference:** ICICDAQ 2026  
**Authors:** Dr. V.S. Saranya · Harsh Saharan · Divyansh Praveen — SRMIST, Chengalpattu  

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Pipeline Run Order](#pipeline-run-order)
- [Controllers](#controllers)
- [Model Training](#model-training)
- [Cross-Network Validation](#cross-network-validation)
- [Outputs](#outputs)
- [File Reference](#file-reference)
- [Known Limitations](#known-limitations)
- [Citation](#citation)

---

## Overview

Dual-fuel diesel-CNG vehicles are common in Indian urban fleets but typically use static fuel assignment — running on CNG until the tank empties, then switching to diesel. This approach ignores real-time traffic conditions, road type, and driving dynamics.

This project replaces static assignment with a trained **XGBoost classifier** that decides the optimal fuel mode (diesel or CNG) at every simulation second based on 14 environment and traffic features. The controller is deployed inside SUMO via the TraCI interface and compared against four baselines: Diesel-only, CNG-only, a Rule-Based CNG-priority controller, and a PID controller.

**Key design decisions:**
- Environment-only features — no internal engine parameters required
- Offline training on Potheri (Chennai) data, zero-shot deployment on Uttam Nagar (Delhi)
- Stress-aware switching gate with 15-second dwell constraint prevents mechanical oscillation
- Robust training via 20% noise injection on mode-state feature + per-fold class-weight balancing (Option D)

---

## Key Results

| Network | Accuracy | Stress vs diesel | Fuel vs diesel | CO₂ vs diesel |
|---|---|---|---|---|
| Potheri, Chennai | 89.96% | −53.6% | +11.7% | +10.9% |
| Uttam Nagar, Delhi (zero-shot) | 90.95% | −53.7% | −0.7% | −0.4% |

**Notable findings:**
- The AI Supervisor is the **only strategy that reduces fuel below the diesel baseline** on the Uttam Nagar network
- The Rule-Based controller shows +75.1% fuel on Uttam Nagar vs ~0% on Potheri — fixed thresholds fail on dense Delhi stop-start traffic
- Zero-shot transfer achieves *higher* accuracy (90.95%) than Potheri training accuracy (89.96%), confirming the learned policy is network-agnostic
- Feature importance after Option D: `rolling_mean_speed_3s` and `accel` are the top environment features, confirming the model learns driving physics rather than mode-state inertia

---

## System Architecture

The system has two clearly separated phases:

**Offline training pipeline** (run once on Potheri data):
```
SUMO Baseline Sims → merge_baseline_logs → compute_cost_labels
    → finalize_labels → engineer_features → train_model → model.json
```

**Online real-time control loop** (deployed on any city):
```
SUMO Live Sim → TraCI (14 features) → XGBoost predict
    → Stress Gate (avg stress < 0.7, dwell ≥ 15s) → setType(diesel/CNG)
    → CNG Tank depletion check → Per-step logger → [feedback to SUMO]
```

The closed-loop feedback arrow from `setType()` back to SUMO is what makes this a control system, not just a data pipeline.

---

## Repository Structure

```
dual-fuel-ai-supervisor/
│
├── pipeline/                        # All simulation + ML scripts — shared across both cities
│   ├── simulate_baseline.py         # Run SUMO with diesel-only or CNG-only config, log output
│   ├── simulate_ai_controller.py    # Real-time XGBoost controller with stress gate + dwell
│   ├── simulate_pid_controller.py   # PID baseline: speed-error feedback, anti-windup ±5.0
│   ├── simulate_rule_controller.py  # Rule-Based: CNG-priority with load-based diesel fallback
│   ├── merge_baseline_logs.py       # Align diesel + CNG logs by vehicle ID and timestep
│   ├── compute_cost_labels.py       # J = W_F·fuel + W_CO2·CO2 + W_S·stress per timestep
│   ├── finalize_labels.py           # 5s dwell smoothing, remove tie rows, add history features
│   ├── engineer_features.py         # Build 14-feature matrix from training_dataset.csv
│   ├── train_model.py               # XGBoost training with Option D (noise + class balancing)
│   ├── aggregate_metrics.py         # Single source of metrics for all 5 strategies
│   └── generate_plots.py            # 3 comparison charts: absolute, relative, SSI
│
├── cities/
│   ├── potheri_chennai/             # Main training network — Chengalpattu, Tamil Nadu
│   │   ├── sim_cng.sumocfg          # SUMO config for CNG baseline simulation
│   │   ├── sim_diesel.sumocfg       # SUMO config for diesel baseline simulation
│   │   ├── vtypes.xml               # Vehicle types: HBEFA3/PC_D_EU4, HBEFA3/PC_G_EU4
│   │   ├── routes_base.rou.xml      # Base routes generated by duarouter
│   │   ├── routes_cng.rou.xml       # All vehicles typed as CNG
│   │   ├── routes_diesel.rou.xml    # All vehicles typed as diesel
│   │   ├── generate_routes.py       # Generates CNG + diesel route files from base routes
│   │   ├── map.osm                  # Raw OSM download (gitignored — large)
│   │   └── data/                    # All CSVs gitignored — regenerate by running pipeline
│   │       ├── log_diesel.csv
│   │       ├── log_cng.csv
│   │       ├── log_ai_last.csv
│   │       ├── log_pid_controller.csv
│   │       ├── log_rule_based.csv
│   │       ├── merged_logs.csv
│   │       ├── merged_with_J_norm.csv
│   │       ├── training_dataset.csv
│   │       └── features_final.csv
│   │
│   └── uttam_nagar_delhi/           # Zero-shot transfer network — Dwarka, Delhi
│       ├── sim_cng.sumocfg
│       ├── sim_diesel.sumocfg
│       ├── vtypes.xml
│       ├── routes_cng.rou.xml
│       ├── routes_diesel.rou.xml
│       ├── generate_routes.py       # Generates CNG + diesel routes from base routes
│       ├── map.osm                  # Gitignored — bbox 77.0467,28.6124,77.0747,28.6364
│       └── data/                    # Gitignored — regenerate by running pipeline on this city
│
├── models/
│   ├── potheri/                     # Trained on Potheri data
│   │   ├── model.json               # Best XGBoost model (serialized, 6.3 MB)
│   │   ├── model_rf_comparison.pkl  # Random Forest baseline for paper comparison
│   │   ├── training_report.txt      # Per-fold accuracy, F1, precision, recall
│   │   └── feature_importance.txt   # Gain scores for all 14 features
│   │
│   └── uttam_nagar/                 # Transfer evaluation — NO separate model trained
│       ├── transfer_results.txt     # 90.95% accuracy achieved without retraining
│       └── feature_importance.txt   # Feature importance on Delhi data
│
├── outputs/
│   ├── potheri/
│   │   ├── comparison_table.csv     # Fuel, CO2, stress, SSI for all 5 strategies
│   │   ├── plot_absolute.png        # Absolute fuel/CO2/stress comparison
│   │   ├── plot_relative.png        # % change vs diesel baseline
│   │   └── plot_ssi.png             # Switching Stability Index bar chart
│   │
│   └── uttam_nagar/
│       ├── comparison_table.csv
│       ├── plot_absolute.png
│       ├── plot_relative.png
│       └── plot_ssi.png
│
├── README.md
├── .gitignore
└── requirements.txt
```

---

## Installation

**Prerequisites:**
- Python 3.10 or higher
- SUMO 1.x with TraCI — download at https://sumo.dlr.de/docs/Downloads.php
- Set the `SUMO_HOME` environment variable after installing SUMO

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Verify SUMO is accessible:**
```bash
sumo --version
python -c "import traci; print('TraCI OK')"
```

---

## Pipeline Run Order

Point `SUMO_CFG` and log file paths at the correct city folder before running each script. All scripts read from and write to the paths configured at the top of each file.

### Step 1 — Baseline simulations
```bash
# Run twice — once for each fuel config
python pipeline/simulate_baseline.py   # set SUMO_CFG = cities/potheri_chennai/sim_cng.sumocfg
python pipeline/simulate_baseline.py   # set SUMO_CFG = cities/potheri_chennai/sim_diesel.sumocfg
```
Produces: `log_cng.csv`, `log_diesel.csv`

### Step 2 — Merge logs
```bash
python pipeline/merge_baseline_logs.py
```
Produces: `merged_logs.csv` with `diesel_speed`, `diesel_accel`, `cng_fuel`, `cng_co2` columns aligned per vehicle-timestep

### Step 3 — Compute cost labels
```bash
python pipeline/compute_cost_labels.py
```
Cost function: `J = W_F·fuel_norm + W_CO2·CO2_norm + W_S·stress`  
Default weights: `W_F=1.0`, `W_CO2=0.5`, `W_S=0.3` (configurable at top of file)  
Produces: `merged_with_J_norm.csv` with `J_diesel`, `J_cng`, `best_mode` columns

### Step 4 — Finalize labels
```bash
python pipeline/finalize_labels.py
```
Applies 5-second dwell smoothing per vehicle, removes tie rows, adds `prev_speed`, `prev_accel`, rolling statistics  
Produces: `training_dataset.csv` — label balance printed (target: 55–70% diesel)

### Step 5 — Engineer features
```bash
python pipeline/engineer_features.py
```
Builds the 14-feature matrix. Features (in exact training order):

| # | Feature | Description |
|---|---|---|
| 0 | speed | Vehicle speed (m/s) |
| 1 | accel | Acceleration (m/s²) |
| 2 | speed_limit | Road speed limit (m/s) |
| 3 | speed_norm | speed / speed_limit |
| 4 | hard_accel_flag | abs(accel) > 1.5 m/s² |
| 5 | prev_speed | Speed at t−1 |
| 6 | prev_accel | Acceleration at t−1 |
| 7 | rolling_mean_speed_3s | Mean speed over last 3 seconds |
| 8 | rolling_std_accel_3s | Std of acceleration over last 3 seconds |
| 9 | current_mode_encoded | Current fuel mode (0=diesel, 1=CNG) — noise-injected in training |
| 10 | edge_type_code | Road type encoding (motorway/arterial/residential) |
| 11 | lane_max_speed | Maximum speed for current lane |
| 12 | traffic_density_on_edge | Vehicles per metre on current edge |
| 13 | avg_speed_on_edge | Mean speed of all vehicles on current edge |

Produces: `features_final.csv`

### Step 6 — Train model
```bash
python pipeline/train_model.py
```
**Option D robust training:**
- 20% noise injection on `current_mode_encoded` — prevents leakage from dwell-smoothing inertia
- Per-fold `scale_pos_weight` — handles class imbalance
- 5-fold Group K-Fold cross-validation grouped by `veh_id`
- Hyperparameters: `n_estimators=500`, `lr=0.03`, `max_depth=8`

Produces: `models/potheri/model.json`, `training_report.txt`, `feature_importance.txt`

### Step 7 — Run controllers
```bash
python pipeline/simulate_ai_controller.py
python pipeline/simulate_pid_controller.py
python pipeline/simulate_rule_controller.py
```
Each controller reads `model.json` (AI only), runs the SUMO simulation, and writes its own log CSV.

### Step 8 — Aggregate metrics
```bash
python pipeline/aggregate_metrics.py
```
Reads all 5 log files, computes fuel total, CO₂ total, avg stress, switch count, SSI  
Produces: `outputs/potheri/comparison_table.csv`

### Step 9 — Generate plots
```bash
python pipeline/generate_plots.py
```
Produces 3 PNG charts in `outputs/potheri/`

---

## Controllers

### AI Supervisor
Loads `models/potheri/model.json` and calls `model.predict()` every simulation second using 14 live features from TraCI. Every predicted switch is gated by:
- **Stress gate:** rolling average stress over last 5 steps must be < 0.7
- **Dwell constraint:** minimum 15 seconds must have elapsed since the last switch

CNG tank: 12 kg capacity, depleted at `(fuel_mg_s / 1,000,000) × step` kg per second. Vehicle is forced to diesel when tank reaches 0.

### Rule-Based Controller
CNG-priority hybrid logic:
```python
if a_norm > 0.6:                          → diesel   # high torque demand
elif v_norm > 0.7 and a_norm > 0.3:       → diesel   # sustained load
else:                                      → CNG      # default
```
Thresholds match the stress formula (V_THRESH=0.7, A_THRESH=0.6). Same 15s dwell and CNG tank logic as AI.

### PID Controller
Proportional-integral-derivative feedback on normalised speed error targeting ratio 0.6.  
`Kp=1.2, Ki=0.05, Kd=0.8`, integral anti-windup clamp `±5.0`.  
Same stress formula, dwell constraint, and CNG tank logic as AI.

### Stress Formula (shared across all controllers)
```
S = max(0, v_norm − 0.7) + 1.5 · max(0, a_norm − 0.6) + 0.1 · |jerk|
  + 0.2 · mode_penalty   [only when v_norm > 0.7 OR a_norm > 0.6]

mode_penalty: diesel = 1.0, CNG = 0.4
A_MAX = 3.0 m/s²
```

---

## Model Training

The XGBoost model is trained exclusively on Potheri data. Before training, **Option D** is applied to prevent label leakage:

**Why leakage occurs:** The 5-second dwell smoothing in `finalize_labels.py` makes `current_mode_encoded` almost always equal to the label (the vehicle stays in its current mode for at least 5 seconds). Without mitigation, the model learns "predict current mode" rather than learning the driving environment — achieving 99.67% accuracy by exploiting inertia.

**Option D fix:**
1. Randomly flip 20% of `current_mode_encoded` values before training (noise injection)
2. Compute `scale_pos_weight = diesel_count / cng_count` per fold

**Result:** Feature gain gap reduced from 377× (leaky) to 11× (robust). `rolling_mean_speed_3s` becomes the dominant environment feature, confirming the model learns driving physics.

---

## Cross-Network Validation

The Potheri-trained model (`models/potheri/model.json`) is deployed on Uttam Nagar **without retraining**. To run the Uttam Nagar evaluation:

**1. Download the OSM file:**
```
https://overpass-api.de/api/map?bbox=77.0467,28.6124,77.0747,28.6364
```
Save as `cities/uttam_nagar_delhi/map.osm`

**2. Build the network:**
```bash
netconvert --osm-files map.osm --output-file net.net.xml \
  --geometry.remove --roundabouts.guess --ramps.guess \
  --junctions.join --tls.guess-signals --tls.discard-simple --tls.join

python cities/uttam_nagar_delhi/generate_routes.py
```

**3. Run all 5 simulations** pointing each script at `cities/uttam_nagar_delhi/sim_cng.sumocfg`

**4. Aggregate and plot:**
```bash
python pipeline/aggregate_metrics.py    # point LOG_FILES at uttam_nagar data/
python pipeline/generate_plots.py
```

**Why the Rule-Based controller fails on Uttam Nagar (+75% fuel):** Dense Delhi stop-start traffic keeps `a_norm` and `v_norm` permanently below the diesel thresholds, so the controller defaults to CNG for almost the entire simulation. Since HBEFA3 reports CNG fuel mass higher than diesel at equivalent speed-load conditions, permanent CNG = 75% more fuel. The AI Supervisor, trained on cost signals rather than fixed thresholds, adapts correctly.

---

## Outputs

Each city produces the same set of output files:

| File | Description |
|---|---|
| `comparison_table.csv` | Fuel (g), CO₂ (g), avg stress, switch count, SSI, fuel%, CO₂% for all 5 strategies |
| `plot_absolute.png` | Side-by-side bar chart of absolute fuel, CO₂, stress |
| `plot_relative.png` | % change relative to diesel baseline for all strategies |
| `plot_ssi.png` | Switching Stability Index bar chart |

**SSI definition:**  
`SSI = 100 × switches / (num_vehicles × total_timesteps)`  
Static baselines (Diesel, CNG) have SSI = 0. Higher SSI = more responsive switching.

---

## File Reference

| Old name | New name | Location |
|---|---|---|
| `run.py` | `simulate_baseline.py` | `pipeline/` |
| `run_ai_controller_stress_safe.py` | `simulate_ai_controller.py` | `pipeline/` |
| `run_pid_controller.py` | `simulate_pid_controller.py` | `pipeline/` |
| `run_rule.py` | `simulate_rule_controller.py` | `pipeline/` |
| `j_log.py` | `compute_cost_labels.py` | `pipeline/` |
| `step3_labels.py` | `finalize_labels.py` | `pipeline/` |
| `build_features.py` | `engineer_features.py` | `pipeline/` |
| `graph.py` | `generate_plots.py` | `pipeline/` |
| `build_comparison_table.py` | `aggregate_metrics.py` | `pipeline/` |
| `best_xgboost_env_only.json` | `model.json` | `models/potheri/` |
| `1_absolute_comparison.png` | `plot_absolute.png` | `outputs/<city>/` |
| `2_improvement_vs_baseline.png` | `plot_relative.png` | `outputs/<city>/` |
| `3_ssi_comparison.png` | `plot_ssi.png` | `outputs/<city>/` |

---

## Known Limitations

1. **HBEFA3 CNG proxy:** SUMO HBEFA3 has no dedicated CNG emission class. `PC_G_EU4` (gasoline) is used as the standard proxy, reporting higher fuel mass than diesel. Absolute fuel comparisons should be interpreted with this in mind. HBEFA4 introduces a dedicated `PC_CNG_Euro-4` class but requires SUMO 1.23+.

2. **No combustion transient modelling:** The stress metric is a heuristic (normalised speed + acceleration + jerk). It does not physically model engine thermal transients, exhaust temperature spikes, or fuel delivery delays during mode switching.

3. **Partial feature leakage:** After Option D, `current_mode_encoded` retains an 11× gain advantage over the next feature (down from 377×). Full elimination requires raw (non-dwell-smoothed) labels or a two-stage prediction architecture — identified as future work.

4. **Empirical dwell constraint:** The 15-second minimum dwell was chosen empirically to suppress oscillation observed in preliminary runs. Principled derivation from engine thermal time constants is future work.

5. **Two networks evaluated:** Potheri (Chennai) and Uttam Nagar (Delhi). Extension to a third network (e.g. Bangalore Koramangala) is planned for the journal version.

---

## Citation

```bibtex
@inproceedings{saranya2026dualfuel,
  title     = {AI-Driven Dual-Fuel Engine Optimization for Sustainable Mobility and Improved Efficiency},
  author    = {Saranya, V.S. and Saharan, Harsh and Praveen, Divyansh},
  booktitle = {Proceedings of ICICDAQ 2026},
  year      = {2026},
  institution = {SRMIST, Chengalpattu}
}
```

---

## Acknowledgements

Eclipse SUMO traffic simulator — German Aerospace Center (DLR)  
XGBoost — Chen and Guestrin, KDD 2016  
OpenStreetMap contributors — road network data  
Department of Computing Technologies, SRMIST