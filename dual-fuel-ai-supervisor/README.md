# AI-Driven Dual-Fuel Engine Optimization

XGBoost-based supervisory controller for real-time diesel-CNG fuel-mode switching,
evaluated on two Indian road networks using SUMO microscopic traffic simulation.

**Paper:** AI-Driven Dual-Fuel Engine Optimization for Sustainable Mobility  
**Conference:** ICICDAQ 2026  
**Authors:** Dr. V.S. Saranya, Harsh Saharan, Divyansh Praveen — SRMIST

---

## Key Results

| Network | Accuracy | Stress vs diesel | Fuel vs diesel |
|---|---|---|---|
| Potheri, Chennai | 89.96% | -53.6% | +11.7% |
| Uttam Nagar, Delhi (zero-shot) | 90.95% | -53.7% | -0.7% |

---

## Structure

```
dual-fuel-ai-supervisor/
├── pipeline/               Simulation + ML scripts (shared, city-agnostic)
├── cities/
│   ├── potheri_chennai/    SUMO configs, routes, data — training network
│   └── uttam_nagar_delhi/  SUMO configs, routes, data — zero-shot transfer
├── models/
│   ├── potheri/            Trained XGBoost + Random Forest comparison model
│   └── uttam_nagar/        Transfer evaluation results (no retraining)
└── outputs/
    ├── potheri/            Plots and comparison table for Potheri
    └── uttam_nagar/        Plots and comparison table for Uttam Nagar
```

---

## Pipeline Run Order

```
1.  pipeline/simulate_baseline.py         (x2: diesel + CNG configs)
2.  pipeline/merge_baseline_logs.py
3.  pipeline/compute_cost_labels.py
4.  pipeline/finalize_labels.py
5.  pipeline/engineer_features.py
6.  pipeline/train_model.py               (Potheri only)
7.  pipeline/simulate_ai_controller.py
    pipeline/simulate_pid_controller.py
    pipeline/simulate_rule_controller.py
8.  pipeline/aggregate_metrics.py
9.  pipeline/generate_plots.py
```

## Requirements

```bash
pip install -r requirements.txt
```

SUMO 1.x with TraCI: https://sumo.dlr.de
