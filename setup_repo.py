# setup_repo.py
# Place this script in the parent folder containing both Phothri and 4, then run:
#   python setup_repo.py

import os
import shutil

BASE        = os.path.dirname(os.path.abspath(__file__))
SRC_POTHERI = os.path.join(BASE, "Phothri")
SRC_UTTAM   = os.path.join(BASE, "4")
REPO        = os.path.join(BASE, "dual-fuel-ai-supervisor")

FOLDERS = [
    "pipeline",
    "cities/potheri_chennai/data",
    "cities/uttam_nagar_delhi/data",
    "models/potheri",
    "models/uttam_nagar",
    "outputs/potheri",
    "outputs/uttam_nagar",
]

# (source_folder, current_filename, repo_destination)
MOVES = [

    # ── Pipeline scripts — taken from Phothri (canonical source) ─────────────
    (SRC_POTHERI, "run.py",                           "pipeline/simulate_baseline.py"),
    (SRC_POTHERI, "run_ai_controller_stress_safe.py", "pipeline/simulate_ai_controller.py"),
    (SRC_POTHERI, "run_pid_controller.py",            "pipeline/simulate_pid_controller.py"),
    (SRC_POTHERI, "run_rule.py",                      "pipeline/simulate_rule_controller.py"),
    (SRC_POTHERI, "merge_logs.py",                    "pipeline/merge_baseline_logs.py"),
    (SRC_POTHERI, "j_log.py",                         "pipeline/compute_cost_labels.py"),
    (SRC_POTHERI, "step3_labels.py",                  "pipeline/finalize_labels.py"),
    (SRC_POTHERI, "build_features.py",                "pipeline/engineer_features.py"),
    (SRC_POTHERI, "train_xgboost_env_only.py",        "pipeline/train_model.py"),
    (SRC_POTHERI, "build_comparison_table.py",        "pipeline/aggregate_metrics.py"),
    (SRC_POTHERI, "graph.py",                         "pipeline/generate_plots.py"),

    # ── Potheri city config ───────────────────────────────────────────────────
    (SRC_POTHERI, "sim_cng.sumocfg",     "cities/potheri_chennai/sim_cng.sumocfg"),
    (SRC_POTHERI, "sim_diesel.sumocfg",  "cities/potheri_chennai/sim_diesel.sumocfg"),
    (SRC_POTHERI, "vtypes.xml",          "cities/potheri_chennai/vtypes.xml"),
    (SRC_POTHERI, "routes_base.rou",     "cities/potheri_chennai/routes_base.rou.xml"),
    (SRC_POTHERI, "routes_base.rou.xml", "cities/potheri_chennai/routes_base.rou.xml"),
    (SRC_POTHERI, "routes_cng.rou",      "cities/potheri_chennai/routes_cng.rou.xml"),
    (SRC_POTHERI, "routes_cng.rou.xml",  "cities/potheri_chennai/routes_cng.rou.xml"),
    (SRC_POTHERI, "routes_diesel.rou",   "cities/potheri_chennai/routes_diesel.rou.xml"),
    (SRC_POTHERI, "routes_diesel.rou.xml","cities/potheri_chennai/routes_diesel.rou.xml"),
    (SRC_POTHERI, "trips.trips",         "cities/potheri_chennai/trips.trips.xml"),
    (SRC_POTHERI, "map.osm",             "cities/potheri_chennai/map.osm"),
    (SRC_POTHERI, "make_fuel_routes.py", "cities/potheri_chennai/generate_routes.py"),

    # ── Potheri data ──────────────────────────────────────────────────────────
    (SRC_POTHERI, "log_diesel.csv",         "cities/potheri_chennai/data/log_diesel.csv"),
    (SRC_POTHERI, "log_cng.csv",            "cities/potheri_chennai/data/log_cng.csv"),
    (SRC_POTHERI, "log_ai_last.csv",        "cities/potheri_chennai/data/log_ai_last.csv"),
    (SRC_POTHERI, "log_pid_controller.csv", "cities/potheri_chennai/data/log_pid_controller.csv"),
    (SRC_POTHERI, "log_rule_based.csv",     "cities/potheri_chennai/data/log_rule_based.csv"),
    (SRC_POTHERI, "merged_logs.csv",        "cities/potheri_chennai/data/merged_logs.csv"),
    (SRC_POTHERI, "merged_with_J_norm.csv", "cities/potheri_chennai/data/merged_with_J_norm.csv"),
    (SRC_POTHERI, "training_dataset.csv",   "cities/potheri_chennai/data/training_dataset.csv"),
    (SRC_POTHERI, "features_final.csv",     "cities/potheri_chennai/data/features_final.csv"),
    (SRC_POTHERI, "cng.csv",                "cities/potheri_chennai/data/cng_raw.csv"),
    (SRC_POTHERI, "diesel.csv",             "cities/potheri_chennai/data/diesel_raw.csv"),

    # ── Potheri models ────────────────────────────────────────────────────────
    (SRC_POTHERI, "best_xgboost_env_only.json",     "models/potheri/model.json"),
    (SRC_POTHERI, "best_random_forest_env_only.pkl", "models/potheri/model_rf_comparison.pkl"),
    (SRC_POTHERI, "xgboost_env_only_results.txt",   "models/potheri/training_report.txt"),
    (SRC_POTHERI, "feature_importance_env_only.txt", "models/potheri/feature_importance.txt"),

    # ── Potheri outputs ───────────────────────────────────────────────────────
    (SRC_POTHERI, "comparison_table.csv",          "outputs/potheri/comparison_table.csv"),
    (SRC_POTHERI, "1_absolute_comparison.png",     "outputs/potheri/plot_absolute.png"),
    (SRC_POTHERI, "2_improvement_vs_baseline.png", "outputs/potheri/plot_relative.png"),
    (SRC_POTHERI, "3_ssi_comparison.png",          "outputs/potheri/plot_ssi.png"),

    # ── Uttam Nagar city config ───────────────────────────────────────────────
    (SRC_UTTAM, "sim_cng.sumocfg",         "cities/uttam_nagar_delhi/sim_cng.sumocfg"),
    (SRC_UTTAM, "sim_cng_map2.sumocfg",    "cities/uttam_nagar_delhi/sim_cng.sumocfg"),
    (SRC_UTTAM, "sim_diesel.sumocfg",      "cities/uttam_nagar_delhi/sim_diesel.sumocfg"),
    (SRC_UTTAM, "sim_diesel_map2.sumocfg", "cities/uttam_nagar_delhi/sim_diesel.sumocfg"),
    (SRC_UTTAM, "vtypes.xml",              "cities/uttam_nagar_delhi/vtypes.xml"),
    (SRC_UTTAM, "make_fuel_routes2.py",    "cities/uttam_nagar_delhi/generate_routes.py"),
    (SRC_UTTAM, "map2.osm",               "cities/uttam_nagar_delhi/map.osm"),
    (SRC_UTTAM, "trips2.trips",           "cities/uttam_nagar_delhi/trips.trips.xml"),
    (SRC_UTTAM, "routes_base.rou",        "cities/uttam_nagar_delhi/routes_base.rou.xml"),
    (SRC_UTTAM, "routes2_base.rou",       "cities/uttam_nagar_delhi/routes_base.rou.xml"),
    (SRC_UTTAM, "routes2_base.rou.xml",   "cities/uttam_nagar_delhi/routes_base.rou.xml"),
    (SRC_UTTAM, "routes2_cng.rou",        "cities/uttam_nagar_delhi/routes_cng.rou.xml"),
    (SRC_UTTAM, "routes2_cng.rou.xml",    "cities/uttam_nagar_delhi/routes_cng.rou.xml"),
    (SRC_UTTAM, "routes_diesel.rou",      "cities/uttam_nagar_delhi/routes_diesel.rou.xml"),
    (SRC_UTTAM, "routes2_diesel.rou",     "cities/uttam_nagar_delhi/routes_diesel.rou.xml"),
    (SRC_UTTAM, "routes2_diesel.rou.xml", "cities/uttam_nagar_delhi/routes_diesel.rou.xml"),

    # ── Uttam Nagar data ──────────────────────────────────────────────────────
    (SRC_UTTAM, "log_diesel.csv",         "cities/uttam_nagar_delhi/data/log_diesel.csv"),
    (SRC_UTTAM, "log_cng.csv",            "cities/uttam_nagar_delhi/data/log_cng.csv"),
    (SRC_UTTAM, "log_ai_last.csv",        "cities/uttam_nagar_delhi/data/log_ai_last.csv"),
    (SRC_UTTAM, "log_pid_controller.csv", "cities/uttam_nagar_delhi/data/log_pid_controller.csv"),
    (SRC_UTTAM, "log_rule_based.csv",     "cities/uttam_nagar_delhi/data/log_rule_based.csv"),
    (SRC_UTTAM, "merged_logs.csv",        "cities/uttam_nagar_delhi/data/merged_logs.csv"),
    (SRC_UTTAM, "merged_with_J_norm.csv", "cities/uttam_nagar_delhi/data/merged_with_J_norm.csv"),
    (SRC_UTTAM, "training_dataset.csv",   "cities/uttam_nagar_delhi/data/training_dataset.csv"),
    (SRC_UTTAM, "features_final.csv",     "cities/uttam_nagar_delhi/data/features_final.csv"),

    # ── Uttam Nagar transfer results ──────────────────────────────────────────
    (SRC_UTTAM, "xgboost_env_only_results.txt",    "models/uttam_nagar/transfer_report.txt"),
    (SRC_UTTAM, "feature_importance_env_only.txt",  "models/uttam_nagar/feature_importance.txt"),

    # ── Uttam Nagar outputs ───────────────────────────────────────────────────
    (SRC_UTTAM, "comparison_table.csv",          "outputs/uttam_nagar/comparison_table.csv"),
    (SRC_UTTAM, "1_absolute_comparison.png",     "outputs/uttam_nagar/plot_absolute.png"),
    (SRC_UTTAM, "2_improvement_vs_baseline.png", "outputs/uttam_nagar/plot_relative.png"),
    (SRC_UTTAM, "3_ssi_comparison.png",          "outputs/uttam_nagar/plot_ssi.png"),
]

README = """\
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
"""

GITIGNORE = """\
# Python
.venv/
__pycache__/
*.pyc

# Large SUMO files — regenerate with netconvert
cities/*/net.net.xml
cities/*/map.osm
cities/*/*.trips.xml
*.rou.alt.xml

# All CSV data — regenerate by running the pipeline
cities/*/data/*.csv

# Node (not part of project)
node_modules/

# OS / IDE
.DS_Store
Thumbs.db
.vscode/
.idea/
"""

REQUIREMENTS = """\
xgboost>=2.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
openpyxl>=3.1
"""

TRANSFER = """\
Zero-Shot Transfer — Uttam Nagar, Delhi
========================================
Model: models/potheri/model.json (trained on Potheri, NO retraining on Delhi)

Accuracy:   90.95%
F1 Score:   89.82%
Precision:  88.14%
Recall:     91.55%

Simulation vs diesel baseline (Uttam Nagar):
  Stress:  -53.7%
  Fuel:    -0.7%   <- AI uniquely achieves fuel reduction below diesel
  CO2:     -0.4%

Key finding: Rule-Based shows +75.1% fuel on Uttam Nagar (vs +0% on Potheri).
Fixed thresholds fail on dense Delhi stop-start traffic. AI generalises correctly.
"""

def run():
    print(f"Phothri source : {SRC_POTHERI}")
    print(f"Uttam source   : {SRC_UTTAM}")
    print(f"Repo target    : {REPO}")
    print()

    if not os.path.isdir(SRC_POTHERI):
        print(f"ERROR: 'Phothri' folder not found at {SRC_POTHERI}")
        return
    if not os.path.isdir(SRC_UTTAM):
        print(f"ERROR: Folder '4' not found at {SRC_UTTAM}")
        return

    print("Creating folders...")
    for folder in FOLDERS:
        os.makedirs(os.path.join(REPO, folder), exist_ok=True)
        print(f"  [OK] {folder}/")
    print()

    print("Copying files...")
    done = set()
    skipped = []
    for src_dir, src_name, dst_rel in MOVES:
        if dst_rel in done:
            continue
        src = os.path.join(src_dir, src_name)
        dst = os.path.join(REPO, dst_rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            label = "Phothri" if src_dir == SRC_POTHERI else "4      "
            print(f"  [COPY] [{label}] {src_name:44s} -> {dst_rel}")
            done.add(dst_rel)
        else:
            skipped.append((src_dir, src_name))
    print()

    print("Generating new files...")
    def write(rel, content):
        path = os.path.join(REPO, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [GEN]  {rel}")

    write("README.md",                               README)
    write(".gitignore",                              GITIGNORE)
    write("requirements.txt",                        REQUIREMENTS)
    write("models/uttam_nagar/transfer_results.txt", TRANSFER)
    print()

    print("=" * 60)
    print(f"Repo created at:\n  {REPO}")
    if skipped:
        not_found = [(d, n) for d, n in skipped
                     if not any(d2 == d and n2 == n and dst in done
                                for d2, n2, dst in MOVES)]
        unique = list({n for _, n in skipped})
        print(f"\nSkipped {len(unique)} files not found (may have different names):")
        for _, name in skipped:
            print(f"  {name}")
    print()
    print("Next steps:")
    print(f'  cd "{REPO}"')
    print("  git init")
    print("  git add .")
    print('  git commit -m "Initial commit"')
    print("  git remote add origin <your-github-url>")
    print("  git push -u origin main")

if __name__ == "__main__":
    run()