"""
build_comparison_table.py

Single source of truth for all simulation metrics.
Reads the 5 log CSVs, computes metrics, and saves comparison_table.csv.
graph.py imports compute_all_metrics() from this file — no duplication.

Expected log columns (all 5 controllers):
    time, veh_id, mode, fuel, co2, stress, cng_tank_kg
    (diesel/cng baselines from run.py do not have cng_tank_kg — that is fine)
"""

import os
import pandas as pd

# --------------------------------------------------
# Log file mapping — keys match STRATEGY_LABELS in graph.py
# --------------------------------------------------
LOG_FILES = {
    "diesel": "log_diesel.csv",
    "cng":    "log_cng.csv",
    "rule":   "log_rule_based.csv",
    "pid":    "log_pid_controller.csv",
    "ai":     "log_ai_last.csv",
}

OUTPUT_FILE = "comparison_table.csv"


def compute_metrics(file_path):
    """
    Reads one log CSV and returns a metrics dict.
    Returns None if the file is missing or malformed.
    """
    if not os.path.exists(file_path):
        print(f"  SKIP (not found): {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)

        required = {"time", "veh_id", "mode", "fuel", "co2", "stress"}
        missing  = required - set(df.columns)
        if missing:
            print(f"  SKIP (missing columns {missing}): {file_path}")
            return None

        total_fuel = df["fuel"].sum()
        total_co2  = df["co2"].sum()
        avg_stress = df["stress"].mean()

        # Switch counting — sort first to ensure correct shift
        df = df.sort_values(["veh_id", "time"])
        df["prev_mode"] = df.groupby("veh_id")["mode"].shift(1)

        switches     = int(df[(df["mode"] != df["prev_mode"]) & df["prev_mode"].notna()].shape[0])
        total_steps  = df["time"].nunique()
        num_vehicles = df["veh_id"].nunique()

        ssi = (100 * switches / (num_vehicles * total_steps)
               if total_steps > 0 and num_vehicles > 0 else 0.0)

        return {
            "fuel":     total_fuel,
            "co2":      total_co2,
            "stress":   avg_stress,
            "switches": switches,
            "ssi":      round(ssi, 4),
        }

    except Exception as e:
        print(f"  ERROR reading {file_path}: {e}")
        return None


def compute_all_metrics():
    """
    Computes metrics for all 5 strategies.
    Returns a dict keyed by strategy key (diesel/cng/rule/pid/ai).
    Only includes strategies whose log files exist and are valid.
    """
    print("Computing metrics...")
    metrics = {}
    for key, path in LOG_FILES.items():
        result = compute_metrics(path)
        if result is not None:
            metrics[key] = result
            print(f"  {key:8s} -> fuel={result['fuel']:.3e}  "
                  f"co2={result['co2']:.3e}  "
                  f"stress={result['stress']:.4f}  "
                  f"switches={result['switches']}  "
                  f"ssi={result['ssi']:.4f}")
    return metrics


def build_table(metrics):
    """
    Builds and saves the comparison table CSV.
    Adds Fuel(%) and CO2(%) columns normalised to the diesel baseline.
    Returns the DataFrame.
    """
    if not metrics:
        print("No metrics to build table from.")
        return pd.DataFrame()

    df = pd.DataFrame(metrics).T

    # Normalise vs diesel baseline
    if "diesel" in df.index:
        df["fuel_pct"] = (df["fuel"] / df.loc["diesel", "fuel"] * 100).round(2)
        df["co2_pct"]  = (df["co2"]  / df.loc["diesel", "co2"]  * 100).round(2)

    df = df.rename(columns={
        "fuel":     "Total Fuel (g)",
        "co2":      "Total CO2 (g)",
        "stress":   "Avg Stress",
        "switches": "Switch Count",
        "ssi":      "SSI",
        "fuel_pct": "Fuel (%)",
        "co2_pct":  "CO2 (%)",
    })

    # Consistent row order
    order = ["diesel", "cng", "rule", "pid", "ai"]
    df    = df.reindex([k for k in order if k in df.index])

    df = df.round(4)

    print("\n=== FINAL COMPARISON TABLE ===\n")
    print(df.to_string())

    df.to_csv(OUTPUT_FILE)
    print(f"\nSaved: {OUTPUT_FILE}")

    return df


if __name__ == "__main__":
    metrics = compute_all_metrics()
    build_table(metrics)