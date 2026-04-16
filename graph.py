"""
graph.py

Generates all comparison graphs. Imports metrics from build_comparison_table.py
so there is a single source of truth — no duplicated metric computation.

Outputs:
    1_absolute_comparison.png
    2_improvement_vs_baseline.png
"""

import numpy as np
import matplotlib.pyplot as plt

from build_comparison_table import compute_all_metrics, build_table

# --------------------------------------------------
# Strategy display labels — keys match LOG_FILES in build_comparison_table.py
# --------------------------------------------------
STRATEGY_LABELS = {
    "diesel": "Diesel",
    "cng":    "CNG",
    "rule":   "Rule-Based",
    "pid":    "PID",
    "ai":     "AI Supervisor",
}

plt.rcParams.update({
    "figure.dpi":          300,
    "font.family":         "DejaVu Sans",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
})


# --------------------------------------------------
# Absolute comparison — fuel, CO2, stress
# --------------------------------------------------
def plot_absolute(data):

    keys   = [k for k in ["diesel", "cng", "rule", "pid", "ai"] if k in data]
    labels = [STRATEGY_LABELS[k] for k in keys]

    fuel   = [data[k]["fuel"]   for k in keys]
    co2    = [data[k]["co2"]    for k in keys]
    stress = [data[k]["stress"] for k in keys]

    x     = np.arange(len(keys))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 5))

    bars1 = ax1.bar(x - width, fuel,   width, label="Fuel (g)",  color="#378ADD")
    bars2 = ax1.bar(x,         co2,    width, label="CO\u2082 (g)", color="#EF9F27")
    ax2   = ax1.twinx()
    bars3 = ax2.bar(x + width, stress, width, label="Avg Stress", color="#639922", alpha=0.8)

    # Value labels
    for b in bars1 + bars2:
        ax1.text(b.get_x() + b.get_width() / 2,
                 b.get_height() * 1.01,
                 f"{b.get_height():.2e}",
                 ha="center", va="bottom", fontsize=7)

    for b in bars3:
        ax2.text(b.get_x() + b.get_width() / 2,
                 b.get_height() * 1.01,
                 f"{b.get_height():.2f}",
                 ha="center", va="bottom", fontsize=7)

    ax1.set_ylabel("Fuel / CO\u2082 (g)")
    ax2.set_ylabel("Avg Stress")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="upper center", bbox_to_anchor=(0.5, 1.18),
               ncol=3, frameon=False)

    plt.title("Absolute Fuel, CO\u2082 and Stress Comparison", pad=15)
    plt.tight_layout()
    plt.savefig("1_absolute_comparison.png")
    plt.close()
    print("Saved: 1_absolute_comparison.png")


# --------------------------------------------------
# Relative comparison — % change vs diesel baseline
# --------------------------------------------------
def plot_relative(data, baseline_key="diesel"):

    if baseline_key not in data:
        print(f"Baseline '{baseline_key}' not found — skipping relative plot.")
        return

    base = data[baseline_key]

    keys   = [k for k in ["diesel", "cng", "rule", "pid", "ai"] if k in data]
    labels = [STRATEGY_LABELS[k] for k in keys]

    def pct(val, base_val):
        return (val - base_val) / base_val * 100 if base_val != 0 else 0.0

    fuel_pct   = [pct(data[k]["fuel"],   base["fuel"])   for k in keys]
    co2_pct    = [pct(data[k]["co2"],    base["co2"])    for k in keys]
    stress_pct = [pct(data[k]["stress"], base["stress"]) for k in keys]

    x     = np.arange(len(keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - width, fuel_pct,   width, label="Fuel %",   color="#378ADD")
    b2 = ax.bar(x,         co2_pct,    width, label="CO\u2082 %", color="#EF9F27")
    b3 = ax.bar(x + width, stress_pct, width, label="Stress %", color="#639922")

    # Value labels — place above positive bars, below negative
    for bars in [b1, b2, b3]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2,
                    h + (0.5 if h >= 0 else -1.5),
                    f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=7)

    ax.axhline(0, linestyle="--", linewidth=1, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(f"% Change vs {STRATEGY_LABELS[baseline_key]}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)

    plt.title("Percentage Change Relative to Diesel Baseline")
    plt.tight_layout()
    plt.savefig("2_improvement_vs_baseline.png")
    plt.close()
    print("Saved: 2_improvement_vs_baseline.png")


# --------------------------------------------------
# SSI bar chart (bonus — useful for paper)
# --------------------------------------------------
def plot_ssi(data):

    keys   = [k for k in ["diesel", "cng", "rule", "pid", "ai"] if k in data]
    labels = [STRATEGY_LABELS[k] for k in keys]
    ssi    = [data[k]["ssi"] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(labels, ssi, color="#7F77DD", alpha=0.85)

    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() * 1.02,
                f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("SSI (% per vehicle-step)")
    ax.set_xticklabels(labels, rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.title("Switching Stability Index (SSI)")
    plt.tight_layout()
    plt.savefig("3_ssi_comparison.png")
    plt.close()
    print("Saved: 3_ssi_comparison.png")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    metrics = compute_all_metrics()

    if not metrics:
        print("No metrics computed — check that log CSV files are present.")
    else:
        build_table(metrics)
        plot_absolute(metrics)
        plot_relative(metrics, baseline_key="diesel")
        plot_ssi(metrics)
        print("\nAll graphs generated successfully.")