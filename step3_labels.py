# step3_finalize_labels.py
# Produce training_dataset.csv from merged_with_J_norm.csv
# Applies 5s dwell smoothing, removes 'tie' rows, adds history/control features.
#
# Output columns passed to build_features.py:
#   time, veh_id, speed, accel, speed_limit, edge, lane
#   prev_speed, prev_accel, rolling_mean_speed_3s, rolling_std_accel_3s
#   previous_mode, time_since_last_switch
#   diesel_fuel, diesel_co2, diesel_stress
#   cng_fuel, cng_co2, cng_stress
#   J_diesel, J_cng, label_mode, label_num

import pandas as pd
import numpy as np

IN_FILE  = "merged_with_J_norm.csv"
OUT_FILE = "training_dataset.csv"

# 5s dwell for label smoothing — intentionally shorter than the 15s
# runtime dwell in controllers (5s avoids over-constraining training labels,
# 15s is a hard mechanical safety constraint enforced at deployment)
MIN_DWELL = 5.0
TIME_COL  = "time"
VEH_COL   = "veh_id"


def process_vehicle(g):
    g = g.sort_values(by=TIME_COL).copy().reset_index(drop=True)

    times = g[TIME_COL].values
    dt    = np.diff(times, prepend=times[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0

    # --------------------------------------------------
    # History features
    # --------------------------------------------------
    g["prev_speed"] = g["speed"].shift(1).fillna(g["speed"])
    g["prev_accel"] = g["accel"].shift(1).fillna(g["accel"])

    # Rolling 3-second window (assumes 1s step; scales automatically otherwise)
    mean_dt     = max(1e-9, dt.mean())
    window_rows = max(1, int(round(3.0 / mean_dt)))

    g["rolling_mean_speed_3s"] = (
        g["speed"].rolling(window=window_rows, min_periods=1).mean()
    )
    g["rolling_std_accel_3s"] = (
        g["accel"].rolling(window=window_rows, min_periods=1).std().fillna(0.0)
    )

    # --------------------------------------------------
    # Dwell-smoothed label generation
    # --------------------------------------------------
    current         = g["best_mode"].iloc[0] if "best_mode" in g.columns else "diesel"
    prev_mode       = current
    time_since_last = 1e9   # large — first switch always allowed
    label_modes     = []
    prev_modes      = []
    dwell_times     = []

    for i, row in g.iterrows():
        raw = row.get("best_mode", "diesel" if row["J_diesel"] < row["J_cng"] else "cng")
        candidate = current if raw == "tie" else raw

        if candidate != current and time_since_last >= MIN_DWELL:
            prev_mode       = current
            current         = candidate
            time_since_last = 0.0
        else:
            time_since_last += dt[i]

        label_modes.append(current)
        prev_modes.append(prev_mode)
        dwell_times.append(time_since_last)

    g["label_mode"]            = label_modes
    g["previous_mode"]         = prev_modes
    g["time_since_last_switch"] = dwell_times

    return g


def main():
    print("Loading:", IN_FILE)
    df = pd.read_csv(IN_FILE)

    needed = {"time", "veh_id", "best_mode", "J_diesel", "J_cng"}
    if not needed.issubset(df.columns):
        raise RuntimeError(
            f"Input must contain: {needed}\nFound: {set(df.columns)}"
        )

    # merge_logs.py renames speed/accel to diesel_speed/diesel_accel
    # Restore plain speed and accel columns for step3 processing
    if "speed" not in df.columns and "diesel_speed" in df.columns:
        df["speed"] = df["diesel_speed"]
        print("Using diesel_speed as speed")
    if "accel" not in df.columns and "diesel_accel" in df.columns:
        df["accel"] = df["diesel_accel"]
        print("Using diesel_accel as accel")

    vehicle_count = df[VEH_COL].nunique()
    print(f"Processing {vehicle_count} vehicles with MIN_DWELL = {MIN_DWELL}s ...")

    groups = []
    for vid, g in df.groupby(VEH_COL, sort=False):
        groups.append(process_vehicle(g))

    merged = (
        pd.concat(groups, axis=0)
        .sort_values(by=[VEH_COL, TIME_COL])
        .reset_index(drop=True)
    )

    # Remove tie rows — 2-class classification only
    before = len(merged)
    merged = merged[merged["best_mode"] != "tie"].reset_index(drop=True)
    print(f"Removed {before - len(merged)} tie rows. Remaining: {len(merged)}")

    # --------------------------------------------------
    # Explicit keep list — no conditional column name tricks
    # build_features.py depends on all of these being present
    # --------------------------------------------------
    keep_cols = [
        "time", "veh_id",
        "speed", "accel", "speed_limit", "edge", "lane",
        "prev_speed", "prev_accel",
        "rolling_mean_speed_3s", "rolling_std_accel_3s",
        "previous_mode", "time_since_last_switch",
        "diesel_fuel", "diesel_co2", "diesel_stress",
        "cng_fuel",    "cng_co2",    "cng_stress",
        "J_diesel", "J_cng",
        "label_mode",
    ]

    keep_final = [c for c in keep_cols if c in merged.columns]
    missing    = [c for c in keep_cols if c not in merged.columns]
    if missing:
        print(f"WARNING: these expected columns are missing from merged data: {missing}")

    training = merged[keep_final].copy()

    # Integer target for XGBoost
    training["label_num"] = training["label_mode"].map({"diesel": 0, "cng": 1})

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    print("\nLabel distribution:")
    print(training["label_mode"].value_counts())
    print(f"\nColumns saved ({len(training.columns)}):")
    print(list(training.columns))
    print("\nSample rows (first 3):")
    print(training.head(3).to_string(index=False))

    training.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}  ({len(training)} rows)")


if __name__ == "__main__":
    main()