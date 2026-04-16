import pandas as pd
import numpy as np

INPUT  = "merged_logs.csv"
OUTPUT = "merged_with_J_norm.csv"

# --------------------------------------------------
# Cost function weights (must match paper)
# J = W_F*fuel_n + W_CO2*co2_n + W_S*stress_n
# --------------------------------------------------
W_F   = 1.0
W_CO2 = 0.5
W_S   = 0.3

DELTA_J_FACTOR = 0.01   # tie band fraction of mean J
USE_MEAN_NORM  = True   # True = mean norm, False = minmax

# --------------------------------------------------
# Stress formula constants — MUST match run.py and all controllers
# S = max(0,v_norm-0.7) + 1.5*max(0,a_norm-0.6) + 0.1*|jerk|
#   + 0.2*mode_pen  (only when v_norm>0.7 OR a_norm>0.6)
# --------------------------------------------------
A_MAX    = 3.0
V_THRESH = 0.7
A_THRESH = 0.6
A_COEFF  = 1.5
J_COEFF  = 0.1
PEN_D    = 1.0    # diesel mode penalty
PEN_C    = 0.4    # CNG mode penalty
PEN_W    = 0.2    # penalty weight


def mean_normalize(series):
    m = series.mean()
    return series if m == 0 else series / m


def minmax_normalize(series):
    mn, mx = series.min(), series.max()
    denom  = (mx - mn) if (mx - mn) != 0 else 1.0
    return (series - mn) / denom


def recompute_stress(df, speed_col, accel_col, prev_accel_col,
                     speed_limit_col, mode_str):
    """
    Recomputes stress using the full formula including jerk term.
    mode_str: fixed string "diesel" or "cng" for baseline logs
              (baseline logs are single-mode so mode is constant)
    Returns a Series or None if required columns are missing.
    """
    required = {speed_col, accel_col, prev_accel_col, speed_limit_col}
    if not required.issubset(df.columns):
        return None

    v_norm = df[speed_col] / df[speed_limit_col].replace(0, 0.1)
    a_norm = df[accel_col].abs() / A_MAX
    jerk   = (df[accel_col] - df[prev_accel_col]).abs()

    s = (
        (v_norm - V_THRESH).clip(lower=0) +
        A_COEFF * (a_norm - A_THRESH).clip(lower=0) +
        J_COEFF * jerk
    )

    under_load = (v_norm > V_THRESH) | (a_norm > A_THRESH)
    mode_pen   = PEN_D if mode_str == "diesel" else PEN_C
    s += PEN_W * mode_pen * under_load.astype(float)

    return s


def main():
    df = pd.read_csv(INPUT)

    required = {"diesel_fuel", "diesel_co2", "cng_fuel", "cng_co2"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"Missing required columns: {required - set(df.columns)}\n"
            f"Found: {set(df.columns)}"
        )

    # --------------------------------------------------
    # Stress: recompute from raw columns if available
    # (merge_logs.py now preserves diesel_speed, diesel_accel,
    #  cng_speed, cng_accel and speed_limit)
    # Falls back to pre-computed stress columns with a warning.
    # --------------------------------------------------
    diesel_stress = recompute_stress(
        df,
        speed_col="diesel_speed",
        accel_col="diesel_accel",
        prev_accel_col="diesel_accel",   # prev_accel approximated as same col shifted
        speed_limit_col="speed_limit",
        mode_str="diesel"
    )

    cng_stress = recompute_stress(
        df,
        speed_col="cng_speed",
        accel_col="cng_accel",
        prev_accel_col="cng_accel",
        speed_limit_col="speed_limit",
        mode_str="cng"
    )

    if diesel_stress is not None and cng_stress is not None:
        print("Stress recomputed from raw columns (includes jerk term)")
        df["diesel_stress"] = diesel_stress
        df["cng_stress"]    = cng_stress
    else:
        if "diesel_stress" not in df.columns or "cng_stress" not in df.columns:
            raise RuntimeError(
                "Cannot compute stress: raw speed/accel columns not found "
                "and no pre-computed stress columns available."
            )
        print("WARNING: using pre-computed stress columns (no jerk term).")

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------
    norm = mean_normalize if USE_MEAN_NORM else minmax_normalize

    for col in ["diesel_fuel", "cng_fuel", "diesel_co2",
                "cng_co2", "diesel_stress", "cng_stress"]:
        df[col + "_n"] = norm(df[col])

    # --------------------------------------------------
    # Cost computation
    # --------------------------------------------------
    df["J_diesel"] = (
        W_F   * df["diesel_fuel_n"] +
        W_CO2 * df["diesel_co2_n"] +
        W_S   * df["diesel_stress_n"]
    )

    df["J_cng"] = (
        W_F   * df["cng_fuel_n"] +
        W_CO2 * df["cng_co2_n"] +
        W_S   * df["cng_stress_n"]
    )

    mean_J  = 0.5 * (df["J_diesel"].mean() + df["J_cng"].mean())
    delta_J = DELTA_J_FACTOR * mean_J

    # --------------------------------------------------
    # Label assignment
    # --------------------------------------------------
    diff = (df["J_diesel"] - df["J_cng"]).abs()

    df["best_mode"]   = np.where(
        diff <= delta_J, "tie",
        np.where(df["J_diesel"] < df["J_cng"], "diesel", "cng")
    )
    df["delta_J_abs"] = diff

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    print(f"delta_J threshold  : {delta_J:.6f}")
    print(f"Label distribution :\n{df['best_mode'].value_counts()}")
    print(f"Rows saved         : {len(df)}")

    df.to_csv(OUTPUT, index=False)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()