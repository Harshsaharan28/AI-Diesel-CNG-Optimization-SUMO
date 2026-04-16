import pandas as pd

DIESEL_FILE  = "log_diesel.csv"
CNG_FILE     = "log_cng.csv"
OUTPUT_FILE  = "merged_logs.csv"

# Load baseline logs
diesel = pd.read_csv(DIESEL_FILE)
cng    = pd.read_csv(CNG_FILE)

# Rename columns to avoid collision after merge
diesel = diesel.rename(columns={
    "co2":    "diesel_co2",
    "fuel":   "diesel_fuel",
    "stress": "diesel_stress",
    "speed":  "diesel_speed",
    "accel":  "diesel_accel",
    "speed_limit": "speed_limit"   # keep one copy — both baselines share the same road
})

cng = cng.rename(columns={
    "co2":    "cng_co2",
    "fuel":   "cng_fuel",
    "stress": "cng_stress",
    "speed":  "cng_speed",
    "accel":  "cng_accel"
})

# Merge on (time, veh_id) — inner join keeps only rows present in both baselines
merged = pd.merge(
    diesel,
    cng[[
        "time", "veh_id",
        "cng_co2", "cng_fuel", "cng_stress",
        "cng_speed", "cng_accel"
    ]],
    on=["time", "veh_id"],
    how="inner"
)

merged.to_csv(OUTPUT_FILE, index=False)
print(f"Merged {len(merged)} rows -> {OUTPUT_FILE}")
print(f"Columns: {list(merged.columns)}")