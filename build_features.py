# build_features.py
# Reads training_dataset.csv and net.net.xml and produces features_final.csv
#
# Produces exactly 14 training features + 2 extra context columns:
#
# 14 TRAINING FEATURES (must match train_xgboost_env_only.py feature_cols exactly):
#   0  speed
#   1  accel
#   2  speed_limit
#   3  speed_norm
#   4  hard_accel_flag        (abs(accel) > HARD_ACCEL_THRESH)
#   5  prev_speed
#   6  prev_accel
#   7  rolling_mean_speed_3s
#   8  rolling_std_accel_3s
#   9  current_mode_encoded
#  10  edge_type_code
#  11  lane_max_speed
#  12  traffic_density_on_edge
#  13  avg_speed_on_edge
#
# 2 EXTRA CONTEXT (saved to CSV but NOT used in training):
#  14  time_since_last_switch
#  15  previous_mode_encoded

import os
import pandas as pd
import numpy as np

TRAIN_IN = "training_dataset.csv"
NET_FILE = "net.net.xml"
OUT_FILE = "features_final.csv"

# hard_accel threshold in absolute m/s^2
# Must match AI controller: abs(accel)/A_MAX > 0.5  => 0.5 * 3.0 = 1.5 m/s^2
HARD_ACCEL_THRESH = 1.5   # m/s^2  (= 0.5 * A_MAX)

A_MAX = 3.0               # must match all controllers

EDGE_TYPE_PRIORITY = {
    "residential": 0,
    "tertiary": 1,
    "trunk": 2,
    "motorway": 3,
    "primary": 2,
    "secondary": 1,
    "unclassified": 0
}

try:
    import sumolib
    HAVE_SUMOLIB = True
except Exception:
    HAVE_SUMOLIB = False
    print("sumolib not found — lane_max_speed will be filled from speed_limit.")


def get_edge_meta_from_net(netfile):
    if not HAVE_SUMOLIB:
        return {}, {}

    net = sumolib.net.readNet(netfile)
    edge_to_type = {}
    lane_max_speed = {}
    for edge in net.getEdges():
        etype = None
        try:
            etype = edge.getType()
        except Exception:
            etype = None
        if etype is None:
            etype = getattr(edge, "type", "other")
        edge_to_type[edge.getID()] = str(etype)

        for lane in edge.getLanes():
            try:
                lane_max_speed[lane.getID()] = float(lane.getMaxSpeed())
            except Exception:
                lane_max_speed[lane.getID()] = float(edge.getSpeed()) if edge.getSpeed() is not None else np.nan

    return edge_to_type, lane_max_speed


def map_edge_type_to_code(edge_id, edge_to_type):
    s = edge_to_type.get(edge_id, "")
    for k, v in EDGE_TYPE_PRIORITY.items():
        if k in str(s):
            return v
    if edge_id is None:
        return 4
    eid = str(edge_id)
    if eid.startswith(":") or "#" in eid:
        return 4
    return 4


def main():
    if not os.path.exists(TRAIN_IN):
        raise FileNotFoundError(f"{TRAIN_IN} not found. Run step3_finalize_labels.py first.")

    print("Loading training dataset:", TRAIN_IN)
    df = pd.read_csv(TRAIN_IN)

    needed = {"time", "veh_id", "speed", "accel", "edge", "lane",
              "prev_speed", "prev_accel",
              "rolling_mean_speed_3s", "rolling_std_accel_3s",
              "time_since_last_switch", "previous_mode",
              "label_mode", "label_num"}
    missing = needed - set(df.columns)
    if missing:
        print("Warning: missing columns in training dataset:", missing)

    df["time"]   = df["time"].astype(float)
    df["veh_id"] = df["veh_id"].astype(int)

    edge_to_type, lane_max_speed = get_edge_meta_from_net(NET_FILE)
    if not edge_to_type:
        print("No edge metadata from net — edge_type_code -> 4, lane_max_speed -> speed_limit fallback")

    FEATURES = pd.DataFrame()
    FEATURES["time"]   = df["time"]
    FEATURES["veh_id"] = df["veh_id"]

    # --------------------------------------------------
    # Feature 0-3: basic kinematics
    # --------------------------------------------------
    FEATURES["speed"]       = df["speed"]
    FEATURES["accel"]       = df["accel"]
    FEATURES["speed_limit"] = df.get("speed_limit", np.nan)
    FEATURES["speed_norm"]  = (
        FEATURES["speed"] / FEATURES["speed_limit"].replace(0, np.nan)
    ).fillna(0.0)

    # --------------------------------------------------
    # Feature 4: hard accel flag
    # Uses abs(accel) — must match AI controller threshold:
    #   deploy: abs(accel)/A_MAX > 0.5  => abs(accel) > 1.5 m/s^2
    # --------------------------------------------------
    FEATURES["hard_accel_flag"] = (df["accel"].abs() > HARD_ACCEL_THRESH).astype(int)

    # --------------------------------------------------
    # Features 5-8: temporal context
    # --------------------------------------------------
    FEATURES["prev_speed"]            = df.get("prev_speed", df["speed"])
    FEATURES["prev_accel"]            = df.get("prev_accel", df["accel"])
    FEATURES["rolling_mean_speed_3s"] = df.get("rolling_mean_speed_3s", df["speed"])
    FEATURES["rolling_std_accel_3s"]  = df.get("rolling_std_accel_3s", 0.0)

    # --------------------------------------------------
    # Feature 9: current mode encoded
    # --------------------------------------------------
    prev_mode_series = df["previous_mode"].fillna("diesel").astype(str)
    FEATURES["current_mode_encoded"] = (
        prev_mode_series.map({"diesel": 1, "cng": 0}).fillna(0).astype(int)
    )

    # --------------------------------------------------
    # Features 10-11: road metadata
    # --------------------------------------------------
    FEATURES["edge"] = df["edge"].astype(str)
    FEATURES["lane"] = df["lane"].astype(str)

    FEATURES["edge_type_code"] = FEATURES["edge"].apply(
        lambda e: map_edge_type_to_code(e, edge_to_type)
    )

    def lookup_lane_speed(l):
        if l in lane_max_speed:
            return lane_max_speed[l]
        if "_" in l:
            base = l.rsplit("_", 1)[0]
            if base in lane_max_speed:
                return lane_max_speed[base]
        return np.nan

    FEATURES["lane_max_speed"] = FEATURES["lane"].apply(lookup_lane_speed)

    # Fill NaN lane_max_speed with speed_limit — ensures no NaN reaches XGBoost
    FEATURES["lane_max_speed"] = FEATURES["lane_max_speed"].fillna(FEATURES["speed_limit"])

    # --------------------------------------------------
    # Features 12-13: traffic environment
    # --------------------------------------------------
    print("Computing traffic density and avg_speed per (time, edge)...")
    df["_time_int"] = df["time"].round(6)
    group         = df.groupby(["_time_int", "edge"])
    density       = group.size().rename("traffic_density_on_edge")
    avg_speed_edge = group["speed"].mean().rename("avg_speed_on_edge")

    df2 = df.copy().set_index(["_time_int", "edge"])
    FEATURES["traffic_density_on_edge"] = density.reindex(df2.index, fill_value=0).values
    FEATURES["avg_speed_on_edge"]       = avg_speed_edge.reindex(df2.index, fill_value=np.nan).values
    FEATURES["avg_speed_on_edge"]       = FEATURES["avg_speed_on_edge"].fillna(FEATURES["speed"])

    # --------------------------------------------------
    # Extra context (not used in training — kept for traceability)
    # --------------------------------------------------
    FEATURES["time_since_last_switch"] = df.get("time_since_last_switch", 1e6)
    FEATURES["previous_mode_encoded"]  = (
        prev_mode_series.map({"diesel": 1, "cng": 0}).fillna(0).astype(int)
    )

    # --------------------------------------------------
    # Labels and bookkeeping
    # --------------------------------------------------
    FEATURES["label_mode"] = df["label_mode"]
    FEATURES["label_num"]  = df["label_num"]
    FEATURES["time"]       = df["time"]
    FEATURES["veh_id"]     = df["veh_id"]
    FEATURES["edge"]       = df["edge"]
    FEATURES["lane"]       = df["lane"]

    # --------------------------------------------------
    # Final column order — 14 training features first, then extras
    # --------------------------------------------------
    ordered_cols = [
        # 14 training features
        "speed", "accel", "speed_limit", "speed_norm", "hard_accel_flag",
        "prev_speed", "prev_accel", "rolling_mean_speed_3s", "rolling_std_accel_3s",
        "current_mode_encoded",
        "edge_type_code", "lane_max_speed", "traffic_density_on_edge", "avg_speed_on_edge",
        # extra context
        "time_since_last_switch", "previous_mode_encoded",
        # labels
        "label_mode", "label_num",
        # bookkeeping
        "time", "veh_id", "edge", "lane"
    ]
    OUT = FEATURES[[c for c in ordered_cols if c in FEATURES.columns]]

    print("Saving:", OUT_FILE)
    OUT.to_csv(OUT_FILE, index=False)
    print("Saved. Rows:", len(OUT))
    print("Columns:", list(OUT.columns))

    # Sanity check
    train_features = [
        "speed", "accel", "speed_limit", "speed_norm", "hard_accel_flag",
        "prev_speed", "prev_accel", "rolling_mean_speed_3s", "rolling_std_accel_3s",
        "current_mode_encoded",
        "edge_type_code", "lane_max_speed", "traffic_density_on_edge", "avg_speed_on_edge",
    ]
    nan_counts = OUT[train_features].isna().sum()
    if nan_counts.any():
        print("WARNING: NaN values in training features:")
        print(nan_counts[nan_counts > 0])
    else:
        print("All 14 training features have zero NaN values.")


if __name__ == "__main__":
    main()