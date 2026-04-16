import traci
import xgboost as xgb
import numpy as np
import csv
from collections import deque

SUMO_CFG   = "sim_cng.sumocfg"
MODEL_FILE = "best_xgboost_env_only.json"
STEP       = 1.0
MIN_DWELL  = 15.0

# Stress gate parameters
STRESS_WINDOW  = 5
STRESS_LIMIT   = 0.7
SWITCH_PENALTY = 0.2

A_MAX = 3.0

# CNG tank — 12 kg starting capacity per vehicle
# getFuelConsumption() returns mg/s, divide by 1_000_000 to get kg/s
CNG_TANK_KG = 12.0

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)
print("Loaded AI model (14-feature version)")

# --------------------------------------------------
# State tracking
# --------------------------------------------------
last_switch_time = {}
prev_speed       = {}
prev_accel       = {}
stress_history   = {}
cng_tank         = {}

# --------------------------------------------------
# Logging
# --------------------------------------------------
log_file = open("log_ai_last.csv", "w", newline="")
writer   = csv.writer(log_file)
writer.writerow(["time", "veh_id", "mode", "fuel", "co2", "stress", "cng_tank_kg"])

# --------------------------------------------------
# Shared stress function (identical across all controllers)
# S = max(0,v_norm-0.7) + 1.5*max(0,a_norm-0.6) + 0.1*|jerk|
#   + 0.2*mode_pen  (only when v_norm>0.7 OR a_norm>0.6)
# --------------------------------------------------
def compute_stress(v_norm, a_norm, jerk, mode):
    s = (
        max(0.0, v_norm - 0.7) +
        1.5 * max(0.0, a_norm - 0.6) +
        0.1 * abs(jerk)
    )
    if v_norm > 0.7 or a_norm > 0.6:
        mode_pen = 1.0 if mode == "diesel" else 0.4
        s += 0.2 * mode_pen
    return s

# --------------------------------------------------
# Feature builder (MATCHES TRAINING — 14 features)
# --------------------------------------------------
def build_features(veh):
    speed     = traci.vehicle.getSpeed(veh)
    ps        = prev_speed[veh]           # prev_speed (feature 5)
    accel     = speed - ps                # current accel (feature 1)
    pa        = prev_accel.get(veh, 0.0)  # prev_accel (feature 6)
    prev_speed[veh] = speed

    lane  = traci.vehicle.getLaneID(veh)
    edge  = traci.vehicle.getRoadID(veh)

    speed_limit    = max(traci.lane.getMaxSpeed(lane), 0.1)
    lane_max_speed = speed_limit  # traci does not expose per-lane max separately; use same value
    speed_norm     = speed / speed_limit

    # hard_accel: abs(accel) > 1.5 m/s^2 = 0.5 * A_MAX — matches build_features.py
    hard_accel = 1 if abs(accel) / A_MAX > 0.5 else 0

    rolling_mean_speed = speed       # simplified approximation
    rolling_std_accel  = abs(accel)  # simplified approximation

    vehs            = traci.edge.getLastStepVehicleIDs(edge)
    traffic_density = len(vehs)
    avg_speed_edge  = (
        sum(traci.vehicle.getSpeed(v) for v in vehs) / traffic_density
        if traffic_density > 0 else speed
    )

    current_mode         = traci.vehicle.getTypeID(veh)
    current_mode_encoded = 1 if current_mode == "diesel" else 0
    edge_type_code       = 1

    # Feature order MUST match train_xgboost_env_only.py feature_cols exactly:
    #  0:speed  1:accel  2:speed_limit  3:speed_norm  4:hard_accel_flag
    #  5:prev_speed  6:prev_accel  7:rolling_mean  8:rolling_std
    #  9:current_mode_encoded  10:edge_type_code  11:lane_max_speed
    # 12:traffic_density  13:avg_speed_edge
    features = np.array([[
        speed, accel, speed_limit, speed_norm, hard_accel,
        ps, pa, rolling_mean_speed, rolling_std_accel,
        current_mode_encoded, edge_type_code, lane_max_speed,
        traffic_density, avg_speed_edge
    ]])

    return features

# --------------------------------------------------
# Start SUMO
# --------------------------------------------------
traci.start(["sumo-gui", "-c", SUMO_CFG, "--step-length", str(STEP)])
print("SUMO started with AI controller")

sim_time = 0.0

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    sim_time += STEP

    for veh in traci.vehicle.getIDList():

        if veh not in last_switch_time:
            last_switch_time[veh] = -1e9
            stress_history[veh]   = deque(maxlen=STRESS_WINDOW)
            prev_speed[veh]       = traci.vehicle.getSpeed(veh)
            prev_accel[veh]       = 0.0
            cng_tank[veh]         = CNG_TANK_KG

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------
        features     = build_features(veh)
        pred         = model.predict(features)[0]
        desired_mode = "cng" if pred == 1 else "diesel"

        current_mode      = traci.vehicle.getTypeID(veh)
        time_since_switch = sim_time - last_switch_time[veh]
        fuel              = traci.vehicle.getFuelConsumption(veh)  # mg/s

        # --------------------------------------------------
        # CNG tank logic
        # Deplete tank when running on CNG.
        # fuel is in mg/s — divide by 1_000_000 to convert to kg/s.
        # Block or force switch to diesel when tank reaches zero.
        # --------------------------------------------------
        if current_mode == "cng":
            cng_tank[veh] = max(0.0, cng_tank[veh] - (fuel / 1_000_000) * STEP)

        if current_mode == "cng" and cng_tank[veh] <= 0.0:
            traci.vehicle.setType(veh, "diesel")
            current_mode = "diesel"
            last_switch_time[veh] = sim_time
            print(f"[{sim_time:.1f}s] TANK EMPTY -> {veh} forced to DIESEL")

        if cng_tank[veh] <= 0.0 and desired_mode == "cng":
            desired_mode = "diesel"   # block CNG switch if tank empty

        # --------------------------------------------------
        # Stress computation (independent of build_features)
        # --------------------------------------------------
        speed       = traci.vehicle.getSpeed(veh)
        speed_limit = max(traci.lane.getMaxSpeed(traci.vehicle.getLaneID(veh)), 0.1)
        v_norm      = speed / speed_limit
        ps          = prev_speed.get(veh, speed)
        accel       = speed - ps
        a_norm      = abs(accel) / A_MAX
        jerk        = accel - prev_accel[veh]

        inst_stress = compute_stress(v_norm, a_norm, jerk, current_mode)

        prev_accel[veh] = accel
        stress_history[veh].append(inst_stress)
        avg_recent_stress = sum(stress_history[veh]) / len(stress_history[veh])

        # --------------------------------------------------
        # Switching logic
        # --------------------------------------------------
        allow_switch = (
            desired_mode != current_mode and
            time_since_switch >= MIN_DWELL and
            avg_recent_stress < STRESS_LIMIT
        )

        if allow_switch:
            traci.vehicle.setType(veh, desired_mode)
            last_switch_time[veh] = sim_time
            inst_stress += SWITCH_PENALTY
            traci.vehicle.setColor(
                veh,
                (0, 200, 0) if desired_mode == "cng" else (200, 0, 0)
            )
            print(f"[{sim_time:.1f}s] AI -> {veh} -> {desired_mode.upper()}")

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        co2 = traci.vehicle.getCO2Emission(veh)
        writer.writerow([
            sim_time, veh, traci.vehicle.getTypeID(veh),
            fuel, co2, inst_stress, cng_tank[veh]
        ])

log_file.close()
traci.close()
print("AI simulation finished. Saved log_ai_last.csv")