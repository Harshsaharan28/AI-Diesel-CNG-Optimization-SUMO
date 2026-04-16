import traci
import csv

SUMO_CFG  = "sim_cng.sumocfg"
STEP      = 1.0
MIN_DWELL = 15.0

# PID parameters
Kp = 1.2
Ki = 0.05
Kd = 0.8
TARGET_SPEED_RATIO = 0.6
PID_THRESHOLD      = 0.05
INTEGRAL_CLAMP     = 5.0

A_MAX = 3.0

# CNG tank — 12 kg starting capacity per vehicle
# getFuelConsumption() returns mg/s, divide by 1_000_000 to get kg/s
CNG_TANK_KG = 12.0

# --------------------------------------------------
# Storage
# --------------------------------------------------
last_switch_time = {}
prev_speed       = {}
prev_accel       = {}
pid_integral     = {}
pid_prev_error   = {}
cng_tank         = {}

# --------------------------------------------------
# Logging
# --------------------------------------------------
log_file = open("log_pid_controller.csv", "w", newline="")
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
# Start SUMO
# --------------------------------------------------
traci.start(["sumo-gui", "-c", SUMO_CFG, "--step-length", str(STEP)])
print("SUMO started with PID-based controller")

sim_time = 0.0

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    sim_time += STEP

    for veh in traci.vehicle.getIDList():

        if veh not in last_switch_time:
            last_switch_time[veh] = -1e9
            prev_speed[veh]       = traci.vehicle.getSpeed(veh)
            prev_accel[veh]       = 0.0
            pid_integral[veh]     = 0.0
            pid_prev_error[veh]   = 0.0
            cng_tank[veh]         = CNG_TANK_KG

        speed  = traci.vehicle.getSpeed(veh)
        ps     = prev_speed[veh]
        accel  = speed - ps
        prev_speed[veh] = speed

        lane        = traci.vehicle.getLaneID(veh)
        speed_limit = max(traci.lane.getMaxSpeed(lane), 0.1)
        v_norm      = speed / speed_limit
        a_norm      = abs(accel) / A_MAX

        current_mode      = traci.vehicle.getTypeID(veh)
        time_since_switch = sim_time - last_switch_time[veh]
        fuel              = traci.vehicle.getFuelConsumption(veh)  # mg/s

        # --------------------------------------------------
        # CNG tank logic
        # --------------------------------------------------
        if current_mode == "cng":
            cng_tank[veh] = max(0.0, cng_tank[veh] - (fuel / 1_000_000) * STEP)

        if current_mode == "cng" and cng_tank[veh] <= 0.0:
            traci.vehicle.setType(veh, "diesel")
            current_mode = "diesel"
            last_switch_time[veh] = sim_time
            print(f"[{sim_time:.1f}s] TANK EMPTY -> {veh} forced to DIESEL")

        # --------------------------------------------------
        # PID calculation
        # --------------------------------------------------
        error = TARGET_SPEED_RATIO - v_norm

        pid_integral[veh] += error * STEP
        pid_integral[veh]  = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, pid_integral[veh]))

        derivative          = (error - pid_prev_error[veh]) / STEP
        control             = Kp * error + Ki * pid_integral[veh] + Kd * derivative
        pid_prev_error[veh] = error

        # --------------------------------------------------
        # Decision
        # --------------------------------------------------
        desired_mode = current_mode
        if control > PID_THRESHOLD:
            desired_mode = "diesel"
        elif control < -PID_THRESHOLD:
            desired_mode = "cng"

        if cng_tank[veh] <= 0.0 and desired_mode == "cng":
            desired_mode = "diesel"   # block CNG if tank empty

        # --------------------------------------------------
        # Dwell constraint
        # --------------------------------------------------
        if desired_mode != current_mode and time_since_switch >= MIN_DWELL:
            traci.vehicle.setType(veh, desired_mode)
            last_switch_time[veh] = sim_time
            traci.vehicle.setColor(
                veh,
                (0, 200, 0) if desired_mode == "cng" else (200, 0, 0)
            )
            print(f"[{sim_time:.1f}s] PID -> {veh} -> {desired_mode.upper()}")

        # --------------------------------------------------
        # Stress
        # --------------------------------------------------
        jerk   = accel - prev_accel[veh]
        stress = compute_stress(v_norm, a_norm, jerk, current_mode)
        prev_accel[veh] = accel

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        co2 = traci.vehicle.getCO2Emission(veh)
        writer.writerow([
            sim_time, veh, traci.vehicle.getTypeID(veh),
            fuel, co2, stress, cng_tank[veh]
        ])

log_file.close()
traci.close()
print("PID-based simulation finished. Saved log_pid_controller.csv")