"""
run.py

Purpose:
--------
Runs a SUMO baseline simulation (CNG-only OR Diesel-only) and logs
per-vehicle dynamic, emission, and stress metrics at each timestep.

These are INFINITE FUEL baselines — no tank depletion, no switching.
Tank logic only applies to the adaptive controllers (AI, PID, Rule-Based).

Usage:
    CNG baseline:    set SUMO_CFG = "sim_cng.sumocfg",    LOG_FILE = "log_cng.csv"
    Diesel baseline: set SUMO_CFG = "sim_diesel.sumocfg", LOG_FILE = "log_diesel.csv"

Stress formula (identical across all controllers):
    S = max(0, v_norm-0.7) + 1.5*max(0, a_norm-0.6) + 0.1*|jerk|
      + 0.2*mode_pen  (only when v_norm>0.7 OR a_norm>0.6)
      mode_pen: diesel=1.0, cng=0.4
"""

import traci
import csv

SUMO_CFG = "sim_cng_map2.sumocfg"   # change to sim_diesel.sumocfg for diesel baseline
LOG_FILE  = "log_cng.csv"       # change to log_diesel.csv  for diesel baseline
STEP      = 1.0

A_MAX = 3.0


def compute_stress(speed, accel, prev_accel, mode, speed_limit):
    """
    Unified stress formula — identical to all adaptive controllers.
      S = max(0, v_norm-0.7) + 1.5*max(0, a_norm-0.6) + 0.1*|jerk|
        + 0.2*mode_pen  (only under load: v_norm>0.7 OR a_norm>0.6)
    """
    v_norm = speed / max(speed_limit, 0.1)
    a_norm = abs(accel) / A_MAX
    jerk   = abs(accel - prev_accel)

    s = (
        max(0.0, v_norm - 0.7) +
        1.5 * max(0.0, a_norm - 0.6) +
        0.1 * jerk
    )

    if v_norm > 0.7 or a_norm > 0.6:
        mode_pen = 1.0 if mode == "diesel" else 0.4
        s += 0.2 * mode_pen

    return s


def main():
    traci.start(["sumo", "-c", SUMO_CFG, "--step-length", str(STEP)])

    prev_speed = {}
    prev_accel = {}

    with open(LOG_FILE, "w", newline="") as f:
        fieldnames = [
            "time", "veh_id", "mode",
            "speed", "accel",
            "edge", "lane", "speed_limit",
            "co2", "fuel", "stress"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            current_time = traci.simulation.getTime()

            for vid in traci.vehicle.getIDList():

                # Initialise per-vehicle state
                if vid not in prev_speed:
                    prev_speed[vid] = traci.vehicle.getSpeed(vid)
                    prev_accel[vid] = 0.0

                speed  = traci.vehicle.getSpeed(vid)
                pa     = prev_speed[vid]
                accel  = speed - pa

                lane        = traci.vehicle.getLaneID(vid)
                speed_limit = max(traci.lane.getMaxSpeed(lane), 0.1)
                mode        = traci.vehicle.getTypeID(vid)

                stress = compute_stress(speed, accel, prev_accel[vid], mode, speed_limit)

                writer.writerow({
                    "time":        current_time,
                    "veh_id":      vid,
                    "mode":        mode,
                    "speed":       speed,
                    "accel":       accel,
                    "edge":        traci.vehicle.getRoadID(vid),
                    "lane":        lane,
                    "speed_limit": speed_limit,
                    "co2":         traci.vehicle.getCO2Emission(vid),
                    "fuel":        traci.vehicle.getFuelConsumption(vid),
                    "stress":      stress
                })

                prev_speed[vid] = speed
                prev_accel[vid] = accel

    traci.close()


if __name__ == "__main__":
    main()