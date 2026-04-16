import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATA_FILE  = "features_final.csv"
MODEL_OUT  = "best_xgboost_env_only.json"
REPORT_OUT = "xgboost_env_only_results.txt"

# --------------------------------------------------
# Option D: Robust learning settings
# NOISE_RATE: fraction of rows where current_mode_encoded is randomly
#             flipped before training -- forces model to learn env features
#             rather than relying purely on mode state inertia.
#             Set to 0.0 to disable noise injection.
# --------------------------------------------------
NOISE_RATE = 0.10

print("Loading dataset:", DATA_FILE)
df = pd.read_csv(DATA_FILE)

# --------------------------------------------------
# Target + Groups
# --------------------------------------------------
y      = df["label_num"].astype(int)
groups = df["veh_id"].values

# --------------------------------------------------
# Fill any remaining NaN values in key columns
# --------------------------------------------------
df["current_mode_encoded"]   = df["current_mode_encoded"].fillna(0)
df["time_since_last_switch"] = df["time_since_last_switch"].fillna(1e6)
df["lane_max_speed"]         = df["lane_max_speed"].fillna(df["speed_limit"])

# --------------------------------------------------
# Label distribution diagnostics
# --------------------------------------------------
label_counts = y.value_counts().sort_index()
n_diesel = label_counts.get(0, 1)
n_cng    = label_counts.get(1, 1)
global_spw = n_diesel / max(n_cng, 1)

print("\nLabel distribution:")
print(f"  diesel (0): {n_diesel:>6d}  ({n_diesel/len(y)*100:.1f}%)")
print(f"  cng    (1): {n_cng:>6d}  ({n_cng/len(y)*100:.1f}%)")
print(f"  global scale_pos_weight: {global_spw:.2f}")
print(f"  noise injection rate:    {NOISE_RATE*100:.0f}%")

# --------------------------------------------------
# Option D -- Noise injection on current_mode_encoded
# Randomly flip NOISE_RATE fraction of current_mode_encoded values.
# Prevents the model from exploiting dwell-smoothing inertia and
# forces it to learn from the 13 environment features as backup.
# --------------------------------------------------
if NOISE_RATE > 0:
    np.random.seed(42)
    noise_mask = np.random.rand(len(df)) < NOISE_RATE
    df.loc[noise_mask, "current_mode_encoded"] = (
        1 - df.loc[noise_mask, "current_mode_encoded"]
    )
    print(f"\nNoise injection: flipped {noise_mask.sum()} / {len(df)} rows "
          f"of current_mode_encoded ({noise_mask.mean()*100:.1f}%)")

# --------------------------------------------------
# 14 training features -- order MUST match build_features.py
# and the feature vector in run_ai_controller_stress_safe.py
#
#  0  speed
#  1  accel
#  2  speed_limit
#  3  speed_norm
#  4  hard_accel_flag        abs(accel) > 1.5 m/s^2  (= 0.5 * A_MAX)
#  5  prev_speed
#  6  prev_accel             NOT a duplicate of accel
#  7  rolling_mean_speed_3s
#  8  rolling_std_accel_3s
#  9  current_mode_encoded   (noise-injected -- see Option D above)
# 10  edge_type_code
# 11  lane_max_speed         NOT speed_limit
# 12  traffic_density_on_edge
# 13  avg_speed_on_edge
# --------------------------------------------------
feature_cols = [
    "speed",
    "accel",
    "speed_limit",
    "speed_norm",
    "hard_accel_flag",
    "prev_speed",
    "prev_accel",
    "rolling_mean_speed_3s",
    "rolling_std_accel_3s",
    "current_mode_encoded",
    "edge_type_code",
    "lane_max_speed",
    "traffic_density_on_edge",
    "avg_speed_on_edge",
]

print(f"\nUsing {len(feature_cols)} features:")
for i, f in enumerate(feature_cols):
    print(f"  {i:2d}  {f}")

X = df[feature_cols]

# --------------------------------------------------
# Cross-validation
# --------------------------------------------------
gkf = GroupKFold(n_splits=5)

accs, f1s, precs, recs = [], [], [], []
best_model = None
best_f1    = -1
lines      = []
fold       = 1

for train_idx, val_idx in gkf.split(X, y, groups):
    print(f"\n===== Fold {fold} =====")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Option D -- per-fold scale_pos_weight
    # Computed from THIS fold's training split for accurate balancing
    fold_diesel = (y_train == 0).sum()
    fold_cng    = (y_train == 1).sum()
    spw = fold_diesel / max(fold_cng, 1)
    print(f"  diesel: {fold_diesel}  cng: {fold_cng}  scale_pos_weight: {spw:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=500,      # was 300
        learning_rate=0.03,    # was 0.05
        max_depth=8,           # was 6
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        min_child_weight=5,    # new — reduces overfitting on small groups
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc  = accuracy_score(y_val, y_pred)
    f1   = f1_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec  = recall_score(y_val, y_pred, zero_division=0)

    accs.append(acc)
    f1s.append(f1)
    precs.append(prec)
    recs.append(rec)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

    lines.append(
        f"Fold {fold} -> Acc={acc:.4f}, F1={f1:.4f}, "
        f"Precision={prec:.4f}, Recall={rec:.4f}"
    )

    if f1 > best_f1:
        best_f1    = f1
        best_model = model

    fold += 1

# --------------------------------------------------
# Summary
# --------------------------------------------------
summary = f"""
==============================
XGBOOST RESULTS (14 features, Option D)
Option D: scale_pos_weight per fold + {NOISE_RATE*100:.0f}% noise on current_mode_encoded
==============================

Features used: {len(feature_cols)}
{chr(10).join(f'  {i:2d}  {f}' for i, f in enumerate(feature_cols))}

Label balance: diesel={n_diesel} ({n_diesel/len(y)*100:.1f}%), cng={n_cng} ({n_cng/len(y)*100:.1f}%)

AVG Accuracy:  {np.mean(accs):.4f}
AVG F1 Score:  {np.mean(f1s):.4f}
AVG Precision: {np.mean(precs):.4f}
AVG Recall:    {np.mean(recs):.4f}

Per-fold results:
""" + "\n".join(lines)

print(summary)

with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write(summary)

# --------------------------------------------------
# Save best model
# --------------------------------------------------
best_model.save_model(MODEL_OUT)
print("\nSaved model to:", MODEL_OUT)

# --------------------------------------------------
# Feature importance
# --------------------------------------------------
importance = best_model.get_booster().get_score(importance_type="gain")
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\nTop feature importances (by gain):")
for k, v in importance[:14]:
    print(f"  {k}: {v:.4f}")

with open("feature_importance_env_only.txt", "w") as f:
    for k, v in importance:
        f.write(f"{k}: {v}\n")

print("\nTraining complete.")