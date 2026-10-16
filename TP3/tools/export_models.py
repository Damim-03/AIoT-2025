"""TP3 step 1 - export the two TP2 models for the ESP32.

Loads the pipelines trained in TP2, unchanged:
  models/LR-Pipeline.pkl   StandardScaler + LogisticRegression on the 12 sensor features
  models/XGB-Pipeline.pkl  StandardScaler + XGBClassifier (300 trees of depth 6) on the same features

Logistic Regression: prints the four sets of values used in src/main.cpp (MEAN, STD, WEIGHTS, BIAS).
XGBoost: writes src/xgb_model.h, every node of the 300 trees with the scaler folded into the split thresholds, so the
board compares raw sensor values, and xgb_predict_proba().

Both C computations are checked in float32 against scikit-learn / XGBoost: they must give the same decision on every
row of the dataset. The script ends with the expected serial output for a few DHT22 readings.

Run from the TP3 folder:  python tools/export_models.py
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost
from sklearn import metrics
from sklearn.model_selection import train_test_split

TP3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lr_pipe = pickle.load(open(os.path.join(TP3, "models", "LR-Pipeline.pkl"), "rb"))
xgb_pipe = pickle.load(open(os.path.join(TP3, "models", "XGB-Pipeline.pkl"), "rb"))
features = list(lr_pipe.feature_names_in_)
assert list(xgb_pipe.feature_names_in_) == features

# ---- same data and split as TP1 and TP2 ----
df = pd.read_csv(os.path.join(TP3, "smoke_detection_iot.csv"))
X = df.drop(columns=["Fire Alarm", "Unnamed: 0", "UTC", "CNT"])
X.columns = X.columns.str.replace("[", "_", regex=False).str.replace("]", "", regex=False)
y = df["Fire Alarm"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
assert list(X.columns) == features


def flit(x):
    # C float literal with enough digits to keep the float32 value
    s = f"{x:.9g}"
    return s if any(ch in s for ch in ".e") else s + ".0"


def check(name, p_c, p_ref, X_rows):
    # A decision may only differ where Python's P(fire) is within 1e-4 of the 0.5 threshold (float32 rounding)
    differ = np.where((p_c >= 0.5) != (p_ref >= 0.5))[0]
    print(f"{name}: C formula vs Python on all {len(X_rows)} rows: {len(X_rows) - len(differ)} same decisions, "
          f"max |P diff| = {np.abs(p_c - p_ref).max():.1e}")
    for i in differ:
        print(f"     row {i}: P_python = {p_ref[i]:.6f}, P_float32 = {p_c[i]:.6f} (on the threshold)")
        assert abs(p_ref[i] - 0.5) < 1e-4


for name, pipe in (("LR ", lr_pipe), ("XGB", xgb_pipe)):
    y_pred = pipe.predict(X_test)
    print(f"{name} test set ({len(X_test)} samples): Accuracy {metrics.accuracy_score(y_test, y_pred):.4f}"
          f" | Recall {metrics.recall_score(y_test, y_pred):.4f} | F1 {metrics.f1_score(y_test, y_pred):.4f}")
print()

# =====================================================================================================
# Logistic Regression: the four parameter sets of src/main.cpp
# =====================================================================================================
scaler, lr = lr_pipe.named_steps["scaler"], lr_pipe.named_steps["model"]
mean = scaler.mean_.astype(np.float32)
std = scaler.scale_.astype(np.float32)
w = lr.coef_[0].astype(np.float32)
b = np.float32(lr.intercept_[0])

print("// ---- Logistic Regression, paste into src/main.cpp ----")
print(f"const int N_FEATURES = {len(features)};")
print("// Feature order: " + ", ".join(features))
for name, v in (("MEAN", mean), ("STD", std), ("WEIGHTS", w)):
    print(f"const float {name}[N_FEATURES] = {{{', '.join(flit(x) for x in v)}}};")
print(f"const float BIAS = {flit(b)};\n")


def lr_c_proba(Xv):
    # standardize() + predict() + sigmoid() of main.cpp, in float32
    z = (((np.asarray(Xv, dtype=np.float32) - mean) / std) * w).sum(axis=1, dtype=np.float32) + b
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


check("LR ", lr_c_proba(X.values), lr_pipe.predict_proba(X)[:, 1], X)

# =====================================================================================================
# XGBoost: all the trees as one node table in src/xgb_model.h
# =====================================================================================================
xs = xgb_pipe.named_steps["scaler"]
booster = xgb_pipe.named_steps["model"].get_booster()
trees = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]

# The scaler is folded into the thresholds: (x - mean) / std < t  <=>  x < T. XGBoost puts t exactly on a data value,
# so T = t * std + mean would sit on that value too and float rounding could flip the decision. T is placed instead
# halfway between the largest raw value that goes left and the smallest that goes right.
values = {fi: np.unique(X.iloc[:, fi].values) for fi in range(len(features))}


def raw_threshold(fi, t):
    v = values[fi]
    goes_left = ((v - xs.mean_[fi]) / xs.scale_[fi]).astype(np.float32) < np.float32(t)
    if not goes_left.any():
        return np.float32(v.min())
    if goes_left.all():
        return np.nextafter(np.float32(v.max()), np.float32(np.inf))
    return np.float32((v[goes_left].max() + v[~goes_left].min()) / 2)


value, left, right, feat, roots = [], [], [], [], []   # leaf: feature = -1 and value = leaf score
for t in trees:
    off = len(value)
    roots.append(off)
    for k in range(len(t["left_children"])):
        if t["left_children"][k] == -1:
            value.append(np.float32(t["split_conditions"][k])); left.append(0); right.append(0); feat.append(-1)
        else:
            fi = t["split_indices"][k]
            value.append(raw_threshold(fi, t["split_conditions"][k]))
            left.append(off + t["left_children"][k]); right.append(off + t["right_children"][k]); feat.append(fi)
value, left, right, feat = (np.array(value, np.float32), np.array(left), np.array(right), np.array(feat))
assert len(value) < 65536


def xgb_leaf_sum(Xv):
    # Same traversal as xgb_predict_proba() in C, on raw float32 sensor values
    Xv = np.asarray(Xv, dtype=np.float32)
    rows = np.arange(len(Xv))
    total = np.zeros(len(Xv), dtype=np.float32)
    for r in roots:
        n = np.full(len(Xv), r)
        while (feat[n] >= 0).any():
            inner = feat[n] >= 0
            go_left = Xv[rows, np.maximum(feat[n], 0)] < value[n]
            n = np.where(inner, np.where(go_left, left[n], right[n]), n)
        total += value[n]
    return total


margin = booster.predict(xgboost.DMatrix(xs.transform(X)), output_margin=True)
base = np.float32(np.median(margin - xgb_leaf_sum(X.values)))


def xgb_c_proba(Xv):
    z = (base + xgb_leaf_sum(Xv)).astype(np.float32)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


check("XGB", xgb_c_proba(X.values), xgb_pipe.predict_proba(X)[:, 1], X)

with open(os.path.join(TP3, "src", "xgb_model.h"), "w") as f:
    f.write("// Auto-generated by tools/export_models.py from models/XGB-Pipeline.pkl - do not edit by hand.\n")
    f.write("#pragma once\n#include <math.h>\n#include <stdint.h>\n\n")
    f.write(f"#define XGB_N_TREES {len(roots)}\n#define XGB_N_NODES {len(value)}\n")
    f.write("// Feature order: " + ", ".join(features) + "\n")
    f.write("// Inner node: go to `left` if x[feature] < value (raw sensor units, the scaler is folded in).\n"
            "// Leaf: feature = -1 and value = leaf score.\n")
    f.write("typedef struct { float value; uint16_t left, right; int8_t feature; } XgbNode;\n\n")
    f.write(f"static const float XGB_BASE_MARGIN = {flit(base)};\n")
    f.write(f"static const uint16_t XGB_TREE_ROOT[XGB_N_TREES] = {{{', '.join(map(str, roots))}}};\n")
    f.write("static const XgbNode XGB_NODES[XGB_N_NODES] = {\n")
    for v, l, r, fi in zip(value, left, right, feat):
        f.write(f"  {{{flit(v)}, {l}, {r}, {fi}}},\n")
    f.write("};\n\n")
    f.write("// P(fire) = sigmoid( base + sum of the leaf reached in every tree ), on the raw sensor values\n")
    f.write("static inline float xgb_predict_proba(const float *x) {\n"
            "  float z = XGB_BASE_MARGIN;\n"
            "  for (int t = 0; t < XGB_N_TREES; t++) {\n"
            "    const XgbNode *n = &XGB_NODES[XGB_TREE_ROOT[t]];\n"
            "    while (n->feature >= 0) n = &XGB_NODES[x[n->feature] < n->value ? n->left : n->right];\n"
            "    z += n->value;\n"
            "  }\n"
            "  return 1.0f / (1.0f + expf(-z));\n"
            "}\n")
size = len(value) * 12 + len(roots) * 2 + 4
print(f"\nWrote src/xgb_model.h: {len(roots)} trees, {len(value)} nodes, {size:,} bytes of model data")

# =====================================================================================================
# Expected serial output in Wokwi
# =====================================================================================================
X_FIXED = [20.0, 57.36, 0, 400, 12306, 18520, 939.735, 0.0, 0.0, 0.0, 0.0, 0.0]   # X in main.cpp
print("\nExpected P(fire) in Wokwi (DHT22 temperature / humidity, other 10 features = X in main.cpp):")
for t, h in [(24.0, 40.0), (20.0, 57.36), (42.6, 38.0), (60.0, 10.0), (70.0, 5.0)]:
    row = [[t, h] + X_FIXED[2:]]
    p_lr, p_xgb = lr_c_proba(row)[0], xgb_c_proba(row)[0]
    print(f"  T = {t:5.1f} C, H = {h:5.1f} %  ->  LR {p_lr:.4f} class {int(p_lr >= 0.5)}"
          f" | XGB {p_xgb:.4f} class {int(p_xgb >= 0.5)}")
