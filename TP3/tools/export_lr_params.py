"""TP3 step 1 - train the Temperature/Humidity Logistic Regression and export its parameters for the Arduino Uno.

The Uno only has the DHT22 sensor, so the model uses the 2 features it measures: Temperature and Humidity.
The model is the LR-Pipeline of Practicals 1 and 2 (StandardScaler + LogisticRegression, same 80/20 stratified split,
random_state = 42), trained on these 2 columns only, and saved to models/LR-TH-Pipeline.pkl.

Prints:
  - the test metrics of the model (Accuracy, Precision, Recall, F1, confusion matrix);
  - the four sets of values used in src/main.cpp: MEAN and STD (StandardScaler), WEIGHTS and BIAS (LogisticRegression);
  - a check that the C formula, computed in float32 like on the Uno, gives the same decision as scikit-learn;
  - the expected P(fire) for a few DHT22 readings, to compare with the Wokwi serial monitor.

Run from the TP3 folder:  python tools/export_lr_params.py
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TP3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["Temperature_C", "Humidity_%"]   # feature 0 -> Temperature, feature 1 -> Humidity (same order in C++)
RANDOM_STATE = 42

# ---- same data preparation and split as Practicals 1 and 2 ----
df = pd.read_csv(os.path.join(TP3, "smoke_detection_iot.csv"))
X = df.drop(columns=["Fire Alarm", "Unnamed: 0", "UTC", "CNT"])
X.columns = X.columns.str.replace("[", "_", regex=False).str.replace("]", "", regex=False)
y = df["Fire Alarm"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_test = X_train[FEATURES], X_test[FEATURES]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
]).fit(X_train, y_train)
os.makedirs(os.path.join(TP3, "models"), exist_ok=True)
with open(os.path.join(TP3, "models", "LR-TH-Pipeline.pkl"), "wb") as f:
    pickle.dump(pipe, f)

y_pred = pipe.predict(X_test)
print(f"LR on Temperature + Humidity, test set ({len(X_test)} samples):")
print(f"  Accuracy {metrics.accuracy_score(y_test, y_pred):.4f} | Precision {metrics.precision_score(y_test, y_pred):.4f}"
      f" | Recall {metrics.recall_score(y_test, y_pred):.4f} | F1 {metrics.f1_score(y_test, y_pred):.4f}")
print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics.confusion_matrix(y_test, y_pred).tolist()}\n")

# ---- the four parameter sets, as float32 like on the Uno ----
scaler, lr = pipe.named_steps["scaler"], pipe.named_steps["model"]
mean = scaler.mean_.astype(np.float32)
std = scaler.scale_.astype(np.float32)
w = lr.coef_[0].astype(np.float32)
b = np.float32(lr.intercept_[0])


def flit(x):
    # C float literal with enough digits to keep the float32 value
    s = f"{x:.9g}"
    return s if any(ch in s for ch in ".e") else s + ".0"


def c_array(name, v):
    return f"const float {name}[N_FEATURES] = {{{', '.join(flit(x) for x in v)}}};"


print("// ---- paste into src/main.cpp ----")
print(f"const int N_FEATURES = {len(FEATURES)};")
print(c_array("MEAN", mean) + "     // mu_Temperature, mu_Humidity")
print(c_array("STD", std) + "      // sigma_Temperature, sigma_Humidity")
print(c_array("WEIGHTS", w) + "  // W_Temperature, W_Humidity")
print(f"const float BIAS = {flit(b)};\n")


def c_proba(Xv):
    # standardize() + predict() + sigmoid() of main.cpp, in float32
    z = (((Xv.astype(np.float32) - mean) / std) * w).sum(axis=1, dtype=np.float32) + b
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


p_sk = pipe.predict_proba(X_test)[:, 1]
p_c = c_proba(X_test.values)
agree = ((p_c >= 0.5) == (p_sk >= 0.5)).mean()
print(f"C formula vs scikit-learn on the test set: same decision {agree:.2%}, max |P diff| = {np.abs(p_c - p_sk).max():.1e}")
assert agree == 1.0

print("\nExpected serial output in Wokwi (move the DHT22 sliders to these values):")
for t, h in [(24.0, 40.0), (15.0, 30.0), (35.0, 20.0), (20.0, 57.36), (10.0, 70.0)]:
    p = c_proba(np.array([[t, h]]))[0]
    print(f"  T = {t:5.1f} C, H = {h:5.1f} %  ->  P(fire) = {p:.4f}  class {int(p >= 0.5)}")
