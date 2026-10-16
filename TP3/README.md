# TP3 – Solution: Logistic Regression on the Arduino Uno

Statement: [TP3_Logistic_Arduino.md](TP3_Logistic_Arduino.md) · Setup: [install.md](install.md)

## Files
| File | Role |
|---|---|
| `tools/export_lr_params.py` | **Step 1** – trains the LR-Pipeline (StandardScaler + LogisticRegression, same split as TP1/TP2) on Temperature and Humidity, saves it to `models/LR-TH-Pipeline.pkl`, prints μ, σ, W and b, and checks the C formula against scikit-learn |
| `src/main.cpp` | **Steps 2–4** – the parameters as constants, `standardize()`, `sigmoid()`, `predict()`, and the real-time loop on the DHT22 |
| `diagram.json`, `wokwi.toml`, `platformio.ini` | Wokwi circuit (Uno + DHT22 on pin 2) and build settings, as provided |
| `smoke_detection_iot.csv` | Dataset of TP1, used by the export script |
| `esp32_fire_alarm/` | **Bonus** – the full 12-sensor models (LR and compressed XGBoost) on an ESP32, see its README |

## Why 2 features
The Uno only has the DHT22, which measures Temperature and Humidity, so the model is trained on these 2 features (`N_FEATURES = 2`, as in the statement).
The 12-feature model of TP2, with the 10 other sensors fixed at the values of the first dataset row, would print *fire* for almost every DHT22 reading (P = 1.0 at 20 °C / 57 %, although that row is a normal case).

## Model (test set, 12,526 samples)
| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.7711 | 0.7673 | 0.9755 | 0.8590 |

| Parameter | Temperature | Humidity |
|---|---|---|
| μ (mean) | 15.9531 | 48.5743 |
| σ (std) | 14.3580 | 8.8212 |
| W (weight) | −0.2087 | 0.9680 |
| b (bias) | 0.9694 | |

## Run
1. Open the `TP3` folder in VS Code → PlatformIO **Build**.
2. `F1` → **Wokwi: Start Simulator**, then click the DHT22 to move the temperature and humidity sliders.

Expected serial output (same values as Python):
```
Temperature: 24.00 C | Humidity: 40.00 % | Predicted Probability: 0.4779 | Predicted Class: 0 (normal)
Temperature: 15.00 C | Humidity: 30.00 % | Predicted Probability: 0.2583 | Predicted Class: 0 (normal)
Temperature: 20.00 C | Humidity: 57.36 % | Predicted Probability: 0.8670 | Predicted Class: 1 (FIRE)
Temperature: 10.00 C | Humidity: 70.00 % | Predicted Probability: 0.9679 | Predicted Class: 1 (FIRE)
```
Firmware size: 6,042 bytes of Flash (18.7%) and 219 bytes of RAM (10.7%).
