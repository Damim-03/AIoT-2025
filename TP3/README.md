# TP3 – Solution: the TP2 models on a microcontroller (ESP32 + DHT22)

Statement: [TP3_Logistic_Arduino.md](TP3_Logistic_Arduino.md) · Setup: [install.md](install.md)

## Board: ESP32 instead of the Arduino Uno
The statement uses an Arduino Uno. This solution runs **both** TP2 models, and the XGBoost one does not fit in the Uno:

| | Arduino Uno | ESP32 |
|---|---|---|
| Flash / RAM | 32 KB / 2 KB | 4 MB / 520 KB |
| Logistic Regression (37 numbers, 148 bytes) | fits | fits |
| XGBoost (300 trees, 4,922 nodes, 59,668 bytes) | does not fit | fits |

So the provided Wokwi project is kept, with the board changed: `platformio.ini` targets `esp32dev`, `diagram.json` wires the DHT22 to an ESP32 DevKit (data on GPIO 15) and `wokwi.toml` points to the ESP32 firmware.
Firmware: 332,609 bytes of Flash (25.4%) and 21,532 bytes of RAM (6.6%).

## Files
| File | Role |
|---|---|
| `models/LR-Pipeline.pkl`, `models/XGB-Pipeline.pkl` | The two pipelines trained in TP2, unchanged (StandardScaler + model, 12 sensor features) |
| `tools/export_models.py` | **Step 1** – prints μ, σ, W and b of the LR, writes the XGBoost trees to `src/xgb_model.h`, and checks both C computations against Python on the 62,630 dataset rows |
| `src/main.cpp` | **Steps 2–4** – the LR parameters as constants, `standardize()`, `sigmoid()`, `predict()`, the XGBoost call, and the real-time loop on the DHT22 |
| `src/xgb_model.h` | XGBoost exported to C: every node of the 300 trees, the scaler folded into the split thresholds |
| `diagram.json`, `wokwi.toml`, `platformio.ini` | Wokwi circuit and build settings (ESP32 + DHT22) |
| `smoke_detection_iot.csv` | Dataset of TP1, used by the export script for the checks |

## Models (TP2 pipelines, test set of 12,526 samples)
| Model | Accuracy | Recall | F1 | On the board |
|---|---|---|---|---|
| Logistic Regression | 0.8951 | 0.9482 | 0.9281 | 37 numbers in `main.cpp` |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 4,922 nodes in `xgb_model.h` |

C vs Python, on all 62,630 rows: XGBoost gives the same decision on every row. Logistic Regression differs on one row only, whose probability is 0.500003 in Python and 0.499995 in float32: a sample sitting on the threshold.

LR parameters:

| Feature | μ (mean) | σ (std) | W (weight) |
|---|---|---|---|
| Temperature_C | 15.9531 | 14.3580 | −1.1063 |
| Humidity_% | 48.5743 | 8.8212 | 1.3782 |
| TVOC_ppb | 1922.8844 | 7755.8950 | −18.2296 |
| eCO2_ppm | 667.7439 | 1903.7822 | 5.9192 |
| Raw H2 | 12942.3008 | 271.1686 | 3.3515 |
| Raw Ethanol | 19754.2500 | 606.9760 | −7.8964 |
| Pressure_hPa | 938.6320 | 1.3245 | −2.4279 |
| PM1.0 | 100.8873 | 925.3035 | −0.3308 |
| PM2.5 | 186.0739 | 1991.5278 | −0.1946 |
| NC0.5 | 490.2093 | 4259.4155 | −0.4543 |
| NC1.0 | 205.4578 | 2232.4849 | −0.1843 |
| NC2.5 | 81.3198 | 1095.2900 | −0.0682 |

Bias b = **−0.3844**.

## What the DHT22 changes
Only Temperature and Humidity come from the DHT22; the 10 other features keep the values of the `X` array of the provided `main.cpp` (first row of the dataset, a *normal* case).
- **Logistic Regression** answers *fire* for most readings, and *normal* only for hot and dry air (about T ≥ 43 °C with H ≤ 18 %).
- **XGBoost** answers *normal* for every reading of the DHT22 range (−40…80 °C, 0…100 %; highest P = 0.0019): its decision depends on the gas and particle sensors, which stay at the values of that normal row.

## Run
1. Open the `TP3` folder in VS Code → PlatformIO **Build**.
2. `F1` → **Wokwi: Start Simulator**, then click the DHT22 to move the temperature and humidity sliders.

Expected serial output (same probabilities as Python; the times are measured on the board):
```
Temperature: 24.00 C | Humidity: 40.00 %
  LR : P(fire) = 0.9995 -> Class 1 (FIRE)    [.. us]
  XGB: P(fire) = 0.0000 -> Class 0 (normal)  [.. us]
Temperature: 70.00 C | Humidity: 5.00 %
  LR : P(fire) = 0.2026 -> Class 0 (normal)  [.. us]
  XGB: P(fire) = 0.0000 -> Class 0 (normal)  [.. us]
```
