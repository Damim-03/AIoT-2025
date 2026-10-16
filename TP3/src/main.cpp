// TP3 - Logistic Regression fire alarm on the Arduino Uno (Wokwi)
// The model (StandardScaler + LogisticRegression) was trained in Python on the 2 features the DHT22 measures,
// Temperature and Humidity. Its parameters are printed by tools/export_lr_params.py.
#include <math.h>
#include "DHT.h"
#define DHTPIN 2      // Digital pin connected to the DHT sensor
#define DHTTYPE DHT22 // DHT 22  (AM2302), AM2321
DHT dht(DHTPIN, DHTTYPE);

// ---- Model parameters (feature 0 = Temperature in C, feature 1 = Humidity in %) ----
const int N_FEATURES = 2;
const float MEAN[N_FEATURES] = {15.9531155, 48.5743027};     // mu_Temperature, mu_Humidity
const float STD[N_FEATURES] = {14.3580389, 8.82118797};      // sigma_Temperature, sigma_Humidity
const float WEIGHTS[N_FEATURES] = {-0.208676055, 0.96797812};  // W_Temperature, W_Humidity
const float BIAS = 0.969380438;                               // b
const float THRESHOLD = 0.5;                                  // same decision rule as Pipeline.predict()

// x_scaled = (x_raw - mu) / sigma
float standardize(float x_raw, int idx)
{
  return (x_raw - MEAN[idx]) / STD[idx];
}

// y = 1 / (1 + e^-z)
float sigmoid(float z)
{
  return 1.0 / (1.0 + exp(-z));
}

// z = sum_i W_i * x_scaled_i + b, then P(fire) = sigmoid(z)
float predict(float features[])
{
  float z = 0.0;
  for (int i = 0; i < N_FEATURES; i++)
  {
    z += WEIGHTS[i] * features[i];
  }
  z += BIAS;
  return sigmoid(z);
}

void setup()
{
  Serial.begin(9600);
  Serial.println(F("TP3 - Logistic Regression fire alarm (Arduino Uno + DHT22)"));
  dht.begin();
}

void loop()
{
  delay(2000);

  // 1. Read the DHT sensor (about 250 ms; the values may be up to 2 s old)
  float h = dht.readHumidity();
  float t = dht.readTemperature(); // Celsius

  // Check if any reads failed and exit early (to try again).
  if (isnan(h) || isnan(t))
  {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }

  // 2. Standardize, in the training order: Temperature, then Humidity
  float x_scaled[N_FEATURES];
  x_scaled[0] = standardize(t, 0);
  x_scaled[1] = standardize(h, 1);

  // 3. Predict P(fire)
  float y_pred = predict(x_scaled);

  // 4. Print the readings, the probability and the class
  Serial.print(F("Temperature: "));
  Serial.print(t);
  Serial.print(F(" C | Humidity: "));
  Serial.print(h);
  Serial.print(F(" % | Predicted Probability: "));
  Serial.print(y_pred, 4);
  if (y_pred >= THRESHOLD)
    Serial.println(F(" | Predicted Class: 1 (FIRE)"));
  else
    Serial.println(F(" | Predicted Class: 0 (normal)"));
}
