#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"

const char* ssid = "Wokwi-GUEST";
const char* broker = "broker.hivemq.com";
#ifndef PATIENT_ID
// Change cet identifiant avant de compiler le firmware de chaque patient.
#define PATIENT_ID "patient1"
#endif
char TOPIC_RAW[96];
char TOPIC_BPM[96];
char TOPIC_MODE[96];
char TOPIC_PULSE[96];
char TOPIC_SPO2[96];
char TOPIC_TEMP[96];
char TOPIC_MOTION[96];
const int I2C_SDA = 21;
const int I2C_SCL = 22;
const int TEMP_PIN = 4;

WiFiClient espClient;
PubSubClient mqtt(espClient);
MAX30105 pulseSensor;
Adafruit_MPU6050 mpu;
OneWire oneWire(TEMP_PIN);
DallasTemperature skinTemperature(&oneWire);
bool pulseSensorReady = false;
bool mpuReady = false;
float pulseBpm = 0;
float oxygenPercent = -1;
uint32_t redSamples[100];
uint32_t irSamples[100];
int opticalSampleCount = 0;
unsigned long lastTemperatureRequest = 0;
bool temperatureConversionPending = false;

const int FS = 100;
const int PKT = 20;                 // 0,2 s par message
const int TBL = 200;
const float BRUIT = 0.0;            // bruit d'électrode (essayez 0.05 à 0.2)

float tblP[TBL];                    // battement avec onde P
float tblN[TBL];                    // battement sans onde P (fibrillation)
int buf[PKT];
int n = 0;
float phase = 0;
float rr = 0.8;                     // durée du battement en cours (s)
bool avecP = true;
int compteur = 0;
float bpmBase = 75;
unsigned long lastSample = 0, lastSec = 0;

bool faMode = false, extraMode = false;   // modes activés par les boutons
bool lastB1 = HIGH, lastB2 = HIGH;
unsigned long t1 = 0, t2 = 0;             // anti-rebond

void configureTopics() {
  snprintf(TOPIC_RAW, sizeof(TOPIC_RAW), "ecg-doua/%s/raw", PATIENT_ID);
  snprintf(TOPIC_BPM, sizeof(TOPIC_BPM), "ecg-doua/%s/bpm", PATIENT_ID);
  snprintf(TOPIC_MODE, sizeof(TOPIC_MODE), "ecg-doua/%s/mode", PATIENT_ID);
  snprintf(TOPIC_PULSE, sizeof(TOPIC_PULSE), "ecg-doua/%s/pulse", PATIENT_ID);
  snprintf(TOPIC_SPO2, sizeof(TOPIC_SPO2), "ecg-doua/%s/spo2", PATIENT_ID);
  snprintf(TOPIC_TEMP, sizeof(TOPIC_TEMP), "ecg-doua/%s/temperature", PATIENT_ID);
  snprintf(TOPIC_MOTION, sizeof(TOPIC_MOTION), "ecg-doua/%s/motion", PATIENT_ID);
}

float g(float x, float c, float w, float a) {
  return a * exp(-((x - c) * (x - c)) / (2 * w * w));
}

float qrst(float p) {
  return g(p, 0.37, 0.010, -0.12) + g(p, 0.40, 0.012, 1.00) +
         g(p, 0.43, 0.010, -0.25) + g(p, 0.65, 0.045, 0.30);
}

float prochainRR(bool fa, bool extra) {
  float base = 60.0 / bpmBase;
  if (fa) return base * (0.5 + random(0, 101) / 100.0);   // 0,5 à 1,5 x base
  if (extra) {
    int k = compteur % 5;
    if (k == 4) return base * 0.55;                       // battement prématuré
    if (k == 0 && compteur > 0) return base * 1.45;       // pause compensatrice
  }
  return base;
}

void reconnect() {
  while (!mqtt.connected()) {
    String id = "esp32-ecg-doua-" + String(random(100000));
    if (mqtt.connect(id.c_str())) Serial.println("MQTT OK");
    else delay(2000);
  }
}

void readPulseSensor() {
  if (!pulseSensorReady) return;

  pulseSensor.check();
  while (pulseSensor.available()) {
    uint32_t ir = pulseSensor.getIR();
    uint32_t red = pulseSensor.getRed();
    pulseSensor.nextSample();
    redSamples[opticalSampleCount] = red;
    irSamples[opticalSampleCount] = ir;
    opticalSampleCount++;
    if (opticalSampleCount == 100) {
      int32_t calculatedSpo2, calculatedHeartRate;
      int8_t validSpo2, validHeartRate;
      maxim_heart_rate_and_oxygen_saturation(irSamples, 100, redSamples,
                                             &calculatedSpo2, &validSpo2,
                                             &calculatedHeartRate, &validHeartRate);
      pulseBpm = validHeartRate ? calculatedHeartRate : -1;
      oxygenPercent = validSpo2 ? calculatedSpo2 : -1;
      for (int i = 25; i < 100; i++) {
        redSamples[i - 25] = redSamples[i];
        irSamples[i - 25] = irSamples[i];
      }
      opticalSampleCount = 75;
    }
  }
}

void publishAdditionalSensors() {
  if (pulseSensorReady) {
    mqtt.publish(TOPIC_PULSE, String(pulseBpm, 1).c_str());
    mqtt.publish(TOPIC_SPO2, String(oxygenPercent, 0).c_str());
  } else {
    mqtt.publish(TOPIC_PULSE, "-1");
    mqtt.publish(TOPIC_SPO2, "-1");
  }

  if (mpuReady) {
    sensors_event_t acceleration, gyro, sensorTemperature;
    mpu.getEvent(&acceleration, &gyro, &sensorTemperature);
    float ax = acceleration.acceleration.x;
    float ay = acceleration.acceleration.y;
    float az = acceleration.acceleration.z;
    float magnitude = sqrt(ax * ax + ay * ay + az * az);
    bool moving = magnitude < 7.0f || magnitude > 12.5f ||
                  fabs(gyro.gyro.x) > 1.0f || fabs(gyro.gyro.y) > 1.0f || fabs(gyro.gyro.z) > 1.0f;
    char payload[112];
    snprintf(payload, sizeof(payload), "{\"ax\":%.2f,\"ay\":%.2f,\"az\":%.2f,\"motion\":%s}",
             ax, ay, az, moving ? "true" : "false");
    mqtt.publish(TOPIC_MOTION, payload);
  }

  unsigned long now = millis();
  if (temperatureConversionPending && now - lastTemperatureRequest >= 100) {
    float tempC = skinTemperature.getTempCByIndex(0);
    if (tempC != DEVICE_DISCONNECTED_C) {
      mqtt.publish(TOPIC_TEMP, String(tempC, 2).c_str());
    }
    temperatureConversionPending = false;
  }
  if (!temperatureConversionPending && now - lastTemperatureRequest >= 1000) {
    skinTemperature.requestTemperatures();
    lastTemperatureRequest = now;
    temperatureConversionPending = true;
  }
}

void setup() {
  Serial.begin(115200);
  configureTopics();
  Serial.println("Version anti-rebond 500ms");
  Serial.printf("Patient MQTT : %s\n", PATIENT_ID);
  Wire.begin(I2C_SDA, I2C_SCL);
  pulseSensorReady = pulseSensor.begin(Wire, I2C_SPEED_FAST);
  if (pulseSensorReady) {
    pulseSensor.setup(60, 4, 2, 100, 411, 4096);
    pulseSensor.setPulseAmplitudeRed(0x2A);
    pulseSensor.setPulseAmplitudeIR(0x2A);
    pulseSensor.setPulseAmplitudeGreen(0);
    Serial.println("MAX30102 OK");
  } else {
    Serial.println("MAX30102 absent (non simule dans Wokwi)");
  }
  mpuReady = mpu.begin(0x68, &Wire);
  if (mpuReady) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("MPU6050 OK");
  } else {
    Serial.println("MPU6050 absent");
  }
  skinTemperature.begin();
  skinTemperature.setResolution(9);
  skinTemperature.setWaitForConversion(false);
  pinMode(13, INPUT_PULLUP);          // bouton rouge : fibrillation (bascule)
  pinMode(12, INPUT_PULLUP);          // bouton bleu : extrasystole (bascule)
  pinMode(2, OUTPUT);

  for (int i = 0; i < TBL; i++) {
    float p = (float)i / TBL;
    tblN[i] = qrst(p);
    tblP[i] = qrst(p) + g(p, 0.20, 0.025, 0.15);
  }

  WiFi.begin(ssid, "", 6);
  while (WiFi.status() != WL_CONNECTED) delay(200);
  Serial.println("Wi-Fi OK");
  mqtt.setServer(broker, 1883);
  mqtt.setBufferSize(1024);
}

void loop() {
  if (!mqtt.connected()) reconnect();
  mqtt.loop();
  readPulseSensor();

  // Boutons à bascule : un seul changement par 500 ms maximum
  bool b1 = digitalRead(13), b2 = digitalRead(12);
  if (b1 == LOW && lastB1 == HIGH && millis() - t1 > 500) {
    t1 = millis();
    faMode = !faMode;
    if (faMode) extraMode = false;
    Serial.println(faMode ? "Mode : FIBRILLATION" : "Mode : NORMAL");
  }
  if (b2 == LOW && lastB2 == HIGH && millis() - t2 > 500) {
    t2 = millis();
    extraMode = !extraMode;
    if (extraMode) faMode = false;
    Serial.println(extraMode ? "Mode : EXTRASYSTOLE" : "Mode : NORMAL");
  }
  lastB1 = b1;
  lastB2 = b2;
  bool fa = faMode;
  bool extra = extraMode;

  if (millis() - lastSample >= 1000 / FS) {
    lastSample = millis();
    phase += 1.0 / (rr * FS);
    if (phase >= 1) {                 // nouveau battement
      phase -= 1;
      compteur++;
      rr = prochainRR(fa, extra);
      avecP = !fa;
    }
    float v = (avecP ? tblP : tblN)[(int)(phase * TBL)];
    if (fa) v += random(-25, 26) / 1000.0;
    if (BRUIT > 0) v += BRUIT * random(-100, 101) / 100.0;
    buf[n++] = (int)(v * 1000);

    if (n == PKT) {
      String s = "";
      for (int i = 0; i < PKT; i++) { s += buf[i]; if (i < PKT - 1) s += ","; }
      if (!mqtt.publish(TOPIC_RAW, s.c_str())) Serial.println("ECHEC raw");
      n = 0;
    }
  }

  if (millis() - lastSec >= 1000) {
    lastSec = millis();
    bpmBase = map(analogRead(34), 0, 4095, 40, 180);
    mqtt.publish(TOPIC_BPM, String(bpmBase, 0).c_str());
    mqtt.publish(TOPIC_MODE, fa ? "fa" : (extra ? "extrasystole" : "normal"));
    publishAdditionalSensors();
    digitalWrite(2, (fa || extra || bpmBase > 120 || bpmBase < 50) ? HIGH : LOW);
  }
}