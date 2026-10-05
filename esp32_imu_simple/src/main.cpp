// MPU-6050 (GY-521) reader for ESP32.
//
// Wiring: VCC -> 3V3, GND -> GND, SDA -> GPIO21, SCL -> GPIO22, AD0 -> GND or open (address 0x68).
//
// Prints one CSV line per sample over USB serial (115200 baud) at 100 Hz:
//   ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps

#include <Arduino.h>
#include <Wire.h>

constexpr uint8_t MPU_ADDR = 0x68;
constexpr float ACCEL_LSB_PER_G = 16384.0f;  // default range +-2 g
constexpr float GYRO_LSB_PER_DPS = 131.0f;   // default range +-250 deg/s

void wakeSensor() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1
  Wire.write(0x00);  // clear the sleep bit
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  delay(100);  // give the sensor time to power up
  wakeSensor();
}

void loop() {
  // Read 14 bytes: accel x/y/z, temperature, gyro x/y/z
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14) {
    Serial.println("# sensor not responding");
    delay(1000);
    return;
  }
  int16_t raw[7];
  for (int i = 0; i < 7; i++) raw[i] = (Wire.read() << 8) | Wire.read();

  // A sensor that lost power restarts asleep and reads all zeros; gravity never reads 0 on every axis.
  if (raw[0] == 0 && raw[1] == 0 && raw[2] == 0) {
    Serial.println("# sensor asleep (power glitch?), waking it");
    wakeSensor();
    delay(100);
    return;
  }

  Serial.printf("%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
                raw[0] / ACCEL_LSB_PER_G, raw[1] / ACCEL_LSB_PER_G, raw[2] / ACCEL_LSB_PER_G,
                raw[4] / GYRO_LSB_PER_DPS, raw[5] / GYRO_LSB_PER_DPS, raw[6] / GYRO_LSB_PER_DPS);
  delay(10);
}
