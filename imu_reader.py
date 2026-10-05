"""Minimal reader for the ESP32 IMU stream (see esp32_imu_simple/src/main.cpp).

Each line from the ESP32 holds 6 values per sensor: ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps.
Prints each sensor's readings plus roll/pitch estimated from the accelerometer. Ctrl+C to stop.
"""

import math

import serial

PORT = "COM4"     # the ESP32's COM port (Device Manager -> Ports)
BAUD = 115200
NUM_SENSORS = 1   # 1-3, must match how many sensors the ESP32 prints
VALUES_PER_SENSOR = 6


def read_sensors(line):
    """Return a list of (acc_g, gyro_dps) per sensor, or None if the line isn't a data line."""
    try:
        values = [float(v) for v in line.split(",")]
    except ValueError:
        return None
    if len(values) != NUM_SENSORS * VALUES_PER_SENSOR:
        return None
    sensors = []
    for i in range(0, len(values), VALUES_PER_SENSOR):
        sensors.append((values[i:i + 3], values[i + 3:i + 6]))
    return sensors


def tilt_deg(acc):
    """Roll and pitch in degrees from the gravity direction (only valid while not accelerating)."""
    ax, ay, az = acc
    roll = math.degrees(math.atan2(ay, az))
    pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az)))
    return roll, pitch


def main():
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = PORT, BAUD, 1
    # Keep DTR/RTS released so opening the port doesn't hold the ESP32 in reset.
    ser.dtr = False
    ser.rts = False
    ser.open()
    print(f"Reading {PORT}, Ctrl+C to stop")

    try:
        while True:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith("#"):
                print(line)
                continue
            sensors = read_sensors(line)
            if sensors is None:
                continue
            parts = []
            for n, (acc, gyro) in enumerate(sensors, start=1):
                roll, pitch = tilt_deg(acc)
                parts.append(f"S{n} acc {acc[0]:+.2f} {acc[1]:+.2f} {acc[2]:+.2f}  "
                             f"gyro {gyro[0]:+7.1f} {gyro[1]:+7.1f} {gyro[2]:+7.1f}  "
                             f"roll {roll:+6.1f} pitch {pitch:+6.1f}")
            print(" | ".join(parts))
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
