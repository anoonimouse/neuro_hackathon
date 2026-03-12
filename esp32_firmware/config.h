#pragma once

// WiFi credentials
static const char* WIFI_SSID = "YOUR_WIFI_SSID";
static const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// TCP server settings
static const int TCP_PORT = 5005;

// PCA9685 settings
static const uint8_t PCA9685_I2C_ADDRESS = 0x40;
static const uint16_t PCA9685_FREQ_HZ = 50;

// Servo channels
static const uint8_t CHANNEL_BASE = 0;
static const uint8_t CHANNEL_SHOULDER = 1;
static const uint8_t CHANNEL_ELBOW = 2;
static const uint8_t CHANNEL_GRIPPER = 3;

// Servo angle constraints
static const float BASE_MIN_DEG = 0.0f;
static const float BASE_MAX_DEG = 180.0f;
static const float SHOULDER_MIN_DEG = 15.0f;
static const float SHOULDER_MAX_DEG = 165.0f;
static const float ELBOW_MIN_DEG = 15.0f;
static const float ELBOW_MAX_DEG = 170.0f;
static const float GRIPPER_MIN_DEG = 0.0f;
static const float GRIPPER_MAX_DEG = 90.0f;

// PWM calibration range for common hobby servos at 50Hz
static const uint16_t SERVO_PULSE_MIN_US = 500;
static const uint16_t SERVO_PULSE_MAX_US = 2500;

// Motion smoothing
static const int MOTION_STEP_DELAY_MS = 15;
static const float MOTION_STEP_DEG = 1.5f;

// Home pose
static const float HOME_BASE_DEG = 90.0f;
static const float HOME_SHOULDER_DEG = 85.0f;
static const float HOME_ELBOW_DEG = 95.0f;
static const float HOME_GRIPPER_DEG = 40.0f;

// Gripper presets
static const float GRIPPER_OPEN_DEG = 40.0f;
static const float GRIPPER_CLOSED_DEG = 18.0f;
