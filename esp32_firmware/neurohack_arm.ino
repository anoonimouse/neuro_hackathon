#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_PWMServoDriver.h>
#include "config.h"

WiFiServer server(TCP_PORT);
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(PCA9685_I2C_ADDRESS);

struct ArmPose {
  float base;
  float shoulder;
  float elbow;
  float gripper;
};

ArmPose currentPose = {
  HOME_BASE_DEG,
  HOME_SHOULDER_DEG,
  HOME_ELBOW_DEG,
  HOME_GRIPPER_DEG
};

float clampFloat(float value, float minimum, float maximum) {
  if (value < minimum) return minimum;
  if (value > maximum) return maximum;
  return value;
}

uint16_t pulseUsToTicks(uint16_t pulseUs) {
  // 50Hz period = 20000us; PCA9685 has 4096 ticks
  return static_cast<uint16_t>((static_cast<float>(pulseUs) / 20000.0f) * 4096.0f);
}

uint16_t angleToPulseUs(float angleDeg, float minDeg, float maxDeg) {
  float clamped = clampFloat(angleDeg, minDeg, maxDeg);
  float ratio = (clamped - minDeg) / (maxDeg - minDeg);
  return static_cast<uint16_t>(SERVO_PULSE_MIN_US + ratio * (SERVO_PULSE_MAX_US - SERVO_PULSE_MIN_US));
}

void writeServo(uint8_t channel, float angleDeg, float minDeg, float maxDeg) {
  uint16_t pulseUs = angleToPulseUs(angleDeg, minDeg, maxDeg);
  uint16_t ticks = pulseUsToTicks(pulseUs);
  pca9685.setPWM(channel, 0, ticks);
}

void applyPose(const ArmPose& pose) {
  writeServo(CHANNEL_BASE, pose.base, BASE_MIN_DEG, BASE_MAX_DEG);
  writeServo(CHANNEL_SHOULDER, pose.shoulder, SHOULDER_MIN_DEG, SHOULDER_MAX_DEG);
  writeServo(CHANNEL_ELBOW, pose.elbow, ELBOW_MIN_DEG, ELBOW_MAX_DEG);
  writeServo(CHANNEL_GRIPPER, pose.gripper, GRIPPER_MIN_DEG, GRIPPER_MAX_DEG);
}

float stepToward(float current, float target, float step) {
  if (fabs(target - current) <= step) {
    return target;
  }
  return current + ((target > current) ? step : -step);
}

void moveSmooth(const ArmPose& targetPose) {
  ArmPose target = {
    clampFloat(targetPose.base, BASE_MIN_DEG, BASE_MAX_DEG),
    clampFloat(targetPose.shoulder, SHOULDER_MIN_DEG, SHOULDER_MAX_DEG),
    clampFloat(targetPose.elbow, ELBOW_MIN_DEG, ELBOW_MAX_DEG),
    clampFloat(targetPose.gripper, GRIPPER_MIN_DEG, GRIPPER_MAX_DEG)
  };

  while (true) {
    bool done = true;

    float nextBase = stepToward(currentPose.base, target.base, MOTION_STEP_DEG);
    float nextShoulder = stepToward(currentPose.shoulder, target.shoulder, MOTION_STEP_DEG);
    float nextElbow = stepToward(currentPose.elbow, target.elbow, MOTION_STEP_DEG);
    float nextGripper = stepToward(currentPose.gripper, target.gripper, MOTION_STEP_DEG);

    if (nextBase != currentPose.base || nextShoulder != currentPose.shoulder || nextElbow != currentPose.elbow || nextGripper != currentPose.gripper) {
      done = false;
      currentPose.base = nextBase;
      currentPose.shoulder = nextShoulder;
      currentPose.elbow = nextElbow;
      currentPose.gripper = nextGripper;
      applyPose(currentPose);
      delay(MOTION_STEP_DELAY_MS);
    }

    if (done) {
      break;
    }
  }
}

String readLine(WiFiClient& client) {
  String line;
  while (client.connected()) {
    while (client.available()) {
      char c = static_cast<char>(client.read());
      if (c == '\n') {
        line.trim();
        return line;
      }
      line += c;
      if (line.length() > 1024) {
        line = "";
        return line;
      }
    }
    delay(1);
  }
  line.trim();
  return line;
}

bool parsePose(JsonObject poseObj, ArmPose& outPose) {
  if (!poseObj.containsKey("base") || !poseObj.containsKey("shoulder") || !poseObj.containsKey("elbow") || !poseObj.containsKey("gripper")) {
    return false;
  }
  outPose.base = poseObj["base"].as<float>();
  outPose.shoulder = poseObj["shoulder"].as<float>();
  outPose.elbow = poseObj["elbow"].as<float>();
  outPose.gripper = poseObj["gripper"].as<float>();
  return true;
}

void sendAck(WiFiClient& client, const String& status, const String& message) {
  StaticJsonDocument<256> response;
  response["status"] = status;
  response["message"] = message;
  serializeJson(response, client);
  client.println();
}

void handleJsonCommand(WiFiClient& client, const String& payload) {
  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    sendAck(client, "error", "invalid_json");
    return;
  }

  const char* command = doc["command"] | "";

  if (strcmp(command, "SET_ANGLES") == 0) {
    ArmPose pose;
    JsonObject angles = doc["angles"].as<JsonObject>();
    if (!parsePose(angles, pose)) {
      sendAck(client, "error", "missing_angles");
      return;
    }
    moveSmooth(pose);
    sendAck(client, "ok", "set_angles_applied");
    return;
  }

  if (strcmp(command, "OPEN") == 0) {
    ArmPose pose = currentPose;
    pose.gripper = GRIPPER_OPEN_DEG;
    moveSmooth(pose);
    sendAck(client, "ok", "gripper_opened");
    return;
  }

  if (strcmp(command, "CLOSE") == 0) {
    ArmPose pose = currentPose;
    pose.gripper = GRIPPER_CLOSED_DEG;
    moveSmooth(pose);
    sendAck(client, "ok", "gripper_closed");
    return;
  }

  if (strcmp(command, "HOME") == 0) {
    ArmPose pose = {HOME_BASE_DEG, HOME_SHOULDER_DEG, HOME_ELBOW_DEG, GRIPPER_OPEN_DEG};
    moveSmooth(pose);
    sendAck(client, "ok", "home_reached");
    return;
  }

  if (strcmp(command, "PING") == 0) {
    sendAck(client, "ok", "pong");
    return;
  }

  sendAck(client, "error", "unknown_command");
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin();
  pca9685.begin();
  pca9685.setPWMFreq(PCA9685_FREQ_HZ);

  applyPose(currentPose);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("[WIFI] Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("[WIFI] Connected. IP: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.printf("[TCP] Listening on port %d\n", TCP_PORT);
}

void loop() {
  WiFiClient client = server.available();
  if (!client) {
    delay(5);
    return;
  }

  Serial.println("[TCP] Client connected");
  while (client.connected()) {
    String line = readLine(client);
    if (line.length() == 0) {
      if (!client.connected()) {
        break;
      }
      continue;
    }
    handleJsonCommand(client, line);
  }
  client.stop();
  Serial.println("[TCP] Client disconnected");
}
