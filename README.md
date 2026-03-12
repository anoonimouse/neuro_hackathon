# NeuroHack Setup and Config Guide

## Project Structure

- `configs/` contains all runtime configuration JSON files.
- `esp32_firmware/` contains ESP32 + PCA9685 firmware.
- `dashboard.py` is the integrated live demo UI.
- `start_demo.bat` and `start_demo.ps1` are one-click launch scripts.

## Config Files You Need to Tune

### configs/signal_config.json

Use this file for NPG Lite signal acquisition and command detection.

- `serial.port`: Set your NPG Lite COM port (for example, `COM3`).
- `serial.baudrate`: Keep at `115200` unless your firmware uses another value.
- `signal.sample_rate_hz`: Match your true incoming sample rate.
- `signal.emg_channel_index`: EMG channel index in incoming packet.
- `signal.eog_channel_index`: EOG channel index in incoming packet.
- `emg.*`: Tune EMG detection behavior (bandpass, thresholds, short/long flex timing).
- `eog.*`: Tune EOG detection behavior (bandpass, polarity hold, cooldown).

### configs/vision_config.json

Use this file for single-camera object detection and tabletop coordinate conversion.

- `camera.index`: Laptop webcam index (`0` usually works).
- `camera.width`, `camera.height`, `camera.fps`: Camera capture settings.
- `yolo.model_path`: YOLO model file path (currently `yolov8n.pt`).
- `yolo.confidence_threshold`: Raise to reduce false detections.
- `yolo.target_classes`: Keep only classes relevant to your objects.
- `calibration.file_path`: Points to `configs/calibration_data.json`.
- `reach_zone_cm.*`: Defines valid object area on table in arm coordinates.
- `selection.stabilization_seconds`: Stability requirement before target locks.

### configs/calibration_data.json

Stores camera calibration output used by vision conversion.

- `origin_u`, `origin_v`: Arm base origin in image pixels.
- `pixels_per_cm_x`, `pixels_per_cm_y`: Pixel-to-centimeter scale.
- `table_z_cm`: Fixed table height reference.

This file is updated by calibration mode.

### configs/motion_config.json

Use this file for arm geometry, servo limits, and planner behavior.

- `geometry.link1_cm`, `geometry.link2_cm`: Physical arm link lengths.
- `geometry.base_height_cm`: Height of shoulder joint over table.
- `planner.look_step_deg`: Base step size for LOOK_LEFT/LOOK_RIGHT.
- `poses.home`: Home joint angles.
- `poses.safe`: Safe fallback joint angles.
- `poses.gripper_open`, `poses.gripper_closed`: Gripper preset angles.
- `servo.*.min/max`: Real servo limits to prevent mechanical stress.
- `motion.step_delay_ms`: Delay between planned motion steps.

### configs/arm_network_config.json

Use this file for laptop-to-ESP32 command transport.

- `esp32.host`: ESP32 IP address on your WiFi network.
- `esp32.port`: TCP server port (default `5005`).
- `esp32.connect_timeout_seconds`: Socket connect timeout.
- `esp32.response_timeout_seconds`: Read timeout for command acknowledgments.
- `motion.planner_config_path`: Points to `configs/motion_config.json`.

### configs/dashboard_config.json

Use this file for dashboard refresh and worker toggles.

- `ui.refresh_interval_ms`: Dashboard rerender interval.
- `ui.max_signal_points`: Length of live EMG/EOG plot buffers.
- `ui.max_command_history`: Number of command events retained.
- `runtime.enable_signal_worker`: Enable/disable signal acquisition thread.
- `runtime.enable_vision_worker`: Enable/disable vision thread.
- `runtime.enable_arm_worker`: Enable/disable arm execution thread.

## Demo-Day Quick Checklist

1. Update WiFi credentials in `esp32_firmware/config.h`.
2. Flash firmware in `esp32_firmware/neurohack_arm.ino`.
3. Set `configs/arm_network_config.json -> esp32.host` to your board IP.
4. Set `configs/signal_config.json -> serial.port` to NPG Lite COM port.
5. Run camera calibration once:
   - `python vision_pipeline.py --mode calibrate`
6. Start full dashboard:
   - `python -m streamlit run dashboard.py`

## Optional One-Click Start

- Command Prompt:
  - `start_demo.bat`
  - `start_demo.bat calibrate`
- PowerShell:
  - `./start_demo.ps1`
  - `./start_demo.ps1 calibrate`
