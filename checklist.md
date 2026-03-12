# Project Checklist

## Current Progress
- [x] Created external configuration file for all tunable settings in `signal_config.json`.
- [x] Implemented serial acquisition with reconnect handling in `signal_processing.py`.
- [x] Implemented CSV/TSV/whitespace parser for incoming ADC stream lines.
- [x] Implemented EMG pipeline: bandpass, rectify, envelope, adaptive threshold, GRAB/RELEASE detection.
- [x] Implemented EOG pipeline: bandpass, polarity thresholds, LOOK_LEFT/LOOK_RIGHT detection.
- [x] Added runtime dependencies list in `requirements.txt`.
- [x] Added single-camera vision configuration in `vision_config.json`.
- [x] Added calibration storage in `calibration_data.json`.
- [x] Implemented laptop-camera YOLOv8n pipeline in `vision_pipeline.py`.
- [x] Implemented one-time calibration mode for arm-base origin and pixel-to-cm scale.
- [x] Implemented table coordinate conversion to (X, Y, Z) using calibration values.
- [x] Implemented nearest object selection with 0.5s stabilization.
- [x] Added motion planner configuration in `motion_config.json`.
- [x] Implemented 4-DOF closed-form IK solver in `motion_planner.py`.
- [x] Implemented command mapping for GRAB, RELEASE, LOOK_LEFT, and LOOK_RIGHT.
- [x] Implemented out-of-reach fallback to safe pose.
- [x] Implemented configurable home/safe/gripper poses and servo limit clamping.
- [x] Added CLI and interactive runner for motion planning tests.
- [x] Added ESP32 firmware with WiFi TCP server and JSON command handling.
- [x] Added PCA9685 servo control with smooth per-joint stepping.
- [x] Added firmware commands: SET_ANGLES, OPEN, CLOSE, HOME, PING.
- [x] Added laptop-side network config in `arm_network_config.json`.
- [x] Added laptop command runner in `arm_commander.py` to execute motion plans on ESP32.
- [x] Added dashboard runtime config in `dashboard_config.json`.
- [x] Added integrated Streamlit dashboard in `dashboard.py`.
- [x] Added live EMG/EOG plots and biosignal command history panel.
- [x] Added live single-camera vision panel with stable target telemetry.
- [x] Added integrated arm execution worker with queue-based command dispatch.
- [x] Added manual simulation buttons for GRAB/RELEASE/LOOK_LEFT/LOOK_RIGHT/OPEN/HOME.
- [x] Added one-click Windows launcher script `start_demo.bat`.
- [x] Added one-click PowerShell launcher script `start_demo.ps1`.
- [x] Added optional calibration launch mode before dashboard startup.
- [x] Centralized all runtime JSON configs under `configs/` and updated all code defaults.
- [x] Added README documentation for all centralized config files and demo startup steps.
- [ ] Validate with live NPG Lite stream and tune thresholds from real data.
- [ ] Validate live camera calibration and tune reach-zone limits for the physical table.
- [ ] Validate IK output against real arm geometry and tune servo offsets.
- [ ] Flash ESP32 firmware and verify TCP command round-trip on local WiFi.

## Hardware and Setup
- Assemble the 4-DOF arm (base, shoulder, elbow, gripper) and mount it on the demo table with a fixed orientation.
- Measure and record link lengths for base-to-shoulder, shoulder-to-elbow, and elbow-to-gripper.
- Mount the laptop webcam on a stable stand facing the table reach zone.
- Mark a rectangular reach zone on the table and ensure objects are placed within it.
- Wire PCA9685 to ESP32 and connect all servos with stable power.

## NPG Lite Serial Acquisition
- Connect NPG Lite via USB-C and confirm serial streaming on the laptop.
- Implement a Python serial reader for Channel A0 (EMG) and A1 (EOG).
- Add reconnect handling to recover from cable disconnects.

## EMG Processing and Commands
- Implement EMG bandpass filter (20–450 Hz).
- Rectify EMG and compute a 50 ms moving average envelope.
- Record a 5-second resting baseline and compute an adaptive threshold.
- Detect short flex (>50 ms and <800 ms) and emit GRAB.
- Detect long flex (>=800 ms) and emit RELEASE.
- Add a 300 ms refractory window after EMG commands.

## EOG Processing and Commands
- Implement EOG bandpass filter (0.5–10 Hz).
- Detect positive deflection (>100 ms) and emit LOOK_RIGHT.
- Detect negative deflection (>100 ms) and emit LOOK_LEFT.
- Add a 400 ms cooldown between EOG commands.

## Vision Pipeline (Single Laptop Camera)
- Run YOLOv8n on the laptop webcam at ~30 fps.
- Filter detections to common tabletop objects.
- Compute bounding box center (u, v) for each detection.
- Perform camera calibration to derive pixel-to-cm scale at table height.
- Convert (u, v) to table-relative (X, Y) and set Z to fixed table height.

## Object Selection
- Select the nearest detected object within the reach zone using (X, Y) distance.
- Keep the current target stable for at least 0.5 seconds to prevent jitter.

## Inverse Kinematics (4-DOF)
- Compute base angle = atan2(Y, X).
- Compute reach r = sqrt(X^2 + Y^2).
- Solve shoulder and elbow angles with planar two-link IK.
- Clamp all joint angles to servo limits.
- Fallback to a safe pose when the target is out of reach.

## Motion Planner
- GRAB: move to target, close gripper.
- RELEASE: open gripper, return to home pose.
- LOOK_LEFT: decrement base by 15 degrees.
- LOOK_RIGHT: increment base by 15 degrees.

## ESP32 Firmware
- Implement a TCP listener to receive joint angles.
- Convert angles to PCA9685 PWM values per joint.
- Apply per-joint speed limiting for smooth motion.
- Implement HOME, OPEN, and CLOSE commands.

## Integration
- Combine signal processing, command dispatcher, vision, IK, and TCP in one Python app.
- Queue commands and execute sequentially.
- Log every command, target coordinate, IK solution, and servo message.

## Dashboard (Streamlit)
- Plot live EMG waveform with command markers.
- Plot live EOG waveform with command markers.
- Display camera feed with YOLO bounding boxes.
- Display current arm state, target (X, Y, Z), and joint angles.

## Demo Rehearsal
- Run the full sequence: look left/right, short flex to grab, long flex to release.
- Repeat at least 10 times and adjust thresholds if needed.
- Enable a keyboard-triggered signal simulation mode for backup.
