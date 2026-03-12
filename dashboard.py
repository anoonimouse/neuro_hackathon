from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None

try:
    import serial
except ModuleNotFoundError:
    serial = None

from arm_commander import execute_plan, load_network_config, send_command
from signal_processing import EmgCommandDetector, EogCommandDetector, load_config as load_signal_config
from vision_pipeline import (
    TargetStabilizer,
    in_reach_zone,
    load_calibration,
    load_config as load_vision_config,
    pixel_to_table_cm,
    quantized_key,
)


@dataclass
class DashboardConfig:
    refresh_interval_ms: int
    max_signal_points: int
    max_command_history: int
    max_log_entries: int
    enable_signal_worker: bool
    enable_vision_worker: bool
    enable_arm_worker: bool


class SharedState:
    def __init__(self, max_signal_points: int, max_command_history: int, max_log_entries: int) -> None:
        self.lock = threading.Lock()
        self.emg_raw = deque(maxlen=max_signal_points)
        self.emg_env = deque(maxlen=max_signal_points)
        self.eog_raw = deque(maxlen=max_signal_points)
        self.eog_filt = deque(maxlen=max_signal_points)
        self.signal_ts = deque(maxlen=max_signal_points)

        self.command_history = deque(maxlen=max_command_history)
        self.system_logs = deque(maxlen=max_log_entries)

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_fps: float = 0.0
        self.stable_target: Optional[dict[str, float | str]] = None
        self.vision_candidates_count: int = 0

        self.arm_state: str = "IDLE"
        self.last_arm_result: dict[str, Any] = {}

    def log(self, text: str) -> None:
        with self.lock:
            self.system_logs.appendleft(f"{time.strftime('%H:%M:%S')} {text}")


class SignalWorker(threading.Thread):
    def __init__(self, shared: SharedState, stop_event: threading.Event, command_queue: queue.Queue[str]) -> None:
        super().__init__(daemon=True)
        self.shared = shared
        self.stop_event = stop_event
        self.command_queue = command_queue
        self.config = load_signal_config(Path("configs/signal_config.json"))

        self.emg_detector = EmgCommandDetector(self.config, on_command=self._on_command)
        self.eog_detector = EogCommandDetector(self.config, on_command=self._on_command)

    def _on_command(self, command: str, timestamp: float) -> None:
        event = {
            "timestamp": time.strftime("%H:%M:%S", time.localtime(timestamp)),
            "source": "biosignal",
            "command": command,
        }
        with self.shared.lock:
            self.shared.command_history.appendleft(event)
        self.command_queue.put(command)
        self.shared.log(f"Command detected: {command}")

    def _parse_line(self, line_bytes: bytes) -> Optional[list[float]]:
        line = line_bytes.decode("utf-8", errors="ignore").strip()
        if not line:
            return None

        separator_mode = self.config.serial.separator_mode.lower()
        if separator_mode == "comma":
            parts = line.split(",")
        elif separator_mode == "tab":
            parts = line.split("\t")
        else:
            if "\t" in line:
                parts = line.split("\t")
            elif "," in line:
                parts = line.split(",")
            else:
                parts = line.split()

        tokens = [segment.strip() for segment in parts if segment.strip()]
        if not tokens:
            return None

        values: list[float] = []
        for token in tokens:
            value = float(token)
            if np.isnan(value) or np.isinf(value):
                return None
            values.append(value)
        return values

    def run(self) -> None:
        if serial is None:
            self.shared.log("Signal worker unavailable: pyserial is not installed in active Python environment")
            return

        port = self.config.serial.port
        baud = self.config.serial.baudrate
        reconnect_delay = self.config.serial.reconnect_delay_seconds

        self.shared.log(f"Signal worker started on {port} @ {baud}")

        while not self.stop_event.is_set():
            ser: Optional[serial.Serial] = None
            try:
                ser = serial.Serial(port, baud, timeout=self.config.serial.timeout_seconds)
                self.shared.log("Signal serial connected")

                while not self.stop_event.is_set():
                    line = ser.readline()
                    if not line:
                        continue
                    parsed = self._parse_line(line)
                    if parsed is None:
                        continue

                    emg_idx = self.config.signal.emg_channel_index
                    eog_idx = self.config.signal.eog_channel_index
                    if len(parsed) <= max(emg_idx, eog_idx):
                        continue

                    now = time.time()
                    emg_raw = parsed[emg_idx]
                    eog_raw = parsed[eog_idx]
                    emg_env = self.emg_detector.process(emg_raw, now)
                    eog_filt = self.eog_detector.process(eog_raw, now)

                    with self.shared.lock:
                        self.shared.signal_ts.append(now)
                        self.shared.emg_raw.append(emg_raw)
                        self.shared.emg_env.append(emg_env)
                        self.shared.eog_raw.append(eog_raw)
                        self.shared.eog_filt.append(eog_filt)

            except (serial.SerialException, ValueError) as exc:
                self.shared.log(f"Signal worker reconnecting after error: {exc}")
                time.sleep(reconnect_delay)
            finally:
                if ser is not None and ser.is_open:
                    ser.close()

        self.shared.log("Signal worker stopped")


class VisionWorker(threading.Thread):
    def __init__(self, shared: SharedState, stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.shared = shared
        self.stop_event = stop_event
        self.config = load_vision_config(Path("configs/vision_config.json"))
        self.calibration = load_calibration(self.config)
        self.stabilizer = TargetStabilizer(
            stabilization_seconds=self.config.selection.stabilization_seconds,
            max_missing_seconds=self.config.selection.max_missing_seconds,
        )
        self.model = YOLO(self.config.yolo.model_path) if YOLO is not None else None

    def run(self) -> None:
        if YOLO is None or self.model is None:
            self.shared.log("Vision worker unavailable: ultralytics is not installed in active Python environment")
            return

        cap = cv2.VideoCapture(self.config.camera.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)

        if not cap.isOpened():
            self.shared.log("Vision worker failed: camera open error")
            return

        self.shared.log("Vision worker started")
        frame_count = 0
        last_fps_ts = time.time()

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            result = self.model.predict(
                source=frame,
                conf=self.config.yolo.confidence_threshold,
                iou=self.config.yolo.iou_threshold,
                max_det=self.config.yolo.max_detections,
                verbose=False,
            )[0]

            names = result.names
            candidates = []
            boxes = result.boxes

            if boxes is not None and boxes.xyxy is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    class_name = names.get(cls_id, str(cls_id))
                    if class_name not in self.config.yolo.target_classes:
                        continue

                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    u = 0.5 * (x1 + x2)
                    v = 0.5 * (y1 + y2)
                    x_cm, y_cm, z_cm = pixel_to_table_cm(u, v, self.calibration)
                    if not in_reach_zone(x_cm, y_cm, self.config.reach_zone_cm):
                        continue

                    distance_cm = float(np.hypot(x_cm, y_cm))
                    key = quantized_key(class_name, u, v, self.config.selection.position_quantization_px)
                    candidates.append(
                        {
                            "key": key,
                            "class_name": class_name,
                            "confidence": conf,
                            "x_cm": x_cm,
                            "y_cm": y_cm,
                            "z_cm": z_cm,
                            "distance_cm": distance_cm,
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                        }
                    )

            candidates.sort(key=lambda item: item["distance_cm"])
            current_candidate = candidates[0] if candidates else None
            stable = self.stabilizer.update(_candidate_to_obj(current_candidate), now)

            for obj in candidates:
                x1, y1, x2, y2 = obj["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                text = f"{obj['class_name']} {obj['confidence']:.2f} X={obj['x_cm']:.1f} Y={obj['y_cm']:.1f}"
                cv2.putText(frame, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            stable_payload = None
            if stable is not None:
                stable_payload = {
                    "class_name": stable.class_name,
                    "x_cm": float(stable.x_cm),
                    "y_cm": float(stable.y_cm),
                    "z_cm": float(stable.z_cm),
                    "distance_cm": float(stable.distance_cm),
                }
                if candidates:
                    sx1, sy1, sx2, sy2 = candidates[0]["box"]
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)
                cv2.putText(
                    frame,
                    f"TARGET {stable.class_name} X={stable.x_cm:.1f} Y={stable.y_cm:.1f} Z={stable.z_cm:.1f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            frame_count += 1
            elapsed = now - last_fps_ts
            fps = 0.0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_ts = now

            with self.shared.lock:
                self.shared.latest_frame = frame
                self.shared.stable_target = stable_payload
                self.shared.vision_candidates_count = len(candidates)
                if fps > 0:
                    self.shared.latest_fps = fps

        cap.release()
        self.shared.log("Vision worker stopped")


class ArmWorker(threading.Thread):
    def __init__(self, shared: SharedState, stop_event: threading.Event, command_queue: queue.Queue[str]) -> None:
        super().__init__(daemon=True)
        self.shared = shared
        self.stop_event = stop_event
        self.command_queue = command_queue
        self.network_config = load_network_config(Path("configs/arm_network_config.json"))

    def run(self) -> None:
        self.shared.log("Arm worker started")
        direct_commands = {"OPEN", "CLOSE", "HOME", "PING"}
        while not self.stop_event.is_set():
            try:
                command = self.command_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self.shared.lock:
                stable_target = self.shared.stable_target
                self.shared.arm_state = f"EXECUTING {command}"

            try:
                if command in direct_commands:
                    response = send_command(self.network_config.esp32, {"command": command})
                    result = {"status": response.get("status", "unknown"), "response": response}
                else:
                    target = None
                    if command == "GRAB":
                        if stable_target is None:
                            self.shared.log("GRAB skipped: no stable target")
                            with self.shared.lock:
                                self.shared.arm_state = "IDLE"
                            continue
                        target = [stable_target["x_cm"], stable_target["y_cm"], stable_target["z_cm"]]
                    result = execute_plan(self.network_config, command, target)

                with self.shared.lock:
                    self.shared.last_arm_result = result
                    self.shared.arm_state = "IDLE"
                status = result.get("status", "unknown")
                self.shared.log(f"Arm command {command} finished with status={status}")
            except Exception as exc:
                with self.shared.lock:
                    self.shared.last_arm_result = {"status": "error", "message": str(exc)}
                    self.shared.arm_state = "ERROR"
                self.shared.log(f"Arm command failed: {exc}")

        self.shared.log("Arm worker stopped")


def _candidate_to_obj(candidate: Optional[dict[str, Any]]):
    if candidate is None:
        return None

    class _Obj:
        def __init__(self, item: dict[str, Any]) -> None:
            self.key = item["key"]
            self.class_name = item["class_name"]
            self.confidence = item["confidence"]
            self.x_cm = item["x_cm"]
            self.y_cm = item["y_cm"]
            self.z_cm = item["z_cm"]
            self.distance_cm = item["distance_cm"]
            self.box = item["box"]

    return _Obj(candidate)


def load_dashboard_config(path: Path) -> DashboardConfig:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return DashboardConfig(
        refresh_interval_ms=int(data["ui"]["refresh_interval_ms"]),
        max_signal_points=int(data["ui"]["max_signal_points"]),
        max_command_history=int(data["ui"]["max_command_history"]),
        max_log_entries=int(data["ui"]["max_log_entries"]),
        enable_signal_worker=bool(data["runtime"]["enable_signal_worker"]),
        enable_vision_worker=bool(data["runtime"]["enable_vision_worker"]),
        enable_arm_worker=bool(data["runtime"]["enable_arm_worker"]),
    )


def init_runtime_state(cfg: DashboardConfig) -> None:
    if "runtime" in st.session_state:
        return

    shared = SharedState(
        max_signal_points=cfg.max_signal_points,
        max_command_history=cfg.max_command_history,
        max_log_entries=cfg.max_log_entries,
    )
    stop_event = threading.Event()
    command_queue: queue.Queue[str] = queue.Queue()

    st.session_state.runtime = {
        "shared": shared,
        "stop_event": stop_event,
        "command_queue": command_queue,
        "signal_worker": None,
        "vision_worker": None,
        "arm_worker": None,
        "running": False,
    }


def start_workers(cfg: DashboardConfig) -> None:
    runtime = st.session_state.runtime
    if runtime["running"]:
        return

    shared: SharedState = runtime["shared"]
    stop_event: threading.Event = runtime["stop_event"]
    command_queue: queue.Queue[str] = runtime["command_queue"]

    stop_event.clear()

    if cfg.enable_signal_worker:
        runtime["signal_worker"] = SignalWorker(shared, stop_event, command_queue)
        runtime["signal_worker"].start()

    if cfg.enable_vision_worker:
        runtime["vision_worker"] = VisionWorker(shared, stop_event)
        runtime["vision_worker"].start()

    if cfg.enable_arm_worker:
        runtime["arm_worker"] = ArmWorker(shared, stop_event, command_queue)
        runtime["arm_worker"].start()

    runtime["running"] = True
    shared.log("System started")


def stop_workers() -> None:
    runtime = st.session_state.runtime
    if not runtime["running"]:
        return

    stop_event: threading.Event = runtime["stop_event"]
    stop_event.set()

    for key in ["signal_worker", "vision_worker", "arm_worker"]:
        worker = runtime.get(key)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    runtime["running"] = False
    runtime["shared"].log("System stopped")


def queue_manual_command(command: str) -> None:
    runtime = st.session_state.runtime
    shared: SharedState = runtime["shared"]
    command_queue: queue.Queue[str] = runtime["command_queue"]

    event = {
        "timestamp": time.strftime("%H:%M:%S"),
        "source": "manual",
        "command": command,
    }
    with shared.lock:
        shared.command_history.appendleft(event)
    command_queue.put(command)
    shared.log(f"Manual command queued: {command}")


def render_dashboard(cfg: DashboardConfig) -> None:
    runtime = st.session_state.runtime
    shared: SharedState = runtime["shared"]

    st.title("NeuroHack Live Dashboard")

    if serial is None:
        st.error("pyserial is not installed in the Python environment running Streamlit. Install it with: python -m pip install pyserial")
    if YOLO is None:
        st.error("ultralytics is not installed in the Python environment running Streamlit. Install it with: python -m pip install ultralytics")

    top_col1, top_col2, top_col3, top_col4 = st.columns(4)
    with top_col1:
        if st.button("Start System", use_container_width=True):
            start_workers(cfg)
    with top_col2:
        if st.button("Stop System", use_container_width=True):
            stop_workers()
    with top_col3:
        if st.button("Sim GRAB", use_container_width=True):
            queue_manual_command("GRAB")
    with top_col4:
        if st.button("Sim RELEASE", use_container_width=True):
            queue_manual_command("RELEASE")

    mid_col1, mid_col2, mid_col3, mid_col4 = st.columns(4)
    with mid_col1:
        if st.button("Sim LOOK_LEFT", use_container_width=True):
            queue_manual_command("LOOK_LEFT")
    with mid_col2:
        if st.button("Sim LOOK_RIGHT", use_container_width=True):
            queue_manual_command("LOOK_RIGHT")
    with mid_col3:
        if st.button("Sim OPEN", use_container_width=True):
            queue_manual_command("OPEN")
    with mid_col4:
        if st.button("Sim HOME", use_container_width=True):
            queue_manual_command("HOME")

    with shared.lock:
        running = runtime["running"]
        arm_state = shared.arm_state
        fps = shared.latest_fps
        stable_target = shared.stable_target

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    status_col1.metric("System", "RUNNING" if running else "STOPPED")
    status_col2.metric("Arm State", arm_state)
    status_col3.metric("Vision FPS", f"{fps:.1f}")
    status_col4.metric("Target", stable_target["class_name"] if stable_target else "None")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("EMG")
        with shared.lock:
            emg_raw = list(shared.emg_raw)
            emg_env = list(shared.emg_env)
        if emg_raw:
            emg_df = pd.DataFrame({"EMG Raw": emg_raw, "EMG Envelope": emg_env})
            st.line_chart(emg_df, use_container_width=True)
        else:
            st.info("No EMG data yet")

        st.subheader("EOG")
        with shared.lock:
            eog_raw = list(shared.eog_raw)
            eog_filt = list(shared.eog_filt)
        if eog_raw:
            eog_df = pd.DataFrame({"EOG Raw": eog_raw, "EOG Filtered": eog_filt})
            st.line_chart(eog_df, use_container_width=True)
        else:
            st.info("No EOG data yet")

    with right:
        st.subheader("Camera + Detections")
        with shared.lock:
            frame = None if shared.latest_frame is None else shared.latest_frame.copy()
            candidate_count = shared.vision_candidates_count
            target_payload = shared.stable_target

        st.write(f"Candidates in reach zone: {candidate_count}")
        if target_payload is not None:
            st.write(
                f"Stable target: {target_payload['class_name']} | X={target_payload['x_cm']:.2f} cm, "
                f"Y={target_payload['y_cm']:.2f} cm, Z={target_payload['z_cm']:.2f} cm"
            )
        else:
            st.write("Stable target: None")

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(rgb, channels="RGB", use_container_width=True)
        else:
            st.info("No camera frame yet")

    st.subheader("Command History")
    with shared.lock:
        history = list(shared.command_history)
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.info("No command history yet")

    st.subheader("Arm Execution Result")
    with shared.lock:
        arm_result = dict(shared.last_arm_result)
    if arm_result:
        st.json(arm_result)
    else:
        st.info("No arm execution result yet")

    st.subheader("System Logs")
    with shared.lock:
        logs = list(shared.system_logs)
    if logs:
        st.code("\n".join(logs), language="text")
    else:
        st.info("No logs yet")


def main() -> None:
    st.set_page_config(page_title="NeuroHack Dashboard", layout="wide")
    cfg = load_dashboard_config(Path("configs/dashboard_config.json"))
    init_runtime_state(cfg)
    render_dashboard(cfg)

    if st.session_state.runtime["running"]:
        time.sleep(cfg.refresh_interval_ms / 1000.0)
        st.rerun()


if __name__ == "__main__":
    main()
