from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


@dataclass
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int


@dataclass
class YoloConfig:
    model_path: str
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    target_classes: list[str]


@dataclass
class CalibrationConfig:
    file_path: str
    table_z_cm: float
    pixels_per_cm_x: float
    pixels_per_cm_y: float
    origin_u: float
    origin_v: float
    measurement_distance_cm: float


@dataclass
class ReachZoneCm:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class SelectionConfig:
    stabilization_seconds: float
    position_quantization_px: int
    max_missing_seconds: float


@dataclass
class DisplayConfig:
    window_name: str
    show_fps: bool
    draw_reach_zone: bool


@dataclass
class VisionConfig:
    camera: CameraConfig
    yolo: YoloConfig
    calibration: CalibrationConfig
    reach_zone_cm: ReachZoneCm
    selection: SelectionConfig
    display: DisplayConfig


@dataclass
class CalibrationData:
    pixels_per_cm_x: float
    pixels_per_cm_y: float
    origin_u: float
    origin_v: float
    table_z_cm: float
    updated_at_epoch: float


@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    u: float
    v: float
    x_cm: float
    y_cm: float
    z_cm: float
    distance_cm: float
    box: tuple[int, int, int, int]
    key: str


class TargetStabilizer:
    def __init__(self, stabilization_seconds: float, max_missing_seconds: float) -> None:
        self.stabilization_seconds = stabilization_seconds
        self.max_missing_seconds = max_missing_seconds
        self.candidate_key: Optional[str] = None
        self.candidate_start_ts: float = 0.0
        self.stable_target: Optional[DetectedObject] = None
        self.last_seen_ts: float = 0.0

    def update(self, candidate: Optional[DetectedObject], now_ts: float) -> Optional[DetectedObject]:
        if candidate is None:
            if self.stable_target is not None and now_ts - self.last_seen_ts > self.max_missing_seconds:
                self.stable_target = None
            self.candidate_key = None
            self.candidate_start_ts = 0.0
            return self.stable_target

        self.last_seen_ts = now_ts
        if candidate.key != self.candidate_key:
            self.candidate_key = candidate.key
            self.candidate_start_ts = now_ts

        held_long_enough = now_ts - self.candidate_start_ts >= self.stabilization_seconds
        if held_long_enough:
            self.stable_target = candidate

        return self.stable_target


def load_config(config_path: Path) -> VisionConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    return VisionConfig(
        camera=CameraConfig(**data["camera"]),
        yolo=YoloConfig(**data["yolo"]),
        calibration=CalibrationConfig(**data["calibration"]),
        reach_zone_cm=ReachZoneCm(**data["reach_zone_cm"]),
        selection=SelectionConfig(**data["selection"]),
        display=DisplayConfig(**data["display"]),
    )


def load_calibration(config: VisionConfig) -> CalibrationData:
    calibration_path = Path(config.calibration.file_path)
    if calibration_path.exists():
        with calibration_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return CalibrationData(**data)

    return CalibrationData(
        pixels_per_cm_x=config.calibration.pixels_per_cm_x,
        pixels_per_cm_y=config.calibration.pixels_per_cm_y,
        origin_u=config.calibration.origin_u,
        origin_v=config.calibration.origin_v,
        table_z_cm=config.calibration.table_z_cm,
        updated_at_epoch=0.0,
    )


def save_calibration(config: VisionConfig, calibration: CalibrationData) -> None:
    payload = {
        "pixels_per_cm_x": calibration.pixels_per_cm_x,
        "pixels_per_cm_y": calibration.pixels_per_cm_y,
        "origin_u": calibration.origin_u,
        "origin_v": calibration.origin_v,
        "table_z_cm": calibration.table_z_cm,
        "updated_at_epoch": calibration.updated_at_epoch,
    }
    with Path(config.calibration.file_path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def pixel_to_table_cm(u: float, v: float, calibration: CalibrationData) -> tuple[float, float, float]:
    x_cm = (u - calibration.origin_u) / calibration.pixels_per_cm_x
    y_cm = (calibration.origin_v - v) / calibration.pixels_per_cm_y
    z_cm = calibration.table_z_cm
    return x_cm, y_cm, z_cm


def in_reach_zone(x_cm: float, y_cm: float, zone: ReachZoneCm) -> bool:
    return zone.x_min <= x_cm <= zone.x_max and zone.y_min <= y_cm <= zone.y_max


def quantized_key(class_name: str, u: float, v: float, quantization_px: int) -> str:
    q = max(1, quantization_px)
    q_u = int(round(u / q) * q)
    q_v = int(round(v / q) * q)
    return f"{class_name}:{q_u}:{q_v}"


def run_calibration(config: VisionConfig) -> None:
    capture = cv2.VideoCapture(config.camera.index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    capture.set(cv2.CAP_PROP_FPS, config.camera.fps)

    if not capture.isOpened():
        raise RuntimeError("Unable to open camera for calibration")

    clicks: list[tuple[int, int]] = []
    calibration = load_calibration(config)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    window_name = "Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ok, frame = capture.read()
        if not ok:
            continue

        display = frame.copy()
        for point in clicks:
            cv2.circle(display, point, 6, (0, 255, 255), -1)

        instruction = "Click ARM BASE origin, then click reference point. Press S to save, R to reset, Q to quit"
        cv2.putText(display, instruction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            clicks.clear()
        elif key == ord("q"):
            break
        elif key == ord("s") and len(clicks) >= 2:
            origin = clicks[0]
            reference = clicks[1]
            distance_px = math.dist(origin, reference)
            pixels_per_cm = distance_px / config.calibration.measurement_distance_cm
            calibration.origin_u = float(origin[0])
            calibration.origin_v = float(origin[1])
            calibration.pixels_per_cm_x = float(pixels_per_cm)
            calibration.pixels_per_cm_y = float(pixels_per_cm)
            calibration.table_z_cm = config.calibration.table_z_cm
            calibration.updated_at_epoch = time.time()
            save_calibration(config, calibration)
            print(
                f"[CALIBRATION] Saved origin=({calibration.origin_u:.1f}, {calibration.origin_v:.1f}) "
                f"pixels_per_cm={pixels_per_cm:.4f}"
            )
            break

    capture.release()
    cv2.destroyAllWindows()


def run_vision(config: VisionConfig) -> None:
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed in the active Python environment. "
            "Install it with: python -m pip install ultralytics"
        )

    calibration = load_calibration(config)
    model = YOLO(config.yolo.model_path)
    capture = cv2.VideoCapture(config.camera.index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
    capture.set(cv2.CAP_PROP_FPS, config.camera.fps)

    if not capture.isOpened():
        raise RuntimeError("Unable to open camera")

    stabilizer = TargetStabilizer(
        stabilization_seconds=config.selection.stabilization_seconds,
        max_missing_seconds=config.selection.max_missing_seconds,
    )

    print("[VISION] Running single-camera laptop vision pipeline")
    frame_count = 0
    fps_start = time.time()
    fps_value = 0.0

    while True:
        ok, frame = capture.read()
        if not ok:
            continue

        now_ts = time.time()
        result = model.predict(
            source=frame,
            conf=config.yolo.confidence_threshold,
            iou=config.yolo.iou_threshold,
            max_det=config.yolo.max_detections,
            verbose=False,
        )[0]

        names = result.names
        candidates: list[DetectedObject] = []
        boxes = result.boxes

        if boxes is not None and boxes.xyxy is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                class_name = names.get(cls_id, str(cls_id))
                if class_name not in config.yolo.target_classes:
                    continue

                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                u = 0.5 * (x1 + x2)
                v = 0.5 * (y1 + y2)
                x_cm, y_cm, z_cm = pixel_to_table_cm(u, v, calibration)

                if not in_reach_zone(x_cm, y_cm, config.reach_zone_cm):
                    continue

                distance_cm = math.hypot(x_cm, y_cm)
                key = quantized_key(class_name, u, v, config.selection.position_quantization_px)
                candidates.append(
                    DetectedObject(
                        class_name=class_name,
                        confidence=conf,
                        u=u,
                        v=v,
                        x_cm=x_cm,
                        y_cm=y_cm,
                        z_cm=z_cm,
                        distance_cm=distance_cm,
                        box=(int(x1), int(y1), int(x2), int(y2)),
                        key=key,
                    )
                )

        candidates.sort(key=lambda item: item.distance_cm)
        current_candidate = candidates[0] if candidates else None
        stable_target = stabilizer.update(current_candidate, now_ts)

        for obj in candidates:
            x1, y1, x2, y2 = obj.box
            color = (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.class_name} {obj.confidence:.2f} X={obj.x_cm:.1f} Y={obj.y_cm:.1f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if stable_target is not None:
            x1, y1, x2, y2 = stable_target.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            stable_text = (
                f"STABLE TARGET {stable_target.class_name} "
                f"X={stable_target.x_cm:.1f} Y={stable_target.y_cm:.1f} Z={stable_target.z_cm:.1f}"
            )
            cv2.putText(frame, stable_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(
                f"[TARGET] class={stable_target.class_name} x_cm={stable_target.x_cm:.2f} "
                f"y_cm={stable_target.y_cm:.2f} z_cm={stable_target.z_cm:.2f}"
            )

        if config.display.draw_reach_zone:
            zone = config.reach_zone_cm
            corners_cm = [
                (zone.x_min, zone.y_min),
                (zone.x_max, zone.y_min),
                (zone.x_max, zone.y_max),
                (zone.x_min, zone.y_max),
            ]
            points_px = []
            for x_cm, y_cm in corners_cm:
                u = int(round(calibration.origin_u + x_cm * calibration.pixels_per_cm_x))
                v = int(round(calibration.origin_v - y_cm * calibration.pixels_per_cm_y))
                points_px.append((u, v))
            cv2.polylines(frame, [cv2.UMat(np_array(points_px))], True, (255, 0, 0), 2)

        frame_count += 1
        elapsed = now_ts - fps_start
        if elapsed >= 1.0:
            fps_value = frame_count / elapsed
            frame_count = 0
            fps_start = now_ts

        if config.display.show_fps:
            cv2.putText(frame, f"FPS {fps_value:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(config.display.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def np_array(points_px: list[tuple[int, int]]):
    import numpy as np

    return np.array(points_px, dtype="int32")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-camera laptop vision pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vision_config.json"),
        help="Path to vision config JSON file",
    )
    parser.add_argument(
        "--mode",
        choices=["run", "calibrate"],
        default="run",
        help="run: detect and select objects, calibrate: set arm origin and pixel scale",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode == "calibrate":
        run_calibration(config)
    else:
        run_vision(config)


if __name__ == "__main__":
    main()
