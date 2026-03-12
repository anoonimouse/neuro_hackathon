from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from motion_planner import MotionPlanner, TargetPoint, load_config as load_motion_config


@dataclass
class Esp32Config:
    host: str
    port: int
    connect_timeout_seconds: float
    response_timeout_seconds: float


@dataclass
class NetworkMotionConfig:
    planner_config_path: str
    default_step_delay_ms: int


@dataclass
class NetworkConfig:
    esp32: Esp32Config
    motion: NetworkMotionConfig


def load_network_config(config_path: Path) -> NetworkConfig:
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return NetworkConfig(
        esp32=Esp32Config(**data["esp32"]),
        motion=NetworkMotionConfig(**data["motion"]),
    )


def send_command(esp: Esp32Config, payload: dict) -> dict:
    request = json.dumps(payload) + "\n"

    with socket.create_connection((esp.host, esp.port), timeout=esp.connect_timeout_seconds) as sock:
        sock.settimeout(esp.response_timeout_seconds)
        sock.sendall(request.encode("utf-8"))

        response_chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response_chunks.append(chunk)
            if b"\n" in chunk:
                break

    raw = b"".join(response_chunks).decode("utf-8", errors="ignore").strip()
    if not raw:
        return {"status": "error", "message": "empty_response"}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "error", "message": "invalid_response", "raw": raw}


def execute_plan(config: NetworkConfig, command: str, target_xyz: Optional[list[float]]) -> dict:
    motion_config = load_motion_config(Path(config.motion.planner_config_path))
    planner = MotionPlanner(motion_config)

    if target_xyz is not None:
        planner.update_target(TargetPoint(x_cm=target_xyz[0], y_cm=target_xyz[1], z_cm=target_xyz[2]))

    plan = planner.handle_command(command)
    if plan.get("result") != "ok" and plan.get("result") != "unreachable":
        return {
            "plan": plan,
            "execution": [],
            "status": "error",
            "message": plan.get("reason", "planner_error"),
        }

    execution = []
    sequence = plan.get("sequence", [])
    step_delay_ms = plan.get("step_delay_ms", config.motion.default_step_delay_ms)

    for step in sequence:
        angles = step["angles_deg"]
        payload = {
            "command": "SET_ANGLES",
            "angles": {
                "base": float(angles["base"]),
                "shoulder": float(angles["shoulder"]),
                "elbow": float(angles["elbow"]),
                "gripper": float(angles["gripper"]),
            },
        }
        response = send_command(config.esp32, payload)
        execution.append({"step": step["name"], "request": payload, "response": response})
        time.sleep(step_delay_ms / 1000.0)

    return {
        "plan": plan,
        "execution": execution,
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Send motion plans to ESP32 arm controller")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/arm_network_config.json"),
        help="Path to network config JSON",
    )
    parser.add_argument(
        "--command",
        required=True,
        choices=["GRAB", "RELEASE", "LOOK_LEFT", "LOOK_RIGHT", "HOME", "OPEN", "CLOSE", "PING"],
        help="Command to execute",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X_CM", "Y_CM", "Z_CM"),
        help="Target coordinates for GRAB command",
    )
    args = parser.parse_args()

    config = load_network_config(args.config)

    if args.command in {"HOME", "OPEN", "CLOSE", "PING"}:
        response = send_command(config.esp32, {"command": args.command})
        print(json.dumps({"command": args.command, "response": response}, indent=2))
        return

    result = execute_plan(config, args.command, args.target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
