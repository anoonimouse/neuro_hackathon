from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GeometryConfig:
    link1_cm: float
    link2_cm: float
    base_height_cm: float
    table_z_cm: float
    grip_height_offset_cm: float
    approach_lift_cm: float


@dataclass
class PlannerConfig:
    look_step_deg: float
    manual_base_offset_min_deg: float
    manual_base_offset_max_deg: float


@dataclass
class PoseConfig:
    home: dict[str, float]
    safe: dict[str, float]
    gripper_open: float
    gripper_closed: float


@dataclass
class ServoJointConfig:
    min: float
    max: float
    invert: bool
    offset_deg: float


@dataclass
class ServoConfig:
    base: ServoJointConfig
    shoulder: ServoJointConfig
    elbow: ServoJointConfig
    gripper: ServoJointConfig


@dataclass
class MotionConfig:
    step_delay_ms: int
    use_approach_pose: bool


@dataclass
class AppConfig:
    geometry: GeometryConfig
    planner: PlannerConfig
    poses: PoseConfig
    servo: ServoConfig
    motion: MotionConfig


@dataclass
class TargetPoint:
    x_cm: float
    y_cm: float
    z_cm: float
    label: str = "object"


@dataclass
class JointAngles:
    base: float
    shoulder: float
    elbow: float
    gripper: float


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class FourDofIkSolver:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def solve(self, target: TargetPoint, base_manual_offset_deg: float) -> Optional[JointAngles]:
        x = target.x_cm
        y = target.y_cm
        z = target.z_cm + self.config.geometry.grip_height_offset_cm

        base_rad = math.atan2(y, x)
        base_deg = math.degrees(base_rad) + base_manual_offset_deg

        r = math.hypot(x, y)
        z_rel = z - self.config.geometry.base_height_cm
        d = math.hypot(r, z_rel)

        l1 = self.config.geometry.link1_cm
        l2 = self.config.geometry.link2_cm

        max_reach = l1 + l2
        min_reach = abs(l1 - l2)
        if d > max_reach or d < min_reach:
            return None

        cos_elbow = (d * d - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_elbow = clamp(cos_elbow, -1.0, 1.0)
        elbow_rad = math.acos(cos_elbow)

        shoulder_to_target = math.atan2(z_rel, r)
        shoulder_offset = math.atan2(l2 * math.sin(elbow_rad), l1 + l2 * math.cos(elbow_rad))
        shoulder_rad = shoulder_to_target - shoulder_offset

        elbow_deg = math.degrees(elbow_rad)
        shoulder_deg = math.degrees(shoulder_rad)

        servo_target = JointAngles(
            base=base_deg,
            shoulder=shoulder_deg,
            elbow=elbow_deg,
            gripper=self.config.poses.gripper_open,
        )
        return self._to_servo_space(servo_target)

    def _to_servo_space(self, joints: JointAngles) -> JointAngles:
        base = self._apply_joint_map(joints.base, self.config.servo.base)
        shoulder = self._apply_joint_map(joints.shoulder, self.config.servo.shoulder)
        elbow = self._apply_joint_map(joints.elbow, self.config.servo.elbow)
        gripper = self._apply_joint_map(joints.gripper, self.config.servo.gripper)
        return JointAngles(base=base, shoulder=shoulder, elbow=elbow, gripper=gripper)

    @staticmethod
    def _apply_joint_map(angle_deg: float, joint_cfg: ServoJointConfig) -> float:
        mapped = -angle_deg if joint_cfg.invert else angle_deg
        mapped += joint_cfg.offset_deg
        mapped = clamp(mapped, joint_cfg.min, joint_cfg.max)
        return mapped


class MotionPlanner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ik_solver = FourDofIkSolver(config)
        self.base_manual_offset_deg = 0.0
        self.current_target: Optional[TargetPoint] = None

    def update_target(self, target: TargetPoint) -> None:
        self.current_target = target

    def handle_command(self, command: str) -> dict:
        command = command.upper().strip()

        if command == "LOOK_LEFT":
            self.base_manual_offset_deg -= self.config.planner.look_step_deg
            self.base_manual_offset_deg = clamp(
                self.base_manual_offset_deg,
                self.config.planner.manual_base_offset_min_deg,
                self.config.planner.manual_base_offset_max_deg,
            )
            return self._single_pose_plan("LOOK_LEFT", self._base_only_pose())

        if command == "LOOK_RIGHT":
            self.base_manual_offset_deg += self.config.planner.look_step_deg
            self.base_manual_offset_deg = clamp(
                self.base_manual_offset_deg,
                self.config.planner.manual_base_offset_min_deg,
                self.config.planner.manual_base_offset_max_deg,
            )
            return self._single_pose_plan("LOOK_RIGHT", self._base_only_pose())

        if command == "RELEASE":
            release_pose = JointAngles(
                base=self.config.poses.home["base"],
                shoulder=self.config.poses.home["shoulder"],
                elbow=self.config.poses.home["elbow"],
                gripper=self.config.poses.gripper_open,
            )
            return {
                "command": "RELEASE",
                "result": "ok",
                "sequence": [
                    self._pose_to_dict("open_gripper", release_pose),
                    self._pose_to_dict("return_home", release_pose),
                ],
                "step_delay_ms": self.config.motion.step_delay_ms,
            }

        if command == "GRAB":
            if self.current_target is None:
                return {
                    "command": "GRAB",
                    "result": "error",
                    "reason": "No target available",
                    "sequence": [],
                }

            ik_solution = self.ik_solver.solve(self.current_target, self.base_manual_offset_deg)
            if ik_solution is None:
                safe_pose = JointAngles(
                    base=self.config.poses.safe["base"],
                    shoulder=self.config.poses.safe["shoulder"],
                    elbow=self.config.poses.safe["elbow"],
                    gripper=self.config.poses.gripper_open,
                )
                return {
                    "command": "GRAB",
                    "result": "unreachable",
                    "reason": "Target outside reachable workspace",
                    "sequence": [self._pose_to_dict("move_safe", safe_pose)],
                    "step_delay_ms": self.config.motion.step_delay_ms,
                }

            sequence = []
            if self.config.motion.use_approach_pose:
                approach = JointAngles(
                    base=ik_solution.base,
                    shoulder=clamp(ik_solution.shoulder - 8.0, self.config.servo.shoulder.min, self.config.servo.shoulder.max),
                    elbow=clamp(ik_solution.elbow - 6.0, self.config.servo.elbow.min, self.config.servo.elbow.max),
                    gripper=self.config.poses.gripper_open,
                )
                sequence.append(self._pose_to_dict("approach", approach))

            grasp_pose = JointAngles(
                base=ik_solution.base,
                shoulder=ik_solution.shoulder,
                elbow=ik_solution.elbow,
                gripper=self.config.poses.gripper_open,
            )
            close_pose = JointAngles(
                base=ik_solution.base,
                shoulder=ik_solution.shoulder,
                elbow=ik_solution.elbow,
                gripper=self.config.poses.gripper_closed,
            )
            sequence.append(self._pose_to_dict("move_to_target", grasp_pose))
            sequence.append(self._pose_to_dict("close_gripper", close_pose))

            return {
                "command": "GRAB",
                "result": "ok",
                "target": {
                    "label": self.current_target.label,
                    "x_cm": self.current_target.x_cm,
                    "y_cm": self.current_target.y_cm,
                    "z_cm": self.current_target.z_cm,
                },
                "sequence": sequence,
                "step_delay_ms": self.config.motion.step_delay_ms,
            }

        return {
            "command": command,
            "result": "error",
            "reason": "Unknown command",
            "sequence": [],
        }

    def _base_only_pose(self) -> JointAngles:
        base_home = self.config.poses.home["base"]
        base = clamp(
            base_home + self.base_manual_offset_deg,
            self.config.servo.base.min,
            self.config.servo.base.max,
        )
        return JointAngles(
            base=base,
            shoulder=self.config.poses.home["shoulder"],
            elbow=self.config.poses.home["elbow"],
            gripper=self.config.poses.gripper_open,
        )

    def _single_pose_plan(self, command: str, pose: JointAngles) -> dict:
        return {
            "command": command,
            "result": "ok",
            "sequence": [self._pose_to_dict("base_adjust", pose)],
            "step_delay_ms": self.config.motion.step_delay_ms,
            "base_manual_offset_deg": self.base_manual_offset_deg,
        }

    @staticmethod
    def _pose_to_dict(name: str, pose: JointAngles) -> dict:
        return {
            "name": name,
            "angles_deg": {
                "base": round(pose.base, 3),
                "shoulder": round(pose.shoulder, 3),
                "elbow": round(pose.elbow, 3),
                "gripper": round(pose.gripper, 3),
            },
        }


def load_config(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    servo = data["servo"]
    return AppConfig(
        geometry=GeometryConfig(**data["geometry"]),
        planner=PlannerConfig(**data["planner"]),
        poses=PoseConfig(**data["poses"]),
        servo=ServoConfig(
            base=ServoJointConfig(**servo["base"]),
            shoulder=ServoJointConfig(**servo["shoulder"]),
            elbow=ServoJointConfig(**servo["elbow"]),
            gripper=ServoJointConfig(**servo["gripper"]),
        ),
        motion=MotionConfig(**data["motion"]),
    )


def run_cli(config: AppConfig, command: str, target: Optional[list[float]]) -> None:
    planner = MotionPlanner(config)

    if target is not None:
        planner.update_target(TargetPoint(x_cm=target[0], y_cm=target[1], z_cm=target[2]))

    plan = planner.handle_command(command)
    print(json.dumps(plan, indent=2))


def run_interactive(config: AppConfig) -> None:
    planner = MotionPlanner(config)
    print("Interactive motion planner started.")
    print("Set target: TARGET x y z")
    print("Commands: GRAB, RELEASE, LOOK_LEFT, LOOK_RIGHT, QUIT")

    while True:
        line = input("> ").strip()
        if not line:
            continue

        if line.upper() == "QUIT":
            break

        tokens = line.split()
        if tokens[0].upper() == "TARGET" and len(tokens) == 4:
            x, y, z = float(tokens[1]), float(tokens[2]), float(tokens[3])
            planner.update_target(TargetPoint(x_cm=x, y_cm=y, z_cm=z))
            print(f"Target set to ({x:.2f}, {y:.2f}, {z:.2f})")
            continue

        plan = planner.handle_command(tokens[0].upper())
        print(json.dumps(plan, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="4-DOF IK and motion planner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/motion_config.json"),
        help="Path to motion config JSON",
    )
    parser.add_argument(
        "--command",
        type=str,
        default="GRAB",
        help="Single command mode: GRAB, RELEASE, LOOK_LEFT, LOOK_RIGHT",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X_CM", "Y_CM", "Z_CM"),
        help="Optional target coordinates in cm for single command mode",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive command loop",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.interactive:
        run_interactive(config)
    else:
        run_cli(config, args.command, args.target)


if __name__ == "__main__":
    main()
