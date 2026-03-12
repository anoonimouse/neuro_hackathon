from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, sosfilt

try:
    import serial
    SerialExceptionType = serial.SerialException
except ModuleNotFoundError:
    serial = None

    class SerialExceptionType(Exception):
        pass


@dataclass
class SerialConfig:
    port: str
    baudrate: int
    timeout_seconds: float
    reconnect_delay_seconds: float
    separator_mode: str


@dataclass
class SignalConfig:
    sample_rate_hz: float
    emg_channel_index: int
    eog_channel_index: int
    baseline_seconds: float


@dataclass
class EmgConfig:
    bandpass_low_hz: float
    bandpass_high_hz: float
    filter_order: int
    envelope_window_ms: int
    threshold_std_multiplier: float
    threshold_minimum: float
    short_flex_min_ms: int
    long_flex_ms: int
    refractory_ms: int


@dataclass
class EogConfig:
    bandpass_low_hz: float
    bandpass_high_hz: float
    filter_order: int
    threshold_std_multiplier: float
    threshold_minimum: float
    hold_ms: int
    cooldown_ms: int


@dataclass
class LoggingConfig:
    print_raw: bool
    print_filtered: bool


@dataclass
class AppConfig:
    serial: SerialConfig
    signal: SignalConfig
    emg: EmgConfig
    eog: EogConfig
    logging: LoggingConfig


class StatefulBandpass:
    def __init__(self, sample_rate_hz: float, low_hz: float, high_hz: float, order: int) -> None:
        nyquist = 0.5 * sample_rate_hz
        high_hz = min(high_hz, nyquist * 0.95)
        if not 0 < low_hz < high_hz < nyquist:
            raise ValueError(
                f"Invalid bandpass range low={low_hz}, high={high_hz}, nyquist={nyquist}"
            )
        self.sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)

    def process(self, sample: float) -> float:
        data = np.array([sample], dtype=np.float64)
        filtered, self.zi = sosfilt(self.sos, data, zi=self.zi)
        return float(filtered[0])


class EmgCommandDetector:
    def __init__(self, config: AppConfig, on_command: Callable[[str, float], None]) -> None:
        self.config = config
        self.on_command = on_command
        self.filter = StatefulBandpass(
            sample_rate_hz=config.signal.sample_rate_hz,
            low_hz=config.emg.bandpass_low_hz,
            high_hz=config.emg.bandpass_high_hz,
            order=config.emg.filter_order,
        )
        window_size = max(1, int((config.emg.envelope_window_ms / 1000.0) * config.signal.sample_rate_hz))
        self.envelope_window = deque(maxlen=window_size)
        self.baseline_samples = deque(maxlen=max(1, int(config.signal.baseline_seconds * config.signal.sample_rate_hz)))
        self.threshold: Optional[float] = None
        self.flex_active = False
        self.flex_start_ts = 0.0
        self.last_command_ts = 0.0
        self.long_flex_sent = False

    def process(self, raw_value: float, now_ts: float) -> float:
        filtered = self.filter.process(raw_value)
        rectified = abs(filtered)
        self.envelope_window.append(rectified)
        envelope = float(np.mean(self.envelope_window))

        if self.threshold is None:
            self.baseline_samples.append(envelope)
            baseline_ready = len(self.baseline_samples) == self.baseline_samples.maxlen
            if baseline_ready:
                baseline_mean = float(np.mean(self.baseline_samples))
                baseline_std = float(np.std(self.baseline_samples))
                adaptive = baseline_mean + self.config.emg.threshold_std_multiplier * baseline_std
                self.threshold = max(adaptive, self.config.emg.threshold_minimum)
                print(f"[EMG] Baseline ready. Threshold={self.threshold:.3f}")
            return envelope

        refractory_s = self.config.emg.refractory_ms / 1000.0
        if now_ts - self.last_command_ts < refractory_s:
            return envelope

        if envelope > self.threshold and not self.flex_active:
            self.flex_active = True
            self.flex_start_ts = now_ts
            self.long_flex_sent = False

        if self.flex_active:
            duration_ms = (now_ts - self.flex_start_ts) * 1000.0
            if duration_ms >= self.config.emg.long_flex_ms and not self.long_flex_sent:
                self.on_command("RELEASE", now_ts)
                self.last_command_ts = now_ts
                self.long_flex_sent = True

            if envelope <= self.threshold:
                if (
                    self.config.emg.short_flex_min_ms <= duration_ms < self.config.emg.long_flex_ms
                    and not self.long_flex_sent
                ):
                    self.on_command("GRAB", now_ts)
                    self.last_command_ts = now_ts
                self.flex_active = False
                self.long_flex_sent = False

        return envelope


class EogCommandDetector:
    def __init__(self, config: AppConfig, on_command: Callable[[str, float], None]) -> None:
        self.config = config
        self.on_command = on_command
        self.filter = StatefulBandpass(
            sample_rate_hz=config.signal.sample_rate_hz,
            low_hz=config.eog.bandpass_low_hz,
            high_hz=config.eog.bandpass_high_hz,
            order=config.eog.filter_order,
        )
        self.baseline_samples = deque(maxlen=max(1, int(config.signal.baseline_seconds * config.signal.sample_rate_hz)))
        self.threshold: Optional[float] = None
        self.last_command_ts = 0.0
        self.state = "CENTER"
        self.state_start_ts = 0.0

    def process(self, raw_value: float, now_ts: float) -> float:
        filtered = self.filter.process(raw_value)

        if self.threshold is None:
            self.baseline_samples.append(filtered)
            baseline_ready = len(self.baseline_samples) == self.baseline_samples.maxlen
            if baseline_ready:
                baseline_std = float(np.std(self.baseline_samples))
                adaptive = self.config.eog.threshold_std_multiplier * baseline_std
                self.threshold = max(adaptive, self.config.eog.threshold_minimum)
                print(f"[EOG] Baseline ready. Threshold={self.threshold:.3f}")
            return filtered

        cooldown_s = self.config.eog.cooldown_ms / 1000.0
        if now_ts - self.last_command_ts < cooldown_s:
            return filtered

        hold_s = self.config.eog.hold_ms / 1000.0
        if filtered > self.threshold:
            if self.state != "RIGHT_CANDIDATE":
                self.state = "RIGHT_CANDIDATE"
                self.state_start_ts = now_ts
            elif now_ts - self.state_start_ts >= hold_s:
                self.on_command("LOOK_RIGHT", now_ts)
                self.last_command_ts = now_ts
                self.state = "CENTER"
        elif filtered < -self.threshold:
            if self.state != "LEFT_CANDIDATE":
                self.state = "LEFT_CANDIDATE"
                self.state_start_ts = now_ts
            elif now_ts - self.state_start_ts >= hold_s:
                self.on_command("LOOK_LEFT", now_ts)
                self.last_command_ts = now_ts
                self.state = "CENTER"
        else:
            self.state = "CENTER"

        return filtered


class SerialSignalRunner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self) -> None:
        if serial is None:
            raise RuntimeError(
                "pyserial is not installed in the active Python environment. "
                "Install it with: python -m pip install pyserial"
            )

        print("[SYSTEM] Starting signal processing pipeline")
        print(
            f"[SYSTEM] Serial port={self.config.serial.port}, baud={self.config.serial.baudrate}, "
            f"sample_rate={self.config.signal.sample_rate_hz}Hz"
        )

        emg_detector = EmgCommandDetector(self.config, on_command=self._on_command)
        eog_detector = EogCommandDetector(self.config, on_command=self._on_command)

        while True:
            ser = None
            try:
                ser = serial.Serial(
                    self.config.serial.port,
                    self.config.serial.baudrate,
                    timeout=self.config.serial.timeout_seconds,
                )
                print(f"[SERIAL] Connected to {self.config.serial.port}")

                while True:
                    line_bytes = ser.readline()
                    if not line_bytes:
                        continue

                    parsed = self._parse_line(line_bytes)
                    if parsed is None:
                        continue

                    if len(parsed) <= max(self.config.signal.emg_channel_index, self.config.signal.eog_channel_index):
                        continue

                    now_ts = time.time()
                    emg_raw = parsed[self.config.signal.emg_channel_index]
                    eog_raw = parsed[self.config.signal.eog_channel_index]

                    emg_envelope = emg_detector.process(emg_raw, now_ts)
                    eog_filtered = eog_detector.process(eog_raw, now_ts)

                    if self.config.logging.print_raw:
                        print(f"[RAW] EMG={emg_raw:.3f} EOG={eog_raw:.3f}")
                    if self.config.logging.print_filtered:
                        print(f"[FILTERED] EMG_ENV={emg_envelope:.3f} EOG_FILT={eog_filtered:.3f}")

            except SerialExceptionType as exc:
                print(f"[SERIAL] Connection error: {exc}")
                print(f"[SERIAL] Reconnecting in {self.config.serial.reconnect_delay_seconds:.1f}s...")
                time.sleep(self.config.serial.reconnect_delay_seconds)
            finally:
                if ser is not None and ser.is_open:
                    ser.close()

    def _on_command(self, command: str, timestamp: float) -> None:
        formatted = time.strftime("%H:%M:%S", time.localtime(timestamp))
        print(f"[COMMAND] {formatted} {command}")

    def _parse_line(self, line_bytes: bytes) -> Optional[list[float]]:
        try:
            line = line_bytes.decode("utf-8", errors="ignore").strip()
        except UnicodeDecodeError:
            return None

        if not line:
            return None

        separator_mode = self.config.serial.separator_mode.lower()
        parts: list[str]

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

        cleaned = [segment.strip() for segment in parts if segment.strip()]
        if not cleaned:
            return None

        values: list[float] = []
        for token in cleaned:
            if token.lower() in {"nan", "inf", "-inf"}:
                return None
            value = float(token)
            if math.isnan(value) or math.isinf(value):
                return None
            values.append(value)

        return values


def load_config(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    return AppConfig(
        serial=SerialConfig(**data["serial"]),
        signal=SignalConfig(**data["signal"]),
        emg=EmgConfig(**data["emg"]),
        eog=EogConfig(**data["eog"]),
        logging=LoggingConfig(**data["logging"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EMG/EOG command detection over serial stream")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/signal_config.json"),
        help="Path to JSON configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    runner = SerialSignalRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
