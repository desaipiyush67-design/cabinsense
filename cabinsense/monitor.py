"""Temporal state machine that turns per-frame signals into driver-safety events.

Per-frame measurements are noisy, so every alert is based on a time window
(PERCLOS for drowsiness, sustained duration for distraction and phone use).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict


@dataclass
class MonitorConfig:
    ear_closed: float = 0.21          # EAR below this counts as "eyes closed" (until calibrated)
    ear_closed_ratio: float = 0.7     # after calibration: closed = EAR < 70% of the driver's open-eye EAR
    perclos_window_s: float = 30.0    # rolling window for PERCLOS
    perclos_alert: float = 0.15       # >15% eyes-closed time in the window = drowsy
    eyes_closed_alert_s: float = 1.5  # a single continuous closure this long = microsleep
    mar_yawn: float = 0.6             # MAR above this counts as mouth wide open
    yawn_min_s: float = 1.0           # open that long = one yawn
    yaw_limit_deg: float = 30.0       # head turned further than this = looking away
    pitch_limit_deg: float = 25.0     # head down/up further than this = looking away
    distraction_alert_s: float = 2.0  # looking away this long = distraction alert
    phone_alert_s: float = 1.0        # phone visible this long = phone-use alert
    cooldown_s: float = 5.0           # minimum gap between two alerts of the same type
    calibration_s: float = 3.0        # learn the driver's neutral head pose over the first seconds


@dataclass
class Event:
    t: float
    kind: str        # drowsiness | microsleep | yawn | distraction | phone_use | no_face
    value: float     # the measurement that triggered it (PERCLOS, seconds, ...)
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameSignals:
    t: float                     # timestamp in seconds
    face: bool                   # was a face found
    ear: float = 0.0
    mar: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    phone: bool = False          # phone detected near the driver


@dataclass
class DriverMonitor:
    cfg: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self) -> None:
        self._closed = deque()          # (t, closed?) samples inside the PERCLOS window
        self._closed_since = None
        self._mouth_since = None
        self._away_since = None
        self._phone_since = None
        self._noface_since = None
        self._last_alert: dict[str, float] = {}
        self._calib: list[tuple[float, float, float]] = []
        self._calib_start = None
        self.neutral = None             # (yaw, pitch) of the driver's normal forward-looking pose
        self.ear_threshold = self.cfg.ear_closed
        self.perclos = 0.0

    def _fire(self, kind: str, t: float, value: float, detail: str = "") -> list[Event]:
        if t - self._last_alert.get(kind, -1e9) < self.cfg.cooldown_s:
            return []
        self._last_alert[kind] = t
        return [Event(t=round(t, 2), kind=kind, value=round(value, 3), detail=detail)]

    @staticmethod
    def _held(since: float | None, active: bool, t: float) -> tuple[float | None, float]:
        """Track how long a condition has been continuously true."""
        if not active:
            return None, 0.0
        since = t if since is None else since
        return since, t - since

    def update(self, s: FrameSignals) -> list[Event]:
        c = self.cfg
        events: list[Event] = []

        self._noface_since, noface_for = self._held(self._noface_since, not s.face, s.t)
        if not s.face:
            if noface_for >= c.distraction_alert_s:
                events += self._fire("no_face", s.t, noface_for, "driver not visible to the camera")
            return events

        # Calibration: learn this driver's neutral head pose and open-eye EAR (faces and camera mounts differ).
        if self.neutral is None:
            self._calib_start = s.t if self._calib_start is None else self._calib_start
            self._calib.append((s.yaw, s.pitch, s.ear))
            if s.t - self._calib_start >= c.calibration_s:
                med = lambda i: sorted(x[i] for x in self._calib)[len(self._calib) // 2]
                self.neutral = (med(0), med(1))
                self.ear_threshold = min(c.ear_closed, c.ear_closed_ratio * med(2))
                self._closed.clear()          # samples so far were judged with the generic threshold
                self._closed_since = None

        closed = s.ear < self.ear_threshold
        self._closed.append((s.t, closed))
        while self._closed and s.t - self._closed[0][0] > c.perclos_window_s:
            self._closed.popleft()
        self.perclos = sum(1 for _, x in self._closed if x) / len(self._closed)
        calibrated = self.neutral is not None  # no fatigue alerts while the baseline is still being learned
        window_full = s.t - self._closed[0][0] >= c.perclos_window_s * 0.5
        if calibrated and window_full and self.perclos >= c.perclos_alert:
            events += self._fire("drowsiness", s.t, self.perclos, f"PERCLOS {self.perclos:.0%} over {c.perclos_window_s:.0f}s")

        self._closed_since, closed_for = self._held(self._closed_since, closed, s.t)
        if calibrated and closed_for >= c.eyes_closed_alert_s:
            events += self._fire("microsleep", s.t, closed_for, f"eyes closed {closed_for:.1f}s")

        self._mouth_since, open_for = self._held(self._mouth_since, s.mar > c.mar_yawn, s.t)
        if open_for >= c.yawn_min_s:
            events += self._fire("yawn", s.t, s.mar, f"mouth open {open_for:.1f}s")

        if self.neutral is not None:
            dyaw, dpitch = s.yaw - self.neutral[0], s.pitch - self.neutral[1]
            away = abs(dyaw) > c.yaw_limit_deg or abs(dpitch) > c.pitch_limit_deg
            self._away_since, away_for = self._held(self._away_since, away, s.t)
            if away_for >= c.distraction_alert_s:
                events += self._fire("distraction", s.t, away_for, f"head turned yaw {dyaw:+.0f} deg, pitch {dpitch:+.0f} deg")

        self._phone_since, phone_for = self._held(self._phone_since, s.phone, s.t)
        if phone_for >= c.phone_alert_s:
            events += self._fire("phone_use", s.t, phone_for, f"phone visible {phone_for:.1f}s")

        return events
