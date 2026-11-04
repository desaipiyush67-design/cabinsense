"""Real-time driver monitoring from a webcam or video file.

    python -m cabinsense.run --source 0                    # webcam
    python -m cabinsense.run --source drive.mp4 --save out.mp4 --no-display
    python -m cabinsense.run --source 0 --backend onnx --model models/yolov8n_int8.onnx
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .monitor import DriverMonitor, FrameSignals
from .perception import FaceAnalyzer, PhoneDetector
from .summary import summarize

COLORS = {"ok": (60, 200, 60), "warn": (0, 165, 255), "alert": (0, 0, 255)}


def draw_hud(frame, sig: FrameSignals, perclos: float, fps: float, phones, active: list[str]) -> None:
    for x1, y1, x2, y2, s in phones:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS["alert"], 2)
        cv2.putText(frame, f"phone {s:.2f}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["alert"], 1)
    lines = [f"FPS {fps:4.1f}"]
    if sig.face:
        lines += [f"EAR {sig.ear:.2f}  MAR {sig.mar:.2f}", f"yaw {sig.yaw:+.0f}  pitch {sig.pitch:+.0f}", f"PERCLOS {perclos:.0%}"]
    else:
        lines.append("no face")
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["ok"], 2)
    if active:
        cv2.putText(frame, "ALERT: " + ", ".join(active), (10, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["alert"], 2)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="CabinSense driver monitoring")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--phone-every", type=int, default=3, help="run the phone detector every N frames")
    ap.add_argument("--events", default="runs/events.jsonl")
    ap.add_argument("--save", help="write the annotated video here")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args(argv)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    live = isinstance(src, int)

    face, phone_det, monitor = FaceAnalyzer(), PhoneDetector(args.model, args.backend), DriverMonitor()
    Path(args.events).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.events, "w", encoding="utf-8")
    writer, events, phones = None, [], []
    n, t0, fps, active_until = 0, time.perf_counter(), 0.0, {}

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        tick = time.perf_counter()
        t = (tick - t0) if live else n / video_fps   # video files use media time, cameras use wall time
        f = face(frame)
        if n % args.phone_every == 0:
            phones = phone_det(frame)
        sig = FrameSignals(t=t, face=f is not None, phone=bool(phones),
                           **({k: f[k] for k in ("ear", "mar", "yaw", "pitch")} if f else {}))
        for ev in monitor.update(sig):
            events.append(ev.to_dict())
            log.write(json.dumps(ev.to_dict()) + "\n")
            log.flush()
            active_until[ev.kind] = t + 2.0
        fps = 0.9 * fps + 0.1 / max(time.perf_counter() - tick, 1e-6) if n else 1.0 / max(time.perf_counter() - tick, 1e-6)
        draw_hud(frame, sig, monitor.perclos, fps, phones, [k for k, until in active_until.items() if until > t])

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("CabinSense", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        n += 1

    duration = (time.perf_counter() - t0) if live else n / video_fps
    cap.release()
    log.close()
    face.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    report = summarize(events, duration)
    report["frames"], report["avg_fps"] = n, round(fps, 1)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
