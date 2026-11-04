"""Latency benchmark: PyTorch vs ONNX Runtime FP32 vs ONNX Runtime INT8 on CPU, plus face-mesh cost.

    python tools/benchmark.py --image samples/test.jpg --runs 50
Prints a Markdown table that README.md quotes.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cabinsense.perception import FaceAnalyzer, PhoneDetector  # noqa: E402


def timeit(fn, frame, runs: int) -> tuple[float, float]:
    for _ in range(5):  # warm-up
        fn(frame)
    ms = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(frame)
        ms.append((time.perf_counter() - t) * 1000)
    return statistics.median(ms), statistics.quantiles(ms, n=20)[18]  # p50, p95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read {args.image}")

    rows = []
    for name, det in [("YOLOv8n PyTorch (FP32)", lambda: PhoneDetector("yolov8n.pt", "torch")),
                      ("YOLOv8n ONNX Runtime (FP32)", lambda: PhoneDetector("models/yolov8n.onnx", "onnx")),
                      ("YOLOv8n ONNX Runtime (INT8)", lambda: PhoneDetector("models/yolov8n_int8.onnx", "onnx"))]:
        try:
            p50, p95 = timeit(det(), frame, args.runs)
            rows.append((name, p50, p95))
        except FileNotFoundError as e:
            print("skip:", e)
    face = FaceAnalyzer()
    p50, p95 = timeit(face, frame, args.runs)
    rows.append(("Face Mesh + EAR/MAR/head pose", p50, p95))

    print(f"\nCPU: {platform.processor() or platform.machine()} | {args.runs} runs | input {frame.shape[1]}x{frame.shape[0]}\n")
    print("| Stage | p50 latency (ms) | p95 latency (ms) |\n|---|---|---|")
    for name, a, b in rows:
        print(f"| {name} | {a:.1f} | {b:.1f} |")


if __name__ == "__main__":
    main()
