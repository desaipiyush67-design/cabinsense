"""Export the YOLO detector to ONNX and build an INT8 (dynamic-quantized) variant for CPU/edge.

    python tools/export_onnx.py                # -> models/yolov8n.onnx, models/yolov8n_int8.onnx
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    from ultralytics import YOLO
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    exported = Path(YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=False))
    fp32 = out / f"{Path(args.weights).stem}.onnx"
    shutil.move(str(exported), fp32)
    int8 = out / f"{Path(args.weights).stem}_int8.onnx"
    # Conv weights -> INT8; keeps the graph runnable on any CPU with ONNX Runtime (no calibration data needed).
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8, op_types_to_quantize=["Conv", "MatMul"])
    for p in (fp32, int8):
        print(f"{p}  {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
