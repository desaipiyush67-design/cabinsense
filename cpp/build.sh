#!/usr/bin/env bash
# Build the C++ engine on Linux (Ubuntu/WSL): installs OpenCV, fetches ONNX Runtime, runs CMake.
set -euo pipefail
ORT_VERSION="${ORT_VERSION:-1.20.1}"
cd "$(dirname "$0")"

if ! pkg-config --exists opencv4 2>/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq libopencv-dev cmake g++
fi
if [ ! -d "third_party/onnxruntime" ]; then
  mkdir -p third_party
  curl -sSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" | tar -xz -C third_party
  mv "third_party/onnxruntime-linux-x64-${ORT_VERSION}" third_party/onnxruntime
fi
cmake -S . -B build -DONNXRUNTIME_DIR="$PWD/third_party/onnxruntime" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
echo "built: cpp/build/cabinsense_engine"
