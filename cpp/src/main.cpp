// CabinSense C++ inference engine: real-time phone-use detection with ONNX Runtime + OpenCV.
// Consumes the same ONNX model as the Python pipeline (tools/export_onnx.py), FP32 or INT8.
//
//   cabinsense_engine <model.onnx> [camera_index|video_path] [--no-display] [--frames N]
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kInput = 640;
constexpr int kPhoneClass = 67;  // COCO "cell phone"
constexpr float kConf = 0.35f;
constexpr float kNms = 0.45f;

struct Letterbox {
  cv::Mat blob;  // 1x3x640x640 float32, RGB, 0..1
  float scale;
  int padX, padY;
};

Letterbox letterbox(const cv::Mat& bgr) {
  Letterbox lb;
  lb.scale = std::min(kInput / float(bgr.cols), kInput / float(bgr.rows));
  int w = int(std::round(bgr.cols * lb.scale)), h = int(std::round(bgr.rows * lb.scale));
  lb.padX = (kInput - w) / 2;
  lb.padY = (kInput - h) / 2;
  cv::Mat canvas(kInput, kInput, CV_8UC3, cv::Scalar(114, 114, 114)), resized;
  cv::resize(bgr, resized, {w, h});
  resized.copyTo(canvas(cv::Rect(lb.padX, lb.padY, w, h)));
  lb.blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0, {}, {}, /*swapRB=*/true, /*crop=*/false);
  return lb;
}

struct Detection {
  cv::Rect box;
  float score;
};

class PhoneDetector {
 public:
  explicit PhoneDetector(const std::string& model)
      : env_(ORT_LOGGING_LEVEL_WARNING, "cabinsense"), session_(nullptr) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef _WIN32
    std::wstring wpath(model.begin(), model.end());
    session_ = Ort::Session(env_, wpath.c_str(), opts);
#else
    session_ = Ort::Session(env_, model.c_str(), opts);
#endif
    Ort::AllocatorWithDefaultOptions alloc;
    inName_ = session_.GetInputNameAllocated(0, alloc).get();
    outName_ = session_.GetOutputNameAllocated(0, alloc).get();
  }

  std::vector<Detection> operator()(const cv::Mat& bgr) {
    Letterbox lb = letterbox(bgr);
    std::array<int64_t, 4> shape{1, 3, kInput, kInput};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, lb.blob.ptr<float>(), lb.blob.total(), shape.data(), shape.size());
    const char* in[] = {inName_.c_str()};
    const char* out[] = {outName_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in, &input, 1, out, 1);

    // Output layout (1, 84, N): rows 0-3 = cx, cy, w, h; rows 4..83 = class scores.
    auto dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int n = int(dims[2]);
    const float* data = outputs[0].GetTensorData<float>();
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (int i = 0; i < n; ++i) {
      float s = data[(4 + kPhoneClass) * n + i];
      if (s < kConf) continue;
      float cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
      float x1 = (cx - w / 2 - lb.padX) / lb.scale, y1 = (cy - h / 2 - lb.padY) / lb.scale;
      boxes.emplace_back(int(x1), int(y1), int(w / lb.scale), int(h / lb.scale));
      scores.push_back(s);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, kConf, kNms, keep);
    std::vector<Detection> dets;
    for (int k : keep) dets.push_back({boxes[k], scores[k]});
    return dets;
  }

 private:
  Ort::Env env_;
  Ort::Session session_;
  std::string inName_, outName_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.onnx> [camera|video] [--no-display] [--frames N]\n", argv[0]);
    return 1;
  }
  std::string model = argv[1], source = argc > 2 && argv[2][0] != '-' ? argv[2] : "0";
  bool display = true;
  long maxFrames = 0;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--no-display") display = false;
    if (a == "--frames" && i + 1 < argc) maxFrames = std::stol(argv[++i]);
  }

  PhoneDetector detector(model);
  cv::VideoCapture cap;
  bool isCam = std::all_of(source.begin(), source.end(), ::isdigit);
  isCam ? cap.open(std::stoi(source)) : cap.open(source);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "cannot open %s\n", source.c_str());
    return 1;
  }

  cv::Mat frame;
  long frames = 0;
  double totalMs = 0, phoneStreakMs = 0;
  auto last = std::chrono::steady_clock::now();
  while (cap.read(frame) && (!maxFrames || frames < maxFrames)) {
    auto t0 = std::chrono::steady_clock::now();
    auto dets = detector(frame);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    totalMs += ms;
    ++frames;

    double dt = std::chrono::duration<double, std::milli>(t0 - last).count();
    last = t0;
    phoneStreakMs = dets.empty() ? 0 : phoneStreakMs + dt;
    bool alert = phoneStreakMs >= 1000;  // same 1 s rule as MonitorConfig.phone_alert_s
    if (alert) std::printf("{\"t_frame\": %ld, \"kind\": \"phone_use\", \"held_ms\": %.0f}\n", frames, phoneStreakMs);

    if (display) {
      for (auto& d : dets) {
        cv::rectangle(frame, d.box, {0, 0, 255}, 2);
        cv::putText(frame, cv::format("phone %.2f", d.score), d.box.tl() + cv::Point(0, -6), cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 255}, 1);
      }
      cv::putText(frame, cv::format("infer %.1f ms", ms), {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {60, 200, 60}, 2);
      if (alert) cv::putText(frame, "ALERT: phone use", {10, frame.rows - 16}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
      cv::imshow("CabinSense (C++)", frame);
      if ((cv::waitKey(1) & 0xFF) == 27) break;
    }
  }
  if (frames) std::fprintf(stderr, "frames=%ld  mean inference=%.1f ms  (%.1f FPS)\n", frames, totalMs / frames, 1000.0 * frames / totalMs);
  return 0;
}
