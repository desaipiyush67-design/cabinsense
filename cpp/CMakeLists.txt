cmake_minimum_required(VERSION 3.16)
project(cabinsense_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn videoio highgui)

# Point ONNXRUNTIME_DIR at an extracted ONNX Runtime release (contains include/ and lib/).
set(ONNXRUNTIME_DIR "" CACHE PATH "ONNX Runtime install directory")
find_path(ORT_INCLUDE onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_DIR}/include ${ONNXRUNTIME_DIR}/include/onnxruntime REQUIRED)
find_library(ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib REQUIRED)

add_executable(cabinsense_engine src/main.cpp)
target_include_directories(cabinsense_engine PRIVATE ${ORT_INCLUDE} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cabinsense_engine PRIVATE ${ORT_LIB} ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(cabinsense_engine PRIVATE /W4)
else()
  target_compile_options(cabinsense_engine PRIVATE -Wall -Wextra -O3)
endif()
