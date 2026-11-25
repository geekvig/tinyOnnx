#!/usr/bin/env bash
# Build MiniOnnx with plain g++ (no CMake). Run from the project root:  bash scripts/build_nocmake.sh
set -e
LIB="src/stage1_format/onnx_io.cpp src/stage1_format/weights.cpp src/stage2_graph/graph.cpp \
     src/stage2_graph/optimizer.cpp src/stage3_ep/cpu_ep.cpp src/stage3_ep/x86_ep.cpp src/stage3_ep/arm_ep.cpp \
     src/stage3_ep/registry.cpp src/stage3_ep/cuda_ep_stub.cpp src/stage4_runtime/plan.cpp src/stage4_runtime/session.cpp"
FLAGS="-std=c++17 -g -O1 -Wall -Wextra -pthread -Isrc"
mkdir -p bin
for prog in tools/make_models tools/minionnx_run tools/inspect_onnx tests/test_main bench/bench; do
  echo "g++ ... $prog.cpp"
  g++ $FLAGS $LIB $prog.cpp -o bin/$(basename $prog)
done
echo "done: binaries are in bin/   try:  ./bin/test_main"
