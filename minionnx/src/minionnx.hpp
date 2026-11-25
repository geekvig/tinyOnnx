// =====================================================================================
// minionnx.hpp  --  include this one header to use MiniOnnx
// =====================================================================================
//   Stage 1  stage1_format/   protobuf wire format, ModelProto, weight store
//   Stage 2  stage2_graph/    Graph IR, validation, topological order, optimizer
//   Stage 3  stage3_ep/       Execution Providers: cpu, x86 (AVX2), arm (NEON), cuda
//   Stage 4  stage4_runtime/  ExecutionPlan, InferenceSession, executors
//   core/                     Tensor, ThreadPool
// =====================================================================================
#pragma once
#include "stage1_format/onnx_model.hpp"
#include "stage1_format/protobuf_wire.hpp"
#include "stage1_format/weights.hpp"
#include "stage2_graph/graph.hpp"
#include "stage2_graph/optimizer.hpp"
#include "stage3_ep/execution_provider.hpp"
#include "stage4_runtime/session.hpp"
