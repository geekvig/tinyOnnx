// =====================================================================================
// STAGE 3 · cuda_ep_stub.cpp  --  compiled INSTEAD of cuda_ep.cu when CUDA is disabled
// =====================================================================================
// CMake picks exactly one of {cuda_ep.cu, cuda_ep_stub.cpp}. The rest of the program calls
// make_cuda_ep() either way; here it just says "not compiled in".
// =====================================================================================
#include "stage3_ep/execution_provider.hpp"

namespace mo {
std::unique_ptr<ExecutionProvider> make_cuda_ep() { return nullptr; }
void cuda_ep_register_weights(ExecutionProvider *, const Graph &) {}
} // namespace mo
