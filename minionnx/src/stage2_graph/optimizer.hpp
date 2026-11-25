// =====================================================================================
// STAGE 2 · optimizer.hpp  --  graph rewrites done ONCE at load time
// =====================================================================================
// An optimizer pass takes a valid graph and returns an equivalent graph that is cheaper to run.
// "Equivalent" is the whole game: a pass that changes the answer is a bug, even if it is fast.
//
//   eliminate_identity    Identity(x) -> y   : consumers read x directly
//   fold_gemm_transposes  Gemm(A, B, transB=1) with constant B : transpose B once now, not every run
//   fuse_matmul_add       MatMul(A,B) -> t ; Add(t, bias) -> y   ==>   Gemm(A, B, bias) -> y
//                         (GUARDED: only when `bias` is provably a bias vector -- see the .cpp)
//   eliminate_dead_nodes  remove nodes whose results never reach a graph output
//   drop_unused_weights   free initializers nothing reads any more
// =====================================================================================
#pragma once
#include "stage2_graph/graph.hpp"

#include <string>

namespace mo {

struct OptimizerStats {
    int identities_removed = 0;
    int transposes_folded = 0;
    int gemms_fused = 0;
    int dead_nodes_removed = 0;
    int weights_dropped = 0;
    std::string to_string() const;
};

int eliminate_identity(Graph &g);
int fold_gemm_transposes(Graph &g);
int fuse_matmul_add(Graph &g);
int eliminate_dead_nodes(Graph &g);
int drop_unused_weights(Graph &g);

// Run all passes in a sensible order and re-validate the result.
OptimizerStats optimize(Graph &g);

} // namespace mo
