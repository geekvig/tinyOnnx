// =====================================================================================
// STAGE 4 · plan.hpp  --  ExecutionPlan: the frozen, integer-indexed graph that Run() uses
// =====================================================================================
// After optimizing and partitioning, everything that never changes again is precomputed
// into an ExecutionPlan. It is IMMUTABLE: built once in load(), then only read -- by any
// number of threads at the same time, with no locks. That is the golden rule:
//
//      After load(), nothing in the session ever changes.
//      Everything that changes during a Run() lives in that Run's private RunState.
//
// What the executors need, per node and per value:
//   kernel[n]          which function to call (already looked up in the chosen EP)
//   indegree[n]        how many input edges of n come from other nodes
//   dependents[n]      who is waiting for n (one entry per edge)
//   consumer_count[v]  how many input slots read v -> when it drops to 0, free v
//   persistent[v]      weights and graph outputs are never freed
// =====================================================================================
#pragma once
#include "stage3_ep/execution_provider.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mo {

struct PlanNode {
    std::string op_type, name, ep_name;
    std::vector<int> in, out;          // value ids (kNoValue for an omitted optional input)
    Attributes attrs;
    const KernelFn *kernel = nullptr;  // points into the EP's kernel table (EPs outlive the plan)
};

struct ExecutionPlan {
    std::vector<std::string> value_names;
    std::unordered_map<std::string, int> value_id;
    std::vector<PlanNode> nodes;
    std::vector<int> topo_order;
    std::vector<std::vector<int>> dependents;
    std::vector<int> indegree;
    std::vector<int> consumer_count;
    std::vector<char> persistent;
    std::vector<int> graph_inputs, graph_outputs;
    std::vector<std::shared_ptr<const Tensor>> weights;   // value id -> weight, or null
};

// `eps` maps an EP name to the EP object that will run that EP's nodes.
ExecutionPlan build_plan(const Graph &g, const std::map<std::string, const ExecutionProvider *> &eps);

} // namespace mo
