// =====================================================================================
// STAGE 4 · session.hpp  --  InferenceSession: load() once, run() many times, from many threads
// =====================================================================================
//   load(path)                                       run(feeds)           (thread-safe, const)
//   ───────────────────────────────────────          ─────────────────────────────────────
//   Stage 1  bytes -> ModelProto, decode weights      make a private RunState
//   Stage 2  Graph: intern, validate, optimize        put weights + feeds in slots
//   Stage 3  create EPs, partition nodes              execute nodes (sequential / parallel)
//   Stage 4  ExecutionPlan + thread pools             return the graph outputs
//
// THREE KINDS OF PARALLELISM (same knobs as ONNX Runtime's SessionOptions):
//   intra_op_threads            split ONE kernel (one big matrix multiply) across threads
//   inter_op_threads + Parallel run INDEPENDENT NODES (graph branches) at the same time
//   many callers                many threads call run() on ONE shared session at once
// =====================================================================================
#pragma once
#include "stage2_graph/optimizer.hpp"
#include "stage4_runtime/plan.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mo {

enum class ExecutionMode { Sequential, Parallel };

struct SessionOptions {
    int intra_op_threads = 1;                          // threads INSIDE one kernel
    int inter_op_threads = 1;                          // threads running different NODES
    ExecutionMode execution_mode = ExecutionMode::Sequential;
    bool enable_optimizations = true;                  // run the Stage 2 passes?
    std::vector<std::string> providers = {"cpu"};      // EP priority list; "cpu" is always appended
    bool verbose = false;                              // print what load() does
};

using Feeds = std::unordered_map<std::string, Tensor>;

struct NodeEvent {
    int node;
    std::string name, op, ep;
    int thread;
    double start_us, end_us;
};
struct RunProfile {
    std::vector<NodeEvent> events;
    int peak_live_values = 0;        // most activations alive at the same moment
    int final_live_values = 0;       // still alive when run() finished (== number of outputs)
    double total_us = 0;
};

class InferenceSession {
public:
    explicit InferenceSession(SessionOptions opts = {});
    ~InferenceSession();
    InferenceSession(const InferenceSession &) = delete;
    InferenceSession &operator=(const InferenceSession &) = delete;

    void load(const std::string &path);
    void load_model(const ModelProto &model, const std::string &base_dir = ".");

    // THREAD-SAFE. Throws std::runtime_error for bad feeds or kernel errors; the session
    // stays usable after a failed run.
    Feeds run(const Feeds &feeds, RunProfile *profile = nullptr) const;

    bool loaded() const { return loaded_; }
    const Graph &graph() const { return graph_; }
    const ExecutionPlan &plan() const { return plan_; }
    const SessionOptions &options() const { return opts_; }
    const OptimizerStats &optimizer_stats() const { return stats_; }
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;
    std::string describe() const;                      // which EP runs which nodes

private:
    struct RunState;                                   // defined in session.cpp
    void init_state(RunState &st, const Feeds &feeds, RunProfile *profile) const;
    void exec_node(RunState &st, int idx) const;
    void run_parallel(const std::shared_ptr<RunState> &stp) const;
    void push_ready(const std::shared_ptr<RunState> &stp, const std::vector<int> &nodes) const;
    void helper(const std::shared_ptr<RunState> &stp) const;
    void process(const std::shared_ptr<RunState> &stp, int idx) const;
    Feeds collect(RunState &st, RunProfile *profile) const;

    SessionOptions opts_;
    bool loaded_ = false;
    Graph graph_;
    OptimizerStats stats_;
    std::vector<std::unique_ptr<ExecutionProvider>> eps_;
    ExecutionPlan plan_;
    std::unique_ptr<ThreadPool> intra_, inter_;
};

} // namespace mo
