// =====================================================================================
// STAGE 4 · session.cpp  --  load(), and the two executors behind run()
// =====================================================================================
#include "stage4_runtime/session.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <exception>
#include <sstream>

namespace mo {

InferenceSession::InferenceSession(SessionOptions opts) : opts_(std::move(opts)) {}
InferenceSession::~InferenceSession() = default;   // inter_ is destroyed first: it joins its
                                                   // workers while plan_/eps_ are still alive

void InferenceSession::load(const std::string &path) {
    std::string base_dir;
    ModelProto m = load_model_file(path, &base_dir);                 // Stage 1: bytes -> ModelProto
    load_model(m, base_dir);
}

void InferenceSession::load_model(const ModelProto &model, const std::string &base_dir) {
    // ---- Stage 1 + 2: weights + graph (validated) ----
    graph_ = Graph::from_model(model, base_dir);
    const size_t before = graph_.nodes.size();
    if (opts_.enable_optimizations) stats_ = optimize(graph_);
    if (opts_.verbose) {
        std::printf("[load] graph '%s': %zu nodes -> %zu after optimization (%s)\n", graph_.graph_name.c_str(), before,
                    graph_.nodes.size(), stats_.to_string().c_str());
        std::printf("[load] weights:\n%s", graph_.weights.summary().c_str());
    }

    // ---- Stage 3: create the requested EPs (+ cpu fallback) and partition ----
    eps_.clear();
    std::vector<std::string> names = opts_.providers;
    if (std::find(names.begin(), names.end(), "cpu") == names.end()) names.push_back("cpu");
    std::vector<const ExecutionProvider *> priority;
    std::map<std::string, const ExecutionProvider *> by_name;
    for (auto &n : names) {
        auto ep = create_execution_provider(n);
        if (!ep) {
            if (opts_.verbose) std::printf("[load] EP '%s' is not compiled into this build: skipped\n", n.c_str());
            continue;
        }
        if (!ep->available()) {
            if (opts_.verbose) std::printf("[load] EP '%s' is not available on this machine: skipped\n", n.c_str());
            continue;
        }
        priority.push_back(ep.get());
        by_name[ep->name()] = ep.get();
        eps_.push_back(std::move(ep));
    }
    assign_providers(graph_, priority);
    for (auto &ep : eps_) cuda_ep_register_weights(ep.get(), graph_);
    if (opts_.verbose) std::printf("%s", describe().c_str());

    // ---- Stage 4: freeze into a plan, create the pools ----
    plan_ = build_plan(graph_, by_name);
    intra_ = std::make_unique<ThreadPool>(opts_.intra_op_threads);
    inter_ = std::make_unique<ThreadPool>(opts_.inter_op_threads);
    loaded_ = true;
}

std::vector<std::string> InferenceSession::input_names() const {
    std::vector<std::string> r;
    for (int v : plan_.graph_inputs) r.push_back(plan_.value_names[size_t(v)]);
    return r;
}
std::vector<std::string> InferenceSession::output_names() const {
    std::vector<std::string> r;
    for (int v : plan_.graph_outputs) r.push_back(plan_.value_names[size_t(v)]);
    return r;
}

std::string InferenceSession::describe() const {
    std::map<std::string, std::map<std::string, int>> count;           // ep -> op -> n
    for (auto &n : graph_.nodes) count[n.ep][n.op_type]++;
    std::ostringstream os;
    os << "[load] partition:";
    for (auto &kv : count) {
        os << "  " << kv.first << "{";
        bool first = true;
        for (auto &op : kv.second) { os << (first ? "" : " ") << op.first << "x" << op.second; first = false; }
        os << "}";
    }
    os << "\n";
    return os.str();
}

// ============================================================================ RunState
// Everything that changes during ONE run() call. Heap-allocated and shared (shared_ptr)
// with the helper tasks, because a helper may start after run() has returned -- it then
// finds an empty queue and leaves, but it must find valid memory, not a dead stack frame.
struct InferenceSession::RunState {
    std::vector<std::shared_ptr<const Tensor>> slots;         // value id -> tensor (null = not produced / freed)
    std::unique_ptr<std::atomic<int>[]> remaining;            // value id -> readers still to come
    std::unique_ptr<std::atomic<int>[]> indeg;                // node -> unmet input edges
    std::atomic<int> pending{0};                              // nodes not yet finished
    std::atomic<int> live{0}, peak{0};                        // memory statistics
    std::atomic<bool> failed{false};
    std::mutex m;                                             // guards ready_q, first_error, cv
    std::condition_variable cv;
    std::deque<int> ready_q;
    std::exception_ptr first_error;
    RunProfile *prof = nullptr;
    std::mutex pm;                                            // guards prof->events and tids
    std::unordered_map<std::thread::id, int> tids;
    std::chrono::steady_clock::time_point t0;

    void bump_live() {                                        // lock-free "maximum so far"
        const int l = live.fetch_add(1) + 1;
        int p = peak.load();
        while (l > p && !peak.compare_exchange_weak(p, l)) {}
    }
    double now_us() const {
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }
    int thread_index() {                                      // caller holds pm
        auto id = std::this_thread::get_id();
        auto it = tids.find(id);
        if (it != tids.end()) return it->second;
        const int k = int(tids.size());
        tids[id] = k;
        return k;
    }
};

Feeds InferenceSession::run(const Feeds &feeds, RunProfile *profile) const {
    if (!loaded_) throw std::runtime_error("run() called before load()");   // (original: null-pointer crash)
    auto stp = std::make_shared<RunState>();
    init_state(*stp, feeds, profile);
    if (opts_.execution_mode == ExecutionMode::Sequential) {
        for (int idx : plan_.topo_order) exec_node(*stp, idx);
    } else {
        run_parallel(stp);
    }
    return collect(*stp, profile);
}

void InferenceSession::init_state(RunState &st, const Feeds &feeds, RunProfile *profile) const {
    const size_t V = plan_.value_names.size(), N = plan_.nodes.size();
    st.t0 = std::chrono::steady_clock::now();
    st.prof = profile;

    // 1. validate the feeds BEFORE touching anything
    for (auto &kv : feeds) {
        auto it = plan_.value_id.find(kv.first);
        if (it == plan_.value_id.end() ||
            std::find(plan_.graph_inputs.begin(), plan_.graph_inputs.end(), it->second) == plan_.graph_inputs.end())
            throw std::runtime_error("unknown input '" + kv.first + "'");
        const Tensor &t = kv.second;
        if (!t.valid()) throw std::runtime_error("input '" + kv.first + "' has no data");
        auto info = graph_.input_info.find(it->second);
        if (info != graph_.input_info.end() && info->second.has_shape) {
            const ValueInfo &vi = info->second;
            bool ok = int(vi.dims.size()) == t.rank();
            for (size_t d = 0; ok && d < vi.dims.size(); d++)
                if (vi.dims[d] >= 0 && vi.dim_names[d].empty() && vi.dims[d] != t.dim(int(d))) ok = false;
            if (!ok) {
                std::string want = "[";
                for (size_t d = 0; d < vi.dims.size(); d++)
                    want += (d ? "," : "") + (vi.dims[d] >= 0 && vi.dim_names[d].empty() ? std::to_string(vi.dims[d])
                                                                                         : (vi.dim_names[d].empty() ? "?" : vi.dim_names[d]));
                throw std::runtime_error("input '" + kv.first + "' expects shape " + want + "] but got " + shape_str(t.shape()));
            }
        }
    }
    // 2. size the per-run arrays from the (immutable) plan
    st.slots.assign(V, nullptr);
    st.remaining.reset(new std::atomic<int>[V]);
    for (size_t v = 0; v < V; v++) st.remaining[v].store(plan_.consumer_count[v], std::memory_order_relaxed);
    st.indeg.reset(new std::atomic<int>[N]);
    for (size_t n = 0; n < N; n++) st.indeg[n].store(plan_.indegree[n], std::memory_order_relaxed);
    st.pending.store(int(N));

    // 3. weights: shared, read-only, never freed
    for (size_t v = 0; v < V; v++) if (plan_.weights[v]) st.slots[v] = plan_.weights[v];
    // 4. feeds: sharing the caller's buffer is safe (kernels never write to their inputs)
    for (int v : plan_.graph_inputs) {
        auto it = feeds.find(plan_.value_names[size_t(v)]);
        if (it == feeds.end()) throw std::runtime_error("missing input '" + plan_.value_names[size_t(v)] + "'");
        st.slots[size_t(v)] = std::make_shared<const Tensor>(it->second);
        st.bump_live();
        if (plan_.consumer_count[size_t(v)] == 0 && !plan_.persistent[size_t(v)]) { st.slots[size_t(v)].reset(); st.live--; }
    }
}

// Run one node: gather inputs, call the kernel, publish the output, free dead inputs.
void InferenceSession::exec_node(RunState &st, int idx) const {
    const PlanNode &n = plan_.nodes[size_t(idx)];
    std::vector<const Tensor *> inputs;
    inputs.reserve(n.in.size());
    for (int v : n.in) {
        if (v == kNoValue) { inputs.push_back(nullptr); continue; }
        const Tensor *p = st.slots[size_t(v)].get();
        if (!p) throw std::logic_error("internal: input '" + plan_.value_names[size_t(v)] + "' not ready for node '" + n.name + "'");
        inputs.push_back(p);
    }

    const double start = st.prof ? st.now_us() : 0;
    Tensor result = (*n.kernel)(KernelContext{intra_.get()}, inputs, n.attrs);
    const double end = st.prof ? st.now_us() : 0;
    if (st.prof) {
        std::lock_guard<std::mutex> lk(st.pm);
        st.prof->events.push_back({idx, n.name, n.op_type, n.ep_name, st.thread_index(), start, end});
    }

    const int out = n.out[0];
    st.slots[size_t(out)] = std::make_shared<const Tensor>(std::move(result));
    st.bump_live();
    if (plan_.consumer_count[size_t(out)] == 0 && !plan_.persistent[size_t(out)]) { st.slots[size_t(out)].reset(); st.live--; }

    // Release inputs AFTER the kernel is done with them. Whoever drops a count to zero frees
    // the tensor; every other reader has already decremented, so nobody still uses it.
    for (int v : n.in)
        if (v != kNoValue && !plan_.persistent[size_t(v)] &&
            st.remaining[size_t(v)].fetch_sub(1, std::memory_order_acq_rel) == 1) {
            st.slots[size_t(v)].reset();
            st.live--;
        }
}

// ============================================================================ parallel executor
// Dependency counting: indeg[n] = unmet input edges. A node is READY at 0. Finishing a node
// decrements its dependents; those reaching 0 become ready. Ready nodes are independent by
// construction, so they can run at the same time on different threads.
//
// Ready nodes go into a PER-RUN queue, drained by the CALLER of run() and by helper tasks
// on the inter-op pool. (Per-run, so a caller only ever runs nodes of its own request.)
void InferenceSession::push_ready(const std::shared_ptr<RunState> &stp, const std::vector<int> &nodes) const {
    RunState &st = *stp;
    {
        std::lock_guard<std::mutex> lk(st.m);
        for (int n : nodes) st.ready_q.push_back(n);
    }
    st.cv.notify_all();                                        // wake the caller if it waits for work
    const int helpers = std::min<int>(int(nodes.size()), inter_->num_workers());
    for (int i = 0; i < helpers; i++) inter_->submit([this, stp] { helper(stp); });
}

void InferenceSession::helper(const std::shared_ptr<RunState> &stp) const {
    RunState &st = *stp;
    for (;;) {
        int idx;
        {
            std::lock_guard<std::mutex> lk(st.m);
            if (st.ready_q.empty()) return;                    // no work: exit (don't sleep in the pool)
            idx = st.ready_q.back();
            st.ready_q.pop_back();
        }
        process(stp, idx);
    }
}

// Run node idx, release its dependents, keep ONE newly-ready dependent on this thread (good
// cache locality, no queue round-trip) and enqueue the rest for whoever is free.
void InferenceSession::process(const std::shared_ptr<RunState> &stp, int idx) const {
    RunState &st = *stp;
    for (;;) {
        if (!st.failed.load(std::memory_order_relaxed)) {
            try { exec_node(st, idx); }
            catch (...) {
                std::lock_guard<std::mutex> lk(st.m);
                if (!st.failed.exchange(true)) st.first_error = std::current_exception();
            }
        }
        std::vector<int> ready;
        for (int d : plan_.dependents[size_t(idx)])
            if (st.indeg[size_t(d)].fetch_sub(1, std::memory_order_acq_rel) == 1) ready.push_back(d);

        // Retire AFTER the dependents are counted, so `pending` reaches 0 only when truly done.
        if (st.pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(st.m);              // lock before notify: no lost wake-up
            st.cv.notify_all();
        }
        if (ready.empty()) return;
        const int keep = ready.back();
        ready.pop_back();
        if (!ready.empty()) push_ready(stp, ready);
        idx = keep;
    }
}

void InferenceSession::run_parallel(const std::shared_ptr<RunState> &stp) const {
    RunState &st = *stp;
    std::vector<int> roots;
    for (size_t i = 0; i < plan_.nodes.size(); i++) if (plan_.indegree[i] == 0) roots.push_back(int(i));
    if (roots.empty()) return;                                  // empty graph
    push_ready(stp, roots);
    for (;;) {                                                  // the caller works too
        int idx;
        {
            std::unique_lock<std::mutex> lk(st.m);
            st.cv.wait(lk, [&] { return !st.ready_q.empty() || st.pending.load(std::memory_order_acquire) == 0; });
            if (st.ready_q.empty()) break;                      // nothing queued, nothing pending: done
            idx = st.ready_q.back();
            st.ready_q.pop_back();
        }
        process(stp, idx);
    }
    std::exception_ptr e;
    {
        std::lock_guard<std::mutex> lk(st.m);
        e = std::move(st.first_error);                          // only the caller owns it from now on
    }
    if (e) std::rethrow_exception(e);
}

Feeds InferenceSession::collect(RunState &st, RunProfile *profile) const {
    Feeds out;
    for (int v : plan_.graph_outputs) {
        const auto &slot = st.slots[size_t(v)];
        if (!slot) throw std::logic_error("internal: output '" + plan_.value_names[size_t(v)] + "' missing");
        if (plan_.weights[size_t(v)]) {                         // never hand out a writable view of a weight
            out[plan_.value_names[size_t(v)]] = Tensor::from(slot->shape(), slot->to_vector());
        } else {
            out[plan_.value_names[size_t(v)]] = *slot;
        }
    }
    if (profile) {
        profile->peak_live_values = st.peak.load();
        profile->final_live_values = st.live.load();
        profile->total_us = st.now_us();
    }
    return out;
}

} // namespace mo
