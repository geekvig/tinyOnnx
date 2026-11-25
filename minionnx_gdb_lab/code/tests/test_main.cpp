// =====================================================================================
// tests/test_main.cpp  --  MiniOnnx unit tests (no framework needed: CHECK + counters)
// =====================================================================================
// Run from the project root (ctest does this for you):   ./build/test_minionnx
// Each group prints "ok" or the failing checks with file:line.
// =====================================================================================
#include "../tools/model_zoo.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <thread>

using namespace mo;

static int g_failures = 0, g_checks = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        g_checks++;                                                                       \
        if (!(cond)) {                                                                    \
            g_failures++;                                                                 \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);               \
        }                                                                                 \
    } while (0)

template <class F>
static bool throws_with(F &&f, const std::string &needle) {
    try { f(); } catch (const std::exception &e) {
        if (std::string(e.what()).find(needle) != std::string::npos) return true;
        std::printf("    (threw a different message: %s)\n", e.what());
        return false;
    }
    return false;
}

static bool same_bits(const Tensor &a, const Tensor &b) {
    // memcmp, not ==:  NaN == NaN is false, but a NaN must still round-trip bit for bit
    return a.shape() == b.shape() && std::memcmp(a.data(), b.data(), size_t(a.numel()) * sizeof(float)) == 0;
}
static double max_diff(const Tensor &a, const Tensor &b) {
    if (a.shape() != b.shape()) return 1e30;
    double m = 0;
    for (int64_t i = 0; i < a.numel(); i++) m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}
static Tensor random_tensor(Shape s, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0, 1);
    Tensor t = Tensor::empty(s);
    for (int64_t i = 0; i < t.numel(); i++) t[i] = nd(rng);
    return t;
}
static Feeds feed_x(const Tensor &t) { return {{"x", t}}; }

static void group(const char *name) { std::printf("[%s]\n", name); }
static int s_before = 0;
static void begin() { s_before = g_failures; }
static void end_group() { if (g_failures == s_before) std::printf("    ok\n"); }

// ------------------------------------------------------------------------------ 1
static void test_thread_pool() {
    group("1 thread pool: coverage, nesting, exceptions"); begin();
    for (int n : {1, 2, 4, 8}) {
        ThreadPool pool(n);
        std::vector<std::atomic<int>> hits(10007);
        pool.parallel_for(0, 10007, 97, [&](int64_t b, int64_t e) { for (int64_t i = b; i < e; i++) hits[size_t(i)]++; });
        bool once = std::all_of(hits.begin(), hits.end(), [](auto &h) { return h.load() == 1; });
        CHECK(once);
        std::atomic<int> total{0};                                    // nested parallel_for must not deadlock
        pool.parallel_for(0, 8, 1, [&](int64_t, int64_t) {
            pool.parallel_for(0, 100, 10, [&](int64_t b, int64_t e) { total += int(e - b); });
        });
        CHECK(total == 800);
        CHECK(throws_with([&] {                                       // a throw on any thread reaches the caller
            pool.parallel_for(0, 1000, 10, [&](int64_t b, int64_t) { if (b == 500) throw std::runtime_error("boom 500"); });
        }, "boom 500"));
    }
    end_group();
}

// ------------------------------------------------------------------------------ 2
static void test_wire() {
    group("2 protobuf wire format"); begin();
    WireWriter w;
    w.varint(300);
    CHECK(w.bytes() == std::string("\xAC\x02", 2));                  // the textbook example
    WireWriter w2;
    for (uint64_t v : {0ull, 1ull, 127ull, 128ull, 16383ull, 16384ull, 0xFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull}) w2.varint(v);
    WireReader r(w2.bytes());
    for (uint64_t v : {0ull, 1ull, 127ull, 128ull, 16383ull, 16384ull, 0xFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull}) CHECK(r.read_varint() == v);
    CHECK(r.at_end());
    const std::string trunc("\x80\x80", 2), past("\x0A\x05" "ab", 4);       // named: a reader never owns bytes
    CHECK(throws_with([&] { WireReader t(trunc); t.read_varint(); }, "truncated varint"));
    CHECK(throws_with([&] { WireReader t(past); uint32_t f; WireType wt; t.next_field(f, wt); t.read_bytes(); }, "runs past end"));
    // unknown fields are skipped; packed and unpacked repeated ints both parse
    WireWriter m;
    m.field_varint(99, 7);                       // unknown varint field
    m.field_bytes(98, "zzz");                    // unknown LEN field
    m.field_varint(1, 5);                        // unpacked int
    WireWriter packed;
    packed.varint(6); packed.varint(7);
    m.field_bytes(1, packed.bytes());            // packed ints
    WireReader mr(m.bytes());
    std::vector<int64_t> ints;
    uint32_t f; WireType wt;
    while (mr.next_field(f, wt)) { if (f == 1) mr.read_repeated_int64(wt, ints); else mr.skip(wt); }
    CHECK((ints == std::vector<int64_t>{5, 6, 7}));
    end_group();
}

// ------------------------------------------------------------------------------ 3
static void test_onnx_roundtrip() {
    group("3 .onnx serialize/parse round trip + weight encodings"); begin();
    ModelProto m = zoo::make_classifier(6, 10, 3);
    m.graph.initializer[1].float_data[0] = 1e-40f;                    // a denormal (broke the old text format)
    m.graph.initializer[1].float_data[1] = std::nanf("");
    ModelProto back = parse_model(serialize_model(m));
    CHECK(back.ir_version == m.ir_version && back.producer_name == m.producer_name);
    CHECK(back.opset_import.size() == 1 && back.opset_import[0].version == 13);
    CHECK(back.graph.node.size() == m.graph.node.size());
    for (size_t i = 0; i < m.graph.node.size(); i++) {
        CHECK(back.graph.node[i].op_type == m.graph.node[i].op_type);
        CHECK(back.graph.node[i].input == m.graph.node[i].input);
        CHECK(back.graph.node[i].attribute.size() == m.graph.node[i].attribute.size());
    }
    CHECK(back.graph.input[0].dim_names[0] == "N" && back.graph.input[0].dims[1] == 6);
    for (size_t i = 0; i < m.graph.initializer.size(); i++) {         // bit-exact, NaN included
        auto &a = m.graph.initializer[i].float_data, &b = back.graph.initializer[i].float_data;
        CHECK(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
    }
    // the three storage formats must decode to identical weights
    SaveOptions fd; fd.use_raw_data = false;
    SaveOptions rd;
    SaveOptions ex; ex.external_data = true; ex.external_threshold = 0;
    save_model_file(m, "test_tmp_fd.onnx", fd);
    save_model_file(m, "test_tmp_rd.onnx", rd);
    save_model_file(m, "test_tmp_ex.onnx", ex);
    std::string base;
    WeightStore a, b, c;
    a.load(load_model_file("test_tmp_fd.onnx", &base).graph, base);
    b.load(load_model_file("test_tmp_rd.onnx", &base).graph, base);
    c.load(load_model_file("test_tmp_ex.onnx", &base).graph, base);
    for (auto &kv : a.all()) {
        CHECK(same_bits(kv.second, *b.find(kv.first)));
        CHECK(same_bits(kv.second, *c.find(kv.first)));
    }
    CHECK(c.records()[0].source.rfind("external:", 0) == 0);
    for (const char *f : {"test_tmp_fd.onnx", "test_tmp_rd.onnx", "test_tmp_ex.onnx", "test_tmp_ex.onnx.data"}) std::remove(f);
    end_group();
}

// ------------------------------------------------------------------------------ 4
static void test_weights() {
    group("4 weight store: alignment and validation"); begin();
    ModelProto m = zoo::make_branchy(3, 5, 7);
    WeightStore ws;
    ws.load(m.graph);
    for (auto &kv : ws.all()) CHECK(reinterpret_cast<uintptr_t>(kv.second.data()) % 64 == 0);
    ModelProto bad = m;
    bad.graph.initializer[0].float_data.pop_back();
    CHECK(throws_with([&] { WeightStore w; w.load(bad.graph); }, "float_data has"));
    bad = m;
    bad.graph.initializer[0].float_data.clear();
    bad.graph.initializer[0].raw_data = "abc";
    CHECK(throws_with([&] { WeightStore w; w.load(bad.graph); }, "raw_data has 3 bytes"));
    bad = m;
    bad.graph.initializer[0].data_location = 1;
    bad.graph.initializer[0].external_data = {{"location", "../../etc/passwd"}};
    CHECK(throws_with([&] { WeightStore w; w.load(bad.graph); }, "unsafe external data location"));
    bad.graph.initializer[0].external_data = {{"location", "no_such_file.bin"}};
    CHECK(throws_with([&] { WeightStore w; w.load(bad.graph); }, "cannot open external file"));
    bad = m;
    bad.graph.initializer[0].data_type = DataType::Int64;
    CHECK(throws_with([&] { WeightStore w; w.load(bad.graph); }, "not supported"));
    end_group();
}

// ------------------------------------------------------------------------------ 5
static void test_graph() {
    group("5 graph: order-independence and validation"); begin();
    ModelProto m = zoo::make_branchy(3, 8, 16);
    ModelProto rev = m;
    std::reverse(rev.graph.node.begin(), rev.graph.node.end());
    InferenceSession s1, s2;
    s1.load_model(m);
    s2.load_model(rev);
    Tensor x = random_tensor({4, 8}, 1);
    CHECK(same_bits(s1.run(feed_x(x)).at("out"), s2.run(feed_x(x)).at("out")));

    ModelProto bad = zoo::make_xor();
    bad.graph.node[2].input[0] = "nobody_makes_this";
    CHECK(throws_with([&] { Graph::from_model(bad); }, "dangling reference"));
    bad = zoo::make_xor();
    bad.graph.node[0].input[0] = "out";                                // out depends on itself
    bad.graph.input.clear();
    CHECK(throws_with([&] { Graph::from_model(bad); }, "cycle"));
    bad = zoo::make_xor();
    bad.graph.node[1].output[0] = "z1";                                // two producers of z1
    CHECK(throws_with([&] { Graph::from_model(bad); }, "more than one producer"));
    bad = zoo::make_xor();
    bad.graph.output[0].name = "ghost";
    CHECK(throws_with([&] { Graph::from_model(bad); }, "never produced"));
    bad = zoo::make_xor();
    bad.graph.node[2].domain = "com.example";
    CHECK(throws_with([&] { Graph::from_model(bad); }, "custom operator domain"));
    end_group();
}

// ------------------------------------------------------------------------------ 6
static ModelProto matmul_add(const std::string &kind) {
    ModelProto m = zoo::empty_model("mm_add");
    auto &g = m.graph;
    std::mt19937 rng(9);
    g.initializer.push_back(zoo::random_tensor("W", {4, 6}, 1.f, rng));
    g.input = {zoo::batch_input("x", 4)};
    g.node.push_back(zoo::node("MatMul", "mm", {"x", "W"}, {"t"}));
    if (kind == "bias" || kind == "bias1d" || kind == "wrongwidth") {
        auto dims = kind == "bias" ? std::vector<int64_t>{1, 6} : kind == "bias1d" ? std::vector<int64_t>{6} : std::vector<int64_t>{1, 5};
        g.initializer.push_back(zoo::random_tensor("b", dims, 1.f, rng));
        g.node.push_back(zoo::node("Add", "add", {"t", "b"}, {"y"}));
        if (kind == "wrongwidth") g.node.back().op_type = "Add";
    } else if (kind == "sameshape") {                                   // branch merge: NOT a bias
        g.initializer.push_back(zoo::random_tensor("W2", {4, 6}, 1.f, rng));
        g.node.push_back(zoo::node("MatMul", "mm2", {"x", "W2"}, {"u"}));
        g.node.push_back(zoo::node("Add", "add", {"t", "u"}, {"y"}));
    } else if (kind == "twoconsumers") {
        g.initializer.push_back(zoo::random_tensor("b", {1, 6}, 1.f, rng));
        g.node.push_back(zoo::node("Add", "add", {"t", "b"}, {"y0"}));
        g.node.push_back(zoo::node("Relu", "r", {"t"}, {"y1"}));
        g.node.push_back(zoo::node("Add", "join", {"y0", "y1"}, {"y"}));
    } else if (kind == "matrix") {
        g.initializer.push_back(zoo::random_tensor("b", {3, 6}, 1.f, rng));
        g.node.push_back(zoo::node("Add", "add", {"t", "b"}, {"y"}));
    }
    g.output = {zoo::batch_input("y", 6)};
    return m;
}

static void test_optimizer() {
    group("6 optimizer: guarded fusion, transpose folding, dead code"); begin();
    auto fused_count = [](const ModelProto &m) { Graph g = Graph::from_model(m); return fuse_matmul_add(g); };
    CHECK(fused_count(matmul_add("bias")) == 1);
    CHECK(fused_count(matmul_add("bias1d")) == 1);
    CHECK(fused_count(matmul_add("sameshape")) == 0);
    CHECK(fused_count(matmul_add("twoconsumers")) == 0);
    CHECK(fused_count(matmul_add("matrix")) == 0);
    CHECK(fused_count(matmul_add("wrongwidth")) == 0);
    // fused and unfused must agree (to float rounding: the bias is added first instead of last)
    for (const char *k : {"bias", "bias1d", "sameshape", "twoconsumers"}) {
        SessionOptions on, off;
        off.enable_optimizations = false;
        InferenceSession a(on), b(off);
        a.load_model(matmul_add(k));
        b.load_model(matmul_add(k));
        Tensor x = random_tensor({3, 4}, 2);
        CHECK(max_diff(a.run(feed_x(x)).at("y"), b.run(feed_x(x)).at("y")) < 1e-5);
    }
    // PyTorch-style Gemm(transB=1): the optimizer pre-transposes the weight; same answer
    {
        SessionOptions off;
        off.enable_optimizations = false;
        InferenceSession a, b(off);
        a.load_model(zoo::make_classifier(6, 10, 3));
        b.load_model(zoo::make_classifier(6, 10, 3));
        CHECK(a.optimizer_stats().transposes_folded == 2);
        CHECK(a.optimizer_stats().weights_dropped == 2);                // the untransposed copies
        Tensor x = random_tensor({5, 6}, 3);
        CHECK(max_diff(a.run(feed_x(x)).at("probs"), b.run(feed_x(x)).at("probs")) < 1e-6);
    }
    // dead node + identity elimination
    {
        ModelProto m = zoo::make_xor();
        m.graph.node.push_back(zoo::node("Relu", "dead", {"h"}, {"unused"}));
        m.graph.node[2].output[0] = "h_pre";
        m.graph.node.push_back(zoo::node("Identity", "id", {"h_pre"}, {"h"}));
        Graph g = Graph::from_model(m);
        OptimizerStats st = optimize(g);
        CHECK(st.identities_removed == 1);
        CHECK(st.dead_nodes_removed == 1);                              // the dead Relu (Identity counted above)
        CHECK(st.gemms_fused == 2);
        CHECK(g.nodes.size() == 4);
    }
    end_group();
}

// ------------------------------------------------------------------------------ 7
static void ref_gemm(const Tensor &A, const Tensor &B, const Tensor *C, float alpha, float beta, std::vector<double> &out) {
    const int64_t M = A.dim(0), K = A.dim(1), N = B.dim(1);
    out.assign(size_t(M * N), 0.0);
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            double s = 0;
            for (int64_t p = 0; p < K; p++) s += double(A[i * K + p]) * double(B[p * N + j]);
            double bias = 0;
            if (C) {
                if (C->numel() == 1) bias = (*C)[0];
                else if (C->rank() == 2 && C->dim(0) == M && C->dim(1) == N) bias = (*C)[i * N + j];
                else if (C->rank() == 2 && C->dim(1) == 1) bias = (*C)[i];
                else bias = (*C)[j];
            }
            out[size_t(i * N + j)] = alpha * s + beta * bias;
        }
}

static void test_kernels() {
    group("7 kernels: every available EP vs a double-precision reference"); begin();
    ThreadPool pool1(1), pool4(4);
    for (auto &[name, avail] : list_execution_providers()) {
        if (!avail) continue;
        auto ep = create_execution_provider(name);
        int shapes = 0;
        for (auto [M, K, N] : std::vector<std::array<int64_t, 3>>{{1, 1, 1}, {5, 7, 3}, {4, 16, 16}, {17, 33, 65}, {64, 128, 40}, {3, 300, 9}}) {
            Tensor A = random_tensor({M, K}, unsigned(M * 7 + K)), B = random_tensor({K, N}, unsigned(N * 3 + 1));
            std::vector<Tensor> biases = {Tensor(), random_tensor({N}, 5), random_tensor({1}, 6), random_tensor({M, 1}, 7), random_tensor({M, N}, 8)};
            for (auto &C : biases) {
                const bool has_c = C.numel() > 0;
                Attributes at;
                at.set_float("alpha", 0.5f);
                at.set_float("beta", 2.f);
                std::vector<const Tensor *> in = {&A, &B};
                if (has_c) in.push_back(&C);
                Tensor y1 = ep->kernel("Gemm")(KernelContext{&pool1}, in, at);
                Tensor y4 = ep->kernel("Gemm")(KernelContext{&pool4}, in, at);
                std::vector<double> ref;
                ref_gemm(A, B, has_c ? &C : nullptr, 0.5f, 2.f, ref);
                double err = 0;
                for (int64_t i = 0; i < y1.numel(); i++) err = std::max(err, std::fabs(y1[i] - ref[size_t(i)]) / (1.0 + std::fabs(ref[size_t(i)])));
                CHECK(err < 1e-4 * std::sqrt(double(K)));
                CHECK(same_bits(y1, y4));                                // thread count never changes the result
                shapes++;
            }
        }
        // elementwise ops: compare against the cpu EP
        auto cpu = make_cpu_ep();
        Tensor a = random_tensor({7, 37}, 11), row = random_tensor({37}, 12), sc = random_tensor({1}, 13);
        for (const char *op : {"Add", "Mul", "Relu"})
            if (ep->ops().end() != std::find(ep->ops().begin(), ep->ops().end(), op)) {
                for (const Tensor *other : {&a, &row, &sc}) {
                    std::vector<const Tensor *> in = std::string(op) == "Relu" ? std::vector<const Tensor *>{&a}
                                                                              : std::vector<const Tensor *>{&a, other};
                    CHECK(max_diff(ep->kernel(op)(KernelContext{&pool4}, in, {}), cpu->kernel(op)(KernelContext{&pool1}, in, {})) < 1e-6);
                    std::vector<const Tensor *> rev = {other, &a};               // small operand first
                    if (std::string(op) != "Relu")
                        CHECK(max_diff(ep->kernel(op)(KernelContext{&pool4}, rev, {}), cpu->kernel(op)(KernelContext{&pool1}, in, {})) < 1e-6);
                }
            }
        std::printf("    EP %-4s: %d gemm cases ok\n", name.c_str(), shapes);
    }
    auto cpu = make_cpu_ep();
    Tensor logits = random_tensor({4, 9}, 21);
    logits[0] = 1000.f;                                                  // would overflow a naive softmax
    Tensor p = cpu->kernel("Softmax")(KernelContext{&pool1}, {&logits}, {});
    for (int r = 0; r < 4; r++) {
        double s = 0;
        for (int j = 0; j < 9; j++) s += p[r * 9 + j];
        CHECK(std::fabs(s - 1.0) < 1e-5);
    }
    CHECK(std::isfinite(p[0]) && p[0] > 0.99f);
    Tensor two = Tensor::from({1}, {2.f}), v = Tensor::from({3}, {1.f, 2.f, 3.f});
    Tensor d = cpu->kernel("Sub")(KernelContext{&pool1}, {&two, &v}, {});   // 2 - v, not v - 2
    CHECK(d[0] == 1.f && d[2] == -1.f);
    end_group();
}

// ------------------------------------------------------------------------------ 8
static void test_partition() {
    group("8 partitioning between EPs"); begin();
    SessionOptions o;
    o.providers = {"x86", "arm", "cpu"};
    InferenceSession s(o);
    s.load_model(zoo::make_xor());
    auto x86 = create_execution_provider("x86"), arm = create_execution_provider("arm");
    const std::string simd = (x86 && x86->available()) ? "x86" : (arm && arm->available()) ? "arm" : "cpu";
    for (auto &n : s.graph().nodes) {
        if (n.op_type == "Sigmoid") CHECK(n.ep == "cpu");                // no SIMD Sigmoid kernel: falls back
        if (n.op_type == "Gemm") CHECK(n.ep == simd);
    }
    Tensor x = Tensor::from({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor y = s.run(feed_x(x)).at("out");
    CHECK(y[0] < 0.1f && y[1] > 0.9f && y[2] > 0.9f && y[3] < 0.1f);   // it still computes XOR
    ModelProto conv = zoo::make_xor();
    conv.graph.node[2].op_type = "Conv";
    CHECK(throws_with([&] { InferenceSession t; t.load_model(conv); }, "no execution provider can run node"));
    SessionOptions bogus;
    bogus.providers = {"tpu", "cuda"};                                  // unknown / maybe-not-compiled: skipped
    InferenceSession t(bogus);
    t.load_model(zoo::make_xor());
    for (auto &n : t.graph().nodes) CHECK(n.ep == "cpu" || n.ep == "cuda");
    end_group();
}

// ------------------------------------------------------------------------------ 9
static void test_executors() {
    group("9 sequential vs parallel executors: bit-identical"); begin();
    ModelProto m = zoo::make_branchy(4, 32, 64);
    InferenceSession ref;
    ref.load_model(m);
    Tensor x = random_tensor({16, 32}, 4);
    Tensor want = ref.run(feed_x(x)).at("out");
    int configs = 0;
    for (auto [inter, intra] : std::vector<std::pair<int, int>>{{1, 1}, {2, 1}, {4, 1}, {2, 2}, {8, 4}}) {
        SessionOptions o;
        o.execution_mode = ExecutionMode::Parallel;
        o.inter_op_threads = inter;
        o.intra_op_threads = intra;
        InferenceSession s(o);
        s.load_model(m);
        for (int r = 0; r < 30; r++) CHECK(same_bits(s.run(feed_x(x)).at("out"), want));
        configs++;
    }
    std::printf("    %d thread configurations x 30 runs\n", configs);
    end_group();
}

// ------------------------------------------------------------------------------ 10
static void test_memory() {
    group("10 memory: tensors freed as soon as their last reader finishes"); begin();
    for (auto mode : {ExecutionMode::Sequential, ExecutionMode::Parallel}) {
        SessionOptions o;
        o.execution_mode = mode;
        o.inter_op_threads = 4;
        InferenceSession s(o);
        s.load_model(zoo::make_chain(30));
        RunProfile p;
        s.run(feed_x(random_tensor({2, 16}, 5)), &p);
        CHECK(p.peak_live_values <= 3);
        CHECK(p.final_live_values == 1);
        // Mul(x, x): x is read twice by ONE node and must be freed exactly once
        ModelProto sq = zoo::empty_model("square");
        sq.graph.input = {zoo::batch_input("x", 3)};
        sq.graph.node = {zoo::node("Mul", "sq", {"x", "x"}, {"y"})};
        sq.graph.output = {zoo::batch_input("y", 3)};
        InferenceSession q(o);
        q.load_model(sq);
        RunProfile p2;
        Tensor y = q.run(feed_x(Tensor::from({1, 3}, {1, 2, 3})), &p2).at("y");
        CHECK(y[0] == 1.f && y[1] == 4.f && y[2] == 9.f);
        CHECK(p2.final_live_values == 1);
    }
    end_group();
}

// ------------------------------------------------------------------------------ 11
static void test_concurrency() {
    group("11 many threads calling run() on ONE shared session"); begin();
    for (auto mode : {ExecutionMode::Sequential, ExecutionMode::Parallel}) {
        SessionOptions o;
        o.execution_mode = mode;
        o.inter_op_threads = 2;
        o.intra_op_threads = 2;
        o.providers = {"x86", "arm", "cpu"};
        InferenceSession s(o);
        s.load_model(zoo::make_branchy(4, 32, 64));
        std::atomic<int> mismatches{0};
        std::vector<std::thread> clients;
        for (int c = 0; c < 8; c++)
            clients.emplace_back([&, c] {
                Tensor x = random_tensor({4, 32}, unsigned(100 + c));
                Tensor want = s.run(feed_x(x)).at("out");
                for (int i = 0; i < 40; i++) if (!same_bits(s.run(feed_x(x)).at("out"), want)) mismatches++;
            });
        for (auto &t : clients) t.join();
        CHECK(mismatches == 0);
    }
    end_group();
}

// ------------------------------------------------------------------------------ 12
static void test_errors() {
    group("12 errors are reported, never crash or hang"); begin();
    // A kernel throws (shape mismatch) on a worker thread -> rethrown on the caller
    ModelProto m = zoo::make_branchy(4, 8, 16);
    m.graph.input[0].has_shape = false;                                 // allow any shape to reach the kernel
    SessionOptions o;
    o.execution_mode = ExecutionMode::Parallel;
    o.inter_op_threads = 4;
    InferenceSession s(o);
    s.load_model(m);
    CHECK(throws_with([&] { s.run(feed_x(random_tensor({2, 5}, 1))); }, "incompatible shapes"));
    CHECK(s.run(feed_x(random_tensor({2, 8}, 1))).at("out").numel() == 16);   // still usable afterwards
    InferenceSession x;
    x.load_model(zoo::make_xor());
    CHECK(throws_with([&] { x.run(feed_x(Tensor::from({4, 3}, std::vector<float>(12, 0.f)))); }, "expects shape [N,2]"));
    CHECK(throws_with([&] { x.run({}); }, "missing input 'x'"));
    CHECK(throws_with([&] { x.run({{"x", Tensor::zeros({1, 2})}, {"y", Tensor::zeros({1})}}); }, "unknown input 'y'"));
    CHECK(throws_with([&] { Tensor::from({4, 2}, {1, 2, 3}); }, "needs 8 values, got 3"));   // the old heap overflow
    CHECK(throws_with([&] { InferenceSession e; e.run({}); }, "before load"));
    CHECK(throws_with([&] { parse_model("this is not protobuf"); }, "format error"));
    CHECK(throws_with([&] { InferenceSession e; e.load("models/does_not_exist.onnx"); }, "cannot open model file"));
    end_group();
}

// ------------------------------------------------------------------------------ 13
static void test_profile() {
    group("13 profile: every node starts after its producers finished"); begin();
    SessionOptions o;
    o.execution_mode = ExecutionMode::Parallel;
    o.inter_op_threads = 4;
    InferenceSession s(o);
    s.load_model(zoo::make_branchy(4, 32, 64));
    const ExecutionPlan &plan = s.plan();
    for (int r = 0; r < 20; r++) {
        RunProfile p;
        s.run(feed_x(random_tensor({8, 32}, 6)), &p);
        CHECK(p.events.size() == plan.nodes.size());
        std::vector<double> start(plan.nodes.size()), end(plan.nodes.size());
        for (auto &e : p.events) { start[size_t(e.node)] = e.start_us; end[size_t(e.node)] = e.end_us; }
        for (size_t n = 0; n < plan.nodes.size(); n++)
            for (int d : plan.dependents[n]) CHECK(start[size_t(d)] >= end[n]);
    }
    end_group();
}

// ------------------------------------------------------------------------------ 14
static void test_against_onnxruntime() {
    group("14 cross-check against reference outputs from real onnxruntime"); begin();
    int checked = 0;
    for (const char *name : {"xor", "classifier", "branchy", "branchy_external", "ort_linear", "ort_mlp"}) {
        const std::string model = std::string("models/") + name + ".onnx", ref = std::string("models/") + name + ".ref";
        std::ifstream f(ref);
        if (!f) continue;
        Feeds inputs, outputs;
        std::string kind, nm;
        while (f >> kind >> nm) {
            int rank;
            f >> rank;
            Shape sh(static_cast<size_t>(rank));
            for (auto &d : sh) f >> d;
            std::vector<float> v(size_t(shape_numel(sh)));
            for (auto &q : v) f >> q;
            (kind == "input" ? inputs : outputs)[nm] = Tensor::from(sh, v);
        }
        for (auto mode : {ExecutionMode::Sequential, ExecutionMode::Parallel}) {
            SessionOptions o;
            o.providers = {"x86", "arm", "cpu"};
            o.execution_mode = mode;
            o.inter_op_threads = 3;
            InferenceSession s(o);
            s.load(model);
            Feeds got = s.run(inputs);
            for (auto &kv : outputs) CHECK(max_diff(got.at(kv.first), kv.second) < 1e-4);
        }
        checked++;
    }
    std::printf("    %d model(s) compared%s\n", checked, checked ? "" : " (no .ref files: run python/crosscheck_ort.py)");
    end_group();
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);              // unbuffered: see output even if we crash
    std::printf("MiniOnnx tests. EPs in this build:");
    for (auto &ep : list_execution_providers()) std::printf(" %s(%s)", ep.first.c_str(), ep.second ? "available" : "not available");
    std::printf("\n");
    using TestFn = void (*)();
    for (TestFn t : {test_thread_pool, test_wire, test_onnx_roundtrip, test_weights, test_graph, test_optimizer, test_kernels, test_partition, test_executors, test_memory, test_concurrency, test_errors, test_profile, test_against_onnxruntime}) {
        try { t(); }
        catch (const std::exception &e) { g_failures++; std::printf("    FAIL: unexpected exception: %s\n", e.what()); }
        std::fflush(stdout);
    }
    std::printf("\n%s: %d checks, %d failures\n", g_failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
