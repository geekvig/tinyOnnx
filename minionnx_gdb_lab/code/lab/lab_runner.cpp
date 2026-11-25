// =====================================================================================
// lab/lab_runner.cpp  --  one scenario per bug of the GDB lab
// =====================================================================================
//   lab_runner <N>        run scenario N (1..20). With clean code every scenario prints
//                         "SCENARIO N: PASS". With bug N applied it fails, crashes or hangs --
//                         and your job is to find out WHY with gdb.
//   lab_runner all        run every scenario
//
// Run it from the code/ folder (it reads models/*.onnx).
// =====================================================================================
#include "../tools/model_zoo.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

using namespace mo;

namespace {

int g_fail = 0;
void expect(bool ok, const char *what) {
    if (!ok) { g_fail++; std::printf("   expected: %s  -> NOT TRUE\n", what); }
}
bool close(const Tensor &a, const Tensor &b, double tol = 1e-4) {
    if (a.shape() != b.shape()) return false;
    for (int64_t i = 0; i < a.numel(); i++) if (!(std::fabs(a[i] - b[i]) <= tol)) return false;   // NaN-safe
    return true;
}

// Reads a .ref file (input + expected output from onnxruntime) and checks the model.
bool check_reference(const std::string &model, SessionOptions o = {}) {
    std::ifstream f("models/" + model + ".ref");
    if (!f) { std::printf("   cannot open models/%s.ref (run from the code/ folder)\n", model.c_str()); return false; }
    Feeds in, want;
    std::string kind, name;
    while (f >> kind >> name) {
        int rank;
        f >> rank;
        Shape sh(static_cast<size_t>(rank));
        for (auto &d : sh) f >> d;
        std::vector<float> v(size_t(shape_numel(sh)));
        for (auto &x : v) f >> x;
        (kind == "input" ? in : want)[name] = Tensor::from(sh, v);
    }
    InferenceSession s(o);
    s.load("models/" + model + ".onnx");
    Feeds got = s.run(in);
    bool ok = true;
    for (auto &kv : want) {
        const Tensor &g = got.at(kv.first);
        double worst = 0;
        for (int64_t i = 0; i < g.numel(); i++) {
            const double d = std::fabs(double(g[i]) - double(kv.second[i]));
            if (std::isnan(d) || d > worst) worst = d;        // std::max would silently drop a NaN
            if (std::isnan(worst)) break;
        }
        std::printf("   %s: max |difference| vs onnxruntime = %g\n", model.c_str(), worst);
        ok = ok && close(g, kv.second);
    }
    return ok;
}

Tensor xor_input() { return Tensor::from({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1}); }

// ------------------------------------------------------------------------------ scenarios
void s1() {   // load a real .onnx file and compute XOR
    InferenceSession s;
    s.load("models/xor.onnx");
    Tensor y = s.run({{"x", xor_input()}}).at("out");
    std::printf("   XOR outputs: %.4f %.4f %.4f %.4f\n", y[0], y[1], y[2], y[3]);
    expect(y[0] < 0.1f && y[1] > 0.9f && y[2] > 0.9f && y[3] < 0.1f, "XOR truth table");
}
void s2() { expect(check_reference("ort_mlp"), "ort_mlp matches onnxruntime"); }
void s3() { expect(check_reference("branchy"), "branchy (raw_data weights) matches onnxruntime"); }
void s4() { expect(check_reference("branchy_external"), "branchy_external matches onnxruntime"); }
void s5() { expect(check_reference("classifier"), "classifier (PyTorch-style Gemm, 1-D biases) matches onnxruntime"); }
void s6() {   // y = Relu(x); z = Add(y, y): one node reads the same producer twice
    ModelProto m = zoo::empty_model("double_read");
    m.graph.input = {zoo::batch_input("x", 3)};
    m.graph.node = {zoo::node("Relu", "relu", {"x"}, {"y"}), zoo::node("Add", "twice", {"y", "y"}, {"z"})};
    m.graph.output = {zoo::batch_input("z", 3)};
    InferenceSession s;
    s.load_model(m);
    Tensor z = s.run({{"x", Tensor::from({1, 3}, {-1, 2, 3})}}).at("z");
    expect(z[0] == 0.f && z[1] == 4.f && z[2] == 6.f, "z = 2 * relu(x) = [0, 4, 6]");
}
void s7() {   // a MatMul whose result is used TWICE must not be fused away
    ModelProto m = zoo::empty_model("two_readers");
    std::mt19937 rng(1);
    m.graph.initializer = {zoo::random_tensor("W", {4, 6}, 1.f, rng), zoo::random_tensor("b", {1, 6}, 1.f, rng)};
    m.graph.input = {zoo::batch_input("x", 4)};
    m.graph.node = {zoo::node("MatMul", "mm", {"x", "W"}, {"t"}), zoo::node("Add", "add", {"t", "b"}, {"y0"}),
                    zoo::node("Relu", "relu", {"t"}, {"y1"}), zoo::node("Add", "join", {"y0", "y1"}, {"y"})};
    m.graph.output = {zoo::batch_input("y", 6)};
    SessionOptions off;
    off.enable_optimizations = false;
    InferenceSession a, b(off);
    a.load_model(m);
    b.load_model(m);
    Tensor x = zoo::pattern_input({2, 4});
    expect(close(a.run({{"x", x}}).at("y"), b.run({{"x", x}}).at("y")), "optimized == unoptimized");
}
void s8() {   // dead-code elimination
    ModelProto m = zoo::make_xor();
    m.graph.node.push_back(zoo::node("Relu", "dead1", {"h"}, {"unused1"}));
    m.graph.node.push_back(zoo::node("Relu", "dead2", {"unused1"}, {"unused2"}));
    Graph g = Graph::from_model(m);
    int removed = eliminate_dead_nodes(g);
    std::printf("   removed %d dead node(s), %zu left\n", removed, g.nodes.size());
    expect(removed == 2, "2 dead nodes removed");
}
void s9() {   // the AVX2 kernel vs the reference kernel on a shape that uses the 4x16 micro-kernel
    auto x86 = create_execution_provider("x86");
    if (!x86 || !x86->available()) { std::printf("   (no AVX2 on this machine: scenario skipped)\n"); return; }
    auto cpu = make_cpu_ep();
    ThreadPool pool(1);
    Tensor A = zoo::pattern_input({8, 5}, 0.3f), B = zoo::pattern_input({5, 16}, 1.1f);
    Tensor y1 = x86->kernel("MatMul")(KernelContext{&pool}, {&A, &B}, {});
    Tensor y2 = cpu->kernel("MatMul")(KernelContext{&pool}, {&A, &B}, {});
    for (int64_t i = 0; i < y1.numel(); i++)
        if (std::fabs(y1[i] - y2[i]) > 1e-4) { std::printf("   first difference at row %lld col %lld\n", (long long)(i / 16), (long long)(i % 16)); break; }
    expect(close(y1, y2), "x86 MatMul == cpu MatMul");
}
void s10() {  // MatMul without bias, twice: results must be identical and correct
    auto cpu = make_cpu_ep();
    ThreadPool pool(1);
    Tensor A = Tensor::from({2, 2}, {1, 2, 3, 4}), B = Tensor::from({2, 2}, {1, 0, 0, 1});
    Tensor junk = Tensor::from({2, 2}, {7777, 7777, 7777, 7777});          // leaves garbage in the allocator
    junk = Tensor();
    Tensor y = cpu->kernel("MatMul")(KernelContext{&pool}, {&A, &B}, {});
    std::printf("   A x I = [%g %g; %g %g]\n", y[0], y[1], y[2], y[3]);
    expect(close(y, A), "A x identity == A");
}
void s11() {  // softmax with huge logits
    auto cpu = make_cpu_ep();
    ThreadPool pool(1);
    Tensor z = Tensor::from({1, 3}, {1000.f, 0.f, -1000.f});
    Tensor p = cpu->kernel("Softmax")(KernelContext{&pool}, {&z}, {});
    std::printf("   softmax([1000, 0, -1000]) = [%g %g %g]\n", p[0], p[1], p[2]);
    expect(std::fabs(p[0] - 1.f) < 1e-6f && p[1] >= 0.f && p[1] < 1e-6f, "softmax = [1, 0, 0]");
}
void s12() {  // PyTorch-style Gemm(transB=1) without the optimizer: must still work on the x86 machine
    SessionOptions o;
    o.enable_optimizations = false;
    o.providers = {"x86", "arm", "cpu"};
    bool ok = false;
    try { ok = check_reference("classifier", o); }
    catch (const std::exception &e) { std::printf("   exception: %s\n", e.what()); }
    expect(ok, "classifier (unoptimized) matches onnxruntime");
}
void s13() {  // 'h' is a graph output AND is read by another node
    ModelProto m = zoo::empty_model("output_also_used");
    m.graph.input = {zoo::batch_input("x", 2)};
    m.graph.node = {zoo::node("Relu", "relu", {"x"}, {"h"}), zoo::node("Sigmoid", "sig", {"h"}, {"y"})};
    m.graph.output = {zoo::batch_input("h", 2), zoo::batch_input("y", 2)};
    InferenceSession s;
    s.load_model(m);
    Feeds out = s.run({{"x", Tensor::from({1, 2}, {-1.f, 1.f})}});
    expect(out.at("h")[0] == 0.f && out.at("h")[1] == 1.f, "h = relu(x) = [0, 1]");
}
void s14() {  // a chain whose intermediate tensors get freed along the way
    ModelProto m = zoo::empty_model("chain_add");
    m.graph.input = {zoo::batch_input("x", 4)};
    m.graph.initializer = {zoo::tensor("one", {1}, {1.f})};
    std::string prev = "x";
    for (int i = 0; i < 6; i++) {
        std::string out = i == 5 ? "y" : "t" + std::to_string(i);
        m.graph.node.push_back(zoo::node("Add", "add" + std::to_string(i), {prev, "one"}, {out}));
        prev = out;
    }
    m.graph.output = {zoo::batch_input("y", 4)};
    InferenceSession s;
    s.load_model(m);
    Tensor big = Tensor::from({256, 4}, std::vector<float>(1024, 10.f));
    Tensor y = s.run({{"x", big}}).at("y");
    std::printf("   y[0] = %g (expected 16), y[1023] = %g\n", y[0], y[1023]);
    expect(y[0] == 16.f && y[1023] == 16.f, "x + 6 = 16 everywhere");
}
void s15() {  // parallel executor + profiling, many runs
    SessionOptions o;
    o.execution_mode = ExecutionMode::Parallel;
    o.inter_op_threads = 4;
    InferenceSession s(o);
    s.load_model(zoo::make_branchy(8, 4, 4));
    Tensor x = zoo::pattern_input({1, 4});
    int bad = 0;
    for (int r = 0; r < 3000; r++) {
        RunProfile p;
        s.run({{"x", x}}, &p);
        if (p.events.size() != s.plan().nodes.size()) bad++;
    }
    std::printf("   runs with a wrong number of profile events: %d / 3000\n", bad);
    expect(bad == 0, "every run records one event per node");
}
void s16() {  // nested parallel_for on a small pool
    ThreadPool pool(2);
    std::atomic<int> total{0};
    pool.parallel_for(0, 4, 1, [&](int64_t, int64_t) {
        pool.parallel_for(0, 1000, 10, [&](int64_t b, int64_t e) { total += int(e - b); });
    });
    std::printf("   total = %d\n", total.load());
    expect(total == 4000, "4 x 1000 iterations");
}
void s17() {  // the parallel executor, many runs (a lost wake-up would hang here)
    SessionOptions o;
    o.execution_mode = ExecutionMode::Parallel;
    o.inter_op_threads = 4;
    InferenceSession s(o);
    s.load_model(zoo::make_branchy(4, 8, 8));
    Tensor x = zoo::pattern_input({2, 8});
    for (int r = 0; r < 20000; r++) {
        s.run({{"x", x}});
        if (r % 2000 == 0) { std::printf("   run %d\n", r); std::fflush(stdout); }
    }
}
void s18() {  // many branches merged by many Adds, many runs
    SessionOptions o;
    o.execution_mode = ExecutionMode::Parallel;
    o.inter_op_threads = 8;
    InferenceSession ref, s(o);
    ModelProto m = zoo::make_branchy(16, 4, 4);
    ref.load_model(m);
    s.load_model(m);
    Tensor x = zoo::pattern_input({1, 4});
    Tensor want = ref.run({{"x", x}}).at("out");
    int bad = 0;
    for (int r = 0; r < 5000; r++) {
        try { if (!close(s.run({{"x", x}}).at("out"), want, 0)) bad++; }
        catch (const std::exception &e) { bad++; if (bad < 3) std::printf("   run %d: %s\n", r, e.what()); }
        if (r % 1000 == 0) { std::printf("   run %d\n", r); std::fflush(stdout); }
    }
    std::printf("   bad runs: %d / 5000\n", bad);
    expect(bad == 0, "all runs correct");
}
void s19() {  // a kernel error on a worker thread must come back to the caller as an exception
    ModelProto m = zoo::make_branchy(4, 8, 8);
    m.graph.input[0].has_shape = false;
    SessionOptions o;
    o.execution_mode = ExecutionMode::Parallel;
    o.inter_op_threads = 4;
    InferenceSession s(o);
    s.load_model(m);
    bool caught = false;
    for (int r = 0; r < 50; r++) {
        try { s.run({{"x", zoo::pattern_input({2, 5})}}); }       // wrong width: 5 instead of 8
        catch (const std::exception &e) { caught = true; if (r == 0) std::printf("   caught: %s\n", e.what()); }
    }
    expect(caught, "the error reached the caller");
}
void s20() {  // create and destroy many sessions (their thread pools must shut down)
    for (int i = 0; i < 50; i++) {
        SessionOptions o;
        o.intra_op_threads = 2;
        o.inter_op_threads = 2;
        InferenceSession s(o);
        s.load_model(zoo::make_xor());
        s.run({{"x", xor_input()}});
        std::this_thread::sleep_for(std::chrono::milliseconds(2));   // let the workers fall asleep
    }
    std::printf("   50 sessions created and destroyed\n");
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    void (*scenarios[])() = {s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20};
    const int n = int(sizeof(scenarios) / sizeof(scenarios[0]));
    if (argc < 2) { std::printf("usage: lab_runner <1..%d | all>\n", n); return 2; }
    const std::string arg = argv[1];
    int first = 1, last = n;
    if (arg != "all") first = last = std::atoi(argv[1]);
    if (first < 1 || last > n) { std::printf("no scenario %s\n", argv[1]); return 2; }
    for (int k = first; k <= last; k++) {
        const int before = g_fail;
        std::printf("SCENARIO %d\n", k);
        try { scenarios[k - 1](); }
        catch (const std::exception &e) { g_fail++; std::printf("   unexpected exception: %s\n", e.what()); }
        std::printf("SCENARIO %d: %s\n", k, g_fail == before ? "PASS" : "FAIL");
    }
    return g_fail ? 1 : 0;
}
