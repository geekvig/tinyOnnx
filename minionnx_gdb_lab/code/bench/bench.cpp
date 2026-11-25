// =====================================================================================
// bench/bench.cpp  --  what do the EPs and the threads actually buy?
// =====================================================================================
// Build in Release (-O2/-O3) before trusting any number:
//   cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && cmake --build build-rel -j
//   ./build-rel/bench_minionnx
//
//   A. one big matrix multiply: cpu EP vs SIMD EP (x86 AVX2 or arm NEON), in GFLOP/s
//   B. intra-op threads on that matrix multiply (1, 2, all cores)
//   C. inter-op threads: sequential vs parallel executor on a 4-branch model
//   D. request-level: many client threads sharing one session
// Numbers depend completely on the machine. Measure, don't guess.
// =====================================================================================
#include "../tools/model_zoo.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace mo;

template <class F>
static double best_ms(int reps, F &&f) {
    f();                                                   // warm-up (caches, page faults, thread start)
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

int main() {
    const int hw = std::max(1u, std::thread::hardware_concurrency());
    std::printf("MiniOnnx benchmark  (hardware threads: %d)\n", hw);

    // ---------------------------------------------------------------- A + B
    const int64_t M = 256, K = 512, N = 512;
    Tensor A = zoo::pattern_input({M, K}, 0.1f), B = zoo::pattern_input({K, N}, 0.7f);
    const double gflop = 2.0 * M * K * N / 1e9;
    std::printf("\nA. MatMul %lldx%lldx%lld (%.2f GFLOP)\n", (long long)M, (long long)K, (long long)N, gflop);
    std::vector<std::string> eps;
    for (auto &[name, avail] : list_execution_providers()) if (avail && name != "cuda") eps.push_back(name);
    for (auto &name : eps) {
        auto ep = create_execution_provider(name);
        ThreadPool pool(1);
        double ms = best_ms(5, [&] { ep->kernel("MatMul")(KernelContext{&pool}, {&A, &B}, {}); });
        std::printf("   %-4s 1 thread : %8.2f ms  %6.1f GFLOP/s\n", name.c_str(), ms, gflop / (ms / 1e3));
    }
    std::printf("\nB. intra-op threads on the same MatMul (best EP: %s)\n", eps.front().c_str());
    {
        auto ep = create_execution_provider(eps.front());
        double base = 0;
        std::vector<int> counts = {1, 2, hw};
        std::sort(counts.begin(), counts.end());
        counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
        for (int t : counts) {
            ThreadPool pool(t);
            double ms = best_ms(5, [&] { ep->kernel("MatMul")(KernelContext{&pool}, {&A, &B}, {}); });
            if (t == 1) base = ms;
            std::printf("   intra=%-2d : %8.2f ms  speedup %.2fx\n", t, ms, base / ms);
        }
    }

    // ---------------------------------------------------------------- C
    std::printf("\nC. 4-branch model (batch 64, D=256, H=512): executors\n");
    ModelProto m = zoo::make_branchy(4, 256, 512);
    Tensor x = zoo::pattern_input({64, 256});
    double seq_ms = 0;
    for (auto [mode, inter] : std::vector<std::pair<ExecutionMode, int>>{{ExecutionMode::Sequential, 1},
                                                                          {ExecutionMode::Parallel, 2},
                                                                          {ExecutionMode::Parallel, 4}}) {
        SessionOptions o;
        o.execution_mode = mode;
        o.inter_op_threads = inter;
        o.providers = {"x86", "arm", "cpu"};
        InferenceSession s(o);
        s.load_model(m);
        double ms = best_ms(10, [&] { s.run({{"x", x}}); });
        if (mode == ExecutionMode::Sequential) seq_ms = ms;
        std::printf("   %-10s inter=%d : %8.2f ms  speedup %.2fx\n", mode == ExecutionMode::Sequential ? "sequential" : "parallel",
                    inter, ms, seq_ms / ms);
    }

    // ---------------------------------------------------------------- D
    std::printf("\nD. request throughput: client threads sharing ONE session\n");
    {
        SessionOptions o;
        o.providers = {"x86", "arm", "cpu"};
        InferenceSession s(o);
        s.load_model(m);
        Tensor small = zoo::pattern_input({8, 256});
        double base = 0;
        std::vector<int> counts = {1, 2, hw, 2 * hw};
        std::sort(counts.begin(), counts.end());
        counts.erase(std::unique(counts.begin(), counts.end()), counts.end());   // hw may be 2
        for (int clients : counts) {
            const int per = 40;
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> ts;
            for (int c = 0; c < clients; c++) ts.emplace_back([&] { for (int i = 0; i < per; i++) s.run({{"x", small}}); });
            for (auto &t : ts) t.join();
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double rps = clients * per / sec;
            if (clients == 1) base = rps;
            std::printf("   %2d client(s): %8.0f requests/s  (%.2fx)\n", clients, rps, rps / base);
        }
    }
    return 0;
}
