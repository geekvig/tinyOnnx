// =====================================================================================
// tools/minionnx_run.cpp  --  load a model, run it, print the result
// =====================================================================================
//   minionnx_run model.onnx [options]
//     --providers x86,cpu   EP priority list (default: cuda,x86,arm,cpu -> best available)
//     --intra N             threads inside one kernel          (default 1)
//     --inter N             threads running different nodes    (default 1)
//     --parallel            use the parallel executor          (default: sequential)
//     --batch B             value for symbolic dims like "N"   (default 4)
//     --values a,b,c,...    input values (default: a deterministic sin() pattern)
//     --no-opt              disable the optimizer
//     --profile             print a per-node timeline
//     --check file.ref      compare with a reference file (made by python/crosscheck_ort.py);
//                           exit code 1 if the max difference is above 1e-4
//     --repeat R            run R times and report the best time
// =====================================================================================
#include "minionnx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace mo;

namespace {

std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, sep)) if (!t.empty()) out.push_back(t);
    return out;
}

// Reference file format (text):
//   input <name> <rank> <dims...>      followed by a line of values
//   output <name> <rank> <dims...>     followed by a line of values
struct Reference { Feeds inputs, outputs; };

Reference read_reference(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open reference file " + path);
    Reference r;
    std::string kind, name;
    while (f >> kind >> name) {
        int rank;
        f >> rank;
        Shape shape(static_cast<size_t>(rank));
        for (auto &d : shape) f >> d;
        std::vector<float> v(size_t(shape_numel(shape)));
        for (auto &x : v) f >> x;
        (kind == "input" ? r.inputs : r.outputs)[name] = Tensor::from(shape, v);
    }
    return r;
}

void print_tensor(const std::string &name, const Tensor &t) {
    std::printf("  %s %s\n", name.c_str(), shape_str(t.shape()).c_str());
    const int64_t R = std::min<int64_t>(t.rows(), 8), C = std::min<int64_t>(t.cols(), 8);
    for (int64_t r = 0; r < R; r++) {
        std::printf("    ");
        for (int64_t c = 0; c < C; c++) std::printf("%9.4f ", t[r * t.cols() + c]);
        std::printf(t.cols() > C ? "...\n" : "\n");
    }
    if (t.rows() > R) std::printf("    ...\n");
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::printf("usage: minionnx_run model.onnx [--providers x86,cpu] [--intra N] [--inter N] [--parallel]\n"
                    "                    [--batch B] [--values v1,v2,...] [--no-opt] [--profile] [--check file] [--repeat R]\n");
        return 2;
    }
    SessionOptions opt;
    opt.providers = {"cuda", "x86", "arm", "cpu"};
    opt.verbose = true;
    int64_t batch = 4;
    int repeat = 1;
    bool profile = false;
    std::string values, check;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        try {
            if (a == "--providers") opt.providers = split(next(), ',');
            else if (a == "--intra") opt.intra_op_threads = std::stoi(next());
            else if (a == "--inter") opt.inter_op_threads = std::stoi(next());
            else if (a == "--parallel") opt.execution_mode = ExecutionMode::Parallel;
            else if (a == "--batch") batch = std::stoll(next());
            else if (a == "--values") values = next();
            else if (a == "--no-opt") opt.enable_optimizations = false;
            else if (a == "--profile") profile = true;
            else if (a == "--check") check = next();
            else if (a == "--repeat") repeat = std::max(1, std::stoi(next()));
            else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "bad arguments: %s\n", e.what());
            return 2;
        }
    }

    try {
        std::printf("available execution providers in this build:");
        for (auto &ep : list_execution_providers()) std::printf("  %s(%s)", ep.first.c_str(), ep.second ? "yes" : "no");
        std::printf("\n");

        InferenceSession session(opt);
        session.load(argv[1]);

        Feeds feeds;
        Reference ref;
        if (!check.empty()) {
            ref = read_reference(check);
            feeds = ref.inputs;
        } else {
            for (const std::string &name : session.input_names()) {
                Shape shape = {batch, 1};
                auto it = session.graph().input_info.find(session.graph().find(name));
                if (it != session.graph().input_info.end() && it->second.has_shape) {
                    shape.clear();
                    for (size_t d = 0; d < it->second.dims.size(); d++)
                        shape.push_back(it->second.dims[d] >= 0 && it->second.dim_names[d].empty() ? it->second.dims[d] : batch);
                }
                Tensor t = Tensor::empty(shape);
                if (!values.empty()) {
                    auto v = split(values, ',');
                    std::vector<float> f;
                    for (auto &s : v) f.push_back(std::stof(s));
                    t = Tensor::from(shape, f);                    // throws if the count is wrong
                } else {
                    for (int64_t k = 0; k < t.numel(); k++) t[k] = std::sin(0.37f * float(k));
                }
                feeds[name] = t;
            }
        }
        for (auto &kv : feeds) print_tensor("input " + kv.first, kv.second);

        RunProfile prof;
        Feeds out;
        double best = 1e30;
        for (int r = 0; r < repeat; r++) {
            prof = RunProfile{};
            auto t0 = std::chrono::steady_clock::now();
            out = session.run(feeds, profile ? &prof : nullptr);
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        for (auto &kv : out) print_tensor("output " + kv.first, kv.second);
        std::printf("run time: %.3f ms (best of %d)\n", best, repeat);

        if (profile) {
            std::sort(prof.events.begin(), prof.events.end(), [](auto &a, auto &b) { return a.start_us < b.start_us; });
            std::printf("  %-26s %-8s %-5s %6s %10s %10s\n", "node", "op", "ep", "thread", "start_us", "end_us");
            for (auto &e : prof.events)
                std::printf("  %-26s %-8s %-5s %6d %10.1f %10.1f\n", e.name.c_str(), e.op.c_str(), e.ep.c_str(), e.thread, e.start_us, e.end_us);
            std::printf("  peak live tensors: %d, live at end: %d\n", prof.peak_live_values, prof.final_live_values);
        }

        if (!check.empty()) {
            double worst = 0;
            for (auto &kv : ref.outputs) {
                const Tensor &want = kv.second, &got = out.at(kv.first);
                if (want.shape() != got.shape()) throw std::runtime_error("output '" + kv.first + "' shape mismatch");
                for (int64_t k = 0; k < want.numel(); k++) worst = std::max(worst, double(std::fabs(want[k] - got[k])));
            }
            std::printf("check vs %s: max |difference| = %.3g  ->  %s\n", check.c_str(), worst, worst < 1e-4 ? "PASS" : "FAIL");
            return worst < 1e-4 ? 0 : 1;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
