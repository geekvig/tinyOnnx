# MiniOnnx — a small, multithreaded ONNX inference runtime in C++17

MiniOnnx reads **real `.onnx` files** (hand-written protobuf decoder, no dependencies), optimizes the graph,
splits it between **execution providers** (portable CPU, x86 AVX2+FMA, ARM NEON, optional CUDA), and runs it
with three kinds of parallelism (intra-op, inter-op, many concurrent callers).

It is the companion code of the PDF *"MiniOnnx: design document + tutorial"*. Read the PDF chapter by chapter
with this folder open in VS Code.

```
 .onnx bytes ──► Stage 1 FORMAT ──► Stage 2 GRAPH ──► Stage 3 EXECUTION PROVIDERS ──► Stage 4 RUNTIME
                 protobuf wire       intern names       partition: cuda > x86/arm > cpu    plan, thread pools,
                 ModelProto          validate, topo     SIMD / GPU kernels                 seq/parallel executor
                 weight arena        optimize
```

## 1. Quick start on Windows (VS Code + Windows Terminal + g++)

1. Install **MSYS2** from <https://www.msys2.org> (default folder `C:\msys64`).
2. Open the **MSYS2 UCRT64** terminal and install the tools:
   ```
   pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gdb mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-make
   ```
3. Add `C:\msys64\ucrt64\bin` to your Windows **PATH** (Start → "Edit the system environment variables" →
   Environment Variables → Path → New). Open a **new** Windows Terminal and check:
   ```
   g++ --version
   gdb --version
   cmake --version
   ```
4. Build and test (from this folder, in Windows Terminal / PowerShell):
   ```
   cmake --preset mingw-debug               # configure  (= cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug)
   cmake --build build -j 8                 # compile
   .\build\make_models.exe models           # (re)write the example .onnx files
   ctest --test-dir build --output-on-failure
   .\build\minionnx_run.exe models\xor.onnx --batch 4 --values 0,0,0,1,1,0,1,1
   ```
5. In VS Code: open the folder, install the recommended extensions (C/C++, CMake Tools), press
   **Ctrl+Shift+B** to build, **F5** with "Windows: debug tests" to debug in gdb.

No CMake? `powershell -ExecutionPolicy Bypass -File scripts\build_nocmake.ps1` builds everything with plain g++.

## 2. Linux / macOS / WSL2

```bash
cmake --preset linux-debug && cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --preset linux-tsan  && cmake --build build-tsan -j && ./build-tsan/test_minionnx   # data-race detector
cmake --preset linux-asan  && cmake --build build-asan -j && ./build-asan/test_minionnx   # memory-error detector
```
(ThreadSanitizer and AddressSanitizer are not shipped with MinGW: use WSL2 for them.)

## 3. ARM (NEON EP) — without ARM hardware

```bash
sudo apt install g++-aarch64-linux-gnu qemu-user
cmake --preset arm64-cross && cmake --build build-arm -j
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm/test_minionnx
```
On a Raspberry Pi 4/5, Apple-silicon Linux VM, or ARM cloud machine just use `linux-debug` natively.

## 4. CUDA EP (optional)

```bash
cmake --preset linux-cuda && cmake --build build-cuda -j
./build-cuda/minionnx_run models/branchy.onnx --providers cuda,cpu
```
Needs the CUDA toolkit. On Windows, nvcc requires MSVC (not MinGW g++), so use WSL2 or Linux. **Honest status:** the
CUDA EP was compile-checked (device and host code) but not executed on a GPU while this tutorial was written. When no
GPU is present the EP reports "not available" and every node falls back to the CPU EPs automatically.

## 5. Tools

| Program | What it does |
|---|---|
| `make_models [dir]` | writes `xor.onnx` (float_data), `classifier.onnx` (raw_data, PyTorch-style Gemm), `branchy.onnx`, `branchy_external.onnx` + `.data` |
| `minionnx_run model.onnx [--providers x86,cpu] [--parallel --inter 4] [--profile] [--check model.ref]` | load, partition, run, print outputs/timeline |
| `inspect_onnx model.onnx [--wire N \| --plan]` | summary, **protobuf X-ray** (every field byte-by-byte), or the optimized/partitioned plan |
| `test_minionnx` | 14 test groups, ~760 checks |
| `bench_minionnx` | GFLOP/s per EP, intra-op, inter-op and request-level scaling |
| `python/crosscheck_ort.py models` | (optional, needs `pip install onnx onnxruntime`) validates every file with `onnx.checker`, writes `*.ref` outputs from real onnxruntime |

## 6. Source map

| Stage | Files | Teaches |
|---|---|---|
| core | `core/tensor.hpp`, `core/thread_pool.hpp` | aligned buffers, shared ownership, thread pool, `parallel_for`, exception hand-off |
| 1 Format | `stage1_format/protobuf_wire.hpp`, `onnx_model.hpp`, `onnx_io.cpp`, `weights.hpp/.cpp` | varints, wire types, field numbers, ONNX schema, float_data/raw_data/external weights, arena |
| 2 Graph | `stage2_graph/graph.*`, `optimizer.*` | interning, validation, Kahn's topological sort, fusion, constant folding, dead-code elimination |
| 3 EPs | `stage3_ep/execution_provider.hpp`, `kernel_utils.hpp`, `cpu_ep.cpp`, `x86_ep.cpp`, `arm_ep.cpp`, `cuda_ep.cu`, `registry.cpp` | EP interface, GetCapability/partitioning, SIMD micro-kernels, CUDA tiling |
| 4 Runtime | `stage4_runtime/plan.*`, `session.*` | immutable plan, RunState, sequential + parallel executors, atomic ref-count freeing |

## 7. What was verified (in a Linux x86-64 sandbox, g++ 13)

* 760 unit checks pass (Debug and Release) — also under **ThreadSanitizer** (0 warnings, 3 runs) and
  **AddressSanitizer + UBSan** (0 reports).
* ARM NEON EP: cross-compiled with aarch64 g++ and the full test suite run under QEMU: all checks pass.
* `onnx.checker` accepts every file MiniOnnx writes; files written by the official `onnx` library load in MiniOnnx;
  outputs match **onnxruntime 1.29** within 7e-8 on 6 models.
* Windows/MinGW: the CMake presets, scripts and VS Code files follow MSYS2 UCRT64 conventions but were not executed on
  a Windows machine here — if a path differs on yours (e.g. MSYS2 not in `C:\msys64`), edit `.vscode/*.json`.
