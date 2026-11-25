# MiniOnnx GDB Lab — 20 bugs to catch

*Companion lab to "MiniOnnx: design document + tutorial". You will deliberately break MiniOnnx in 20 different
ways — 14 single-threaded bugs and 6 multithreaded ones — and catch each bug with **gdb** (command line and VS Code).
Every transcript in this guide is a real gdb session recorded on the real code.*

## 0. What is in this folder

```
minionnx_gdb_lab/
  LAB_GUIDE.md / LAB_GUIDE.pdf   this guide
  code/                          a complete, CLEAN copy of MiniOnnx + lab/lab_runner.cpp (one scenario per bug)
  bugs/B01_*.patch ... B20_*.patch   each patch introduces exactly one bug
  scripts/bug.ps1, scripts/bug.sh    apply / reset / check bugs
```

`lab_runner <N>` runs scenario *N*. On clean code all 20 scenarios print `SCENARIO N: PASS`. With bug *N* applied,
scenario *N* fails, crashes or hangs. Your job: find out **why** — with the debugger, not by reading the patch.

> **Rule of the lab:** do not open the `.patch` file before you have found the bug. The patch *is* the answer.

## 1. Setup

### Windows (MSYS2 g++ + gdb, Windows Terminal, VS Code)

**Step 1.** Install MSYS2 and the tools (see the main README). In the *MSYS2 UCRT64* terminal:
```
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gdb mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-make git
```

**Step 2.** Put `C:\msys64\ucrt64\bin` on PATH; check `g++ --version`, `gdb --version`, `git --version` in Windows Terminal.

**Step 3.** Build the lab in **Debug** (no optimization, full symbols — essential for debugging):
```
cd minionnx_gdb_lab\code
cmake --preset mingw-debug
cmake --build build -j 8
.\build\lab_runner.exe all          # every scenario must PASS before you start
```

**Step 4.** VS Code: open the `code` folder. *Run and Debug* → **"LAB Windows: debug lab_runner scenario"** → F5 → type the
scenario number. If gdb is not in `C:\msys64\ucrt64\bin`, fix `miDebuggerPath` in `.vscode/launch.json`.

### Linux / WSL2
```bash
cd minionnx_gdb_lab/code
cmake --preset linux-debug && cmake --build build -j
./build/lab_runner all
```
Use **"LAB Linux/WSL: debug lab_runner scenario"** in VS Code. WSL also gives you AddressSanitizer and
ThreadSanitizer (`linux-asan`, `linux-tsan` presets), which MinGW does not have.

### Windows differences you should expect
* Heap corruption looks different: instead of glibc's `double free or corruption` you get a silent crash with exit code
  `0xC0000374` (heap corruption) or `0xC0000005` (access violation). gdb still stops on it: `Thread N received signal SIGSEGV`.
* Thread ids look like `Thread 1234.0x2a1c`. `info threads` / `thread apply all bt` work the same.
* If **Ctrl+C** in Windows Terminal does not interrupt the program, use the **pause** button in VS Code (or start gdb
  first, `run`, then Ctrl+C in the gdb window).
* Transcripts in this guide were recorded on Linux; addresses, thread ids and line numbers of *system* libraries
  will differ. Line numbers of MiniOnnx files are the same as yours (same code + same patch).

## 2. The lab loop (do this for every bug)

```
 1. apply      powershell -ExecutionPolicy Bypass -File ..\scripts\bug.ps1 apply 7      (or: bash ../scripts/bug.sh apply 7)
 2. build      cmake --build build -j 8
 3. observe    .\build\lab_runner.exe 7          -> read the symptom, write down a guess
 4. hunt       gdb --args .\build\lab_runner.exe 7     (or VS Code F5)  -> prove or kill your guess
 5. explain    write ONE sentence: "X happens because Y"; then read the walkthrough (section 6)
 6. reset      ...\bug.ps1 reset 7   ;  rebuild  ;  lab_runner 7 must PASS again
```
`bug.ps1 check` lists the bugs that are currently applied. Apply **one bug at a time**.

No git? Each walkthrough shows the exact *before → after* code, so you can introduce the bug by hand in VS Code.

## 3. GDB crash course

gdb is a program that runs your program *under its control*: it can stop it at any line, show any variable, and
continue one line at a time. Think of it as a pause button plus an X-ray machine for your program.

### Starting
| Command | What it does |
|---|---|
| `gdb --args ./build/lab_runner 7` | load the program with its arguments (nothing runs yet) |
| `run` (`r`) | start it |
| `start` | start and stop at the first line of `main` |
| `quit` (`q`) | leave gdb |
| `gdb -p <pid>` / `attach <pid>` | attach to an already-running (e.g. hung) process |

### Stopping: breakpoints, catchpoints, watchpoints
| Command | What it does |
|---|---|
| `break session.cpp:211` (`b`) | stop when line 211 is about to run |
| `break mo::InferenceSession::run` | stop at the start of a function |
| `break weights.cpp:121 if i == 2` | **conditional** breakpoint |
| `tbreak ...` | one-shot breakpoint |
| `catch throw` | stop the moment ANY C++ exception is thrown (before unwinding!) |
| `watch x` / `watch -l g.Y[0]` | stop whenever `x` / that memory location changes |
| `rwatch x` | stop whenever `x` is read |
| `info breakpoints`, `delete 2`, `disable 2` | manage them |

### Moving
| Command | What it does |
|---|---|
| `next` (`n`) | run the current line, stepping OVER calls |
| `step` (`s`) | run the current line, stepping INTO calls |
| `finish` | run until the current function returns; prints the return value |
| `until 40` / `advance 40` | run until line 40 |
| `continue` (`c`) | run until the next stop |
| **Ctrl+C** | interrupt a running (or hung) program |

### Looking
| Command | What it does |
|---|---|
| `print x` (`p`) | show a value; STL containers are pretty-printed |
| `p/x n`, `p/t n`, `p/f bits` | hex, binary, float |
| `p arr[0]@8` | 8 consecutive elements |
| `p *ptr`, `p obj.method()` | dereference, call a member function |
| `x/16xb ptr`, `x/8f ptr` | examine raw memory: 16 hex bytes, 8 floats |
| `display expr` | print `expr` at every stop |
| `info locals`, `info args` | all local variables / arguments |
| `ptype obj`, `whatis x` | show a type |
| `set print object on` | show the real (dynamic) type behind a base pointer |
| `bt` / `bt 8` | backtrace: who called whom |
| `frame 3`, `up`, `down`, `frame function NAME` | move between stack frames |

### Threads
| Command | What it does |
|---|---|
| `info threads` | list threads; `*` = current |
| `thread 2` | switch to thread 2 |
| `thread apply all bt` | backtrace of EVERY thread (first thing to do on a hang) |
| `set scheduler-locking step` | while stepping, freeze the other threads |
| `set print thread-events off` | silence "New Thread ..." messages |

### The same in VS Code
| VS Code | gdb |
|---|---|
| F9 / click left of a line number | `break file:line` |
| right-click a breakpoint → *Edit Breakpoint* | condition / hit count |
| F5 / F10 / F11 / Shift+F11 | `continue` / `next` / `step` / `finish` |
| pause button (‖) | Ctrl+C |
| *Variables*, *Watch*, *Call Stack* panels | `info locals`, `display`, `bt` and `info threads` |
| *Debug Console*: `-exec <command>` | any gdb command, e.g. `-exec catch throw`, `-exec x/8f g.Y` |

## 4. Toolkit for multithreaded bugs

* **Hang?** Interrupt, then `thread apply all bt`. Look for the *cycle*: who waits for whom. Every thread in
  `cond_wait` = deadlock or lost wake-up; one thread spinning = livelock.
* **Inspect the condition** a waiting thread is waiting for (`p st.pending`, `p st.ready_q`). If the condition is true
  but the thread still waits, nobody notified it: a *lost wake-up*.
* **Random crashes that change every run** = a data race until proven otherwise. gdb shows where the damage is
  *discovered*, not where it was done. Use **ThreadSanitizer** (WSL: `cmake --preset linux-tsan`): it reports the two
  conflicting accesses with their stacks.
* **Heisenbugs:** printf and breakpoints change timing and can make a race disappear. Prefer stress loops (the
  scenarios run thousands of times) and post-mortem inspection after interrupting.
* `set scheduler-locking step` lets you single-step one thread without the others running ahead.

Real ThreadSanitizer output for bug 18 (first lines):
```

WARNING: ThreadSanitizer: data race (pid=8735)
  Read of size 4 at 0x7240000004f4 by thread T2:
    #0 mo::InferenceSession::process(std::shared_ptr<mo::InferenceSession::RunState> const&, int) const src/stage4_runtime/session.cpp:272 (
    #1 mo::InferenceSession::helper(std::shared_ptr<mo::InferenceSession::RunState> const&) const src/stage4_runtime/session.cpp:254 (lab_ru
    ...
  Previous write of size 4 at 0x7240000004f4 by thread T6:
    #0 mo::InferenceSession::process(std::shared_ptr<mo::InferenceSession::RunState> const&, int) const src/stage4_runtime/session.cpp:272 (
```
Line 272 is exactly the `--st.indeg[d]` of the bug. TSan found it on the first run.

## 5. The 20 challenges

| # | Stage | Kind | Title | gdb skill you practice |
|---|---|---|---|---|
| 1 | 1 Format | single-thread | The varint that counts in 8s | conditional breakpoint, `p/x`, `finish` (shows the return value) |
| 2 | 1 Format | single-thread | One wrong shift in read_fixed32 | `x` (examine memory) with format letters, `p/x` |
| 3 | 1 Format | single-thread | Bytes or floats? memcpy copies too little | breakpoint on a line, `next`, printing array elements, casting in gdb |
| 4 | 1 Format | single-thread | External weights read from the wrong offset | breakpoint + `continue` to iterate, printing `std::string`s, conditional breakpoints |
| 5 | 3 EP | single-thread | A mystery std::out_of_range (1-D bias) | `catch throw`, `bt`, `frame N`, printing a vector |
| 6 | 2 Graph | single-thread | 'Graph has a cycle' -- but it doesn't | pretty-printing STL vectors (`p order`, nested vectors) |
| 7 | 2 Graph | single-thread | The optimizer fuses away a tensor someone still needs | calling functions from gdb, `display` (try `display consumers[t]`) |
| 8 | 2 Graph | single-thread | A backwards loop that never ends | crash analysis, `frame function NAME`, reading compiler warnings |
| 9 | 3 EP | single-thread | SIMD copy-paste: the wrong register is stored | inspecting SIMD vector variables (`p c01`, `p $ymm0`), debugging at -O0 |
| 10 | 3 EP | single-thread | Garbage in, garbage out: the accumulator is never zeroed | `x/4f` on a buffer, watchpoints (`watch -l g.Y[0]` tells you every write) |
| 11 | 3 EP | single-thread | Softmax returns NaN | printing arrays with `@`, reasoning about float overflow |
| 12 | 3 EP | single-thread | The wrong execution provider gets the node | debugging virtual dispatch, `set print object on`, `up`/`down` |
| 13 | 4 Runtime | single-thread | A graph output disappears | `catch throw` + reading state; (advanced) a watchpoint on the slot's pointer |
| 14 | 4 Runtime | single-thread | Use-after-free: inputs released before the kernel runs | reading a deep backtrace, recognizing use-after-free; (WSL) AddressSanitizer pinpoints it |
| 15 | 4 Runtime | **multithreaded** | Data race: the profile lock was 'too slow' | `info threads`, `thread N`, `bt` per thread; ThreadSanitizer |
| 16 | core | **multithreaded** | Deadlock: the caller just waits | interrupting a hung program, `info threads`, `thread apply all bt` |
| 17 | 4 Runtime | **multithreaded** | Lost wake-up: notify before the state changes | inspecting state of a hung program, condition-variable reasoning |
| 18 | 4 Runtime | **multithreaded** | Lost update: a plain int counter shared by threads | reasoning about lost updates; ThreadSanitizer |
| 19 | 4 Runtime | **multithreaded** | An exception escapes a worker thread | `catch throw` in multithreaded code, identifying the current thread |
| 20 | core | **multithreaded** | The program never exits: workers never hear 'stop' | `thread apply all bt` on a hang at shutdown |

### Challenge 1 — The varint that counts in 8s

*Stage 1 Format · single-threaded · apply with* `bug.ps1 apply 1` *· run* `lab_runner 1`

**Symptom.** Loading `xor.onnx` fails: `format error: length 638 runs past end of message (at byte 30)`. The file is fine -- onnxruntime loads it.

**Your mission.** Find out why the reader thinks the graph is 638 bytes long.

<details><summary>Hint 1</summary>

The graph message starts at byte 27. Its length is a varint (2 bytes: `0xFE 0x02`). Work out by hand what 382 looks like in base-128.

</details>

<details><summary>Hint 2</summary>

Stop inside `read_varint` only when a byte has its continuation bit set: a *conditional breakpoint* `if (b & 0x80)`.

</details>

### Challenge 2 — One wrong shift in read_fixed32

*Stage 1 Format · single-threaded · apply with* `bug.ps1 apply 2` *· run* `lab_runner 2`

**Symptom.** `ort_mlp` gives slightly wrong probabilities: max difference vs onnxruntime 0.053. No crash, no error -- the worst kind of bug.

**Your mission.** The model has two float *attributes* (Gemm alpha = 0.75, beta = 2.0). Check that they are read correctly.

<details><summary>Hint 1</summary>

Floats are wire type I32: 4 little-endian bytes. 0.75f is `0x3F400000`, stored as bytes `00 00 40 3F`.

</details>

<details><summary>Hint 2</summary>

`x/4xb p_ - 4` shows the 4 bytes just consumed; `p/x bits` shows the integer built from them.

</details>

### Challenge 3 — Bytes or floats? memcpy copies too little

*Stage 1 Format · single-threaded · apply with* `bug.ps1 apply 3` *· run* `lab_runner 3`

**Symptom.** `branchy` (weights stored as `raw_data`) produces NaN (`max |difference| = nan`).

**Your mission.** The weights are copied from the file into the arena by `decode_le_floats`. Are all of them copied?

<details><summary>Hint 1</summary>

Compare the first and the LAST weight of a tensor: `p dst[0]` vs `p ((float*)src)[0]` and the same for `[n-1]`.

</details>

<details><summary>Hint 2</summary>

`n` counts floats. How many *bytes* does `memcpy` need?

</details>

### Challenge 4 — External weights read from the wrong offset

*Stage 1 Format · single-threaded · apply with* `bug.ps1 apply 4` *· run* `lab_runner 4`

**Symptom.** `branchy_external` (weights in a separate `.data` file) computes wrong numbers (difference 0.58) while `branchy` -- the same weights inside the .onnx -- is correct.

**Your mission.** Each external tensor has `location`, `offset`, `length`. Check the offset that is really used.

<details><summary>Hint 1</summary>

Stop where the bytes are decoded and print `i`, `off_s`, `len_s`, `off`. Then `continue` to the next tensor.

</details>

<details><summary>Hint 2</summary>

A breakpoint can have a condition, e.g. `break weights.cpp:121 if i == 2`.

</details>

### Challenge 5 — A mystery std::out_of_range (1-D bias)

*Stage 3 EP · single-threaded · apply with* `bug.ps1 apply 5` *· run* `lab_runner 5`

**Symptom.** Running the PyTorch-style `classifier` throws `vector::_M_range_check: __n (which is 1) >= this->size() (which is 1)` -- a message that tells you nothing about *where*.

**Your mission.** Find which line of MiniOnnx threw.

<details><summary>Hint 1</summary>

`catch throw` stops at the moment ANY exception is thrown, before the stack unwinds.

</details>

<details><summary>Hint 2</summary>

Then `bt` and `frame N` to look at the first frame that is MiniOnnx code.

</details>

### Challenge 6 — 'Graph has a cycle' -- but it doesn't

*Stage 2 Graph · single-threaded · apply with* `bug.ps1 apply 6` *· run* `lab_runner 6`

**Symptom.** A perfectly valid model `y = Relu(x); z = Add(y, y)` is rejected: `graph has a cycle: not a valid DAG`.

**Your mission.** Kahn's algorithm only reached some nodes. Find which node never got to in-degree 0, and why.

<details><summary>Hint 1</summary>

Break on the `throw` line of `topological_order` and print `order`, `indegree`, `dependents`.

</details>

<details><summary>Hint 2</summary>

Node 1 reads `y` twice. How many edges go into it? How many times is it listed in `dependents[0]`?

</details>

### Challenge 7 — The optimizer fuses away a tensor someone still needs

*Stage 2 Graph · single-threaded · apply with* `bug.ps1 apply 7` *· run* `lab_runner 7`

**Symptom.** With the optimizer on, the model fails to load: `dangling reference: node 'relu' reads 't' but nothing produces it`. With `enable_optimizations = false` it works.

**Your mission.** The MatMul output `t` is read by TWO nodes. Watch what the fusion pass decides for it.

<details><summary>Hint 1</summary>

Break at the first guard inside the MatMul loop of `fuse_matmul_add` and print `consumers[t]`.

</details>

<details><summary>Hint 2</summary>

`p g.name(t)` -- gdb can call C++ member functions.

</details>

### Challenge 8 — A backwards loop that never ends

*Stage 2 Graph · single-threaded · apply with* `bug.ps1 apply 8` *· run* `lab_runner 8`

**Symptom.** `std::bad_alloc` (on other machines: a segmentation fault) inside dead-code elimination. The compiler printed a warning -- did you read it?

**Your mission.** Find out what value the loop counter had when it went wrong.

<details><summary>Hint 1</summary>

The warning: `comparison of unsigned expression in '>= 0' is always true`.

</details>

<details><summary>Hint 2</summary>

`catch throw`, then `frame function mo::eliminate_dead_nodes` and `p i`.

</details>

### Challenge 9 — SIMD copy-paste: the wrong register is stored

*Stage 3 EP · single-threaded · apply with* `bug.ps1 apply 9` *· run* `lab_runner 9`

**Symptom.** The AVX2 MatMul disagrees with the reference kernel, starting at **row 1, column 8**. Rows 0, 2, 3 are fine.

**Your mission.** Look inside the SIMD registers of the 4x16 micro-kernel.

<details><summary>Hint 1</summary>

gdb prints a `__m256` as 8 floats: `p c01`, `p c11`.

</details>

<details><summary>Hint 2</summary>

Which register should hold row 1, columns 8..15? Which one is stored there?

</details>

### Challenge 10 — Garbage in, garbage out: the accumulator is never zeroed

*Stage 3 EP · single-threaded · apply with* `bug.ps1 apply 10` *· run* `lab_runner 10`

**Symptom.** `A x Identity` returns `[7.04e+21, 2; 3, 4]` instead of `[1, 2; 3, 4]`. Sometimes it is right -- it depends on what memory was used before.

**Your mission.** Look at the output buffer just before the products are accumulated into it.

<details><summary>Hint 1</summary>

Break at the start of the `p0` loop and examine `x/4f g.Y`.

</details>

<details><summary>Hint 2</summary>

What is `g.mode` for a MatMul (no bias)?

</details>

### Challenge 11 — Softmax returns NaN

*Stage 3 EP · single-threaded · apply with* `bug.ps1 apply 11` *· run* `lab_runner 11`

**Symptom.** `softmax([1000, 0, -1000])` returns `[nan, 0, 0]`.

**Your mission.** Find where the first `inf` appears.

<details><summary>Hint 1</summary>

Break before the division and print `mx`, `sum`, `o[0]@3` (the `@` prints 3 consecutive elements).

</details>

<details><summary>Hint 2</summary>

exp(1000) in float is...?

</details>

### Challenge 12 — The wrong execution provider gets the node

*Stage 3 EP · single-threaded · apply with* `bug.ps1 apply 12` *· run* `lab_runner 12`

**Symptom.** Without the optimizer, the classifier fails on an x86 machine: `Gemm: this kernel needs transA=transB=0`. With the optimizer, or on ARM, it works.

**Your mission.** Find out why the x86 EP accepted a transposed Gemm.

<details><summary>Hint 1</summary>

Break in `X86ExecutionProvider::supports` and print `n.attrs`.

</details>

<details><summary>Hint 2</summary>

`set print object on` + `up` + `p *ep` shows the *dynamic* type behind a base-class pointer.

</details>

### Challenge 13 — A graph output disappears

*Stage 4 Runtime · single-threaded · apply with* `bug.ps1 apply 13` *· run* `lab_runner 13`

**Symptom.** `internal: output 'h' missing` -- `h` is a graph output that is ALSO read by another node.

**Your mission.** Find out who freed `h`.

<details><summary>Hint 1</summary>

`catch throw`, `frame function mo::InferenceSession::collect`, `p plan_.persistent`.

</details>

<details><summary>Hint 2</summary>

Graph outputs are marked persistent. Who checks that flag before freeing?

</details>

### Challenge 14 — Use-after-free: inputs released before the kernel runs

*Stage 4 Runtime · single-threaded · apply with* `bug.ps1 apply 14` *· run* `lab_runner 14`

**Symptom.** `malloc(): invalid size (unsorted)` and the program aborts. Classic heap corruption.

**Your mission.** The crash is inside `malloc`, but the bug is not. Read the backtrace upwards until you reach MiniOnnx code and look at the arguments.

<details><summary>Hint 1</summary>

Frame `Tensor::empty (shape=std::vector of length 0)` -- why would a kernel create an output with an EMPTY shape?

</details>

<details><summary>Hint 2</summary>

Where did that shape come from? Which object has already been destroyed?

</details>

### Challenge 15 — Data race: the profile lock was 'too slow'

*Stage 4 Runtime · multithreaded · apply with* `bug.ps1 apply 15` *· run* `lab_runner 15`

**Symptom.** Random crashes during profiled parallel runs: sometimes `double free or corruption`, sometimes a segfault in `~NodeEvent`. Different every run.

**Your mission.** Several threads write to the same `std::vector`. Prove it.

<details><summary>Hint 1</summary>

`info threads` -- how many threads were running nodes?

</details>

<details><summary>Hint 2</summary>

Which shared object do they all `push_back` into? What protects it?

</details>

### Challenge 16 — Deadlock: the caller just waits

*Stage core · multithreaded · apply with* `bug.ps1 apply 16` *· run* `lab_runner 16`

**Symptom.** Scenario 16 hangs forever (0% CPU).

**Your mission.** Interrupt the program (Ctrl+C in gdb, or the pause button in VS Code) and find out what every thread is waiting for.

<details><summary>Hint 1</summary>

`thread apply all bt`.

</details>

<details><summary>Hint 2</summary>

Thread 2 (the only worker) runs an OUTER chunk which calls an INNER `parallel_for`. Who runs the inner chunks?

</details>

### Challenge 17 — Lost wake-up: notify before the state changes

*Stage 4 Runtime · multithreaded · apply with* `bug.ps1 apply 17` *· run* `lab_runner 17`

**Symptom.** The parallel executor hangs after a few hundred runs. The CPU is idle.

**Your mission.** Interrupt, go to the caller's frame in `run_parallel` and check the condition it is waiting for.

<details><summary>Hint 1</summary>

`p st.pending` and `p st.ready_q`. If pending is 0 and the queue is empty, why is the caller still waiting?

</details>

<details><summary>Hint 2</summary>

Look at the order of `notify_all()` and `pending.fetch_sub()` in `process()`.

</details>

### Challenge 18 — Lost update: a plain int counter shared by threads

*Stage 4 Runtime · multithreaded · apply with* `bug.ps1 apply 18` *· run* `lab_runner 18`

**Symptom.** With 16 branches and 8 inter-op threads the executor hangs after a few thousand runs (or, rarely, returns wrong results).

**Your mission.** Interrupt and compare the in-degree counters with what they should be.

<details><summary>Hint 1</summary>

In `run_parallel`: `p st.pending` -- several nodes never finished, the queue is empty.

</details>

<details><summary>Hint 2</summary>

Two parent nodes of the same merge node finish at the same time and both decrement `indeg[merge]`. With a plain `int`...

</details>

### Challenge 19 — An exception escapes a worker thread

*Stage 4 Runtime · multithreaded · apply with* `bug.ps1 apply 19` *· run* `lab_runner 19`

**Symptom.** `terminate called after throwing an instance of ...` / `terminate called recursively` -- the process aborts instead of reporting the shape error.

**Your mission.** Find on which thread the exception is thrown.

<details><summary>Hint 1</summary>

`catch throw`, then `info threads` -- the `*` marks the current thread.

</details>

<details><summary>Hint 2</summary>

What happens to an exception that leaves a `std::thread`'s function?

</details>

### Challenge 20 — The program never exits: workers never hear 'stop'

*Stage core · multithreaded · apply with* `bug.ps1 apply 20` *· run* `lab_runner 20`

**Symptom.** Scenario 20 prints nothing after the first session and never exits.

**Your mission.** Interrupt and look at the main thread's stack.

<details><summary>Hint 1</summary>

Main is in `~ThreadPool` -> `std::thread::join()`.

</details>

<details><summary>Hint 2</summary>

The workers are in `cv.wait`. Who is supposed to wake them up?

</details>

## 6. Walkthroughs and solutions (spoilers!)

Each walkthrough shows the gdb commands, the **real** transcript, the root cause, the fix, and how to introduce the bug by hand.

### Walkthrough 1 — The varint that counts in 8s

**gdb commands**
```
gdb --args ./build/lab_runner 1
(gdb) break src/stage1_format/protobuf_wire.hpp:92 if (b & 0x80)
(gdb) run
(gdb) p/x b
(gdb) p shift
(gdb) finish
```

**Real transcript**
```
Breakpoint 1 at 0x7fe61: file src/stage1_format/protobuf_wire.hpp, line 92.
SCENARIO 1

Breakpoint 1, mo::WireReader::read_varint (this=0x7fffffff9750) at src/stage1_format/protobuf_wire.hpp:92
92	            result |= uint64_t(b & 0x7F) << shift;
$1 = 0xfe
$2 = 0
0x00005555555d4982 in mo::WireReader::read_bytes_at (this=0x7fffffff9750, payload_offset=@0x7fffffff96e0: 140737488328448) at src/stage1_format/protob ...
165	        uint64_t n = read_varint();
Value returned is $3 = 638
```

**Root cause.** `shift += 8` instead of `+= 7`. Each varint byte carries **7** payload bits. `0xFE 0x02` = 0x7E + (2 << 7) = 126 + 256 = **382**, but with the bug it is 126 + (2 << 8) = **638**.

**Fix.** `for (int shift = 0; shift < 64; shift += 7)`

**Lesson.** Small numbers (< 128) fit in one byte, so most fields were read correctly and the bug only showed for long messages. Always test the boundary: 127, 128, 16383, 16384 (test group 2 does exactly that).

**Try also.** Run `inspect_onnx models/xor.onnx --wire 10` with the clean code and compare the offsets with what you saw.

**Introduce it by hand** (instead of the patch):

In `src/stage1_format/protobuf_wire.hpp` replace
```cpp
for (int shift = 0; shift < 64; shift += 7) {
```
with
```cpp
for (int shift = 0; shift < 64; shift += 8) {
```

### Walkthrough 2 — One wrong shift in read_fixed32

**gdb commands**
```
gdb --args ./build/lab_runner 2
(gdb) break src/stage1_format/protobuf_wire.hpp:111
(gdb) run
(gdb) p/x bits
(gdb) x/4xb p_ - 4
(gdb) next
(gdb) p f
```

**Real transcript**
```
Breakpoint 1 at 0x8005c: file src/stage1_format/protobuf_wire.hpp, line 111.
SCENARIO 2

Breakpoint 1, mo::WireReader::read_float (this=0x7fffffff9140) at src/stage1_format/protobuf_wire.hpp:111
111	        std::memcpy(&f, &bits, 4);                 // memcpy = the legal way to reinterpret bits
$1 = 0x7f0000
0x555555644226:	0x00	0x00	0x40	0x3f
112	        return f;
$2 = 1.1663108e-38
```

**Root cause.** In `read_fixed32` the last byte is shifted by 16 instead of 24, so `0x3F` lands on top of `0x40`: bits = `0x007F0000` -> 1.17e-38 instead of 0.75.

**Fix.** `... | uint32_t(p_[3]) << 24;`

**Lesson.** When numbers are 'a bit off', look at the raw bytes. `x/4xb` + knowing the IEEE-754 layout beats staring at decimal numbers.

**Try also.** `p/t bits` prints the bits in binary. Find the sign, exponent and mantissa of 0.75.

**Introduce it by hand** (instead of the patch):

In `src/stage1_format/protobuf_wire.hpp` replace
```cpp
uint32_t v = uint32_t(p_[0]) | uint32_t(p_[1]) << 8 | uint32_t(p_[2]) << 16 | uint32_t(p_[3]) << 24;
```
with
```cpp
uint32_t v = uint32_t(p_[0]) | uint32_t(p_[1]) << 8 | uint32_t(p_[2]) << 16 | uint32_t(p_[3]) << 16;
```

### Walkthrough 3 — Bytes or floats? memcpy copies too little

**gdb commands**
```
gdb --args ./build/lab_runner 3
(gdb) set print frame-arguments none
(gdb) break src/stage1_format/weights.cpp:41
(gdb) run
(gdb) p n
(gdb) next
(gdb) p dst[0]
(gdb) p ((float*)src)[0]
(gdb) p dst[n-1]
(gdb) p ((float*)src)[n-1]
```

**Real transcript**
```
Breakpoint 1 at 0x86972: file src/stage1_format/weights.cpp, line 41.
SCENARIO 3

Breakpoint 1, mo::decode_le_floats (src=..., n=..., dst=...) at src/stage1_format/weights.cpp:41
41	        if (n) std::memcpy(dst, src, n);
$1 = 2048
42	        return;
$2 = -0.122246101
$3 = -0.122246101
$4 = -1.60789583e+14
$5 = 0.13083756
```

**Root cause.** `memcpy(dst, src, n)` copies `n` **bytes**, i.e. only a quarter of the floats. The rest of the arena is uninitialized garbage (here: huge values -> overflow -> NaN).

**Fix.** `std::memcpy(dst, src, n * 4)` (or `n * sizeof(float)`).

**Lesson.** 'Count of elements' vs 'count of bytes' is one of the most common C/C++ bugs. Name variables `n_floats` / `n_bytes`.

**Try also.** Set a breakpoint on the same line and print `x/4f dst + n/4 - 2` before and after `next`: you can see exactly where the copy stops.

**Introduce it by hand** (instead of the patch):

In `src/stage1_format/weights.cpp` replace
```cpp
        if (n) std::memcpy(dst, src, n * 4);
```
with
```cpp
        if (n) std::memcpy(dst, src, n);
```

### Walkthrough 4 — External weights read from the wrong offset

**gdb commands**
```
gdb --args ./build/lab_runner 4
(gdb) break src/stage1_format/weights.cpp:121
(gdb) run
(gdb) p i
(gdb) p off_s
(gdb) p len_s
(gdb) p off
(gdb) continue
(gdb) p i
(gdb) p off_s
(gdb) p off
```

**Real transcript**
```
Breakpoint 1 at 0x88098: file src/stage1_format/weights.cpp, line 121.
SCENARIO 4

Breakpoint 1, mo::WeightStore::load (this=0x7fffffff9838, g=..., base_dir="models") at src/stage1_format/weights.cpp:121
121	            decode_le_floats(blob.data() + off, n, dst);
$1 = 0
$2 = "0"
$3 = "8192"
$4 = 8192

Breakpoint 1, mo::WeightStore::load (this=0x7fffffff9838, g=..., base_dir="models") at src/stage1_format/weights.cpp:121
121	            decode_le_floats(blob.data() + off, n, dst);
$5 = 2
$6 = "8192"
$7 = 8192
```

**Root cause.** Copy-paste: the offset is parsed from `len_s` instead of `off_s`. Tensor 0 (offset 0, length 8192) is read from byte 8192 -- the next tensor's data.

**Fix.** `size_t(std::stoull(off_s))`

**Lesson.** Copy-pasted lines are where typos hide. If two lines differ in one word, read that word twice.

**Try also.** Make the bug worse: what error do you get when the wrong offset runs past the end of the file?

**Introduce it by hand** (instead of the patch):

In `src/stage1_format/weights.cpp` replace
```cpp
const size_t off = off_s.empty() ? 0 : size_t(std::stoull(off_s));
```
with
```cpp
const size_t off = off_s.empty() ? 0 : size_t(std::stoull(len_s));
```

### Walkthrough 5 — A mystery std::out_of_range (1-D bias)

**gdb commands**
```
gdb --args ./build/lab_runner 5
(gdb) catch throw
(gdb) run
(gdb) bt 7
(gdb) frame 5
(gdb) p c.shape_
(gdb) p M
(gdb) p N
```

**Real transcript**
```
Catchpoint 1 (throw)
SCENARIO 5

Catchpoint 1 (exception thrown), 0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#0  0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#1  0x00007ffff7ca94a0 in std::__throw_out_of_range_fmt(char const*, ...) () from /lib/x86_64-linux-gnu/libstdc++.so.6
   ... (2 system/STL frame(s) not shown)
#4  0x000055555559d2f4 in mo::Tensor::dim (this=0x555555643820, i=1) at src/core/tensor.hpp:91
#5  0x00005555555b6417 in mo::bias_mode_for (c=..., M=3, N=32) at src/stage3_ep/kernel_utils.hpp:42
#6  0x00005555555b748a in mo::prepare_gemm_notrans (in=std::vector of length 3, capacity 3 = {...}, at=..., g=...) at src/stage3_ep/kernel_utils.hpp:7 ...
#5  0x00005555555b6417 in mo::bias_mode_for (c=..., M=3, N=32) at src/stage3_ep/kernel_utils.hpp:42
42	    if (c.dim(1) == N && c.dim(0) == 1) return BiasMode::Row;                          // a [1,N] row
$1 = std::vector of length 1, capacity 1 = {32}
$2 = 3
$3 = 32
```

**Root cause.** `bias_mode_for` calls `c.dim(1)` assuming the bias is 2-D `[1,N]`. PyTorch exports 1-D biases `[N]`; `dim(1)` -> `vector::at(1)` -> out_of_range.

**Fix.** Check the rank first: `if (c.rank() <= 2 && c.cols() == N && c.rows() == 1)`.

**Lesson.** `catch throw` is the single most useful gdb command for C++. In VS Code it is in the launch.json `setupCommands` of 'Windows: debug tests'.

**Try also.** Try `catch throw std::out_of_range` (only that type) and `catch catch` (where it is caught).

**Introduce it by hand** (instead of the patch):

In `src/stage3_ep/kernel_utils.hpp` replace
```cpp
    if (c.rank() <= 2 && c.cols() == N && c.rows() == 1) return BiasMode::Row;          // {N} or {1,N}
```
with
```cpp
    if (c.dim(1) == N && c.dim(0) == 1) return BiasMode::Row;                          // a [1,N] row
```

### Walkthrough 6 — 'Graph has a cycle' -- but it doesn't

**gdb commands**
```
gdb --args ./build/lab_runner 6
(gdb) break src/stage2_graph/graph.cpp:131
(gdb) run
(gdb) p order
(gdb) p indegree
(gdb) p dependents
(gdb) p nodes[1].inputs
```

**Real transcript**
```
Breakpoint 1 at 0x3c48a: file src/stage2_graph/graph.cpp, line 131.
SCENARIO 6

Breakpoint 1, mo::Graph::topological_order (this=0x7fffffff9a50) at src/stage2_graph/graph.cpp:131
131	    if (int(order.size()) != N) throw std::runtime_error("graph has a cycle: not a valid DAG");
$1 = std::vector of length 1, capacity 2 = {0}
$2 = std::vector of length 2, capacity 2 = {0, 1}
$3 = std::vector of length 2, capacity 2 = {std::vector of length 1, capacity 1 = {1}, std::vector of length 0, capacity 0}
$4 = std::vector of length 2, capacity 2 = {1, 1}
```

**Root cause.** An 'optimization' removed duplicate entries from `dependents`, but `indegree` still counts every edge. Node 1 has indegree 2 and is decremented only once -> stuck at 1 -> looks like a cycle.

**Fix.** Keep one entry per edge (remove the `std::find` check) -- or count indegree per distinct producer. The two must agree.

**Lesson.** Two data structures that must stay consistent should be built in ONE place. This is the same reason the runtime counts consumers per *input slot* (Mul(x,x)).

**Try also.** `p dependents[0]` before and after the bug: one element vs two.

**Introduce it by hand** (instead of the patch):

In `src/stage2_graph/graph.cpp` replace
```cpp
            dependents[size_t(p)].push_back(i);
            indegree[size_t(i)]++;
```
with
```cpp
            if (std::find(dependents[size_t(p)].begin(), dependents[size_t(p)].end(), i) == dependents[size_t(p)].end())
                dependents[size_t(p)].push_back(i);                // avoid duplicate entries
            indegree[size_t(i)]++;
```

### Walkthrough 7 — The optimizer fuses away a tensor someone still needs

**gdb commands**
```
gdb --args ./build/lab_runner 7
(gdb) break src/stage2_graph/optimizer.cpp:89
(gdb) run
(gdb) p t
(gdb) p g.name(t)
(gdb) p consumers[t]
```

**Real transcript**
```
Breakpoint 1 at 0x4809f: file src/stage2_graph/optimizer.cpp, line 89.
SCENARIO 7

Breakpoint 1, mo::fuse_matmul_add (g=...) at src/stage2_graph/optimizer.cpp:89
89	        if (is_graph_output(g, t)) continue;                                        // (1)
$1 = 3
$2 = "t"
$3 = 2
```

**Root cause.** The guard `consumers[t] != 1` was removed. The pass fused `MatMul + Add` into a Gemm, so `t` no longer exists -- but the Relu still needs it.

**Fix.** Restore `if (consumers[size_t(t)] != 1 || is_graph_output(g, t)) continue;`

**Lesson.** An optimizer pass must prove it is safe. 'Every guard looks unnecessary until you remove it.' (The original tinyONNX had a real bug of this family -- see the PDF.)

**Try also.** Why does the session re-run `g.validate()` after optimizing? What would happen without it?

**Introduce it by hand** (instead of the patch):

In `src/stage2_graph/optimizer.cpp` replace
```cpp
        if (consumers[size_t(t)] != 1 || is_graph_output(g, t)) continue;          // (1)
```
with
```cpp
        if (is_graph_output(g, t)) continue;                                        // (1)
```

### Walkthrough 8 — A backwards loop that never ends

**gdb commands**
```
gdb --args ./build/lab_runner 8
(gdb) catch throw
(gdb) run
(gdb) bt 6
(gdb) frame function mo::eliminate_dead_nodes
(gdb) p i
(gdb) p g.nodes.size()
```

**Real transcript**
```
Catchpoint 1 (throw)
SCENARIO 8

Catchpoint 1 (exception thrown), 0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#0  0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#1  0x00007ffff7ca5ac8 in ?? () from /lib/x86_64-linux-gnu/libstdc++.so.6
#2  0x00007ffff7d6b85b in void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char*>(char*, char*, std: ...
#3  0x00005555555a1448 in mo::Node::Node (this=0x7fffffff9d08) at src/stage2_graph/graph.hpp:47
   ... (2 system/STL frame(s) not shown)
#8  0x000055555559cc61 in mo::eliminate_dead_nodes (g=...) at src/stage2_graph/optimizer.cpp:141
141	    for (size_t i = g.nodes.size() - 1; i >= 0; i--) if (live[i]) kept.insert(kept.begin(), g.nodes[i]);
$1 = 18446744073709551608
$2 = 8
```

**Root cause.** `size_t` is unsigned: `i >= 0` is always true. After `i = 0`, `i--` wraps around to 18446744073709551615 and the loop reads far outside the vector.

**Fix.** Loop forwards, or `for (size_t i = n; i-- > 0;)`, or use a signed index.

**Lesson.** Build with `-Wall -Wextra` and treat warnings as bugs (`-DMINIONNX_WARNINGS_AS_ERRORS=ON`).

**Try also.** `p/x i` -- what does 0xfffffffffffffff8 mean?

**Introduce it by hand** (instead of the patch):

In `src/stage2_graph/optimizer.cpp` replace
```cpp
    for (size_t i = 0; i < g.nodes.size(); i++) if (live[i]) kept.push_back(g.nodes[i]);
```
with
```cpp
    // walk backwards and insert at the front, so the original order is kept
    for (size_t i = g.nodes.size() - 1; i >= 0; i--) if (live[i]) kept.insert(kept.begin(), g.nodes[i]);
```

### Walkthrough 9 — SIMD copy-paste: the wrong register is stored

**gdb commands**
```
gdb --args ./build/lab_runner 9
(gdb) break src/stage3_ep/x86_ep.cpp:95
(gdb) run
(gdb) p i
(gdb) p j
(gdb) p c01
(gdb) p c11
```

**Real transcript**
```
Breakpoint 1 at 0x8c0cc: file src/stage3_ep/x86_ep.cpp, line 95.
SCENARIO 9

Breakpoint 1, mo::(anonymous namespace)::gemm_rows_avx2 (g=..., rb=0, re=8) at src/stage3_ep/x86_ep.cpp:95
95	            store8(g, i, j, c00);     store8(g, i, j + 8, c01);
$1 = 0
$2 = 0
$3 = {-0.0604905039, -1.27438366, -2.31579638, -3.04377484, -3.35979462, -3.22108078, -2.64640856, -1.71355987}
$4 = {-1.27858663, -1.45498276, -1.43445349, -1.2197789, -0.840011895, -0.346553832, 0.193809345, 0.707940936}
```

**Root cause.** Copy-paste in the store sequence: row 1, columns 8..15 is written from `c01` (row 0's accumulator) instead of `c11`. `c11` is computed and thrown away.

**Fix.** `store8(g, i + 1, j + 8, c11);`

**Lesson.** Unrolled SIMD code is copy-paste heaven. Test shapes that exercise every unrolled path (the unit test uses N = 16, 40, 65...).

**Try also.** `info registers ymm0` shows a raw register (in Release builds variables live in registers, not memory).

**Introduce it by hand** (instead of the patch):

In `src/stage3_ep/x86_ep.cpp` replace
```cpp
store8(g, i + 1, j, c10); store8(g, i + 1, j + 8, c11);
```
with
```cpp
store8(g, i + 1, j, c10); store8(g, i + 1, j + 8, c01);
```

### Walkthrough 10 — Garbage in, garbage out: the accumulator is never zeroed

**gdb commands**
```
gdb --args ./build/lab_runner 10
(gdb) break src/stage3_ep/cpu_ep.cpp:29
(gdb) run
(gdb) p g.mode
(gdb) p g.M
(gdb) p g.N
(gdb) x/4f g.Y
```

**Real transcript**
```
Breakpoint 1 at 0x4fd65: file src/stage3_ep/cpu_ep.cpp, line 29.
SCENARIO 10

Breakpoint 1, operator() (__closure=0x7fffffff9d00, rb=0, re=2) at src/stage3_ep/cpu_ep.cpp:29
29	        for (int64_t p0 = 0; p0 < g.K; p0 += KB) {
$1 = mo::BiasMode::None
$2 = 2
$3 = 2
0x55555563fb80:	1.46604001e+13	7.00649232e-45	0	0
```

**Root cause.** The initialization loop was skipped when there is no bias, so the accumulators start from whatever the allocator left in the memory (the test deliberately leaves 7777s behind).

**Fix.** Always initialize: bias if there is one, otherwise 0.

**Lesson.** `Tensor::empty` is uninitialized on purpose (speed). Every kernel must write every output element before reading it. AddressSanitizer does not catch this; MemorySanitizer/Valgrind do.

**Try also.** `watch -l g.Y[0]` then `continue`: see each write to that float.

**Introduce it by hand** (instead of the patch):

In `src/stage3_ep/cpu_ep.cpp` replace
```cpp
        for (int64_t i = rb; i < re; i++)
            for (int64_t j = 0; j < g.N; j++) g.Y[i * g.N + j] = alpha_one ? g.bias_at(i, j) : 0.f;
```
with
```cpp
        if (g.mode != BiasMode::None)                                  // nothing to add without a bias
            for (int64_t i = rb; i < re; i++)
                for (int64_t j = 0; j < g.N; j++) g.Y[i * g.N + j] = alpha_one ? g.bias_at(i, j) : 0.f;
```

### Walkthrough 11 — Softmax returns NaN

**gdb commands**
```
gdb --args ./build/lab_runner 11
(gdb) break src/stage3_ep/cpu_ep.cpp:138
(gdb) run
(gdb) p mx
(gdb) p sum
(gdb) p o[0]@3
```

**Real transcript**
```
Breakpoint 1 at 0x50f70: file src/stage3_ep/cpu_ep.cpp, line 138.
SCENARIO 11

Breakpoint 1, operator() (__closure=0x7fffffff9e00, rb=0, re=1) at src/stage3_ep/cpu_ep.cpp:138
138	            for (int64_t j = 0; j < N; j++) o[j] /= sum;
$1 = 1000
$2 = inf
$3 = {inf, 1, 0}
```

**Root cause.** The 'unnecessary' `- mx` was removed. exp(1000) = inf, inf / inf = NaN.

**Fix.** `o[j] = std::exp(x[j] - mx)` -- mathematically identical, numerically safe.

**Lesson.** Numerically stable formulas look redundant. Comment them so nobody 'simplifies' them.

**Try also.** Conditional breakpoint on a float: `break cpu_ep.cpp:138 if sum > 3e38` stops only when the sum overflowed (and `if x != x` is true only for NaN).

**Introduce it by hand** (instead of the patch):

In `src/stage3_ep/cpu_ep.cpp` replace
```cpp
for (int64_t j = 0; j < N; j++) { o[j] = std::exp(x[j] - mx); sum += o[j]; }
```
with
```cpp
for (int64_t j = 0; j < N; j++) { o[j] = std::exp(x[j]); sum += o[j]; }
```

### Walkthrough 12 — The wrong execution provider gets the node

**gdb commands**
```
gdb --args ./build/lab_runner 12
(gdb) set print object on
(gdb) break src/stage3_ep/x86_ep.cpp:219
(gdb) run
(gdb) p n.name
(gdb) p n.op_type
(gdb) p n.attrs
(gdb) up
(gdb) p *ep
```

**Real transcript**
```
Breakpoint 1 at 0x8dc28: file src/stage3_ep/x86_ep.cpp, line 219.
SCENARIO 12

Breakpoint 1, mo::(anonymous namespace)::X86ExecutionProvider::supports (this=0x555555642690, n=..., g=...) at src/stage3_ep/x86_ep.cpp:219
219	        return ExecutionProvider::supports(n, g);
$1 = "fc1"
$2 = "Gemm"
$3 = {map_ = std::map with 1 element = {["transB"] = {name = "transB", f = 0, i = 1, s = "", floats = std::vector of length 0, capacity 0, ints = std: ...
#1  0x00005555555ba79e in mo::assign_providers (g=..., eps=std::vector of length 2, capacity 2 = {...}) at src/stage3_ep/registry.cpp:29
29	            if (ep->available() && ep->supports(n, g)) { n.ep = ep->name(); break; }
$4 = (mo::(anonymous namespace)::X86ExecutionProvider) {<mo::ExecutionProvider> = {_vptr.ExecutionProvider = 0x55555562b368 <vtable for mo::(anonymous ...
```

**Root cause.** The capability check was removed: `supports()` only checks the op name, so a `Gemm(transB=1)` is given to a kernel that only handles the non-transposed form.

**Fix.** Restore the `transA/transB` check in `supports()` (GetCapability).

**Lesson.** Partitioning must ask 'can you run THIS node?', not 'do you know this op?'. ORT's GetCapability does the same.

**Try also.** Step into `ep->supports(n, g)` with `step` and watch gdb jump to the right override.

**Introduce it by hand** (instead of the patch):

In `src/stage3_ep/x86_ep.cpp` replace
```cpp
    // GetCapability: the fast Gemm kernel only handles the non-transposed form.
    bool supports(const Node &n, const Graph &g) const override {
        if (n.op_type == "Gemm" && (n.attrs.get_int("transA", 0) || n.attrs.get_int("transB", 0))) return false;
        return ExecutionProvider::supports(n, g);
    }
```
with
```cpp
    // GetCapability: the fast Gemm kernel only handles the non-transposed form.
    bool supports(const Node &n, const Graph &g) const override {
        return ExecutionProvider::supports(n, g);
    }
```

### Walkthrough 13 — A graph output disappears

**gdb commands**
```
gdb --args ./build/lab_runner 13
(gdb) catch throw
(gdb) run
(gdb) bt 3
(gdb) frame function mo::InferenceSession::collect
(gdb) p plan_.value_names[v]
(gdb) p plan_.persistent
(gdb) p plan_.consumer_count
```

**Real transcript**
```
Catchpoint 1 (throw)
SCENARIO 13

Catchpoint 1 (exception thrown), 0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#0  0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#1  0x00005555555bf604 in mo::InferenceSession::collect[abi:cxx11](mo::InferenceSession::RunState&, mo::RunProfile*) const (this=0x7fffffff9f50, st=.. ...
#2  0x00005555555bcddd in mo::InferenceSession::run (this=0x7fffffff9f50, feeds=std::unordered_map with 1 element = {...}, profile=0x0) at src/stage4_ ...
#1  0x00005555555bf604 in mo::InferenceSession::collect[abi:cxx11](mo::InferenceSession::RunState&, mo::RunProfile*) const (this=0x7fffffff9f50, st=.. ...
316	        if (!slot) throw std::logic_error("internal: output '" + plan_.value_names[size_t(v)] + "' missing");
$1 = "h"
$2 = std::vector of length 3, capacity 3 = {0 '\000', 1 '\001', 1 '\001'}
$3 = std::vector of length 3, capacity 3 = {1, 1, 0}
```

**Root cause.** The `!plan_.persistent[v]` test was removed from the input-release loop, so the last reader (the Sigmoid) freed `h` although it must be returned.

**Fix.** Restore the persistent check.

**Lesson.** Reference counting needs a notion of 'owned elsewhere'. Here: weights and outputs are never freed by the run.

**Try also.** Advanced: `break exec_node if idx == 1`, then `watch -l st.slots[1]._M_ptr` -- gdb stops at the exact `reset()`.

**Introduce it by hand** (instead of the patch):

In `src/stage4_runtime/session.cpp` replace
```cpp
        if (v != kNoValue && !plan_.persistent[size_t(v)] &&
            st.remaining[size_t(v)].fetch_sub(1, std::memory_order_acq_rel) == 1) {
```
with
```cpp
        if (v != kNoValue &&
            st.remaining[size_t(v)].fetch_sub(1, std::memory_order_acq_rel) == 1) {
```

### Walkthrough 14 — Use-after-free: inputs released before the kernel runs

**gdb commands**
```
gdb --args ./build/lab_runner 14
(gdb) run
(gdb) bt 22
```

**Real transcript**
```
SCENARIO 14
malloc(): invalid size (unsorted)

Program received signal SIGABRT, Aborted.
__pthread_kill_implementation (no_tid=0, signo=6, threadid=<optimized out>) at ./nptl/pthread_kill.c:44
   ... (12 system/STL frame(s) not shown)
#12 0x00007ffff7cbb9f6 in operator new(unsigned long, std::align_val_t) () from /lib/x86_64-linux-gnu/libstdc++.so.6
#13 0x000055555556b047 in mo::aligned_bytes (bytes=64) at src/core/tensor.hpp:47
#14 0x000055555556b2ba in mo::Tensor::empty (shape=std::vector of length 0, capacity 0) at src/core/tensor.hpp:61
#15 0x00005555555a7462 in mo::(anonymous namespace)::binary<mo::(anonymous namespace)::CpuExecutionProvider::CpuExecutionProvider()::<lambda(auto:2&,  ...
#16 0x00005555555a53b8 in operator()<mo::KernelContext const, std::vector<mo::Tensor const*> const, mo::Attributes const> (__closure=0x555555641520, c ...
   ... (4 system/STL frame(s) not shown)
#21 0x00005555555be5f3 in mo::InferenceSession::exec_node (this=0x7fffffff9f50, st=..., idx=1) at src/stage4_runtime/session.cpp:211
```

**Root cause.** Inputs were released BEFORE the kernel ran. For the last reader, `slots[v].reset()` destroyed the input tensor; the kernel then read its shape and data from freed memory and corrupted the heap.

**Fix.** Release inputs only after the kernel returned (original order).

**Lesson.** Raw pointers (`inputs` holds `const Tensor*`) do not keep anything alive. Lifetime must be guaranteed by the owner (the slot) until the user is done.

**Try also.** On WSL: build with the `linux-asan` preset -- ASan reports `heap-use-after-free` with the line that freed and the line that used.

**Introduce it by hand** (instead of the patch):

In `src/stage4_runtime/session.cpp` replace
```cpp
        inputs.push_back(p);
    }

    const double start
```
with
```cpp
        inputs.push_back(p);
    }
    // release inputs early so memory is reused sooner
    for (int v : n.in)
        if (v != kNoValue && !plan_.persistent[size_t(v)] &&
            st.remaining[size_t(v)].fetch_sub(1, std::memory_order_acq_rel) == 1) {
            st.slots[size_t(v)].reset();
            st.live--;
        }

    const double start
```

In `src/stage4_runtime/session.cpp` replace
```cpp
    // Release inputs AFTER the kernel is done with them. Whoever drops a count to zero frees
    // the tensor; every other reader has already decremented, so nobody still uses it.
    for (int v : n.in)
        if (v != kNoValue && !plan_.persistent[size_t(v)] &&
            st.remaining[size_t(v)].fetch_sub(1, std::memory_order_acq_rel) == 1) {
            st.slots[size_t(v)].reset();
            st.live--;
        }
}
```
with
```cpp
}
```

### Walkthrough 15 — Data race: the profile lock was 'too slow'

**gdb commands**
```
gdb --args ./build/lab_runner 15
(gdb) run
(gdb) bt 14
(gdb) info threads
```

**Real transcript**
```
SCENARIO 15

Thread 1 "lab_runner" received signal SIGSEGV, Segmentation fault.
0x00007ffff78ade35 in __GI___libc_free (mem=0xe261d91fc42d50d7) at ./malloc/malloc.c:3375
   ... (1 system/STL frame(s) not shown)
#1  0x000055555558734c in mo::NodeEvent::~NodeEvent (this=0x555555643aa0, __in_chrg=<optimized out>) at src/stage4_runtime/session.hpp:40
   ... (5 system/STL frame(s) not shown)
#7  0x0000555555572a5e in mo::RunProfile::~RunProfile (this=0x7fffffff9d40, __in_chrg=<optimized out>) at src/stage4_runtime/session.hpp:46
#8  0x0000555555565386 in (anonymous namespace)::s15 () at lab/lab_runner.cpp:195
#9  0x0000555555567591 in main (argc=2, argv=0x7fffffffa4d8) at lab/lab_runner.cpp:281
  Id   Target Id                                     Frame 
* 1    Thread 0x7ffff7e91740 (LWP 8458) "lab_runner" 0x00007ffff78ade35 in __GI___libc_free (mem=0xe261d91fc42d50d7) at ./malloc/malloc.c:3375
  2    Thread 0x7ffff77ff6c0 (LWP 8461) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=0, cancel=true, abstime=0x0, op=393 ...
  3    Thread 0x7ffff6ffe6c0 (LWP 8462) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=0, cancel=true, abstime=0x0, op=393 ...
  4    Thread 0x7ffff67fd6c0 (LWP 8463) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=21845, cancel=true, abstime=0x0, op ...
```

**Root cause.** The `lock_guard` on `st.pm` was removed 'for speed'. `std::vector::push_back` and `unordered_map::find` are not thread-safe: two threads reallocating the same vector corrupt the heap.

**Fix.** Restore `std::lock_guard<std::mutex> lk(st.pm);`

**Lesson.** A data race is undefined behaviour: the symptom changes from run to run, and appears far from the cause (here: in a destructor). TSan finds the cause directly.

**Try also.** Run the TSan build on WSL (`linux-tsan` preset). The first report names `thread_index()` and `exec_node` -- the unprotected accesses.

**Introduce it by hand** (instead of the patch):

In `src/stage4_runtime/session.cpp` replace
```cpp
    if (st.prof) {
        std::lock_guard<std::mutex> lk(st.pm);
        st.prof->events.push_back(
```
with
```cpp
    if (st.prof) {
        st.prof->events.push_back(
```

### Walkthrough 16 — Deadlock: the caller just waits

**gdb commands**
```
gdb --args ./build/lab_runner 16
(gdb) run
(gdb) info threads
(gdb) thread apply all bt 7
```

*(the program hangs: press Ctrl+C / the VS Code pause button after a few seconds, then type the rest)*

**Real transcript**
```
SCENARIO 16

Thread 1 "lab_runner" received signal SIGINT, Interrupt.
0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=0, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0x55555563f710) at ./nptl/fu ...
  Id   Target Id                                     Frame 
* 1    Thread 0x7ffff7e91740 (LWP 7950) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=0, cancel=true, abstime=0x0, op=393 ...
  2    Thread 0x7ffff77ff6c0 (LWP 7953) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=21845, cancel=true, abstime=0x0, op ...

Thread 2 (Thread 0x7ffff77ff6c0 (LWP 7953) "lab_runner"):
   ... (5 system/STL frame(s) not shown)
#5  0x0000555555568af1 in std::condition_variable::wait<mo::ThreadPool::parallel_for<(anonymous namespace)::s16()::<lambda(int64_t, int64_t)>::<lambda ...
#6  0x0000555555567ead in mo::ThreadPool::parallel_for<(anonymous namespace)::s16()::<lambda(int64_t, int64_t)>::<lambda(int64_t, int64_t)> >(int64_t, ...

Thread 1 (Thread 0x7ffff7e91740 (LWP 7950) "lab_runner"):
   ... (5 system/STL frame(s) not shown)
#5  0x0000555555568c47 in std::condition_variable::wait<mo::ThreadPool::parallel_for<(anonymous namespace)::s16()::<lambda(int64_t, int64_t)> >(int64_ ...
#6  0x0000555555568837 in mo::ThreadPool::parallel_for<(anonymous namespace)::s16()::<lambda(int64_t, int64_t)> >(int64_t, int64_t, int64_t, struct {. ...
```

**Root cause.** The caller no longer works in `parallel_for`, it only waits. The single worker is busy running the outer chunk, which waits for an inner helper task that is queued behind it -- nobody is free to run it. Deadlock.

**Fix.** The caller claims chunks too (`run();`), so a `parallel_for` can always finish on its own thread.

**Lesson.** Nested parallelism on a bounded pool deadlocks unless waiting threads help. This is rule 2 of the thread pool.

**Try also.** Try with `ThreadPool pool(8)`: does it still hang? Why does a bigger pool only hide the bug?

**Introduce it by hand** (instead of the patch):

In `src/core/thread_pool.hpp` replace
```cpp
        run();                                                       // the caller works too (rule 2)
```
with
```cpp
        // the caller simply waits for the helpers to finish the chunks
```

### Walkthrough 17 — Lost wake-up: notify before the state changes

**gdb commands**
```
gdb --args ./build/lab_runner 17
(gdb) run
(gdb) info threads
(gdb) thread 1
(gdb) frame function mo::InferenceSession::run_parallel
(gdb) p st.pending
(gdb) p st.ready_q
```

*(the program hangs: press Ctrl+C / the VS Code pause button after a few seconds, then type the rest)*

**Real transcript**
```
SCENARIO 17
   run 0

Thread 1 "lab_runner" received signal SIGINT, Interrupt.
0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=21845, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0x5555556520f8) at ./npt ...
  Id   Target Id                                     Frame 
* 1    Thread 0x7ffff7e91740 (LWP 8036) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=21845, cancel=true, abstime=0x0, op ...
  2    Thread 0x7ffff77ff6c0 (LWP 8039) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=21845, cancel=true, abstime=0x0, op ...
  3    Thread 0x7ffff6ffe6c0 (LWP 8040) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=21845, cancel=true, abstime=0x0, op ...
  4    Thread 0x7ffff67fd6c0 (LWP 8041) "lab_runner" 0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=21845, cancel=true, abstime=0x0, op ...
[Switching to thread 1 (Thread 0x7ffff7e91740 (LWP 8036))]
   ... (1 system/STL frame(s) not shown)
#6  0x00005555555bf2a4 in mo::InferenceSession::run_parallel (this=0x7fffffff9f50, stp=std::shared_ptr<mo::InferenceSession::RunState> (use count 1, w ...
295	            st.cv.wait(lk, [&] { return !st.ready_q.empty() || st.pending.load(std::memory_order_acquire) == 0; });
$1 = std::atomic<int> = { 0 }
$2 = std::deque with 0 elements
```

**Root cause.** Lost wake-up: the helper notifies BEFORE decrementing `pending` (and without the mutex). The caller wakes, sees pending still 1, goes back to sleep; then pending becomes 0 -- but nobody notifies again.

**Fix.** Change the state first, then notify -- and do it under the mutex: `if (pending.fetch_sub(1) == 1) { lock_guard lk(m); cv.notify_all(); }`

**Lesson.** Condition variable rule: modify the predicate, (lock), notify. Never notify about a change that has not happened yet.

**Try also.** Why does the predicate overload `cv.wait(lk, pred)` not save us here?

**Introduce it by hand** (instead of the patch):

In `src/stage4_runtime/session.cpp` replace
```cpp
        if (st.pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(st.m);              // lock before notify: no lost wake-up
            st.cv.notify_all();
        }
```
with
```cpp
        st.cv.notify_all();                                    // wake the caller: we might be the last node
        st.pending.fetch_sub(1, std::memory_order_acq_rel);
```

### Walkthrough 18 — Lost update: a plain int counter shared by threads

**gdb commands**
```
gdb --args ./build/lab_runner 18
(gdb) run
(gdb) thread 1
(gdb) bt 4
(gdb) frame function mo::InferenceSession::run_parallel
(gdb) p st.pending
(gdb) p st.ready_q
(gdb) p plan_.indegree[63]
(gdb) p st.indeg[63]
(gdb) p plan_.nodes[63].name
```

*(the program hangs: press Ctrl+C / the VS Code pause button after a few seconds, then type the rest)*

**Real transcript**
```
SCENARIO 18
   run 0
   run 1000
   run 2000

Thread 1 "lab_runner" received signal SIGINT, Interrupt.
0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=16843093, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0x5555556aec68) at ./ ...
[Switching to thread 1 (Thread 0x7ffff7e91740 (LWP 8090))]
   ... (5 system/STL frame(s) not shown)
#6  0x00005555555bf1e2 in mo::InferenceSession::run_parallel (this=0x7fffffff9f50, stp=std::shared_ptr<mo::InferenceSession::RunState> (use count 1, w ...
297	            st.cv.wait(lk, [&] { return !st.ready_q.empty() || st.pending.load(std::memory_order_acquire) == 0; });
$1 = std::atomic<int> = { 5 }
$2 = std::deque with 0 elements
$3 = 1
$4 = 1
$5 = "sigmoid"
```

**Root cause.** `indeg` became a plain `int` array. `--indeg[d]` is load / subtract / store: two threads can both load 2 and both store 1. The merge node never reaches 0 -> never runs -> `pending` never reaches 0 -> hang.

**Fix.** `std::atomic<int>` with `fetch_sub(1, std::memory_order_acq_rel) == 1`.

**Lesson.** A counter shared by threads must be atomic (or locked). x86 makes this bug rare -- ARM makes it common. Rare is worse than common.

**Try also.** TSan on WSL reports the exact line (`session.cpp:272`) on the first run.

**Introduce it by hand** (instead of the patch):

In `src/stage4_runtime/session.cpp` replace
```cpp
    std::unique_ptr<std::atomic<int>[]> indeg;                // node -> unmet input edges
```
with
```cpp
    std::unique_ptr<int[]> indeg;                             // node -> unmet input edges
```

In `src/stage4_runtime/session.cpp` replace
```cpp
    st.indeg.reset(new std::atomic<int>[N]);
    for (size_t n = 0; n < N; n++) st.indeg[n].store(plan_.indegree[n], std::memory_order_relaxed);
```
with
```cpp
    st.indeg.reset(new int[N]);
    for (size_t n = 0; n < N; n++) st.indeg[n] = plan_.indegree[n];
```

In `src/stage4_runtime/session.cpp` replace
```cpp
            if (st.indeg[size_t(d)].fetch_sub(1, std::memory_order_acq_rel) == 1) ready.push_back(d);
```
with
```cpp
            if (--st.indeg[size_t(d)] == 0) ready.push_back(d);
```

### Walkthrough 19 — An exception escapes a worker thread

**gdb commands**
```
gdb --args ./build/lab_runner 19
(gdb) catch throw
(gdb) run
(gdb) info threads
(gdb) bt 6
```

**Real transcript**
```
Catchpoint 1 (throw)
SCENARIO 19
[Switching to Thread 0x7ffff67fd6c0 (LWP 8154)]

Thread 4 "lab_runner" hit Catchpoint 1 (exception thrown), 0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
  Id   Target Id                                     Frame 
  1    Thread 0x7ffff7e91740 (LWP 8149) "lab_runner" 0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
  2    Thread 0x7ffff77ff6c0 (LWP 8152) "lab_runner" 0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
  3    Thread 0x7ffff6ffe6c0 (LWP 8153) "lab_runner" 0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
* 4    Thread 0x7ffff67fd6c0 (LWP 8154) "lab_runner" 0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#0  0x00007ffff7cbb35a in __cxa_throw () from /lib/x86_64-linux-gnu/libstdc++.so.6
#1  0x00005555555b71e0 in mo::prepare_gemm_notrans (in=std::vector of length 3, capacity 3 = {...}, at=..., g=...) at src/stage3_ep/kernel_utils.hpp:7 ...
#2  0x00005555555a4bf7 in mo::(anonymous namespace)::k_gemm (ctx=..., in=std::vector of length 3, capacity 3 = {...}, at=...) at src/stage3_ep/cpu_ep. ...
   ... (3 system/STL frame(s) not shown)
```

**Root cause.** The `try/catch` in `process()` was removed. A kernel exception on a pool worker unwinds out of the thread function -> `std::terminate()`.

**Fix.** Catch on the worker, store the first `exception_ptr`, rethrow on the caller (`std::rethrow_exception`).

**Lesson.** Exceptions do not cross threads by themselves. You must carry them (`std::exception_ptr`, `std::future`, `std::promise`).

**Try also.** Notice 4 threads hit the catchpoint at once: all branches have the same wrong input.

**Introduce it by hand** (instead of the patch):

In `src/stage4_runtime/session.cpp` replace
```cpp
            try { exec_node(st, idx); }
            catch (...) {
                std::lock_guard<std::mutex> lk(st.m);
                if (!st.failed.exchange(true)) st.first_error = std::current_exception();
            }
```
with
```cpp
            exec_node(st, idx);
```

### Walkthrough 20 — The program never exits: workers never hear 'stop'

**gdb commands**
```
gdb --args ./build/lab_runner 20
(gdb) run
(gdb) thread apply all bt 6
```

*(the program hangs: press Ctrl+C / the VS Code pause button after a few seconds, then type the rest)*

**Real transcript**
```
SCENARIO 20

Thread 1 "lab_runner" received signal SIGINT, Interrupt.
0x00007ffff7898e31 in __futex_abstimed_wait_common64 (private=128, cancel=true, abstime=0x0, op=265, expected=8241, futex_word=0x7ffff6ffe990) at ./np ...

Thread 3 (Thread 0x7ffff6ffe6c0 (LWP 8241) "lab_runner"):
   ... (5 system/STL frame(s) not shown)
#5  0x0000555555574ded in std::condition_variable::wait<mo::ThreadPool::worker_loop()::{lambda()#1}>(std::unique_lock<std::mutex>&, mo::ThreadPool::wo ...

Thread 2 (Thread 0x7ffff77ff6c0 (LWP 8240) "lab_runner"):
   ... (5 system/STL frame(s) not shown)
#5  0x0000555555574ded in std::condition_variable::wait<mo::ThreadPool::worker_loop()::{lambda()#1}>(std::unique_lock<std::mutex>&, mo::ThreadPool::wo ...

Thread 1 (Thread 0x7ffff7e91740 (LWP 8237) "lab_runner"):
   ... (4 system/STL frame(s) not shown)
#4  0x00007ffff7cece33 in std::thread::join() () from /lib/x86_64-linux-gnu/libstdc++.so.6
#5  0x000055555556c244 in mo::ThreadPool::~ThreadPool (this=0x555555643ba0, __in_chrg=<optimized out>) at src/core/thread_pool.hpp:59
```

**Root cause.** The destructor sets `stopping_` but never calls `cv_.notify_all()`. Workers that already went to sleep never re-check the flag -> `join()` waits forever.

**Fix.** Notify after setting the flag (under the lock, as in the original).

**Lesson.** Every 'please stop' flag needs a wake-up. The 200 us spin made this bug invisible when workers were still spinning (that is why the scenario sleeps 2 ms).

**Try also.** Remove the 2 ms sleep from scenario 20: does it still hang? Why only sometimes?

**Introduce it by hand** (instead of the patch):

In `src/core/thread_pool.hpp` replace
```cpp
            stopping_.store(true);
        }
        cv_.notify_all();
```
with
```cpp
            stopping_.store(true);
        }
```

## 7. After the lab

* Re-run `lab_runner all` and `ctest` on clean code: everything must pass.
* Write your own bug: pick any function, break it subtly, write a scenario that exposes it, and hand it to a friend.
* Bonus (compile-time, no gdb needed): change `Shape shape(static_cast<size_t>(rank));` in `tools/minionnx_run.cpp`
  back to `Shape shape(size_t(rank));` and read the error. That is C++'s *most vexing parse* — it bit this project
  three times while it was being written (see the PDF).
