# =====================================================================================
# MiniOnnx -- CMakeLists.txt  (read the PDF chapter "CMake from zero" alongside this file)
# =====================================================================================
# CMake does not compile anything itself. It READS this file and WRITES a build system
# (Makefiles, Ninja files, a Visual Studio solution...) that then calls the compiler.
#
#   cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug     # 1. configure
#   cmake --build build -j                                                 # 2. build
#   ctest --test-dir build --output-on-failure                             # 3. test
# =====================================================================================
cmake_minimum_required(VERSION 3.16)
project(MiniOnnx VERSION 1.0.0 LANGUAGES CXX)

# ---- Options: switches you flip on the command line with -DNAME=VALUE --------------------
option(MINIONNX_ENABLE_CUDA "Build the CUDA execution provider (needs the CUDA toolkit)" OFF)
option(MINIONNX_BUILD_TESTS "Build the unit tests" ON)
option(MINIONNX_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(MINIONNX_SANITIZE "" CACHE STRING "Sanitizer: address, thread, undefined, or empty (Linux/WSL only)")

# ---- Language standard ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)                 # -std=c++17, not -std=gnu++17
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)         # build/compile_commands.json for VS Code IntelliSense

# Default to a Debug build (with -g) when nothing was chosen.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)                # gives us the imported target Threads::Threads (-pthread)

# ---- The library: all four stages ---------------------------------------------------------
add_library(minionnx STATIC
  src/stage1_format/onnx_io.cpp
  src/stage1_format/weights.cpp
  src/stage2_graph/graph.cpp
  src/stage2_graph/optimizer.cpp
  src/stage3_ep/cpu_ep.cpp
  src/stage3_ep/x86_ep.cpp            # compiles to a stub on non-x86 CPUs
  src/stage3_ep/arm_ep.cpp            # compiles to a stub on non-ARM CPUs
  src/stage3_ep/registry.cpp
  src/stage4_runtime/plan.cpp
  src/stage4_runtime/session.cpp
)
# PUBLIC = "I need this, AND anyone who links me needs it too".
target_include_directories(minionnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(minionnx PUBLIC Threads::Threads)

if(MINIONNX_ENABLE_CUDA)
  enable_language(CUDA)
  target_sources(minionnx PRIVATE src/stage3_ep/cuda_ep.cu)
  set_target_properties(minionnx PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES "70;80;86")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(minionnx PUBLIC CUDA::cudart)
  target_compile_definitions(minionnx PUBLIC MINIONNX_HAS_CUDA=1)
else()
  target_sources(minionnx PRIVATE src/stage3_ep/cuda_ep_stub.cpp)
endif()

# ---- Warnings, and platform quirks --------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(minionnx PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${MINIONNX_WARNINGS_AS_ERRORS}>>:-Werror>)
endif()
if(MINGW)
  # GCC on Windows cannot guarantee 32-byte stack alignment for AVX registers that get
  # spilled to the stack (GCC bug 54412) -> random crashes in AVX2 code. This assembler flag
  # turns every aligned vector move into an unaligned one, which is always safe.
  target_compile_options(minionnx PUBLIC -Wa,-muse-unaligned-vector-move)
endif()

# ---- Sanitizers (GCC/Clang on Linux, macOS, WSL; MinGW does not ship them) ------------------
if(MINIONNX_SANITIZE)
  target_compile_options(minionnx PUBLIC -fsanitize=${MINIONNX_SANITIZE} -fno-omit-frame-pointer)
  target_link_options(minionnx PUBLIC -fsanitize=${MINIONNX_SANITIZE})
endif()

# ---- Programs -------------------------------------------------------------------------------
# A helper function so each program is one line. Every program links the library, and
# through it inherits the include path, -pthread, sanitizer flags...
function(minionnx_program name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE minionnx)
endfunction()

minionnx_program(make_models    tools/make_models.cpp)
minionnx_program(minionnx_run   tools/minionnx_run.cpp)
minionnx_program(inspect_onnx   tools/inspect_onnx.cpp)
minionnx_program(bench_minionnx bench/bench.cpp)

if(MINIONNX_BUILD_TESTS)
  enable_testing()
  minionnx_program(test_minionnx tests/test_main.cpp)
  # Run from the source folder so the tests find models/*.onnx
  add_test(NAME unit_tests COMMAND test_minionnx WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME run_xor COMMAND minionnx_run models/xor.onnx --providers x86,arm,cpu
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---- A friendly summary at configure time ---------------------------------------------------
message(STATUS "MiniOnnx: build type = ${CMAKE_BUILD_TYPE}, CPU = ${CMAKE_SYSTEM_PROCESSOR}, "
               "CUDA EP = ${MINIONNX_ENABLE_CUDA}, sanitizer = '${MINIONNX_SANITIZE}'")
