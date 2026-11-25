# Build MiniOnnx with plain g++ (no CMake), from Windows Terminal / PowerShell in the project root:
#   powershell -ExecutionPolicy Bypass -File scripts\build_nocmake.ps1
# Needs MSYS2 UCRT64 g++ on PATH (C:\msys64\ucrt64\bin).
$ErrorActionPreference = "Stop"
$lib = @("src/stage1_format/onnx_io.cpp","src/stage1_format/weights.cpp","src/stage2_graph/graph.cpp",
         "src/stage2_graph/optimizer.cpp","src/stage3_ep/cpu_ep.cpp","src/stage3_ep/x86_ep.cpp","src/stage3_ep/arm_ep.cpp",
         "src/stage3_ep/registry.cpp","src/stage3_ep/cuda_ep_stub.cpp","src/stage4_runtime/plan.cpp","src/stage4_runtime/session.cpp")
# -Wa,-muse-unaligned-vector-move: work around GCC-on-Windows AVX stack alignment bug 54412
$flags = @("-std=c++17","-g","-O1","-Wall","-Wextra","-pthread","-Isrc","-Wa,-muse-unaligned-vector-move")
New-Item -ItemType Directory -Force -Path bin | Out-Null
foreach ($prog in @("tools/make_models","tools/minionnx_run","tools/inspect_onnx","tests/test_main","bench/bench")) {
  $name = Split-Path $prog -Leaf
  Write-Host "g++ ... $prog.cpp"
  & g++ @flags @lib "$prog.cpp" -o "bin/$name.exe"
  if ($LASTEXITCODE -ne 0) { throw "compile failed: $prog" }
}
Write-Host "done: binaries are in bin\   try:  .\bin\test_main.exe"
