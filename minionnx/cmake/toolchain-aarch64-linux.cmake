# Cross-compile for 64-bit ARM Linux from an x86 Linux/WSL machine.
#   sudo apt install g++-aarch64-linux-gnu qemu-user
#   cmake -S . -B build-arm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux.cmake
#   cmake --build build-arm -j
#   qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm/test_minionnx     # run the ARM binary on x86
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64 -L /usr/aarch64-linux-gnu)   # lets ctest run ARM binaries
