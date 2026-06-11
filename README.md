# GEMM Kernel

A C++ general matrix multiply (GEMM) kernel, built as a study in performance optimisation through the memory hierarchy.

## Status
- [x] Naive baseline (triple nested loop) with correctness check
- [x] Benchmark harness (GFLOP/s across N = 256, 512, 1024)
- [ ] Cache tiling
- [ ] Loop reordering

## Build

clang++ -O2 -std=c++17 src/main.cpp -o gemm
./gemm
All benchmarks compiled with clang `-O2`.

## Baseline results (Apple clang 15, arm64)
| N    | GFLOP/s |
|------|---------|
| 256  | 1.56    |
| 512  | 2.74    |
| 1024 | 2.58    |
