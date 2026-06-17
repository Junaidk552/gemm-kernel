# GEMM Kernel

A hand-written C++ general matrix multiply (GEMM) kernel, built as a study in
performance optimisation through the memory hierarchy. Starting from a naive
triple-loop baseline, the kernel is optimised in two stages — cache blocking
and loop reordering — with every version validated for numerical correctness
and benchmarked in GFLOP/s.

## What this is

GEMM (general matrix multiply, `C = A·B`) is the core operation underneath
modern machine learning: every dense neural network layer is, fundamentally, a
matrix multiply. This project implements GEMM by hand and makes it fast, with
the optimisations chosen to demonstrate an understanding of how memory access
patterns and the CPU cache hierarchy govern real-world performance.

## Build and run

```
clang++ -O2 -std=c++17 src/main.cpp -o gemm
./gemm
```

All benchmarks compiled with clang `-O2`. Matrices are square `float`,
row-major, filled from a fixed random seed so runs are reproducible. Each
kernel's output is validated against an independent reference multiply before
its timing is reported.

## Optimisations

**1. Naive baseline.** The standard triple-nested loop (`i`, `j`, `k`), one
output cell finished at a time. Correct, and the honest baseline everything
else is measured against. Its inner loop reads `B[k*N + j]`, walking _down_ a
column of B; because B is stored row-major, each step jumps N floats in memory,
landing in a fresh cache line that is evicted before reuse. This cache
thrashing is why naive performance does not scale with matrix size.

**2. Cache blocking (tiling).** The loops are restructured to operate on small
sub-blocks of A, B and C that fit together in cache, so each value loaded is
reused before eviction. The block size was tuned with a sweep (8–128); a fixed
block of 16 was chosen as the most consistent single configuration across all
sizes, rather than cherry-picking the best block per size — a kernel ships one
configuration, not a different one per input. Result: a steady ~2.5x over
naive.

**3. Loop reordering (ikj).** Swapping the loop order to `i`, `k`, `j` makes the
inner loop walk contiguous memory across all three matrices. This both improves
cache locality _and_ exposes a vectorisable pattern (a fixed scalar multiplied
across a contiguous run), which the compiler auto-vectorises at `-O2`. The two
effects together give a consistent ~13x over naive — the largest win, and the
clearest evidence that the naive baseline was limited as much by blocked
vectorisation as by cache misses.

## Results

Apple clang 15, arm64 (Apple Silicon). Figures are the steady values across six
runs; all kernels PASS the correctness check at every size.

| N    | Naive (GFLOP/s) | Tiled, block=16 | Reordered (ikj) | Best speedup |
| ---- | --------------- | --------------- | --------------- | ------------ |
| 256  | ~1.8            | ~4.8 (~2.7x)    | ~25 (~14x)      | ~14x         |
| 512  | ~2.9            | ~6.8 (~2.4x)    | ~34 (~12x)      | ~12x         |
| 1024 | ~2.6            | ~6.7 (~2.6x)    | ~34 (~13x)      | ~13x         |

**Headline:** a consistent ~13x speedup over a naive baseline from loop
reordering, holding steady from N=256 to N=1024, driven by contiguous memory
access that both improves cache locality and enables compiler
auto-vectorisation.

## Correctness

Every kernel is checked against an independent, deliberately-plain reference
multiply on the same inputs, with a tolerance for normal floating-point
rounding. A fast kernel that is wrong is worthless, so timing is only reported
once a kernel passes.

## What's next

- Explicit SIMD intrinsics (rather than relying on the compiler's auto-vectoriser)
- Multithreading across blocks
- `-O3` and architecture-specific tuning
- A small Python script to drive the benchmarks and plot the results

These are noted as the natural next steps rather than implemented, to keep the
current results cleanly attributable to the three optimisations above.
