#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>    // for timing
#include <random>    // for filling matrices with random values
#include <algorithm> // for std::min and std::fill

// ---- The naive kernel (unchanged logic, this is what we measure) ----
void gemm_naive(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ---- Tiled (cache-blocked) GEMM ----
// Same maths as naive, same 2*N^3 operations, but reordered so we work
// on small blocks that stay resident in cache and get reused.
void gemm_tiled(const std::vector<float>& A,
                const std::vector<float>& B,
                std::vector<float>& C,
                int N,
                int block) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (int ii = 0; ii < N; ii += block) {
        for (int jj = 0; jj < N; jj += block) {
            for (int kk = 0; kk < N; kk += block) {

                int i_max = std::min(ii + block, N);
                int j_max = std::min(jj + block, N);
                int k_max = std::min(kk + block, N);

                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < k_max; ++k) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// ---- Reordered (ikj) GEMM ----
// Same maths, loops reordered to i-k-j so the inner loop walks memory
// contiguously across all three matrices. Cache-friendly by access order.
void gemm_reordered(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int N) {
    std::fill(C.begin(), C.end(), 0.0f);  // we accumulate, so start at zero

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a = A[i * N + k];          // fixed for the whole inner loop
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];  // all contiguous walks
            }
        }
    }
}

// ---- Fill a matrix with random floats between -1 and 1 ----
void fill_random(std::vector<float>& M, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : M) {
        x = dist(rng);
    }
}

// ---- A separate, simple reference multiply we trust, to check against ----
void gemm_reference(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ---- Compare two result matrices, allowing tiny floating-point error ----
bool matrices_match(const std::vector<float>& X,
                    const std::vector<float>& Y,
                    int N) {
    for (int idx = 0; idx < N * N; ++idx) {
        if (std::fabs(X[idx] - Y[idx]) > 1e-3f) {
            return false;
        }
    }
    return true;
}

int main() {
    std::mt19937 rng(42);  // fixed seed, so runs are repeatable

    std::vector<int> sizes = {256, 512, 1024};

    for (int N : sizes) {
        std::vector<float> A(N * N), B(N * N);
        std::vector<float> C(N * N, 0.0f);
        std::vector<float> C_ref(N * N, 0.0f);

        fill_random(A, rng);
        fill_random(B, rng);

        // ---- Naive run ----
        auto start = std::chrono::high_resolution_clock::now();
        gemm_naive(A, B, C, N);
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double gflops = (2.0 * N * N * N) / seconds / 1e9;

        gemm_reference(A, B, C_ref, N);
        bool ok = matrices_match(C, C_ref, N);

        std::cout << "N = " << N
                  << "  naive: time = " << seconds << " s"
                  << "  GFLOP/s = " << gflops
                  << "  " << (ok ? "PASS" : "FAIL") << "\n";

        // ---- Tiled run (fixed block = 16, chosen by sweep) ----
        int block = 16;
        std::vector<float> C_tiled(N * N, 0.0f);

        auto start_t = std::chrono::high_resolution_clock::now();
        gemm_tiled(A, B, C_tiled, N, block);
        auto end_t = std::chrono::high_resolution_clock::now();

        double seconds_t = std::chrono::duration<double>(end_t - start_t).count();
        double gflops_t = (2.0 * N * N * N) / seconds_t / 1e9;
        bool ok_t = matrices_match(C_tiled, C_ref, N);

        std::cout << "        tiled(16):   GFLOP/s = " << gflops_t
                  << "  speedup = " << (seconds / seconds_t) << "x"
                  << "  " << (ok_t ? "PASS" : "FAIL") << "\n";

        // ---- Reordered run ----
        std::vector<float> C_re(N * N, 0.0f);

        auto start_r = std::chrono::high_resolution_clock::now();
        gemm_reordered(A, B, C_re, N);
        auto end_r = std::chrono::high_resolution_clock::now();

        double seconds_r = std::chrono::duration<double>(end_r - start_r).count();
        double gflops_r = (2.0 * N * N * N) / seconds_r / 1e9;
        bool ok_r = matrices_match(C_re, C_ref, N);

        std::cout << "        reordered:   GFLOP/s = " << gflops_r
                  << "  speedup = " << (seconds / seconds_r) << "x"
                  << "  " << (ok_r ? "PASS" : "FAIL") << "\n";
    }

    return 0;
}