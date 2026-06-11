#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>    // for timing
#include <random>    // for filling matrices with random values

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

// ---- Fill a matrix with random floats between -1 and 1 ----
void fill_random(std::vector<float>& M, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : M) {
        x = dist(rng);
    }
}

// ---- A separate, simple reference multiply we trust, to check against ----
// Deliberately plain. Its only job is to be obviously correct.
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

    // The sizes the checklist names.
    std::vector<int> sizes = {256, 512, 1024};

    for (int N : sizes) {
        std::vector<float> A(N * N), B(N * N);
        std::vector<float> C(N * N, 0.0f);
        std::vector<float> C_ref(N * N, 0.0f);

        fill_random(A, rng);
        fill_random(B, rng);

        // Time the naive kernel.
        auto start = std::chrono::high_resolution_clock::now();
        gemm_naive(A, B, C, N);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();

        // GFLOP/s = (2 * N^3) / time / 1e9.
        // A matrix multiply does N^3 multiply-add pairs; each pair is 2
        // floating-point operations, hence 2 * N^3. Divide by time for
        // operations per second, by 1e9 to express it in billions (giga).
        double gflops = (2.0 * N * N * N) / seconds / 1e9;

        // Correctness: compare against the reference.
        gemm_reference(A, B, C_ref, N);
        bool ok = matrices_match(C, C_ref, N);

        std::cout << "N = " << N
                  << "  time = " << seconds << " s"
                  << "  GFLOP/s = " << gflops
                  << "  " << (ok ? "PASS" : "FAIL")
                  << "\n";
    }

    return 0;
}