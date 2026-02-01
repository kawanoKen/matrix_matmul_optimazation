// matmul_block_unroll.cpp
// Minimal blocked + unrolled GEMM-like kernel with timing.
//
// Build (no OpenMP):
//   clang++ -O3 -std=c++17 matmul_block_unroll.cpp -o matmul
//
// Build (with OpenMP) (if libomp is installed):
//   clang++ -O3 -std=c++17 -Xpreprocessor -fopenmp matmul_block_unroll.cpp -lomp -o matmul
//
// Run example:
//   ./matmul --M 1024 --N 1024 --K 1024 --BM 64 --BN 64 --BK 32 --U 4 --T 4 --repeat 5

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

static void usage(const char* prog) {
  std::cerr
    << "Usage: " << prog << " [--M int --N int --K int]"
    << " --BM int --BN int --BK int --U int --T int [--repeat int]\n";
}

static int get_arg_int(int argc, char** argv, const std::string& key, int def) {
  for (int i = 1; i + 1 < argc; i++) {
    if (argv[i] == key) return std::stoi(argv[i + 1]);
  }
  return def;
}

static bool has_arg(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; i++) if (argv[i] == key) return true;
  return false;
}

// Simple aligned allocation (64B)
static float* aligned_alloc_floats(size_t n) {
#if defined(_MSC_VER)
  return (float*)_aligned_malloc(n * sizeof(float), 64);
#else
  void* p = nullptr;
  if (posix_memalign(&p, 64, n * sizeof(float)) != 0) return nullptr;
  return (float*)p;
#endif
}

static void aligned_free_floats(float* p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}

// Access row-major matrices
// A: MxK, B: KxN, C: MxN
inline float& A_at(float* A, int M, int K, int i, int k) { (void)M; return A[(size_t)i * K + k]; }
inline float& B_at(float* B, int K, int N, int k, int j) { (void)K; return B[(size_t)k * N + j]; }
inline float& C_at(float* C, int M, int N, int i, int j) { (void)M; return C[(size_t)i * N + j]; }

// Blocked + unrolled inner-k GEMM
static void matmul_block_unroll(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    int BM, int BN, int BK,
    int U, int T
) {
  // Initialize C=0 (caller can do it, but keep safe here)
  std::fill(C, C + (size_t)M * N, 0.0f);

#ifdef _OPENMP
  if (T > 0) omp_set_num_threads(T);
#endif

  // Parallelize over (ii,jj) tiles (coarse-grain)
  // collapse(2) is useful but may not be supported everywhere; keep simple.
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int ii = 0; ii < M; ii += BM) {
    for (int jj = 0; jj < N; jj += BN) {
      int i_end = std::min(ii + BM, M);
      int j_end = std::min(jj + BN, N);

      for (int kk = 0; kk < K; kk += BK) {
        int k_end = std::min(kk + BK, K);

        // Main tile compute
        for (int i = ii; i < i_end; i++) {
          for (int j = jj; j < j_end; j++) {
            float sum = C[(size_t)i * N + j];

            int k = kk;
            // Unrolled loop (k += U)
            // U should be 1,2,4,8,... but any positive int works.
            int k_unroll_end = k_end - (U - 1);
            for (; k < k_unroll_end; k += U) {
              // Manually unroll U steps
              // Use a switch to keep it minimal and fast for common U values.
              // For other U, fall back to small loop.
              switch (U) {
                case 1:
                  sum += A[(size_t)i * K + k] * B[(size_t)k * N + j];
                  break;
                case 2:
                  sum += A[(size_t)i * K + (k+0)] * B[(size_t)(k+0) * N + j];
                  sum += A[(size_t)i * K + (k+1)] * B[(size_t)(k+1) * N + j];
                  break;
                case 4:
                  sum += A[(size_t)i * K + (k+0)] * B[(size_t)(k+0) * N + j];
                  sum += A[(size_t)i * K + (k+1)] * B[(size_t)(k+1) * N + j];
                  sum += A[(size_t)i * K + (k+2)] * B[(size_t)(k+2) * N + j];
                  sum += A[(size_t)i * K + (k+3)] * B[(size_t)(k+3) * N + j];
                  break;
                case 8:
                  sum += A[(size_t)i * K + (k+0)] * B[(size_t)(k+0) * N + j];
                  sum += A[(size_t)i * K + (k+1)] * B[(size_t)(k+1) * N + j];
                  sum += A[(size_t)i * K + (k+2)] * B[(size_t)(k+2) * N + j];
                  sum += A[(size_t)i * K + (k+3)] * B[(size_t)(k+3) * N + j];
                  sum += A[(size_t)i * K + (k+4)] * B[(size_t)(k+4) * N + j];
                  sum += A[(size_t)i * K + (k+5)] * B[(size_t)(k+5) * N + j];
                  sum += A[(size_t)i * K + (k+6)] * B[(size_t)(k+6) * N + j];
                  sum += A[(size_t)i * K + (k+7)] * B[(size_t)(k+7) * N + j];
                  break;
                default:
                  for (int u = 0; u < U; u++) {
                    sum += A[(size_t)i * K + (k + u)] * B[(size_t)(k + u) * N + j];
                  }
                  break;
              }
            }
            // Remainder
            for (; k < k_end; k++) {
              sum += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            }

            C[(size_t)i * N + j] = sum;
          }
        }
      }
    }
  }
}

// simple checksum to prevent dead-code elimination
static double checksum(const float* C, int M, int N) {
  double s = 0.0;
  for (int i = 0; i < M; i += std::max(1, M / 32)) {
    for (int j = 0; j < N; j += std::max(1, N / 32)) {
      s += C[(size_t)i * N + j];
    }
  }
  return s;
}

int main(int argc, char** argv) {
  if (has_arg(argc, argv, "--help")) {
    usage(argv[0]);
    return 0;
  }

  int M = get_arg_int(argc, argv, "--M", 1024);
  int N = get_arg_int(argc, argv, "--N", 1024);
  int K = get_arg_int(argc, argv, "--K", 1024);

  int BM = get_arg_int(argc, argv, "--BM", 64);
  int BN = get_arg_int(argc, argv, "--BN", 64);
  int BK = get_arg_int(argc, argv, "--BK", 32);
  int U  = get_arg_int(argc, argv, "--U", 4);
  int T  = get_arg_int(argc, argv, "--T", 1);
  int repeat = get_arg_int(argc, argv, "--repeat", 5);

  if (BM <= 0 || BN <= 0 || BK <= 0 || U <= 0 || T <= 0 || repeat <= 0) {
    usage(argv[0]);
    return 1;
  }

  size_t szA = (size_t)M * K;
  size_t szB = (size_t)K * N;
  size_t szC = (size_t)M * N;

  float* A = aligned_alloc_floats(szA);
  float* B = aligned_alloc_floats(szB);
  float* C = aligned_alloc_floats(szC);
  if (!A || !B || !C) {
    std::cerr << "Allocation failed\n";
    return 1;
  }

  // Initialize A, B with reproducible random values
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < szA; i++) A[i] = dist(rng);
  for (size_t i = 0; i < szB; i++) B[i] = dist(rng);

  // Warm-up (reduce cold-start effects)
  matmul_block_unroll(A, B, C, M, N, K, BM, BN, BK, U, T);

  // Timed runs
  std::vector<double> times_ms;
  times_ms.reserve(repeat);

  for (int r = 0; r < repeat; r++) {
    auto t0 = std::chrono::steady_clock::now();
    matmul_block_unroll(A, B, C, M, N, K, BM, BN, BK, U, T);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    times_ms.push_back(ms);
  }

  std::sort(times_ms.begin(), times_ms.end());
  double median_ms = times_ms[times_ms.size() / 2];
  double chk = checksum(C, M, N);

  // Print one-line summary (CSV-like)
  std::cout
    << "M,N,K,BM,BN,BK,U,T,median_ms,checksum\n"
    << M << "," << N << "," << K << ","
    << BM << "," << BN << "," << BK << ","
    << U  << "," << T  << ","
    << median_ms << "," << chk << "\n";

  aligned_free_floats(A);
  aligned_free_floats(B);
  aligned_free_floats(C);
  return 0;
}
