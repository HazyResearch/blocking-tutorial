/*******************************************************************************
 *
 * This tutorial implements single-precision matrix multiplication (SGEMM) with
 * various levels of optimization. It follows the algorithm described in:
 *
 * http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
 *
 * Only a subset of BLAS is supported, e.g. specific matrix sizes, no transpose,
 * etc. It is meant just as a tutorial introduction to blocking and SIMD.
 * Experiments compare against OpenBLAS.
 *
 ******************************************************************************/

#include "my_timer.h"
#include <immintrin.h>
#include "cblas.h"
#include "omp.h"
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib>
#include <random>
#include <iostream>
#include <vector>
#include <time.h>
using namespace std;

#ifdef OMP_NUM_THREADS_MACRO
constexpr int OMP_NUM_THREADS = OMP_NUM_THREADS_MACRO;
#else
constexpr int OMP_NUM_THREADS = 8;
#endif

const int REPEAT = 10;
typedef float afloat __attribute__ ((__aligned__(256)));

#define SGEMM_FN sgemm_simd_block_parallel

#include "matrix_kernel_vectorized.cpp"

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

// This code doesn't implement full BLAS so make sure nothing illegal is being passed in
// BLAS Reference manual:
// https://software.intel.com/sites/products/documentation/doclib/iss/2013/mkl/mklman/GUID-97718E5C-6E0A-44F0-B2B1-A551F0F164B2.htm#GUID-97718E5C-6E0A-44F0-B2B1-A551F0F164B2
inline void assert_sgemm_parameters
(
    const enum CBLAS_ORDER Order,
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const blasint N,
    const blasint M,
    const blasint K,
    const blasint lda,
    const blasint ldb,
    const blasint ldc,
    bool & transpose_A,
    bool & transpose_B
)
{
    // Assert for Order
    assert(Order == CblasRowMajor);

    // Transpose
    if (TransA == CblasNoTrans) {
        transpose_A = false;
    } else if (TransA == CblasTrans) {
        transpose_A = true;
    } else if (TransA == CblasConjTrans || TransA == CblasConjNoTrans) {
        std::cout << "Unsupported transpose\n";
        exit(0);
    }

    if (TransB == CblasNoTrans) {
        transpose_B = false;
    } else if (TransB == CblasTrans) {
        transpose_B = true;
    } else if (TransB == CblasConjTrans || TransB == CblasConjNoTrans) {
        std::cout << "Unsupported transpose\n";
        exit(0);
    }

    // Assert for lda, ldb and ldc
    if (transpose_A) {
        assert(lda==M);
    } else {
        assert(lda==K);
    }
    if (transpose_B) {
        assert(ldb==K);
    } else {
        assert(ldb==N);
    }
    assert(ldc==N);
}


// -----------------------------------------------------------------------------
// Version 1 -- Naive iteration
// -----------------------------------------------------------------------------
// I also handled some transposes here but gave up in more optimized versions
inline void sgemm_naive
(
    const blasint M,
    const blasint N,
    const blasint K,
    const float *A,                       // m x k (after transpose if TransA)
    const float *B,                       // k x n (after transpose if TransB)
    float *C                              // m x n
)
{
    bool transpose_A = false;
    bool transpose_B = false;
    assert_sgemm_parameters(/*Order=*/CblasRowMajor, /*TransA=*/CblasNoTrans, /*TransB=*/CblasNoTrans, N, M, K, K, N, N,
                            transpose_A, transpose_B);    
    for (int m=0; m<M; ++m) {
        for (int n=0; n<N; ++n) {
            C[m*N + n] = 0;
            for (int k=0; k<K; ++k) {
                size_t A_idx = 0, B_idx = 0;
                A_idx = m*K + k; // A is m x k
                B_idx = n + k*N; // B is k x n
                C[m*N + n] += A[A_idx] * B[B_idx];
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Version 2 -- SIMD Tiling (6x16)
// -----------------------------------------------------------------------------
template <unsigned regsA=3, unsigned regsB=4>
inline void sgemm_simd
(
    const blasint M,
    const blasint N,
    const blasint K,
    const float *A,
    const float *B,
    float *C
)
{
    bool transpose_A = false;
    bool transpose_B = false;
    assert_sgemm_parameters(/*Order=*/CblasRowMajor, /*TransA=*/CblasNoTrans, /*TransB=*/CblasNoTrans, N, M, K, K, N, N,
                            transpose_A, transpose_B);
    assert(!transpose_A);
    assert(!transpose_B);
    for (int i = 0; i < M * N; i++) C[i] = 0.0;
    for (int m=0; m<M; m+=regsA) {
        for (int n=0; n<N; n+=regsB*8) {
            matmul_dot_inner<regsA, regsB>(A,B,C,M,N,K,m,n);
        }
    }
}


// -----------------------------------------------------------------------------
// Version 3 -- Blocking like in
// http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
// -----------------------------------------------------------------------------
inline void sgemm_simd_block
(
    const blasint M,
    const blasint N,
    const blasint K,
    const float *A,
    const float *B,
    float *C
)
{
    bool transpose_A = false;
    bool transpose_B = false;
    assert_sgemm_parameters(/*Order=*/CblasRowMajor, /*TransA=*/CblasNoTrans, /*TransB=*/CblasNoTrans, N, M, K, K, N, N,
                            transpose_A, transpose_B);
    assert(!transpose_A);
    assert(!transpose_B);
    for (int i = 0; i < M * N; i++) C[i] = 0.0;

    // kc * 16 fits in L1, which is 32 K
    // kc * mc fits in L2, which is 256 K
    // kc * nc fits in L3, which is 4 M
    const int nc = N;
    const int kc = 240;
    const int mc = 120;
    const int nr = 4 * 8;
    const int mr = 3;

    for (int jc=0; jc<N; jc+=nc) {
        for (int pc=0; pc<K; pc+=kc) {
            for (int ic=0; ic<M; ic+=mc) {
                for (int jr=0; jr<nc; jr+=nr) {
                    for (int ir=0; ir<mc; ir+=mr) {
                        matmul_dot_inner_block<3, 4>(A,B,C,M,N,K,jc,nc,pc,kc,ic,mc,jr,nr,ir,mr);
                    }
                }
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Version 4 -- Blocking + Threads
// -----------------------------------------------------------------------------
inline void sgemm_simd_block_parallel
(
    const blasint M,
    const blasint N,
    const blasint K,
    const float *A,
    const float *B,
    float *C
)
{
    bool transpose_A = false;
    bool transpose_B = false;
    assert_sgemm_parameters(/*Order=*/CblasRowMajor, /*TransA=*/CblasNoTrans, /*TransB=*/CblasNoTrans, N, M, K, K, N, N,
                            transpose_A, transpose_B);
    assert(!transpose_A);
    assert(!transpose_B);
    omp_set_num_threads(OMP_NUM_THREADS);
    #pragma omp parallel for
    for (int i = 0; i < M * N; i++) C[i] = 0.0;

    // kc * 16 fits in L1, which is 32 K
    // kc * mc fits in L2, which is 256 K
    // kc * nc fits in L3, which is 4 M
    const int nc = N;
    const int kc = 240;
    const int mc = 120;
    const int nr = 2 * 8;
    const int mr = 6;
    
    // Usually 1 iteration, cannot parallelize
    for (int jc=0; jc<N; jc+=nc) {
        // 8 iterations, not worth parallelizing
        for (int pc=0; pc<K; pc+=kc) {
            // 16 iterations, not worth parallelizing
            for (int ic=0; ic<M; ic+=mc) {
                // 120 iterations, worth parallelizing
                #pragma omp parallel for
                for (int jr=0; jr<nc; jr+=nr) {
                    // 20 iterations, not worth parallelizing
                    for (int ir=0; ir<mc; ir+=mr) {
                        matmul_dot_inner_block<6, 2>(A,B,C,M,N,K,jc,nc,pc,kc,ic,mc,jr,nr,ir,mr);
                    }
                }
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Global Parameters
// -----------------------------------------------------------------------------

// Change this line to try other sizes:
const int n = 16*6*50;  // 16*6*10 (1 million in N^2), 16*6*30 (8 million in N^2)
const int m = n;
const int k = n;

float x[m*k] __attribute__((aligned(256)));
float xr[m*k] __attribute__((aligned(256)));
float y[k*n] __attribute__((aligned(256)));
float out1[m*n]  __attribute__((aligned(256)));
float out2[m*n]  __attribute__((aligned(256)));


// -----------------------------------------------------------------------------
// Comparison against OpenBLAS
// -----------------------------------------------------------------------------
mt19937 rng;
void gen_test(
  int M, int N, int K,
  float* __restrict__ X, float* __restrict__ Y
)
{
  #pragma omp parallel for
  for (int i = 0; i < M; i++)
    for (int j = 0; j < K; j++)
      X[i * K + j] = float(rng() % 128) / 128.0;

  #pragma omp parallel for
  for (int i = 0; i < K; i++)
    for (int j = 0; j < N; j++)
      Y[i * N + j] = float(rng() % 128) / 128.0;
}

//---------------
volatile double dummy_var;
constexpr size_t L2_CACHE_ELEMS = (1 << 20) / sizeof(double); // 1 MB L2 cache per core
vector<double> dummy_cache;

// force (hopefully) all running cores to clear its L2 cache
void __attribute__ ((noinline)) clear_cache() {
  size_t dummy_size = L2_CACHE_ELEMS * OMP_NUM_THREADS;
  dummy_cache.resize(dummy_size);
  omp_set_num_threads(OMP_NUM_THREADS);
  #pragma omp parallel for
  for (size_t i = 0; i < dummy_size; i++) dummy_cache[i] = dummy_var;
  dummy_var = dummy_cache[rng() % dummy_size];
}

void get_stats(int M, int N, float* out1, float* out2, double& total_diff, size_t& total_nZeros)
{
  double diff = 0.0;
  size_t nZeros = 0;
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      double u = (out1[i*n+j] - out2[i*n+j]);
      diff += u*u;
      if (out1[i*n+j] == 0.0) nZeros++;
    }
  }

  total_diff += diff;
  total_nZeros += nZeros;  
}

int main() {
  rng.seed(time(NULL));
  MyTimer timer;
  double my_cost = 0, blas_cost = 0;
  double total_diff = 0;
  size_t total_nZeros = 0;

  for (int t = 1; t <= REPEAT; t++) {
    cout << "Running test " << t << "\n";
    gen_test(m, n, k, x, y);
    
    clear_cache();
    timer.startCounter();
    SGEMM_FN(m, n, k, x, y, out1);
    my_cost += timer.getCounterMsPrecise();
    
    clear_cache();
    timer.startCounter();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m,n,k, 1.0,x,k,y,n,0.0, out2, n);
    blas_cost += timer.getCounterMsPrecise();
    
    get_stats(m, n, out1, out2, total_diff, total_nZeros);
  }

  printf("      My GEMM elapsed time: %f ms, GFlops=%f\n", my_cost, ((double) REPEAT*2*m*n*k) / (my_cost*1e6));
  printf("OpenBLAS GEMM elapsed time: %f ms, GFlops=%f\n", blas_cost, ((double) REPEAT*2*m*n*k) / (blas_cost*1e6));
  printf("Average square error = %f\n", total_diff / REPEAT);
  printf("Average nZeros = %f\n", (double)total_nZeros / REPEAT);
  return 0;
}

