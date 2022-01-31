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
#include <chrono>

const int REPEAT = 1;
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

    // kc * 16 fits in L1, which is 32 K
    // kc * mc fits in L2, which is 256 K
    // kc * nc fits in L3, which is 4 M
    const int nc = N;
    const int kc = 240;
    const int mc = 120;
    const int nr = 2 * 8;
    const int mr = 6;

    omp_set_num_threads(8);

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
const int n = 16*6*30;  // 16*6*10 (1 million in N^2), 16*6*30 (8 million in N^2)
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
int main() {

  // https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  // Generate random data
  //srand((unsigned int)time(NULL));
  srand((unsigned int)0x100);
  std::cout << "Building Matrix: ";
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < k; j++) {
      x[i*k+j] = float(rand()%100) / 100.0;//drand48();
      xr[j*k+i] = x[i*k+j];
    }
  }
  for(int i = 0; i < k; i++) {
    for(int j = 0; j < n; j++) {
      y[i*n+j] = float(rand()%100) / 100.0;//drand48();
    }
  }
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++) {
      out1[i*n+j] = 0.0;
      out2[i*n+j] = 0.0;
    }
  }
  std::cout << "Done." << std::endl;

  // Run manual version
  auto t_start_mine = high_resolution_clock::now();
  for(int i = 0; i < REPEAT; i++) {
    SGEMM_FN(m, n, k, x, y, out1);
    /* Select from 1 of these 4 (same arguments)
            sgemm_naive
            sgemm_simd
            sgemm_simd_block
            sgemm_simd_block_parallel
    */
  }
  auto t_end_mine = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t_end_mine - t_start_mine;
  double mtime = ms_double.count();
  printf("      My GEMM elapsed time: %f ms, GFlops=%f\n", mtime, ((float) REPEAT*2*m*n*k)/(mtime*1e6));

  // Compare to OpenBLAS
  auto t_start_open = high_resolution_clock::now();
  for(int i = 0; i < REPEAT; i++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m,n,k, 1.0,x,k,y,n,0.0, out2, n);
  }
  auto t_end_open = high_resolution_clock::now();
  ms_double = t_end_open - t_start_open;
  mtime = ms_double.count();
  printf("OpenBLAS GEMM elapsed time: %f ms, GFlops=%f\n", mtime, ((float) REPEAT*2*m*n*k)/(mtime*1e6));

  // Compare outputs
  printf("computing diff");
  float diff = 0.0;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      float u = (out1[i*n+j] - out2[i*n+j]);
      diff += u*u;
    }
  }
  printf("\tNorm Squared=%f\n", diff);
  int nZeros = 0;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < n; j++) {
      if(out1[i*n+j] == 0.0) { nZeros++; }
    }
  }
  printf("\tZeros=%d\n", nZeros);
}
