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
 * For any questions contact Hazy Research
 * ( Stefan Hadjis, shadjis@stanford.edu )
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

const int NUMBER = 1;
typedef float afloat __attribute__ ((__aligned__(256)));

#include "matrix_kernel16_6.c"
#include "matrix_kernel16_6_block.c"

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
    const enum CBLAS_ORDER Order,         // {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
    const enum CBLAS_TRANSPOSE TransA,    // {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
    const enum CBLAS_TRANSPOSE TransB,    // {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
    const blasint M,
    const blasint N,
    const blasint K,
    const float alpha,
    const float *A,                       // m x k (after transpose if TransA)
    const blasint lda,                    // leading dimension of a 
    const float *B,                       // k x n (after transpose if TransB)
    const blasint ldb,                    // leading dimension of b 
    const float beta,
    float *C,                             // m x n
    const blasint ldc                     // leading dimension of c
)
{
    bool transpose_A = false;
    bool transpose_B = false;
    assert_sgemm_parameters(Order, TransA, TransB, N, M, K, lda, ldb, ldc, transpose_A, transpose_B);
    
    for (int m=0; m<M; ++m)
    {
        for (int n=0; n<N; ++n)
        {
            C[m*N + n] += beta * C[m*N + n];
            for (int k=0; k<K; ++k)
            {
                size_t A_idx = 0, B_idx = 0;
                if (transpose_A) {
                    A_idx = k*M + m; // A is k x m
                } else {
                    A_idx = m*K + k; // A is m x k
                }
                if (transpose_B) {
                    B_idx = k + n*K; // B is n x k
                } else {
                    B_idx = n + k*N; // B is k x n
                }
                C[m*N + n] += alpha * A[A_idx] * B[B_idx];
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Version 2 -- SIMD Tiling (6x16)
// -----------------------------------------------------------------------------
inline void sgemm_16x6
(
    const enum CBLAS_ORDER Order,     
    const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB,
    const blasint M,
    const blasint N,
    const blasint K,
    const float alpha,
    const float *A,                   
    const blasint lda,                
    const float *B,                   
    const blasint ldb,                
    const float beta,
    float *C,                         
    const blasint ldc                 
)
{
    bool transpose_A = false;
    bool transpose_B = false;
    assert_sgemm_parameters(Order, TransA, TransB, N, M, K, lda, ldb, ldc, transpose_A, transpose_B);
    assert(!transpose_A);
    assert(!transpose_B);
    assert(alpha==1.0);
    assert(beta ==0.0);
    for (int m=0; m<M; m+=6) {
        for (int n=0; n<N; n+=16) {
            matrix_kernel16_6(A,B,C,M,N,K,m,n);
        }
    }
}


// -----------------------------------------------------------------------------
// Version 3 -- Blocking like in
// http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
// -----------------------------------------------------------------------------
inline void sgemm_16x6_block
(
    const enum CBLAS_ORDER Order,      
    const enum CBLAS_TRANSPOSE TransA, 
    const enum CBLAS_TRANSPOSE TransB, 
    const blasint M,
    const blasint N,
    const blasint K,
    const float alpha,
    const float *A,                    
    const blasint lda,                 
    const float *B,                    
    const blasint ldb,                 
    const float beta,
    float *C,                          
    const blasint ldc                  
)
{
    bool transpose_A = false;
    bool transpose_B = false;
    assert_sgemm_parameters(Order, TransA, TransB, N, M, K, lda, ldb, ldc, transpose_A, transpose_B);
    
    assert(!transpose_A);
    assert(!transpose_B);
    assert(alpha==1.0);
    assert(beta ==0.0);
    
    // kc * 16 fits in L1, which is 32 K
    // kc * mc fits in L2, which is 256 K
    // kc * nc fits in L3, which is 4 M
    const int nc = N;
    const int kc = 240;
    const int mc = 120;
    const int nr = 16;
    const int mr = 6;
    
    for (int jc=0; jc<N; jc+=nc) {
        for (int pc=0; pc<K; pc+=kc) {
            for (int ic=0; ic<M; ic+=mc) {
                for (int jr=0; jr<nc; jr+=nr) {
                    for (int ir=0; ir<mc; ir+=mr) {
                        matrix_kernel16_6_block(A,B,C,M,N,K,jc,nc,pc,kc,ic,mc,jr,nr,ir,mr);
                    }
                }
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Version 4 -- Blocking + Threads
// -----------------------------------------------------------------------------
inline void sgemm_16x6_block_parallel
(
    const enum CBLAS_ORDER Order,       
    const enum CBLAS_TRANSPOSE TransA,  
    const enum CBLAS_TRANSPOSE TransB,  
    const blasint M,
    const blasint N,
    const blasint K,
    const float alpha,
    const float *A,                     
    const blasint lda,                  
    const float *B,                     
    const blasint ldb,                  
    const float beta,
    float *C,                           
    const blasint ldc                   
)
{
    bool transpose_A = false;
    bool transpose_B = false;
    assert_sgemm_parameters(Order, TransA, TransB, N, M, K, lda, ldb, ldc, transpose_A, transpose_B);
    
    assert(!transpose_A);
    assert(!transpose_B);
    assert(alpha==1.0);
    assert(beta ==0.0);
    
    // kc * 16 fits in L1, which is 32 K
    // kc * mc fits in L2, which is 256 K
    // kc * nc fits in L3, which is 4 M
    const int nc = N;
    const int kc = 240;
    const int mc = 120;
    const int nr = 16;
    const int mr = 6;
    
    omp_set_num_threads(4);
    
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
                        matrix_kernel16_6_block(A,B,C,M,N,K,jc,nc,pc,kc,ic,mc,jr,nr,ir,mr);
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
const int n = 16*6*20;  // 16*6*10 (1 million in N^2), 16*6*30 (8 million in N^2)
const int m = n;
const int r = n;

float x[m*r] __attribute__((aligned(256)));
float xr[m*r] __attribute__((aligned(256)));
float y[r*n] __attribute__((aligned(256)));
float out1[m*n]  __attribute__((aligned(256)));
float out2[m*n]  __attribute__((aligned(256)));


// -----------------------------------------------------------------------------
// Comparison against OpenBLAS
// -----------------------------------------------------------------------------
int main() {
  
  struct timeval start, end;

  // Generate random data
  //srand((unsigned int)time(NULL));
  srand((unsigned int)0x100);
  std::cout << "Building Matrix: ";
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < r; j++) {
      x[i*r+j] = float(rand()%100) / 100.0;//drand48();
      xr[j*r+i] = x[i*r+j];
    }
  }
  for(int i = 0; i < r; i++) {
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
  gettimeofday(&start, NULL);
  for(int i = 0; i < NUMBER; i++) { 
    sgemm_naive(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m,n,r, 1.0,x,r,y,n,0.0, out1, n);

    /* Select from 1 of these 4 (same arguments)
    
            sgemm_naive
            sgemm_16x6
            sgemm_16x6_block
            sgemm_16x6_block_parallel
    */
  }
  gettimeofday(&end, NULL);
  long seconds  = end.tv_sec  - start.tv_sec;
  long useconds = end.tv_usec - start.tv_usec;  
  float mtime = ((seconds) * 1000 + useconds/1000.0);
  printf("Elapsed time: %f milliseconds GFlops=%f\n", mtime, ((float) NUMBER*2*m*n*r)/(mtime*1e6));

  // Compare to OpenBLAS
  gettimeofday(&start, NULL);
  for(int i = 0; i < NUMBER; i++) { 
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m,n,r, 1.0,x,r,y,n,0.0, out2, n);
  }
  gettimeofday(&end, NULL);
  seconds  = end.tv_sec  - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;  
  mtime = ((seconds) * 1000 + useconds/1000.0);
  printf("Elapsed time: %f milliseconds GFlops=%f\n", mtime, ((float) NUMBER*2*m*n*r)/(mtime*1e6));

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

