--------------------------------------------------------------------------------
Blocking Tutorial
--------------------------------------------------------------------------------

This is a tutorial internal to the Hazy Research group to illustrate SIMD
and cache blocking.

For any questions contact Hazy Research
( Stefan Hadjis, shadjis@stanford.edu )

Based on the paper:
Anatomy of High-Performance Many-Threaded Matrix Multiplication
http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf


--------------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------------

To run the code, you will need:

- A Haswell processor or later (AVX2 and FMA support)
    http://en.wikipedia.org/wiki/Advanced_Vector_Extensions
    http://en.wikipedia.org/wiki/FMA_instruction_set
- The OpenMP library
- I used g++ 4.9.2 but any compiler supporting AVX2 and FMA will work
- OpenBLAS (for comparison vs. OpenBLAS. Disable if not interested)
    http://www.openblas.net/


--------------------------------------------------------------------------------
Instructions
--------------------------------------------------------------------------------

To compile:

1. Change compile.sh to point to OpenBLAS or some other BLAS
    To run without OpenBLAS, modify the .cpp file to not call OpenBLAS
    (remove references to cblas.h, cblas_sgemm)

2. Compile with:
    bash compile.sh

3. Run with:
    ./m

This will run two of the matrix multiply implementations and compare:
- The outputs (to ensure they match)
- The total time and GFLOPS

4. To change which version to compare against, edit which function is called
   (sgemm_naive, sgemm_16x6_block_parallel, etc.)


--------------------------------------------------------------------------------
Experiments
--------------------------------------------------------------------------------

For NxN single precision matrix multiplication
on a Haswell with 2 cores, 4 threads,
    L1 I$/D$: 2 x 32 kB  8-way
    L2$       2 x 256 kB 8-way
    L3$       1 x 4 MB   16-way

Naive       8330.310 milliseconds GFlops=1.699310
+SIMD       1057.134 milliseconds GFlops=13.390711
+Blocking   273.5390 milliseconds GFlops=51.750485
+Threads    138.5209 milliseconds GFlops=102.192277
OpenBLAS    117.7092 milliseconds GFlops=120.259758


For various N:

    N=960 (~1 million elements per matrix):
    This code   101.2 GFLOPS    
    OpenBLAS    98.3 GFLOPS     

    N=1920 (~4 million elements per matrix):
    This code   104.1 GFLOPS    
    OpenBLAS    121.2 GFLOPS    

    N=2880 (~8 million elements per matrix):
    This code   91.7 GFLOPS
    OpenBLAS    103.9 GFLOPS

