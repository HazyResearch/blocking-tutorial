// 6x16 kernel without blocking
// Requires AVX-2 and FMA
// See a full description at:  http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
inline void matrix_kernel16_6( const afloat * __restrict__ A,
		      const afloat * __restrict__ B, afloat * C,
		      const int M, const int N, const int K,
              const int m, const int n
)


{
	__m256 mB0; // __m256 means 256-bit wide. This is introduced in AVX2 (AVX-512, in 2015, has 512, etc.)
	__m256 mB1;
	__m256 mA0;
	__m256 mA1;

    // Chose kernel size 6x16
    // - 16 because SIMD width is 8*32 (so must be multiple of 8)
    // - Also overall 16 registers
    // - Number of registers depends on AVX, AVX-2 or AVX-512
    // - So having 6x16 means 6x2 registers used for C block
    // - This leaves 4 for sections of A and B (needed to do fma)
    // - To use SIMD, need to store in registers
    // - Note: Intel paper uses 30x8, not 6x16
	__m256 result0_0  = _mm256_set1_ps(0); // Broadcast 32-bit (SP) 0 to all 8 elements
	__m256 result1_0  = _mm256_set1_ps(0);
	__m256 result2_0  = _mm256_set1_ps(0);
	__m256 result3_0  = _mm256_set1_ps(0);
	__m256 result4_0  = _mm256_set1_ps(0);
	__m256 result5_0  = _mm256_set1_ps(0);
	__m256 result0_1  = _mm256_set1_ps(0);
	__m256 result1_1  = _mm256_set1_ps(0);
	__m256 result2_1  = _mm256_set1_ps(0);
	__m256 result3_1  = _mm256_set1_ps(0);
	__m256 result4_1  = _mm256_set1_ps(0);
	__m256 result5_1  = _mm256_set1_ps(0);

	// This is the same for loop as in naive implementation, except now instead of the k indexing
	// a single dot product of 2 vectors of size k (a row of A and a col of B),
	// the k is indexing 6 rows of A and 16 cols of B
	// Since the SIMD width is 8 (256 bits), need to do 12 fmas here
	for(int k=0; k<K; k++)
	{
        // Prefetch k+1'th row of B. Gives ~10% speedup
        __builtin_prefetch(&B[N*(k+1)+n+8*0]);
		__builtin_prefetch(&B[N*(k+1)+n+8*1]);
        
		// Load the k'th row of the B block (load twice since in total, it's 16 floats)
		mB0   = _mm256_load_ps(&B[N*k+n+8*0]);
		mB1   = _mm256_load_ps(&B[N*k+n+8*1]);
		
		// Load a single value for the k'th col of A
		// In total, we need to do this 6 times (col of A has height 6)
		// Note: the addresses below must be aligned on a 32-byte boundary
		mA0   = _mm256_set1_ps(A[k+(m+0)*K]);	// Load float @ A's col k, row m+0 into reg
		mA1   = _mm256_set1_ps(A[k+(m+1)*K]);	// Load float @ A's col k, row m+1
		// Now we have the 16 floats of B in mB0|mB1, and the 2 floats
		// of A broadcast in mA0 and mA1.
		result0_0      = _mm256_fmadd_ps(mB0,mA0,result0_0); // result = arg1 .* arg2 .+ arg3
		result0_1      = _mm256_fmadd_ps(mB1,mA0,result0_1);
		result1_0      = _mm256_fmadd_ps(mB0,mA1,result1_0);
		result1_1      = _mm256_fmadd_ps(mB1,mA1,result1_1);
		// result0_0 now contains the final result, for this k,
		// of row 0 and cols 0-7.
		// result0_1 now contains the final result, for this k,
		// of row 0 and cols 8-15.
		// result1_0 now contains the final result, for this k,
		// of row 1 and cols 0-7.
		// result1_1 now contains the final result, for this k,
		// of row 1 and cols 8-15.
        
        // Repeat for the other 4
		
		mA0   = _mm256_set1_ps(A[k+(m+2)*K]);
		mA1   = _mm256_set1_ps(A[k+(m+3)*K]);
		result2_0      = _mm256_fmadd_ps(mB0,mA0,result2_0);
		result2_1      = _mm256_fmadd_ps(mB1,mA0,result2_1);
		result3_0      = _mm256_fmadd_ps(mB0,mA1,result3_0);
		result3_1      = _mm256_fmadd_ps(mB1,mA1,result3_1);
		
		mA0   = _mm256_set1_ps(A[k+(m+4)*K]);
		mA1   = _mm256_set1_ps(A[k+(m+5)*K]);
		result4_0      = _mm256_fmadd_ps(mB0,mA0,result4_0);
		result4_1      = _mm256_fmadd_ps(mB1,mA0,result4_1);
		result5_0      = _mm256_fmadd_ps(mB0,mA1,result5_0);
		result5_1      = _mm256_fmadd_ps(mB1,mA1,result5_1);
	}
    
    // Write registers back to C
	*((__m256*) (&C[(m+0)*N+n+0*8])) = result0_0;
	*((__m256*) (&C[(m+1)*N+n+0*8])) = result1_0;
	*((__m256*) (&C[(m+2)*N+n+0*8])) = result2_0;
	*((__m256*) (&C[(m+3)*N+n+0*8])) = result3_0;
	*((__m256*) (&C[(m+4)*N+n+0*8])) = result4_0;
	*((__m256*) (&C[(m+5)*N+n+0*8])) = result5_0;
	*((__m256*) (&C[(m+0)*N+n+1*8])) = result0_1;
	*((__m256*) (&C[(m+1)*N+n+1*8])) = result1_1;
	*((__m256*) (&C[(m+2)*N+n+1*8])) = result2_1;
	*((__m256*) (&C[(m+3)*N+n+1*8])) = result3_1;
	*((__m256*) (&C[(m+4)*N+n+1*8])) = result4_1;
	*((__m256*) (&C[(m+5)*N+n+1*8])) = result5_1;
}
