// Some of the float8 macros are modeled after https://github.com/pytorch/glow/blob/405e632ef138f1d49db9c3181182f7efd837bccc/lib/Backends/CPU/libjit/libjit_defs.h#L26
// __m256 means 256-bit wide. This is introduced in AVX2 (AVX-512, in 2015, has 512, etc.)
typedef __m256 float8;

/// Loads a simd float8 value from \p ptr.
#define LoadFloat8(PTR) (_mm256_load_ps(PTR))

/// Stores the simd float8 value to \p ptr.
#define StoreFloat8(PTR, VAL) *((float8 *)(PTR)) = (VAL);

/// Accumulate (+=) the simd float8 value to \p ptr.
#define AddFloat8(PTR, VAL) *((float8 *)(PTR)) += (VAL);

/// Broadcast the input value to a float8.
#define BroadcastFloat8(VAL) (_mm256_set1_ps(VAL))

/// Fused-add-multiply: A * B + C
#define FmaddFloat8(A, B, C) (_mm256_fmadd_ps((A), (B), (C)))

// 6x16 kernel without blocking
// Requires AVX-2 and FMA
// See a full description at:  http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
template <unsigned regsA, unsigned regsB>
inline void matmul_dot_inner
(
  const afloat * __restrict__ A,
	const afloat * __restrict__ B, afloat * C,
	const int M, const int N, const int K,
  const int m, const int n
)
{
  // Chose kernel size regsA x (regsB * 8)
  // - SIMD width is 8*32 (so must be multiple of 8)
  // - Also overall regsA * regsB  registers for C
  // - Number of registers depends on AVX, AVX-2 or AVX-512
  // - So for example having 6x16 means 6x2 registers used for C block
  // - This leaves 4 for sections of A and B (needed to do fma)
  // - To use SIMD, need to store in registers
  // - Note: Intel paper uses 30x8
	float8 csum[regsA][regsB] = {{BroadcastFloat8(0)}}; // Broadcast 32-bit (SP) 0 to all 8 elements

	// This is the same for loop as in naive implementation, except now instead of the k indexing
	// a single dot product of 2 vectors of size k (a row of A and a col of B),
	// the k is indexing @regsA rows of A and (@regsB * 8) cols of B
	// Since the SIMD width is 8 (256 bits), need to do regsA * regsB fmas here
	for (int k = 0; k < K; k++) {
    for (unsigned ai = 0; ai < regsA; ai++) {
      float8 aa = BroadcastFloat8(A[(m + ai) * K + k]);
      for (unsigned bi = 0; bi < regsB; bi++) {
        float8 bb = LoadFloat8(&B[k * N + n + bi * 8]);
        csum[ai][bi] = FmaddFloat8(aa, bb, csum[ai][bi]);
      }
    }
	}
  // Write registers back to C
  for (unsigned ai = 0; ai < regsA; ai++) {
    for (unsigned bi = 0; bi < regsB; bi++) {
      StoreFloat8(&C[(m + ai) * N + n + bi * 8], csum[ai][bi]);
    }
  }
}

template <unsigned regsA, unsigned regsB>
inline void matmul_dot_inner_block
(
  const afloat * __restrict__ A,
  const afloat * __restrict__ B, afloat * C,
  const int M, const int N, const int K,
  const int jc, const int nc,
  const int pc, const int kc,
  const int ic, const int mc,
  const int jr, const int nr,
  const int ir, const int mr
)
{
  // Chose kernel size regsA x (regsB * 8)
  // - SIMD width is 8*32 (so must be multiple of 8)
  // - Also overall regsA * regsB  registers for C
  // - Number of registers depends on AVX, AVX-2 or AVX-512
  // - So for example having 6x16 means 6x2 registers used for C block
  // - This leaves 4 for sections of A and B (needed to do fma)
  // - To use SIMD, need to store in registers
  // - Note: Intel paper uses 30x8
	float8 csum[regsA][regsB] = {{BroadcastFloat8(0)}}; // Broadcast 32-bit (SP) 0 to all 8 elements

	// This is the same for loop as in naive implementation, except now instead of the k indexing
	// a single dot product of 2 vectors of size k (a row of A and a col of B),
	// the k is indexing @regsA rows of A and (@regsB * 8) cols of B
	// Since the SIMD width is 8 (256 bits), need to do regsA * regsB fmas here
	for (int k = 0; k < kc; k++) {
    for (unsigned ai = 0; ai < regsA; ai++) {
      float8 aa = BroadcastFloat8(A[(ic + ir + ai) * K + pc + k]);
      for (unsigned bi = 0; bi < regsB; bi++) {
        float8 bb = LoadFloat8(&B[(pc + k) * N + jc + jr + bi * 8]);
        csum[ai][bi] = FmaddFloat8(aa, bb, csum[ai][bi]);
      }
    }
	}
  // Write registers back to C
  for (unsigned ai = 0; ai < regsA; ai++) {
    for (unsigned bi = 0; bi < regsB; bi++) {
      AddFloat8(&C[(ic + ir + ai) * N + jc + jr + bi * 8], csum[ai][bi]);
    }
  }
}
