#pragma once

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------

// Kernel 1: coalesced baseline
__global__ void coalesced_add(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int n);

// Kernel 2: strided (non-coalesced) memory access
__global__ void strided_add(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c,
                             int n,
                             int stride);

// Kernel 3: warp-divergent branching
__global__ void divergent_add(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int n);

// Kernel 4: compute-bound warp divergence
// Even threads do ITERS multiply-adds; odd threads do ITERS divisions.
// Division is ~10x slower than fma on SM 8.6, so half the warp stalls
// while the other half finishes — making divergence clearly visible
// even though both paths read/write the same amount of memory.
__global__ void divergent_compute(const float* __restrict__ a,
                                   float* __restrict__ c,
                                   int n,
                                   int iters);

// Kernel 5: reference for kernel 4 — same fma work, NO divergence.
// All threads take the fast (fma) path.  Comparing runtime(4) vs
// runtime(5) isolates the pure divergence overhead.
__global__ void compute_ref(const float* __restrict__ a,
                             float* __restrict__ c,
                             int n,
                             int iters);
