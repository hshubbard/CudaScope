/*
 * kernels.cu
 *
 * Four intentionally-flawed CUDA kernels for micro-analysis.
 *
 * Kernel 1 – coalesced_add
 *   Baseline: consecutive threads access consecutive addresses.
 *   Expected bottleneck: bandwidth-limited (simple element-wise op).
 *
 * Kernel 2 – strided_add
 *   Each thread strides by STRIDE elements, causing non-coalesced
 *   global memory accesses and cache-line waste.
 *   Expected bottleneck: memory-bound (poor coalescing).
 *
 * Kernel 3 – divergent_add
 *   Alternating threads take different branches (threadIdx.x % 2),
 *   but the work per branch is trivial (one add/sub), so divergence
 *   is hidden by memory latency.  Branch signature visible in PTX only.
 *
 * Kernel 4 – divergent_compute
 *   Compute-bound divergence: both paths do many iterations of heavy
 *   arithmetic (fma vs division).  Memory traffic is minimal (one load,
 *   one store per thread).  Division is ~10-24x slower than fma on
 *   Ampere, so the slow half of each warp serialises execution visibly.
 *   Expected bottleneck: warp divergence (runtime penalty is clear).
 */

#include "kernels.cuh"

// ---------------------------------------------------------------------------
// Kernel 1: coalesced vector add (baseline)
// ---------------------------------------------------------------------------
__global__ void coalesced_add(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// ---------------------------------------------------------------------------
// Kernel 2: strided memory access
// ---------------------------------------------------------------------------
__global__ void strided_add(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c,
                             int n,
                             int stride)
{
    /*
     * Each thread accesses index = idx * stride.
     * With stride > 1 consecutive threads no longer map to consecutive
     * memory addresses, breaking coalescing and wasting cache lines.
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int mem_idx = idx * stride;
    if (mem_idx < n)
        c[mem_idx] = a[mem_idx] + b[mem_idx];
}

// ---------------------------------------------------------------------------
// Kernel 3: branch-divergent add
// ---------------------------------------------------------------------------
__global__ void divergent_add(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int n)
{
    /*
     * threadIdx.x % 2 splits every warp in half:
     *   – even threads: c = a + b
     *   – odd  threads: c = a - b
     * Both paths must execute sequentially within the warp (SIMT
     * serialisation), effectively halving peak throughput.
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (threadIdx.x % 2 == 0)
            c[idx] = a[idx] + b[idx];
        else
            c[idx] = a[idx] - b[idx];
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: compute-bound warp divergence
// ---------------------------------------------------------------------------
__global__ void divergent_compute(const float* __restrict__ a,
                                   float* __restrict__ c,
                                   int n,
                                   int iters)
{
    /*
     * Each thread loads one value, then does `iters` rounds of arithmetic.
     * Memory traffic is negligible (1 load + 1 store per thread).
     *
     * Even threads (threadIdx.x % 2 == 0):
     *   Accumulate with fused multiply-add (fma).
     *   On SM 8.6 fma.f32 has throughput 2 ops/cycle/SM.
     *
     * Odd threads (threadIdx.x % 2 == 1):
     *   Accumulate with division.
     *   On SM 8.6 div.f32 has throughput ~1 op / 10–24 cycles.
     *
     * Within each warp, even and odd threads execute sequentially:
     *   – GPU runs even-thread path  (iters fmas  → fast)
     *   – GPU runs odd-thread  path  (iters divs  → slow)
     * Total warp time ≈ time(fma path) + time(div path).
     *
     * A reference kernel (divergent_compute_ref) doing only fmas on all
     * threads would take ≈ time(fma path) alone.  The ratio measures
     * the real divergence overhead.
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = a[idx];

    if (threadIdx.x % 2 == 0) {
        // Fast path: fused multiply-add
        float acc = val;
        for (int i = 0; i < iters; ++i)
            acc = fmaf(acc, 1.0001f, 0.0001f);
        c[idx] = acc;
    } else {
        // Slow path: repeated division (serialises the warp)
        float acc = val;
        for (int i = 0; i < iters; ++i)
            acc = acc / 1.0001f;
        c[idx] = acc;
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: compute reference (no divergence) — paired with kernel 4
// ---------------------------------------------------------------------------
__global__ void compute_ref(const float* __restrict__ a,
                             float* __restrict__ c,
                             int n,
                             int iters)
{
    /*
     * All threads take the fast fma path — no branch divergence.
     * runtime(divergent_compute) / runtime(compute_ref) ≈ divergence overhead.
     * A ratio near 2× means each warp effectively serialised both paths.
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float acc = a[idx];
    for (int i = 0; i < iters; ++i)
        acc = fmaf(acc, 1.0001f, 0.0001f);
    c[idx] = acc;
}
