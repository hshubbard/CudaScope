"""
add_test_kernels.py
===================
Adds a set of test kernels designed to exercise all dashboard score bands
(HIGH / MEDIUM / LOW risk) and all three analysis passes.

Run from repo root:
    python scripts/add_test_kernels.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import kernel_manager as km

kernels = [

    # -------------------------------------------------------------------------
    # 1. atomic_histogram
    #    Triggers: HIGH non-det (FP atomicAdd order), HIGH resource (atom density),
    #              MEDIUM fragility (warp lane mask), Warp Divergence bottleneck.
    # -------------------------------------------------------------------------
    dict(
        name       = "atomic_histogram",
        block_size = 256,
        n_elements = 16 * 1024 * 1024,
        iters      = 0,
        code       = r"""
__global__ void atomic_histogram(const float* __restrict__ a,
                                  float* __restrict__ bins,
                                  int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Warp-lane extraction: hard-coded mask assuming warpSize == 32
    int lane = threadIdx.x & 31;

    // Thread-divergent bin selection
    int bin = (int)(a[idx] * 8.0f) % 8;
    if (bin < 0) bin = 0;

    // Non-deterministic FP accumulation via global atomic
    atomicAdd(&bins[bin], a[idx]);

    // Extra divergent path that serialises warps
    if (lane < 4) {
        atomicAdd(&bins[bin + 8], 1.0f);
    }
}
""",
        params     = "const float* __restrict__ a, float* __restrict__ bins, int n",
    ),

    # -------------------------------------------------------------------------
    # 2. warp_reduce_shuffle
    #    Triggers: HIGH fragility (shfl.sync / arch-specific), MEDIUM non-det
    #              (warp shuffle feeds FP accumulation), HIGH resource (register
    #              pressure from unrolled reduction).  Bottleneck: Shared Memory Bound.
    # -------------------------------------------------------------------------
    dict(
        name       = "warp_reduce_shuffle",
        block_size = 256,
        n_elements = 16 * 1024 * 1024,
        iters      = 0,
        code       = r"""
__global__ void warp_reduce_shuffle(const float* __restrict__ a,
                                     float* __restrict__ out,
                                     int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? a[idx] : 0.0f;

    // Warp-level reduction via shuffle (requires SM 7.0+)
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val,  8);
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);

    // Block-level via shared memory
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x / 32;
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (lane < (blockDim.x / 32)) ? smem[lane] : 0.0f;
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (lane == 0) out[blockIdx.x] = val;
    }
}
""",
        params     = "const float* __restrict__ a, float* __restrict__ out, int n",
    ),

    # -------------------------------------------------------------------------
    # 3. large_smem_stencil
    #    Triggers: HIGH resource (smem near limit → low occupancy), HIGH fragility
    #              (st.shared not followed by bar.sync in stencil update),
    #              MEDIUM non-det (neighbor reads without barrier).
    #              Bottleneck: Shared Memory Bound.
    # -------------------------------------------------------------------------
    dict(
        name       = "large_smem_stencil",
        block_size = 256,
        n_elements = 16 * 1024 * 1024,
        iters      = 0,
        code       = r"""
__global__ void large_smem_stencil(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n)
{
    // Large shared allocation: 256 * 4 = 1024 floats = 4 KB per block
    // Plus halo: 512 floats = 2 KB → total 6 KB (moderate pressure)
    __shared__ float smem[512];
    __shared__ float halo[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load into shared
    smem[tid] = (idx < n) ? in[idx] : 0.0f;

    // Hazard: write halo WITHOUT syncthreads before reading smem neighbor
    halo[tid] = smem[tid] * 0.5f;  // reads smem before all threads have stored

    __syncthreads();

    // 3-point stencil — reads neighbor that may not have been stored yet
    float left  = (tid > 0)              ? smem[tid - 1] : 0.0f;
    float right = (tid < blockDim.x - 1) ? smem[tid + 1] : 0.0f;
    float res   = 0.25f * left + 0.5f * smem[tid] + 0.25f * right + halo[tid];

    if (idx < n) out[idx] = res;
}
""",
        params     = "const float* __restrict__ in, float* __restrict__ out, int n",
    ),

    # -------------------------------------------------------------------------
    # 4. tiny_block_saxpy
    #    Triggers: HIGH resource (block size 32 → very low occupancy warning),
    #              LOW fragility (loop bound of 32), LOW non-det.
    #              Bottleneck: Bandwidth Limited (simple saxpy).
    # -------------------------------------------------------------------------
    dict(
        name       = "tiny_block_saxpy",
        block_size = 32,   # intentionally tiny — 1 warp per block
        n_elements = 16 * 1024 * 1024,
        iters      = 0,
        code       = r"""
__global__ void tiny_block_saxpy(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n)
{
    // Block size 32 = exactly 1 warp; no latency hiding possible
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Hard-coded loop bound of 32 (warp-size assumption)
    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) {
        acc += 0.001f;
    }

    if (idx < n)
        c[idx] = 2.0f * a[idx] + b[idx] + acc;
}
""",
        params     = "const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n",
    ),

    # -------------------------------------------------------------------------
    # 5. float4_copy
    #    Triggers: MEDIUM fragility (float4* cast alignment assumption),
    #              LOW non-det (global load without guard),
    #              LOW resource (register usage slightly elevated).
    #              Bottleneck: Bandwidth Limited (vectorised copy at near-peak BW).
    # -------------------------------------------------------------------------
    dict(
        name       = "float4_copy",
        block_size = 256,
        n_elements = 16 * 1024 * 1024,
        iters      = 0,
        code       = r"""
__global__ void float4_copy(const float* __restrict__ a,
                              float* __restrict__ c,
                              int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorised load/store assuming 16-byte alignment (fragility flag)
    const float4* a4 = (const float4*)a;
    float4*       c4 = (float4*)c;

    // No bounds guard on the vectorised path — potential OOB for N % 4 != 0
    float4 v = a4[idx];
    c4[idx]  = v;
}
""",
        params     = "const float* __restrict__ a, float* __restrict__ c, int n",
    ),

    # -------------------------------------------------------------------------
    # 6. race_smem_reduce
    #    Triggers: HIGH non-det (shared write then read without barrier),
    #              HIGH fragility (st.shared not followed by bar.sync),
    #              MEDIUM resource.  Bottleneck: Shared Memory Bound.
    # -------------------------------------------------------------------------
    dict(
        name       = "race_smem_reduce",
        block_size = 256,
        n_elements = 16 * 1024 * 1024,
        iters      = 0,
        code       = r"""
__global__ void race_smem_reduce(const float* __restrict__ a,
                                  float* __restrict__ out,
                                  int n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    // BUG: missing __syncthreads() here — race condition
    // Thread tid reads neighbor written by tid+1 which may not be stored yet
    float left = (tid > 0) ? sdata[tid - 1] : 0.0f;

    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0] + left;
}
""",
        params     = "const float* __restrict__ a, float* __restrict__ out, int n",
    ),

]


def main():
    registry = km.load_registry()
    existing = {k["name"] for k in registry}

    added   = []
    skipped = []

    for kw in kernels:
        name = kw["name"]
        if name in existing:
            skipped.append(name)
            continue
        err = km.add_kernel(
            name       = name,
            code       = kw["code"].strip(),
            params     = kw["params"],
            block_size = kw.get("block_size", 256),
            n_elements = kw.get("n_elements", 16 * 1024 * 1024),
            iters      = kw.get("iters", 0),
        )
        if err:
            print(f"  ERROR adding '{name}': {err}")
        else:
            added.append(name)
            print(f"  Added: {name}")

    if skipped:
        print(f"\nSkipped (already exist): {', '.join(skipped)}")
    print(f"\nDone — added {len(added)} kernel(s).")


if __name__ == "__main__":
    main()
