# CUDA Kernel Micro-Analysis Report
_Generated: 2026-02-23 19:25 UTC_
## Methodology
- **Runtime data** – real GPU execution measured with `cudaEventRecord` / `cudaEventElapsedTime` over 100 iterations after 5 warm-up runs.  No CPU-side timing was used.
- **PTX data** – static instruction counts extracted from compiler-generated PTX (`nvcc -O3 --ptx`).  Counts reflect the static instruction mix, not dynamic execution frequency.
- **Derived metrics** – computed algebraically from measured quantities above.  Each metric lists its source explicitly.
- **Bottleneck labels** – produced by a rule-based classifier whose thresholds and triggering metrics are fully documented in `analyze.py`.
## Bottleneck Summary
| Kernel | Runtime (µs) | BW (GB/s) | Bottleneck |
|--------|-------------|-----------|------------|
| `tiled_transpose` | 606.0 ± 93.1 | 0.1 | **Compute Balanced** |
| `atomic_histogram` | 54985.2 ± 339.0 | 3.7 | **Warp Divergence** |
| `warp_reduce_shuffle` | 255.8 ± 1.6 | 787.0 | **Compute Balanced** |
| `large_smem_stencil` | 468.1 ± 84.6 | 430.1 | **Shared Memory Bound** |
| `tiny_block_saxpy` | 1149.2 ± 105.9 | 175.2 | **Bandwidth Limited** |
| `float4_copy` | 456.7 ± 66.0 | 110.2 | **Bandwidth Limited** |
| `race_smem_reduce` | 577.0 ± 75.0 | 348.9 | **Shared Memory Bound** |

## Kernel Results
### tiled_transpose
- **Runtime** (measured): 606.0 µs ± 93.1 µs
- **Runtime share** (derived from measured): 1.0%
- **Approx bandwidth** (derived): 0.1 GB/s
- **Memory op ratio** (PTX static): 0.043
- **Branch ratio** (PTX static): 0.065
- **Arithmetic intensity** (PTX static, derived): 7.00
- **Bottleneck** (inferred): **Compute Balanced**

> Balanced instruction mix: memory ratio=0.043, branch ratio=0.065, arithmetic intensity=7.00 [SOURCE: PTX static]. No single bottleneck dominates. Runtime share=1.0% [SOURCE: measured].

**Suggestion:** Kernel is already well-balanced.  Profile with Nsight Compute for register pressure or occupancy limits before further optimisation.

---
### atomic_histogram
- **Runtime** (measured): 54985.2 µs ± 339.0 µs
- **Runtime share** (derived from measured): 94.0%
- **Approx bandwidth** (derived): 3.7 GB/s
- **Memory op ratio** (PTX static): 0.017
- **Branch ratio** (PTX static): 0.186
- **Arithmetic intensity** (PTX static, derived): 22.00
- **Bottleneck** (inferred): **Warp Divergence**

> Branch ratio=0.186 exceeds threshold 0.1 [SOURCE: PTX static]. Frequent branch instructions force the warp scheduler to serialise diverging paths within each warp, reducing effective SIMT parallelism. Runtime share=94.0% [SOURCE: measured].

**Suggestion:** Restructure the branch condition to use block-level rather than thread-level divergence, or replace the conditional with branchless arithmetic (e.g., `a + sign * b`).

---
### warp_reduce_shuffle
- **Runtime** (measured): 255.8 µs ± 1.6 µs
- **Runtime share** (derived from measured): 0.4%
- **Approx bandwidth** (derived): 787.0 GB/s
- **Memory op ratio** (PTX static): 0.023
- **Branch ratio** (PTX static): 0.070
- **Arithmetic intensity** (PTX static, derived): 7.50
- **Bottleneck** (inferred): **Compute Balanced**

> Balanced instruction mix: memory ratio=0.023, branch ratio=0.070, arithmetic intensity=7.50 [SOURCE: PTX static]. No single bottleneck dominates. Runtime share=0.4% [SOURCE: measured].

**Suggestion:** Kernel is already well-balanced.  Profile with Nsight Compute for register pressure or occupancy limits before further optimisation.

---
### large_smem_stencil
- **Runtime** (measured): 468.1 µs ± 84.6 µs
- **Runtime share** (derived from measured): 0.8%
- **Approx bandwidth** (derived): 430.1 GB/s
- **Memory op ratio** (PTX static): 0.042
- **Branch ratio** (PTX static): 0.104
- **Arithmetic intensity** (PTX static, derived): 7.00
- **Bottleneck** (inferred): **Shared Memory Bound**

> Kernel uses shared memory: shared_loads=4, shared_stores=2 [SOURCE: PTX static]. Elevated branch_ratio=0.104 reflects synchronization loop conditions (e.g. tree-reduction halving), not true warp serialisation — all threads in a warp take the same branch at each step. Effective bandwidth=430.1 GB/s (215% of observed peak) [SOURCE: derived from measured runtime]. Runtime share=0.8% [SOURCE: measured].

**Suggestion:** Kernel is already using shared memory effectively. Consider increasing occupancy (tune block size), using warp-level primitives (`__shfl_down_sync`) to replace shared memory in the final reduction stages, or vectorised loads (`float4`) to improve global memory throughput on the initial load pass.

---
### tiny_block_saxpy
- **Runtime** (measured): 1149.2 µs ± 105.9 µs
- **Runtime share** (derived from measured): 2.0%
- **Approx bandwidth** (derived): 175.2 GB/s
- **Memory op ratio** (PTX static): 0.115
- **Branch ratio** (PTX static): 0.077
- **Arithmetic intensity** (PTX static, derived): 2.33
- **Bottleneck** (inferred): **Bandwidth Limited**

> Arithmetic intensity=2.33 < 3.0 [SOURCE: PTX static, derived]. Compute work is low relative to memory transactions; kernel sits below the roofline ridge point. Effective bandwidth=175.2 GB/s (88% of observed peak) [SOURCE: derived from measured runtime]. Runtime share=2.0% [SOURCE: measured].

**Suggestion:** Fuse adjacent kernels to increase arithmetic intensity, or use half-precision storage to halve bandwidth demand.

---
### float4_copy
- **Runtime** (measured): 456.7 µs ± 66.0 µs
- **Runtime share** (derived from measured): 0.8%
- **Approx bandwidth** (derived): 110.2 GB/s
- **Memory op ratio** (PTX static): 0.100
- **Branch ratio** (PTX static): 0.100
- **Arithmetic intensity** (PTX static, derived): 2.00
- **Bottleneck** (inferred): **Bandwidth Limited**

> Arithmetic intensity=2.00 < 3.0 [SOURCE: PTX static, derived]. Compute work is low relative to memory transactions; kernel sits below the roofline ridge point. Effective bandwidth=110.2 GB/s (55% of observed peak) [SOURCE: derived from measured runtime]. Runtime share=0.8% [SOURCE: measured].

**Suggestion:** Fuse adjacent kernels to increase arithmetic intensity, or use half-precision storage to halve bandwidth demand.

---
### race_smem_reduce
- **Runtime** (measured): 577.0 µs ± 75.0 µs
- **Runtime share** (derived from measured): 1.0%
- **Approx bandwidth** (derived): 348.9 GB/s
- **Memory op ratio** (PTX static): 0.040
- **Branch ratio** (PTX static): 0.140
- **Arithmetic intensity** (PTX static, derived): 4.50
- **Bottleneck** (inferred): **Shared Memory Bound**

> Kernel uses shared memory: shared_loads=4, shared_stores=2 [SOURCE: PTX static]. Elevated branch_ratio=0.140 reflects synchronization loop conditions (e.g. tree-reduction halving), not true warp serialisation — all threads in a warp take the same branch at each step. Effective bandwidth=348.9 GB/s (174% of observed peak) [SOURCE: derived from measured runtime]. Runtime share=1.0% [SOURCE: measured].

**Suggestion:** Kernel is already using shared memory effectively. Consider increasing occupancy (tune block size), using warp-level primitives (`__shfl_down_sync`) to replace shared memory in the final reduction stages, or vectorised loads (`float4`) to improve global memory throughput on the initial load pass.

---
## Limitations
- **No cache hit measurement** – L1/L2 hit rates require hardware performance counters (Nsight / CUPTI). Bandwidth estimates assume full DRAM traffic.
- **No warp stall reason** – stall categories (memory, execution dependency, synchronisation) are not available without CUPTI instrumentation.
- **PTX ≠ SASS** – PTX is a virtual ISA; the final machine code (SASS) may differ after the ptxas compiler pass.  Instruction counts are approximate.
- **Static counts ≠ dynamic counts** – loops and divergent branches cause instruction counts to diverge from PTX static totals at runtime.
- **Architecture-agnostic** – thresholds in the classifier are not tuned to a specific SM architecture.  Results are directionally correct but not quantitatively precise.
- **Single-device** – all results are for GPU 0; multi-GPU systems are not handled.

## Outputs
| File | Contents |
|------|----------|
| `output/data/runtimes.csv` | Measured kernel runtimes |
| `output/data/ptx_stats.csv` | PTX static instruction counts |
| `output/ptx/*.ptx` | Per-kernel PTX assembly |
| `output/plots/heatmap.png` | Metric heatmap |
| `output/plots/summary_table.png` | Summary table |
| `output/report/report.md` | This report |
