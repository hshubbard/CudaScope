# CUDA Kernel Micro-Analysis Report
_Generated: 2026-02-22 22:10 UTC_
## Methodology
- **Runtime data** – real GPU execution measured with `cudaEventRecord` / `cudaEventElapsedTime` over 100 iterations after 5 warm-up runs.  No CPU-side timing was used.
- **PTX data** – static instruction counts extracted from compiler-generated PTX (`nvcc -O3 --ptx`).  Counts reflect the static instruction mix, not dynamic execution frequency.
- **Derived metrics** – computed algebraically from measured quantities above.  Each metric lists its source explicitly.
- **Bottleneck labels** – produced by a rule-based classifier whose thresholds and triggering metrics are fully documented in `analyze.py`.
## Bottleneck Summary
| Kernel | Runtime (µs) | BW (GB/s) | Bottleneck |
|--------|-------------|-----------|------------|
| `tiled_transpose` | 629.0 ± 131.0 | 0.1 | **Shared Memory Bound** |

## Kernel Results
### tiled_transpose
- **Runtime** (measured): 629.0 µs ± 131.0 µs
- **Runtime share** (derived from measured): 100.0%
- **Approx bandwidth** (derived): 0.1 GB/s
- **Memory op ratio** (PTX static): 0.043
- **Branch ratio** (PTX static): 0.065
- **Arithmetic intensity** (PTX static, derived): 7.00
- **Bottleneck** (inferred): **Shared Memory Bound**

> Kernel uses shared memory: shared_loads=1, shared_stores=1 [SOURCE: PTX static]. Elevated branch_ratio=0.065 reflects synchronization loop conditions (e.g. tree-reduction halving), not true warp serialisation — all threads in a warp take the same branch at each step. Effective bandwidth=0.1 GB/s (8% of observed peak) [SOURCE: derived from measured runtime]. Runtime share=100.0% [SOURCE: measured].

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
