# CUDA Kernel Analysis Summary

_Generated: 2026-02-23 19:25 UTC_

> Scores are normalised 0–100 within this run (100 = worst kernel for that dimension).  Full details: `portability.md`, `determinism.md`, `resource_pressure.md`, `report.md`.

## Kernel Scorecard

| Kernel | Bottleneck | Fragility | Non-Det | Resource | Risk | Action |
|--------|-----------|----------:|--------:|---------:|------|--------|
| `warp_reduce_shuffle` | Compute Balanced | 100/100 | 0/100 | 42/100 | **MEDIUM** | Consider: Profile register pressure / occupancy in Nsight Compute. |
| `large_smem_stencil` | Shared Memory Bound | 0/100 | 100/100 | 17/100 | **MEDIUM** | Consider: Tune block size or replace smem reduction with warp shuffles. |
| `race_smem_reduce` | Shared Memory Bound | 0/100 | 100/100 | 17/100 | **MEDIUM** | Consider: Tune block size or replace smem reduction with warp shuffles. |
| `atomic_histogram` | Warp Divergence | 17/100 | 0/100 | 100/100 | **MEDIUM** | Consider: Eliminate per-thread branch with branchless arithmetic. |
| `tiled_transpose` | Compute Balanced | 17/100 | 0/100 | 17/100 | **LOW** | Profile register pressure / occupancy in Nsight Compute. |
| `float4_copy` | Bandwidth Limited | 8/100 | 0/100 | 0/100 | **LOW** | Fuse kernels or use half-precision to cut bandwidth demand. |
| `tiny_block_saxpy` | Bandwidth Limited | 3/100 | 0/100 | 0/100 | **LOW** | Fuse kernels or use half-precision to cut bandwidth demand. |

## Hotspots by Kernel

_Only the top-3 most severe flags per pass are shown. See individual pass reports for the full list._

### `warp_reduce_shuffle`  —  combined score **50/100** [MEDIUM]

- **Bottleneck:** Compute Balanced  |  **Runtime:** 255.8 µs  |  **BW:** 787.0 GB/s
- **Next action:** Consider: Profile register pressure / occupancy in Nsight Compute.

**Fragility — top flags:**

- `[HIGH]` `warp_assumption` @ `ptx:L59` — PTX `and.b32 <reg>, 31` computes lane-ID by masking with 31, assuming warpSize == 32. AMD GPUs use warpSize == 64; this mask will silently produce wrong lane IDs there.
- `[MEDIUM]` `warp_assumption` @ `ptx:L58` — PTX shift by 5 (>> 5 or << 5) used for warp-ID or warp-stride arithmetic implies warpSize == 32. Use `warpSize` runtime value or PTX `%warpsize` special register instead.
- `[MEDIUM]` `arch_specific_instruction` @ `ptx:L36` — shfl.sync requires SM 7.0+; will not compile for older targets.  [Range conflict: requires SM 7.0+, but target min is SM 3.0]

**Resource — top flags:**

- `[MEDIUM]` `register_pressure` @ `ptx:reg_decl` — PTX declares 77 registers per thread. On SM 7.0–9.0 (64 K regs / SM), more than 64 regs/thread limits concurrent warps to <= 32 (50% occupancy). Consider `__launch_bounds__(block_size, min_blocks)` to give the compiler a register budget.


### `large_smem_stencil`  —  combined score **39/100** [MEDIUM]

- **Bottleneck:** Shared Memory Bound  |  **Runtime:** 468.1 µs  |  **BW:** 430.1 GB/s
- **Next action:** Consider: Tune block size or replace smem reduction with warp shuffles.

**Non-Determinism — top flags:**

- `[HIGH]` `timing_dependency` @ `ptx:L62` — Conditional branch driven by a shared-memory load (`ld.shared` -> `setp` -> `bra`) without a preceding `bar.sync`. If the shared memory was written by another thread, the branch outcome depends on when that write completes — a timing-dependent control-flow race.

**Resource — top flags:**

- `[LOW]` `register_pressure` @ `ptx:reg_decl` — PTX declares 42 registers per thread. This is within bounds but will limit occupancy to ~4% on a 64K-register SM. Monitor with Nsight Compute.


### `race_smem_reduce`  —  combined score **39/100** [MEDIUM]

- **Bottleneck:** Shared Memory Bound  |  **Runtime:** 577.0 µs  |  **BW:** 348.9 GB/s
- **Next action:** Consider: Tune block size or replace smem reduction with warp shuffles.

**Non-Determinism — top flags:**

- `[HIGH]` `timing_dependency` @ `ptx:L39` — Conditional branch driven by a shared-memory load (`ld.shared` -> `setp` -> `bra`) without a preceding `bar.sync`. If the shared memory was written by another thread, the branch outcome depends on when that write completes — a timing-dependent control-flow race.

**Resource — top flags:**

- `[LOW]` `register_pressure` @ `ptx:reg_decl` — PTX declares 38 registers per thread. This is within bounds but will limit occupancy to ~5% on a 64K-register SM. Monitor with Nsight Compute.


### `atomic_histogram`  —  combined score **32/100** [MEDIUM]

- **Bottleneck:** Warp Divergence  |  **Runtime:** 54985.2 µs  |  **BW:** 3.7 GB/s
- **Next action:** Consider: Eliminate per-thread branch with branchless arithmetic.

**Fragility — top flags:**

- `[HIGH]` `warp_assumption` @ `ptx:L36` — PTX `and.b32 <reg>, 31` computes lane-ID by masking with 31, assuming warpSize == 32. AMD GPUs use warpSize == 64; this mask will silently produce wrong lane IDs there.

**Resource — top flags:**

- `[MEDIUM]` `sm_monopolization` @ `ptx:atom_density` — 5 global atomic instructions (8% of total). Moderate atomic density may cause memory-bus contention on high-thread-count launches. Consider warp-level pre-reduction before the global atomic to reduce contention.
- `[MEDIUM]` `sm_monopolization` @ `ptx:branch_density` — Branch instruction density 17% (10/59). High branch density in PTX indicates frequent control flow changes; if warps diverge at these branches, the SM serialises both paths, reducing the effective thread count and leaving other warps starved.
- `[LOW]` `register_pressure` @ `ptx:reg_decl` — PTX declares 44 registers per thread. This is within bounds but will limit occupancy to ~4% on a 64K-register SM. Monitor with Nsight Compute.


### `tiled_transpose`  —  combined score **11/100** [LOW]

- **Bottleneck:** Compute Balanced  |  **Runtime:** 606.0 µs  |  **BW:** 0.1 GB/s
- **Next action:** Profile register pressure / occupancy in Nsight Compute.

**Fragility — top flags:**

- `[MEDIUM]` `warp_assumption` @ `ptx:L18` — PTX shift by 5 (>> 5 or << 5) used for warp-ID or warp-stride arithmetic implies warpSize == 32. Use `warpSize` runtime value or PTX `%warpsize` special register instead.

**Resource — top flags:**

- `[LOW]` `register_pressure` @ `ptx:reg_decl` — PTX declares 34 registers per thread. This is within bounds but will limit occupancy to ~5% on a 64K-register SM. Monitor with Nsight Compute.


### `float4_copy`  —  combined score **3/100** [LOW]

- **Bottleneck:** Bandwidth Limited  |  **Runtime:** 456.7 µs  |  **BW:** 110.2 GB/s
- **Next action:** Fuse kernels or use half-precision to cut bandwidth demand.

**Fragility — top flags:**

- `[MEDIUM]` `memory_alignment` @ `ptx:L28` — Vector store `st.v4` assumes 16-byte alignment. Misaligned stores can cause access-fault on SM 3.x and silent partial writes on some architectures.


### `tiny_block_saxpy`  —  combined score **1/100** [LOW]

- **Bottleneck:** Bandwidth Limited  |  **Runtime:** 1149.2 µs  |  **BW:** 175.2 GB/s
- **Next action:** Fuse kernels or use half-precision to cut bandwidth demand.

**Fragility — top flags:**

- `[LOW]` `undefined_behavior` @ `ptx:L31` — Global memory load without a visible bounds-check guard in the preceding PTX. If thread count does not evenly divide the array size, out-of-bounds threads may load garbage or fault. Verify that an `if (idx < n)` guard is present.


## Score Legend

| Score | Meaning |
|------:|---------|
| 0     | No issues detected in this pass |
| 1–24  | Low risk — monitor |
| 25–59 | Medium risk — review before next architecture |
| 60–100 | High risk — fix before porting |

Combined score = 40% Fragility + 35% Non-Determinism + 25% Resource Pressure.

**Detailed reports:** `output/report/portability.md` · `output/report/determinism.md` · `output/report/resource_pressure.md` · `output/report/report.md`
