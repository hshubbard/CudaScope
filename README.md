# CUDA Kernel Micro-Analysis Tool

A minimal, defensible GPU performance analysis tool that runs real CUDA kernels,
collects real execution timing, performs static PTX analysis, and produces
bottleneck reports — with every metric traceable to its source.

---

## Quick Start

```bash
chmod +x analyze.sh
./analyze.sh
```

Outputs land in `./output/`.

---

## Project Files

| File | Role |
|------|------|
| `kernels.cu` | Three intentionally-flawed CUDA kernels |
| `kernels.cuh` | Kernel declarations |
| `benchmark.cpp` | Runtime measurement via CUDA events |
| `ptx_parser.py` | Static PTX instruction analysis |
| `analyze.py` | Metric derivation, bottleneck inference, visualisation |
| `analyze.sh` | One-shot CLI (build → bench → PTX → analyze) |

---

## The Three Kernels

### 1. `coalesced_add` — Baseline
```
thread idx   →   memory address
0            →   A[0]
1            →   A[1]
2            →   A[2]
```
Consecutive threads access consecutive addresses.  The GPU can merge 32 thread
accesses into a single 128-byte cache-line transaction.  This is the
**optimal** memory access pattern.

### 2. `strided_add` — Non-coalesced Memory
```
thread idx   →   memory address  (stride = 32)
0            →   A[0]
1            →   A[32]
2            →   A[64]
```
Consecutive threads are 32 elements apart.  Each thread touches a different
cache line, so 32 threads issue 32 separate transactions instead of 1.
Cache-line utilisation drops to ~3% (1 useful float per 128-byte line).

### 3. `divergent_add` — Warp Divergence
```c
if (threadIdx.x % 2 == 0)
    c[idx] = a[idx] + b[idx];   // even threads
else
    c[idx] = a[idx] - b[idx];   // odd threads
```
All 32 threads in a warp execute in lock-step (SIMT).  The `if` splits the
warp into two groups that must execute sequentially — effectively halving
throughput.

---

## Measurement Methodology

### Runtime (Phase 2)

```
cudaEventRecord(start)
kernel<<<grid, block>>>(...)
cudaEventRecord(stop)
cudaEventSynchronize(stop)
cudaEventElapsedTime(&ms, start, stop)
```

- **Warm-up runs (5):** discarded; ensure JIT compilation and GPU state
  are stable before timing starts.
- **Timed runs (100):** averaged to reduce launch-overhead noise.
- **Synchronisation:** `cudaEventSynchronize` ensures the GPU has completed
  before the timer is read.  No CPU-side `clock()` or `gettimeofday()` is used.
- **Result:** mean ± standard deviation in microseconds.

### PTX Static Analysis (Phase 3)

```bash
nvcc -O3 --ptx kernels.cu -o kernels.ptx
python ptx_parser.py output/*.ptx
```

The parser scans each line of the PTX file, strips comments, and classifies
each instruction into one of:

| Category | PTX opcodes matched |
|---|---|
| `global_loads` | `ld.global.*` |
| `global_stores` | `st.global.*` |
| `shared_loads` | `ld.shared.*` |
| `shared_stores` | `st.shared.*` |
| `arithmetic` | `fma, mad, add, sub, mul, div, neg, …` |
| `branch` | `bra, brx.idx, call, ret, exit` |
| `special` | `bar, atom, red, tex, shfl, vote, prmt` |
| `other` | everything else |

---

## Static vs Dynamic Analysis

| | Static (PTX) | Dynamic (CUDA events) |
|---|---|---|
| **What is measured** | Compiler-emitted instruction text | Wall-clock GPU execution time |
| **When** | After `nvcc --ptx`, offline | During live GPU execution |
| **Accounts for loops** | No — counts each instruction once | Yes — all iterations included |
| **Accounts for branches** | No — both paths counted equally | Yes — runtime reflects actual path |
| **Source label used** | `[SOURCE: PTX static]` | `[SOURCE: measured runtime]` |

---

## Derived Metrics

Every derived metric lists its inputs:

| Metric | Formula | Inputs |
|---|---|---|
| Runtime share | `mean_us / Σ mean_us` | measured runtime |
| Bytes accessed | `n_elements × 4 × 3` | array size (benchmark config) |
| Approx bandwidth | `bytes_accessed / (mean_us × 1e-6) / 1e9` | measured runtime + array size |
| Arithmetic intensity | `arithmetic_ops / (global_loads + global_stores)` | PTX static counts |
| Memory ratio | `(global_loads + global_stores) / total_instr` | PTX static counts |
| Branch ratio | `branch_instr / total_instr` | PTX static counts |

---

## Bottleneck Inference Engine

The classifier applies rules in priority order.  Each rule cites the metric
that triggered it and its source.

```
Rule 1 — Warp Divergence
  IF branch_ratio > 0.05            [SOURCE: PTX static]
  → label: "Warp Divergence"

Rule 2 — Memory Bound
  IF memory_ratio > 0.30            [SOURCE: PTX static]
  AND arith_intensity < 1.0         [SOURCE: PTX static, derived]
  → label: "Memory Bound"

Rule 3 — Bandwidth Limited
  IF arith_intensity < 1.0          [SOURCE: PTX static, derived]
  → label: "Bandwidth Limited"

Rule 4 — Compute Balanced (default)
  → label: "Compute Balanced"
```

Thresholds are deliberately conservative and documented in `analyze.py`.
They produce directionally correct labels — not quantitatively precise ones.

---

## Outputs

| File | Contents |
|------|----------|
| `output/runtimes.csv` | Measured kernel runtimes (mean, std, config) |
| `output/ptx_stats.csv` | PTX instruction counts per kernel |
| `output/heatmap.png` | Metric heatmap (kernels × metrics) |
| `output/summary_table.png` | Colour-coded summary table |
| `output/report.md` | Full analysis report with optimisation suggestions |

---

## Limitations

**No cache hit measurement.**  L1/L2 hit rates and cache-line reuse require
hardware performance counters accessed via CUPTI or Nsight Compute.  Bandwidth
estimates in this tool assume all memory traffic goes to DRAM — this
overestimates effective bandwidth when data is cached.

**No warp stall reason counters.**  Stall causes (long-scoreboard, memory
dependency, synchronisation barrier) require CUPTI instrumentation.  The
divergence label is inferred from PTX branch instruction ratio, not from
warp stall sampling.

**PTX ≠ SASS.**  PTX is a virtual ISA compiled again by `ptxas` into the
final SASS machine code for your specific GPU.  The ptxas pass can fuse,
eliminate, and reorder instructions.  Static PTX counts are approximate.

**Static counts ≠ dynamic counts.**  A PTX instruction inside a loop body
appears once in the static count but executes many times at runtime.  The
arithmetic intensity derived here reflects the static instruction ratio, not
the true dynamic FLOP:byte ratio.

**Architecture-agnostic thresholds.**  The classifier thresholds (memory
ratio > 0.30, branch ratio > 0.05, arithmetic intensity < 1.0) are
heuristic values, not tuned to a specific NVIDIA SM generation.  A Volta
SM may have different effective thresholds than an Ada Lovelace SM.

**Single device.**  All measurements are taken on GPU 0.  Multi-GPU systems
are not handled; device selection is not exposed.

---

## Why This Approach Is Valid

The tool's credibility rests on three principles:

1. **Separation of sources.**  Every metric is tagged as either
   `[SOURCE: measured runtime]`, `[SOURCE: PTX static]`, or
   `[SOURCE: derived from ...]`.  There is no mixing of real and estimated
   data without explicit labelling.

2. **Defensible inference.**  The bottleneck classifier uses simple, documented
   rules with explicit thresholds.  It does not claim precision it cannot
   achieve without hardware counters.

3. **Honest limitations.**  The Limitations section above documents exactly
   what this tool cannot measure and why.  An honest tool that admits its
   gaps is more credible than an impressive-looking tool that hides them.

---

## Requirements

| Tool | Purpose |
|------|---------|
| CUDA toolkit (`nvcc`) | Compile + PTX extraction |
| C++17-capable host compiler | Linked by nvcc |
| Python 3.8+ | ptx_parser.py, analyze.py |
| `pandas`, `numpy`, `matplotlib`, `seaborn` | Analysis + visualisation |

Install Python dependencies:
```bash
pip install pandas numpy matplotlib seaborn
```
