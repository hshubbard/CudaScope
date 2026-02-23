# CUDA Kernel Micro-Analysis Tool

A GPU performance analysis tool that runs real CUDA kernels, collects real
execution timing, performs static PTX analysis, and produces bottleneck
reports, with every metric traceable to its source.

---

## Version History

### v1.1 (current)
- **Three static analysis passes** — portability, determinism, resource pressure
- **Dashboard GUI** — colour-coded risk scores (HIGH / MEDIUM / LOW) per kernel
- **User kernel support** — add your own kernels via `kernel_manager.py` or the GUI
- **Hardware-aware peak BW** — queried live from `nvidia-smi` instead of hardcoded
- **Bottleneck classifier fixes** — smem threshold, float4 OOB, atomic_histogram redesign

### v1.0
- 5 built-in kernels (coalesced_add, strided_add, divergent_add, divergent_compute, compute_ref)
- Runtime measurement via CUDA events
- PTX static analysis
- Bottleneck inference engine
- Heatmap + summary table + Markdown report

---

## Quick Start

```bash
# Install Python dependencies
pip install pandas numpy matplotlib seaborn
```

**Linux / macOS**
```bash
bash scripts/analyze.sh   # build → benchmark → PTX → analyze
python src/gui.py         # launch the GUI
```

**Windows**
```powershell
bash scripts/analyze.sh   # requires Git Bash or WSL
python src/gui.py
```

> The pipeline auto-detects Windows and builds `benchmark_user.exe`. The `.exe`
> is gitignored — it is always rebuilt locally and never committed.

Outputs land in `./output/`.

---

## Project Structure

```
.
├── src/                        # Python analysis scripts
│   ├── gui.py                  # tkinter GUI — Dashboard + Settings tabs
│   ├── kernel_manager.py       # User kernel registry and pipeline runner
│   ├── kernel_add.py           # CLI for adding a single user kernel
│   ├── analyze.py              # Metric derivation, bottleneck inference, visualisation
│   ├── ptx_parser.py           # Static PTX instruction analysis
│   ├── portability_pass.py     # Fragility analysis pass
│   ├── determinism_pass.py     # Non-determinism detection pass
│   ├── resource_pressure_pass.py  # Occupancy / resource pressure pass
│   └── summary_report.py       # Aggregates all three passes into risk scores
│
├── kernels/                    # CUDA source files
│   ├── kernels.cu / .cuh       # Built-in benchmark kernels
│   ├── benchmark.cu            # Built-in benchmark harness
│   ├── user_kernels.cu / .cuh  # Generated user kernel code
│   ├── user_benchmark.cu       # Generated user benchmark harness
│   └── user_kernels.json       # Persistent user kernel registry
│
├── scripts/                    # Utility scripts
│   ├── analyze.sh              # One-shot CLI (build → bench → PTX → analyze)
│   └── add_test_kernels.py     # Adds 6 test kernels covering all risk bands
│
└── output/                     # Generated (gitignored)
    ├── data/                   # runtimes.csv, ptx_stats.csv
    ├── ptx/                    # Per-kernel .ptx files
    ├── plots/                  # heatmap.png, summary_table.png, summary_scores.png
    └── report/                 # report.md + per-pass reports + JSON
```

---

## Built-in Kernels

### `coalesced_add` — Baseline
```
thread idx   →   memory address
0            →   A[0]
1            →   A[1]
2            →   A[2]
```
Consecutive threads access consecutive addresses. The GPU merges 32 accesses
into a single 128-byte cache-line transaction. **Optimal** memory access pattern.

### `strided_add` — Non-coalesced Memory
```
thread idx   →   memory address  (stride = 32)
0            →   A[0]
1            →   A[32]
2            →   A[64]
```
Each thread touches a different cache line — 32 separate transactions instead
of 1. Cache-line utilisation drops to ~3%.

### `divergent_add` — Warp Divergence
```c
if (threadIdx.x % 2 == 0)
    c[idx] = a[idx] + b[idx];   // even threads
else
    c[idx] = a[idx] - b[idx];   // odd threads
```
Splits the warp into two groups that execute sequentially, effectively halving
throughput.

### `divergent_compute` / `compute_ref`
Heavy compute kernel with and without divergence — isolates the cost of warp
serialisation on your hardware.

---

## Measurement Methodology

### Runtime

```
cudaEventRecord(start)
kernel<<<grid, block>>>(...)
cudaEventRecord(stop)
cudaEventSynchronize(stop)
cudaEventElapsedTime(&ms, start, stop)
```

- **Warm-up runs (5):** discarded; ensure JIT compilation and GPU state are stable.
- **Timed runs (100):** averaged to reduce launch-overhead noise.
- **Result:** mean ± standard deviation in microseconds.

### PTX Static Analysis

```bash
nvcc -O3 --ptx kernels.cu -o kernels.ptx
python ptx_parser.py output/ptx/*.ptx
```

| Category | PTX opcodes matched |
|---|---|
| `global_loads` | `ld.global.*` |
| `global_stores` | `st.global.*` |
| `shared_loads` | `ld.shared.*` |
| `shared_stores` | `st.shared.*` |
| `arithmetic` | `fma, mad, add, sub, mul, div, neg, …` |
| `branch` | `bra, brx.idx, call, ret, exit` |
| `special` | `bar, atom, red, tex, shfl, vote, prmt` |

---

## Static vs Dynamic Analysis

| | Static (PTX) | Dynamic (CUDA events) |
|---|---|---|
| **What is measured** | Compiler-emitted instruction text | Wall-clock GPU execution time |
| **Accounts for loops** | No — counts each instruction once | Yes — all iterations included |
| **Accounts for branches** | No — both paths counted equally | Yes — runtime reflects actual path |
| **Source label used** | `[SOURCE: PTX static]` | `[SOURCE: measured runtime]` |

---

## Derived Metrics

| Metric | Formula | Inputs |
|---|---|---|
| Runtime share | `mean_us / Σ mean_us` | measured runtime |
| Bytes accessed | `n_elements × 4 × 3` | array size (benchmark config) |
| Approx bandwidth | `bytes_accessed / (mean_us × 1e-6) / 1e9` | measured runtime + array size |
| Arithmetic intensity | `arithmetic_ops / (global_loads + global_stores)` | PTX static counts |
| Memory ratio | `(global_loads + global_stores) / total_instr` | PTX static counts |
| Branch ratio | `branch_instr / total_instr` | PTX static counts |
| BW efficiency | `approx_bandwidth / nvidia-smi theoretical peak` | derived + hardware spec |

---

## Bottleneck Inference Engine

Rules applied in priority order; first match wins.

```
Rule 0 — Reference kernel (compute_ref only)
  → label: "Compute Bound (Reference)"

Rule 1a — Shared Memory Bound
  IF shared_loads + shared_stores >= 4   [SOURCE: PTX static]
  → label: "Shared Memory Bound"

Rule 1b — Warp Divergence
  IF branch_ratio > 0.10                 [SOURCE: PTX static]
  → label: "Warp Divergence"

Rule 2 — Memory Bound
  IF memory_ratio > 0.30                 [SOURCE: PTX static]
  AND arith_intensity < 3.0              [SOURCE: PTX static, derived]
  → label: "Memory Bound"

Rule 3a — Poor Coalescing
  IF arith_intensity < 3.0              [SOURCE: PTX static, derived]
  AND bw_efficiency < 0.20              [SOURCE: derived + nvidia-smi]
  → label: "Memory Bound (Poor Coalescing)"

Rule 3b — Bandwidth Limited
  IF arith_intensity < 3.0              [SOURCE: PTX static, derived]
  → label: "Bandwidth Limited"

Rule 4 — Compute Balanced (default)
  → label: "Compute Balanced"
```

Thresholds are documented and configurable in `analyzer_config.json` via the
Settings tab in the GUI.

---

## Risk Scoring (v1.1)

Each kernel receives a combined risk score (0–100) from three passes:

| Pass | Weight | Detects |
|------|--------|---------|
| Fragility (`portability_pass.py`) | 40% | Architecture assumptions, alignment casts, missing barriers |
| Non-determinism (`determinism_pass.py`) | 35% | FP atomics, shuffle accumulation, shared memory races |
| Resource pressure (`resource_pressure_pass.py`) | 25% | Low occupancy, high smem, register pressure |

Scores are normalised so 100 = worst kernel in this run, 0 = no issues.

| Band | Score |
|------|-------|
| HIGH | ≥ 60 |
| MEDIUM | 25 – 59 |
| LOW | 1 – 24 |
| None | 0 |

---

## Outputs

| Path | Contents |
|------|----------|
| `output/data/runtimes.csv` | Measured kernel runtimes (mean, std, bottleneck) |
| `output/data/ptx_stats.csv` | PTX instruction counts per kernel |
| `output/ptx/*.ptx` | Per-kernel PTX assembly |
| `output/plots/heatmap.png` | Metric heatmap (kernels × metrics) |
| `output/plots/summary_table.png` | Colour-coded bottleneck summary |
| `output/plots/summary_scores.png` | Risk score bar chart |
| `output/report/report.md` | Full bottleneck analysis with optimisation suggestions |
| `output/report/portability.md` | Fragility pass findings |
| `output/report/determinism.md` | Non-determinism pass findings |
| `output/report/resource_pressure.md` | Resource pressure pass findings |
| `output/report/summary.md` | Combined risk score summary |

---

## Limitations

**No cache hit measurement.** L1/L2 hit rates require hardware performance
counters via CUPTI or Nsight Compute. Bandwidth estimates assume all traffic
goes to DRAM.

**No warp stall reason counters.** Stall causes require CUPTI instrumentation.
Divergence is inferred from PTX branch ratio, not warp stall sampling.

**PTX ≠ SASS.** PTX is a virtual ISA recompiled by `ptxas` into final SASS
machine code. The ptxas pass can fuse, eliminate, and reorder instructions.
Static PTX counts are approximate.

**Static counts ≠ dynamic counts.** An instruction inside a loop appears once
statically but executes many times at runtime.

**Architecture-agnostic thresholds.** Classifier thresholds are heuristic
values, not tuned to a specific NVIDIA SM generation.

**Single device.** All measurements are taken on GPU 0.

---

## Why This Approach Is Valid

1. **Separation of sources.** Every metric is tagged as `[SOURCE: measured runtime]`,
   `[SOURCE: PTX static]`, or `[SOURCE: derived from ...]`. No mixing without labelling.

2. **Defensible inference.** The bottleneck classifier uses simple, documented rules
   with explicit thresholds. It does not claim precision it cannot achieve without
   hardware counters.

3. **Honest limitations.** The Limitations section documents exactly what this tool
   cannot measure and why.

---

## Requirements

| Tool | Purpose |
|------|---------|
| CUDA toolkit (`nvcc`, `nvidia-smi`) | Compile, PTX extraction, hardware BW query |
| C++17-capable host compiler | Linked by nvcc |
| Python 3.8+ | All analysis scripts |
| `pandas`, `numpy`, `matplotlib`, `seaborn` | Analysis + visualisation |
| `tkinter` | GUI (included with most Python distributions) |

```bash
pip install pandas numpy matplotlib seaborn
```

> `tkinter` ships with most Python distributions. If missing: `sudo apt install python3-tk` (Linux).
