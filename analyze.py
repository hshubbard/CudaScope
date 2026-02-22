"""
analyze.py
==========
Metric derivation, bottleneck inference engine, and visualization.

DATA SOURCES
------------
  output/runtimes.csv   – real hardware execution times (CUDA events)
  output/ptx_stats.csv  – static PTX instruction counts (compiler output)

WHAT IS MEASURED vs INFERRED
-----------------------------
MEASURED (runtime):
  - mean_us, std_us per kernel (cudaEventElapsedTime over 100 warm runs)

MEASURED (static PTX):
  - Instruction counts per category (parser over .ptx files)

DERIVED (from measured + array sizes in benchmark.cpp):
  - runtime_share      = kernel_mean / sum(all_means)
  - bytes_accessed     = n_elements * sizeof(float) * (n_reads + n_writes)
                         where n_reads/n_writes = 2/1 for add kernels
  - approx_bandwidth   = bytes_accessed / (mean_us * 1e-6)   [bytes/sec]
  - arith_intensity    = arithmetic_ops_per_element / memory_ops_per_element
                         (from PTX static counts, per-element approximation)

INFERRED (bottleneck classifier):
  Rules are applied in priority order; first match wins.
  Each rule references its triggering metrics explicitly.

OUTPUTS
-------
  output/heatmap.png
  output/summary_table.png
  output/report.md
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

DATA_DIR   = os.path.join("output", "data")    # CSVs (raw measured numbers)
PLOTS_DIR  = os.path.join("output", "plots")   # PNG visualisations
REPORT_DIR = os.path.join("output", "report")  # Markdown report

# ---------------------------------------------------------------------------
# Thresholds for bottleneck classifier
# These are deliberately conservative; label every threshold with its basis.
# ---------------------------------------------------------------------------
# Thresholds calibrated for simple element-wise kernels compiled with -O3.
# nvcc emits bounds-check branches even in "branchless" kernels; we set
# the divergence threshold above the ~8% baseline to catch only kernels
# that have *extra* branches beyond the normal bounds check.
THRESH_MEMORY_RATIO      = 0.30  # >30% global mem ops → memory pressure
THRESH_BRANCH_RATIO      = 0.10  # >10% branch instr → divergence above bounds-check baseline
THRESH_ARITH_INTENSITY   = 3.0   # arith ops / mem ops < 3.0 → bandwidth-limited
# Poor-coalescing detection: if a memory-bound kernel achieves less than
# this fraction of the best observed bandwidth, consecutive threads are
# likely not accessing consecutive cache lines (strided / non-coalesced).
# 0.20 = 20% — tuned so coalesced_add (100% by definition) is not flagged
# while strided_add (~10% of peak) is flagged.
THRESH_BW_EFFICIENCY     = 0.20  # <20% of peak observed BW → poor coalescing suspected

# Load overrides from analyzer_config.json if present (written by GUI settings tab)
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "analyzer_config.json")
if os.path.isfile(_CONFIG_FILE):
    import json as _json
    with open(_CONFIG_FILE, "r") as _cf:
        _cfg = _json.load(_cf)
    THRESH_MEMORY_RATIO    = float(_cfg.get("thresh_memory_ratio",    THRESH_MEMORY_RATIO))
    THRESH_BRANCH_RATIO    = float(_cfg.get("thresh_branch_ratio",    THRESH_BRANCH_RATIO))
    THRESH_ARITH_INTENSITY = float(_cfg.get("thresh_arith_intensity", THRESH_ARITH_INTENSITY))
    THRESH_BW_EFFICIENCY   = float(_cfg.get("thresh_bw_efficiency",   THRESH_BW_EFFICIENCY))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    rt_path  = os.path.join(DATA_DIR, "runtimes.csv")
    ptx_path = os.path.join(DATA_DIR, "ptx_stats.csv")

    missing = [p for p in (rt_path, ptx_path) if not os.path.isfile(p)]
    if missing:
        print("ERROR: missing required files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun the benchmark and PTX parser first.")
        sys.exit(1)

    rt  = pd.read_csv(rt_path)
    ptx = pd.read_csv(ptx_path)
    # Strip stray whitespace/CR from key columns (Windows line-ending safety)
    rt["kernel"]       = rt["kernel"].str.strip()
    ptx["kernel_name"] = ptx["kernel_name"].str.strip()
    return rt, ptx


# ---------------------------------------------------------------------------
# Metric derivation
# ---------------------------------------------------------------------------

def derive_metrics(rt: pd.DataFrame, ptx: pd.DataFrame) -> pd.DataFrame:
    """
    Merge runtime + PTX data and compute derived metrics.
    Every column is annotated with its source in a comment.
    """
    ptx = ptx.rename(columns={"kernel_name": "kernel"})
    df = pd.merge(rt, ptx, on="kernel", how="left")
    # Fill missing PTX columns with 0 (kernel has runtime data but no PTX stats yet)
    ptx_cols = ["global_loads", "global_stores", "shared_loads", "shared_stores",
                "arithmetic", "branch", "special"]
    for col in ptx_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Source: measured runtime
    total_time = df["mean_us"].sum()
    df["runtime_share"] = df["mean_us"] / total_time          # measured runtime

    # bytes_accessed: use n_elements * float_size * 3 arrays.
    # This reflects the number of useful bytes the kernel processes (not the
    # allocation size), so strided kernels show lower effective BW than
    # coalesced ones — which is the correct signal for the classifier.
    FLOAT_BYTES = 4
    df["bytes_accessed"] = df["n_elements"] * FLOAT_BYTES * 3  # measured (array sizes)

    # Approximate memory bandwidth [GB/s]
    # Source: derived from measured runtime + measured array sizes
    df["approx_bandwidth_GBs"] = (
        df["bytes_accessed"] / (df["mean_us"] * 1e-6) / 1e9
    )

    # Arithmetic intensity: arithmetic static ops / memory static ops
    # Source: PTX static counts
    mem_ops = df["global_loads"] + df["global_stores"]
    df["arith_intensity"] = np.where(
        mem_ops > 0,
        df["arithmetic"] / mem_ops,
        np.inf
    )

    # Bandwidth efficiency: kernel BW / reference peak BW.
    # We use coalesced_add as the peak reference — it represents the hardware's
    # achievable bandwidth for a simple element-wise kernel at the same N.
    # If coalesced_add is not in the dataset (user-only run), fall back to max.
    # Source: derived from measured runtime + measured array sizes
    coalesced_rows = df[df["kernel"] == "coalesced_add"]["approx_bandwidth_GBs"]
    if not coalesced_rows.empty:
        peak_bw = float(coalesced_rows.iloc[0])
    else:
        # Exclude obviously-wrong outliers: cap at 99th percentile
        peak_bw = float(df["approx_bandwidth_GBs"].quantile(0.99))
    peak_bw = max(peak_bw, 1.0)  # avoid division by zero
    df["bw_efficiency"] = df["approx_bandwidth_GBs"] / peak_bw

    return df


# ---------------------------------------------------------------------------
# Bottleneck inference engine
# ---------------------------------------------------------------------------

def classify_bottleneck(row) -> tuple[str, str]:
    """
    Rule-based bottleneck classifier.
    Returns (label, explanation).

    Rules are applied in priority order.
    Each rule cites the exact metric (and its source) that triggered it.
    """
    mem_ratio    = row["memory_ratio"]    # SOURCE: PTX static
    branch_ratio = row["branch_ratio"]   # SOURCE: PTX static
    ai           = row["arith_intensity"] # SOURCE: PTX static (derived)
    rt_share     = row["runtime_share"]  # SOURCE: measured runtime
    bw_eff       = row["bw_efficiency"]  # SOURCE: derived from measured runtime
    bw_GBs       = row["approx_bandwidth_GBs"]  # SOURCE: derived from measured runtime
    sh_loads     = row["shared_loads"]   # SOURCE: PTX static
    sh_stores    = row["shared_stores"]  # SOURCE: PTX static
    kernel       = row["kernel"]

    # Rule 0: Reference kernel — labelled explicitly, not inferred
    if kernel == "compute_ref":
        explanation = (
            f"Reference kernel: all threads take the fast fma path, no divergence. "
            f"Arithmetic intensity={ai:.2f} [SOURCE: PTX static]. "
            f"Runtime share={rt_share:.1%} [SOURCE: measured]. "
            f"Compare against divergent_compute to isolate divergence overhead."
        )
        return "Compute Bound (Reference)", explanation

    # Rule 1a: Shared-memory pattern — kernel uses shared memory (reduction/tiling).
    # These kernels have elevated branch ratios from loop conditions (e.g.
    # `if (threadIdx.x < s)`) that are NOT true warp divergence — all active
    # threads in a warp follow the same path at each step of a tree reduction.
    # Detect by: shared_loads + shared_stores > 0 (PTX static).
    if (sh_loads + sh_stores) > 0:
        explanation = (
            f"Kernel uses shared memory: shared_loads={sh_loads}, "
            f"shared_stores={sh_stores} [SOURCE: PTX static]. "
            f"Elevated branch_ratio={branch_ratio:.3f} reflects synchronization "
            f"loop conditions (e.g. tree-reduction halving), not true warp "
            f"serialisation — all threads in a warp take the same branch at each step. "
            f"Effective bandwidth={bw_GBs:.1f} GB/s ({bw_eff:.0%} of observed peak) "
            f"[SOURCE: derived from measured runtime]. "
            f"Runtime share={rt_share:.1%} [SOURCE: measured]."
        )
        return "Shared Memory Bound", explanation

    # Rule 1b: Warp divergence — elevated branch ratio
    if branch_ratio > THRESH_BRANCH_RATIO:
        explanation = (
            f"Branch ratio={branch_ratio:.3f} exceeds threshold {THRESH_BRANCH_RATIO} "
            f"[SOURCE: PTX static]. Frequent branch instructions force the warp "
            f"scheduler to serialise diverging paths within each warp, reducing "
            f"effective SIMT parallelism. "
            f"Runtime share={rt_share:.1%} [SOURCE: measured]."
        )
        return "Warp Divergence", explanation

    # Rule 2: Memory bound — high memory instruction ratio AND low arithmetic intensity
    if mem_ratio > THRESH_MEMORY_RATIO and ai < THRESH_ARITH_INTENSITY:
        explanation = (
            f"Memory ratio={mem_ratio:.3f} > {THRESH_MEMORY_RATIO} [SOURCE: PTX static] "
            f"and arithmetic intensity={ai:.2f} < {THRESH_ARITH_INTENSITY} "
            f"[SOURCE: PTX static, derived]. Kernel issues many global memory "
            f"transactions relative to compute work — likely bandwidth-limited. "
            f"Runtime share={rt_share:.1%} [SOURCE: measured]."
        )
        return "Memory Bound", explanation

    # Rule 3a: Poor coalescing — low arithmetic intensity AND severely degraded bandwidth
    # Effective bandwidth well below the observed peak indicates that cache lines
    # are fetched but most bytes are unused (strided / non-coalesced pattern).
    # SOURCE: derived from measured bandwidth (runtime + array sizes).
    if ai < THRESH_ARITH_INTENSITY and bw_eff < THRESH_BW_EFFICIENCY:
        explanation = (
            f"Arithmetic intensity={ai:.2f} < {THRESH_ARITH_INTENSITY} "
            f"[SOURCE: PTX static, derived] AND effective bandwidth={bw_GBs:.1f} GB/s "
            f"is only {bw_eff:.0%} of the observed peak "
            f"[SOURCE: derived from measured runtime]. "
            f"This large bandwidth shortfall — despite the kernel performing global "
            f"memory ops — points to non-coalesced access: consecutive threads access "
            f"non-consecutive addresses, wasting most bytes of every cache line fetched. "
            f"Runtime share={rt_share:.1%} [SOURCE: measured]."
        )
        return "Memory Bound (Poor Coalescing)", explanation

    # Rule 3b: Bandwidth limited — low arithmetic intensity alone
    if ai < THRESH_ARITH_INTENSITY:
        explanation = (
            f"Arithmetic intensity={ai:.2f} < {THRESH_ARITH_INTENSITY} "
            f"[SOURCE: PTX static, derived]. Compute work is low relative to "
            f"memory transactions; kernel sits below the roofline ridge point. "
            f"Effective bandwidth={bw_GBs:.1f} GB/s ({bw_eff:.0%} of observed peak) "
            f"[SOURCE: derived from measured runtime]. "
            f"Runtime share={rt_share:.1%} [SOURCE: measured]."
        )
        return "Bandwidth Limited", explanation

    # Rule 4: Compute balanced (default for baseline coalesced kernel)
    explanation = (
        f"Balanced instruction mix: memory ratio={mem_ratio:.3f}, "
        f"branch ratio={branch_ratio:.3f}, arithmetic intensity={ai:.2f} "
        f"[SOURCE: PTX static]. No single bottleneck dominates. "
        f"Runtime share={rt_share:.1%} [SOURCE: measured]."
    )
    return "Compute Balanced", explanation


def apply_classifier(df: pd.DataFrame) -> pd.DataFrame:
    tuples = df.apply(classify_bottleneck, axis=1)
    df["bottleneck"]  = [t[0] for t in tuples]
    df["explanation"] = [t[1] for t in tuples]
    return df


# ---------------------------------------------------------------------------
# Visualization: heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame):
    """
    Heatmap: kernels × selected normalised metrics.
    All cells trace to their measurement source in the axis labels.
    """
    if df.empty:
        print("[WARN] No data for heatmap, skipping.")
        return
    heat_cols = {
        "runtime_share":       "Runtime\nShare\n[measured]",
        "memory_ratio":        "Memory\nOp Ratio\n[PTX static]",
        "branch_ratio":        "Branch\nRatio\n[PTX static]",
        "arith_intensity":     "Arithmetic\nIntensity\n[PTX static]",
        "approx_bandwidth_GBs": "Approx BW\n(GB/s)\n[derived]",
        "bw_efficiency":       "BW\nEfficiency\n[derived]",
    }

    heat_df = df.set_index("kernel")[list(heat_cols.keys())].copy()
    # Clamp arith_intensity to avoid inf distorting the colour scale
    heat_df["arith_intensity"] = heat_df["arith_intensity"].clip(upper=10.0)

    # Normalise each column to [0,1] for colour uniformity
    normed = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-12)
    normed.columns = list(heat_cols.values())
    normed.index   = df["kernel"]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(normed, ax=ax, annot=heat_df.values, fmt=".3g",
                cmap="YlOrRd", linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Normalised value (colour only)"})
    ax.set_title("Kernel Metric Heatmap\n"
                 "(annotations = raw values; colour = normalised for visual contrast)",
                 fontsize=11)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "heatmap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Visualization: summary table
# ---------------------------------------------------------------------------

def plot_summary_table(df: pd.DataFrame):
    if df.empty:
        print("[WARN] No data for summary table, skipping.")
        return
    cols = ["kernel", "mean_us", "std_us", "approx_bandwidth_GBs",
            "arith_intensity", "bottleneck"]
    labels = ["Kernel", "Mean (µs)\n[measured]", "Std (µs)\n[measured]",
              "~BW (GB/s)\n[derived]", "Arith Intensity\n[PTX static]",
              "Bottleneck\n[inferred]"]

    tbl = df[cols].copy()
    tbl["mean_us"]               = tbl["mean_us"].map("{:.1f}".format)
    tbl["std_us"]                = tbl["std_us"].map("{:.1f}".format)
    tbl["approx_bandwidth_GBs"]  = tbl["approx_bandwidth_GBs"].map("{:.1f}".format)
    tbl["arith_intensity"]       = tbl["arith_intensity"].map(
        lambda v: f"{v:.2f}" if np.isfinite(v) else "inf")

    fig, ax = plt.subplots(figsize=(13, 2 + len(df) * 0.6))
    ax.axis("off")
    t = ax.table(
        cellText  = tbl.values,
        colLabels = labels,
        loc       = "center",
        cellLoc   = "center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)

    # Colour bottleneck cells
    bottleneck_colours = {
        "Warp Divergence":              "#ffcccc",
        "Shared Memory Bound":          "#d4b8e0",
        "Memory Bound":                 "#ffd9a0",
        "Memory Bound (Poor Coalescing)":"#ffb347",
        "Bandwidth Limited":            "#ffffa0",
        "Compute Balanced":             "#ccffcc",
        "Compute Bound (Reference)":    "#cce5ff",
    }
    for i, val in enumerate(df["bottleneck"]):
        colour = bottleneck_colours.get(val, "white")
        t[i + 1, cols.index("bottleneck")].set_facecolor(colour)

    ax.set_title("Kernel Performance Summary", fontsize=12, pad=10)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "summary_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(df: pd.DataFrame):
    lines = []
    lines.append("# CUDA Kernel Micro-Analysis Report")
    lines.append(f"\n_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_\n")

    lines.append("## Methodology\n")
    lines.append("- **Runtime data** – real GPU execution measured with "
                 "`cudaEventRecord` / `cudaEventElapsedTime` over 100 iterations "
                 "after 5 warm-up runs.  No CPU-side timing was used.\n")
    lines.append("- **PTX data** – static instruction counts extracted from "
                 "compiler-generated PTX (`nvcc -O3 --ptx`).  Counts reflect the "
                 "static instruction mix, not dynamic execution frequency.\n")
    lines.append("- **Derived metrics** – computed algebraically from measured "
                 "quantities above.  Each metric lists its source explicitly.\n")
    lines.append("- **Bottleneck labels** – produced by a rule-based classifier "
                 "whose thresholds and triggering metrics are fully documented "
                 "in `analyze.py`.\n")

    lines.append("## Bottleneck Summary\n")
    lines.append("| Kernel | Runtime (µs) | BW (GB/s) | Bottleneck |\n")
    lines.append("|--------|-------------|-----------|------------|\n")
    for _, row in df.iterrows():
        lines.append(
            f"| `{row['kernel']}` "
            f"| {row['mean_us']:.1f} ± {row['std_us']:.1f} "
            f"| {row['approx_bandwidth_GBs']:.1f} "
            f"| **{row['bottleneck']}** |\n"
        )
    lines.append("\n")

    lines.append("## Kernel Results\n")
    for _, row in df.iterrows():
        lines.append(f"### {row['kernel']}\n")
        lines.append(f"- **Runtime** (measured): {row['mean_us']:.1f} µs "
                     f"± {row['std_us']:.1f} µs\n")
        lines.append(f"- **Runtime share** (derived from measured): "
                     f"{row['runtime_share']:.1%}\n")
        lines.append(f"- **Approx bandwidth** (derived): "
                     f"{row['approx_bandwidth_GBs']:.1f} GB/s\n")
        lines.append(f"- **Memory op ratio** (PTX static): "
                     f"{row['memory_ratio']:.3f}\n")
        lines.append(f"- **Branch ratio** (PTX static): "
                     f"{row['branch_ratio']:.3f}\n")
        ai = row['arith_intensity']
        ai_str = f"{ai:.2f}" if np.isfinite(ai) else "inf"
        lines.append(f"- **Arithmetic intensity** (PTX static, derived): {ai_str}\n")
        lines.append(f"- **Bottleneck** (inferred): **{row['bottleneck']}**\n")
        lines.append(f"\n> {row['explanation']}\n")

        # Optimisation suggestion per bottleneck
        suggestions = {
            "Warp Divergence": (
                "**Suggestion:** Restructure the branch condition to use "
                "block-level rather than thread-level divergence, or replace "
                "the conditional with branchless arithmetic (e.g., `a + sign * b`)."
            ),
            "Shared Memory Bound": (
                "**Suggestion:** Kernel is already using shared memory effectively. "
                "Consider increasing occupancy (tune block size), using warp-level "
                "primitives (`__shfl_down_sync`) to replace shared memory in the "
                "final reduction stages, or vectorised loads (`float4`) to improve "
                "global memory throughput on the initial load pass."
            ),
            "Memory Bound": (
                "**Suggestion:** Improve coalescing by ensuring consecutive "
                "threads access consecutive addresses.  Consider shared-memory "
                "tiling if repeated reads are present."
            ),
            "Memory Bound (Poor Coalescing)": (
                "**Suggestion:** Restructure the access pattern so that consecutive "
                "threads read consecutive memory addresses (coalesced access).  "
                "With a stride-32 pattern, 31 out of every 32 bytes fetched per "
                "cache line are wasted.  Use shared memory as a staging buffer to "
                "coalesce global loads before striding into the buffer, or redesign "
                "the data layout (e.g., SoA instead of AoS)."
            ),
            "Bandwidth Limited": (
                "**Suggestion:** Fuse adjacent kernels to increase arithmetic "
                "intensity, or use half-precision storage to halve bandwidth demand."
            ),
            "Compute Balanced": (
                "**Suggestion:** Kernel is already well-balanced.  Profile with "
                "Nsight Compute for register pressure or occupancy limits before "
                "further optimisation."
            ),
            "Compute Bound (Reference)": (
                "**Note:** This is the no-divergence reference kernel.  "
                "Compare its runtime against `divergent_compute` to quantify "
                "the real cost of warp serialisation on this hardware."
            ),
        }
        lines.append(f"\n{suggestions.get(row['bottleneck'], '')}\n")
        lines.append("\n---\n")

    lines.append("## Limitations\n")
    lines.append("- **No cache hit measurement** – L1/L2 hit rates require "
                 "hardware performance counters (Nsight / CUPTI). "
                 "Bandwidth estimates assume full DRAM traffic.\n")
    lines.append("- **No warp stall reason** – stall categories "
                 "(memory, execution dependency, synchronisation) are not "
                 "available without CUPTI instrumentation.\n")
    lines.append("- **PTX ≠ SASS** – PTX is a virtual ISA; the final machine "
                 "code (SASS) may differ after the ptxas compiler pass.  "
                 "Instruction counts are approximate.\n")
    lines.append("- **Static counts ≠ dynamic counts** – loops and divergent "
                 "branches cause instruction counts to diverge from PTX static "
                 "totals at runtime.\n")
    lines.append("- **Architecture-agnostic** – thresholds in the classifier "
                 "are not tuned to a specific SM architecture.  Results are "
                 "directionally correct but not quantitatively precise.\n")
    lines.append("- **Single-device** – all results are for GPU 0; "
                 "multi-GPU systems are not handled.\n")

    lines.append("\n## Outputs\n")
    lines.append("| File | Contents |\n|------|----------|\n")
    lines.append("| `output/data/runtimes.csv` | Measured kernel runtimes |\n")
    lines.append("| `output/data/ptx_stats.csv` | PTX static instruction counts |\n")
    lines.append("| `output/ptx/*.ptx` | Per-kernel PTX assembly |\n")
    lines.append("| `output/plots/heatmap.png` | Metric heatmap |\n")
    lines.append("| `output/plots/summary_table.png` | Summary table |\n")
    lines.append("| `output/report/report.md` | This report |\n")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Bottleneck summary (printed to stdout, visible in GUI log)
# ---------------------------------------------------------------------------

BOTTLENECK_ICONS = {
    "Bandwidth Limited":             "BW",
    "Memory Bound":                  "MEM",
    "Memory Bound (Poor Coalescing)":"COAL",
    "Shared Memory Bound":           "SMEM",
    "Warp Divergence":               "DIV",
    "Compute Balanced":              "OK",
    "Compute Bound (Reference)":     "REF",
}

def _print_bottleneck_summary(df):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  BOTTLENECK SUMMARY")
    print(sep)
    for _, row in df.iterrows():
        tag  = BOTTLENECK_ICONS.get(row["bottleneck"], "???")
        name = row["kernel"]
        bot  = row["bottleneck"]
        bw   = row["approx_bandwidth_GBs"]
        t    = row["mean_us"]
        print(f"  [{tag:4s}]  {name:26s}  {bot}")
        print(f"          {t:8.1f} us   {bw:6.1f} GB/s")
    print(sep)
    print("  Tags: BW=Bandwidth Limited  MEM=Memory Bound  COAL=Poor Coalescing")
    print("        SMEM=Shared Memory Bound  DIV=Warp Divergence")
    print("        OK=Compute Balanced  REF=Reference kernel")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading data ...")
    rt, ptx = load_data()

    print("Deriving metrics ...")
    df = derive_metrics(rt, ptx)

    print("Classifying bottlenecks ...")
    df = apply_classifier(df)

    print("\n--- Derived metrics ---")
    print(df[["kernel", "mean_us", "runtime_share",
              "approx_bandwidth_GBs", "arith_intensity",
              "memory_ratio", "branch_ratio", "bottleneck"]].to_string(index=False))

    print("\nGenerating plots ...")
    plot_heatmap(df)
    plot_summary_table(df)

    print("\nWriting report ...")
    write_report(df)

    _print_bottleneck_summary(df)

    print("\nDone.  All outputs in ./output/")


if __name__ == "__main__":
    main()
