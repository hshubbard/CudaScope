"""
summary_report.py
=================
Unified, high-signal summary that merges outputs from all four analysis passes:

  - analyze.py        → runtime metrics + bottleneck classification
  - portability_pass  → fragility flags (JSON)
  - determinism_pass  → non-determinism flags (JSON)
  - resource_pressure_pass → resource-pressure flags (JSON)

OUTPUTS
-------
  output/report/summary.md       – concise Markdown (default view)
  output/plots/summary_scores.png – per-kernel score bar chart
  stdout                          – compact terminal table

DESIGN
------
  One glance shows: which kernels matter most, why, and what to do next.
  Verbose per-flag details live in the individual pass reports; this file
  only surfaces the top-3 hotspots per pass per kernel.

  Scores are normalised to 0–100 within each pass across the current run,
  so a score of 100 means "worst kernel in this run for this dimension".
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths (mirrored from analyze.py)
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join("output", "data")
PLOTS_DIR  = os.path.join("output", "plots")
REPORT_DIR = os.path.join("output", "report")

_PASS_JSON = {
    "fragility":     os.path.join(REPORT_DIR, "portability.json"),
    "determinism":   os.path.join(REPORT_DIR, "determinism.json"),
    "resource":      os.path.join(REPORT_DIR, "resource.json"),
}

_PASS_LABELS = {
    "fragility":   "Fragility",
    "determinism": "Non-Determinism",
    "resource":    "Resource Pressure",
}

TOP_N_HOTSPOTS = 3    # hotspots shown per pass in the summary
MAX_RAW_SCORE  = 100  # cap for normalisation denominator


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_runtime() -> Optional[pd.DataFrame]:
    rt_path  = os.path.join(DATA_DIR, "runtimes.csv")
    ptx_path = os.path.join(DATA_DIR, "ptx_stats.csv")
    if not os.path.isfile(rt_path) or not os.path.isfile(ptx_path):
        return None
    rt  = pd.read_csv(rt_path)
    ptx = pd.read_csv(ptx_path)
    rt["kernel"]       = rt["kernel"].str.strip()
    ptx["kernel_name"] = ptx["kernel_name"].str.strip()
    ptx = ptx.rename(columns={"kernel_name": "kernel"})
    return pd.merge(rt, ptx, on="kernel", how="left")


def _load_pass_json(path: str) -> Dict[str, List[Dict]]:
    """Return {kernel_name: [flag_dict, ...]} or {} if file missing."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def _raw_score(flags: List[Dict]) -> int:
    return sum(f.get("score", 0) for f in flags)


def _normalise(raw: int, max_raw: int) -> int:
    """Map raw score to 0–100; max_raw is the highest raw score in this run."""
    if max_raw == 0:
        return 0
    return min(100, round(100 * raw / max(max_raw, 1)))


def _normalise_pass(
    pass_data: Dict[str, List[Dict]]
) -> Dict[str, int]:
    """Return {kernel: normalised_score_0_100} for one pass."""
    raws = {k: _raw_score(v) for k, v in pass_data.items()}
    max_raw = max(raws.values(), default=1) or 1
    return {k: _normalise(r, max_raw) for k, r in raws.items()}


# ---------------------------------------------------------------------------
# Top-N hotspot extraction
# ---------------------------------------------------------------------------

def _top_hotspots(flags: List[Dict], n: int = TOP_N_HOTSPOTS) -> List[Dict]:
    """Return the n highest-score flags, deduplicated by description prefix."""
    sorted_flags = sorted(flags, key=lambda f: -f.get("score", 0))
    seen_descs: set = set()
    result = []
    for f in sorted_flags:
        key = f.get("description", "")[:40]
        if key not in seen_descs:
            seen_descs.add(key)
            result.append(f)
        if len(result) >= n:
            break
    return result


# ---------------------------------------------------------------------------
# Risk level from combined normalised score
# ---------------------------------------------------------------------------

def _risk_level(combined: int) -> str:
    if combined >= 60:
        return "HIGH"
    if combined >= 25:
        return "MEDIUM"
    return "LOW"


def _risk_colour(level: str) -> str:
    return {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#44cc44"}.get(level, "gray")


# ---------------------------------------------------------------------------
# Per-kernel summary dataclass
# ---------------------------------------------------------------------------

@dataclass
class KernelSummary:
    name:           str
    bottleneck:     str
    mean_us:        float
    bw_GBs:         float
    # Raw normalised scores 0–100
    fragility_score:   int
    determinism_score: int
    resource_score:    int
    combined_score:    int       # weighted average
    risk:              str       # HIGH / MEDIUM / LOW
    # Top hotspots per pass
    top_fragility:     List[Dict]
    top_determinism:   List[Dict]
    top_resource:      List[Dict]
    # One-line "next action" note
    action:            str


def _action_from_bottleneck(bottleneck: str, risk: str) -> str:
    if risk == "HIGH":
        prefix = "URGENT: "
    elif risk == "MEDIUM":
        prefix = "Consider: "
    else:
        prefix = ""

    actions = {
        "Warp Divergence":              f"{prefix}Eliminate per-thread branch with branchless arithmetic.",
        "Memory Bound (Poor Coalescing)":f"{prefix}Restructure access pattern for coalesced reads (SoA layout).",
        "Memory Bound":                 f"{prefix}Tile into shared memory to reduce global traffic.",
        "Shared Memory Bound":          f"{prefix}Tune block size or replace smem reduction with warp shuffles.",
        "Bandwidth Limited":            f"{prefix}Fuse kernels or use half-precision to cut bandwidth demand.",
        "Compute Balanced":             f"{prefix}Profile register pressure / occupancy in Nsight Compute.",
        "Compute Bound (Reference)":    f"{prefix}Reference kernel — compare against divergent variant.",
    }
    return actions.get(bottleneck, f"{prefix}Inspect flags below.")


# ---------------------------------------------------------------------------
# Build summary list
# ---------------------------------------------------------------------------

def build_summaries(
    rt_df:    Optional[pd.DataFrame],
    frag:     Dict[str, List[Dict]],
    det:      Dict[str, List[Dict]],
    res:      Dict[str, List[Dict]],
) -> List[KernelSummary]:
    # Normalise each pass independently
    frag_norm = _normalise_pass(frag)
    det_norm  = _normalise_pass(det)
    res_norm  = _normalise_pass(res)

    # Collect all kernel names across all sources
    all_kernels: set = set()
    if rt_df is not None:
        all_kernels.update(rt_df["kernel"].tolist())
    all_kernels.update(frag.keys(), det.keys(), res.keys())
    # Exclude the monolithic combined PTX file entry if present
    all_kernels.discard("kernels")

    summaries = []
    for name in sorted(all_kernels):
        # Runtime row
        bottleneck = "—"
        mean_us    = float("nan")
        bw_GBs     = float("nan")
        if rt_df is not None:
            rows = rt_df[rt_df["kernel"] == name]
            if not rows.empty:
                row        = rows.iloc[0]
                mean_us    = float(row.get("mean_us",  float("nan")))
                bw_GBs     = float(row.get("approx_bandwidth_GBs", float("nan")))
                bottleneck = str(row.get("bottleneck", "—"))
                if pd.isna(mean_us):
                    mean_us = float("nan")

        f_score = frag_norm.get(name, 0)
        d_score = det_norm.get(name,  0)
        r_score = res_norm.get(name,  0)

        # Weighted combined: fragility 40%, determinism 35%, resource 25%
        combined = round(0.40 * f_score + 0.35 * d_score + 0.25 * r_score)
        risk     = _risk_level(combined)
        action   = _action_from_bottleneck(bottleneck, risk)

        summaries.append(KernelSummary(
            name=name,
            bottleneck=bottleneck,
            mean_us=mean_us,
            bw_GBs=bw_GBs,
            fragility_score=f_score,
            determinism_score=d_score,
            resource_score=r_score,
            combined_score=combined,
            risk=risk,
            top_fragility=_top_hotspots(frag.get(name, [])),
            top_determinism=_top_hotspots(det.get(name, [])),
            top_resource=_top_hotspots(res.get(name, [])),
            action=action,
        ))

    # Sort: highest combined score first
    summaries.sort(key=lambda s: -s.combined_score)
    return summaries


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_SEV_TAG = {"high": "HI", "medium": "MD", "low": "LO"}


def print_console_summary(summaries: List[KernelSummary]) -> None:
    W = 74
    sep  = "=" * W
    dash = "-" * W

    print(f"\n{sep}")
    print("  KERNEL ANALYSIS SUMMARY")
    print(sep)
    print(f"  {'Kernel':<26} {'Bottleneck':<26} {'Frag':>4} {'Det':>4} {'Res':>4} {'Risk':>6}")
    print(f"  {dash}")

    for s in summaries:
        bt   = s.bottleneck[:25] if s.bottleneck else "—"
        name = s.name[:25]
        print(
            f"  {name:<26} {bt:<26} "
            f"{s.fragility_score:>3}/100 {s.determinism_score:>3}/100 "
            f"{s.resource_score:>3}/100  {s.risk:>6}"
        )

    print(f"\n{sep}")
    print("  TOP HOTSPOTS PER KERNEL")
    print(sep)

    for s in summaries:
        has_flags = s.top_fragility or s.top_determinism or s.top_resource
        if not has_flags:
            continue

        rt_str = f"{s.mean_us:.1f} us" if not (s.mean_us != s.mean_us) else "n/a"
        print(f"\n  Kernel : {s.name}  [{s.risk}]  {s.bottleneck}  ({rt_str})")
        print(f"  Action : {s.action}")

        for label, flags in [
            ("Fragility",       s.top_fragility),
            ("Non-Determinism", s.top_determinism),
            ("Resource",        s.top_resource),
        ]:
            for f in flags:
                sev  = _SEV_TAG.get(f.get("severity", ""), "??")
                cat  = f.get("category", "")[:28]
                loc  = f.get("location", "")[:22]
                desc = f.get("description", "")[:55]
                print(f"  [{label[:5]:5s}/{sev}] {cat:<30} @ {loc:<22}  {desc}")

    print(f"\n{sep}")
    print("  Frag=Fragility  Det=Non-Determinism  Res=Resource Pressure  (scores 0-100)")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_summary_md(summaries: List[KernelSummary], out_path: str) -> None:
    lines = []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"# CUDA Kernel Analysis Summary\n\n_Generated: {ts}_\n\n")
    lines.append(
        "> Scores are normalised 0–100 within this run (100 = worst kernel for "
        "that dimension).  Full details: "
        "`portability.md`, `determinism.md`, `resource_pressure.md`, `report.md`.\n\n"
    )

    # ---- Master table ----
    lines.append("## Kernel Scorecard\n\n")
    lines.append(
        "| Kernel | Bottleneck | Fragility | Non-Det | Resource | Risk | Action |\n"
    )
    lines.append(
        "|--------|-----------|----------:|--------:|---------:|------|--------|\n"
    )
    for s in summaries:
        rt_str = f"{s.mean_us:.1f} µs" if s.mean_us == s.mean_us else "—"
        lines.append(
            f"| `{s.name}` "
            f"| {s.bottleneck} "
            f"| {s.fragility_score}/100 "
            f"| {s.determinism_score}/100 "
            f"| {s.resource_score}/100 "
            f"| **{s.risk}** "
            f"| {s.action} |\n"
        )
    lines.append("\n")

    # ---- Per-kernel hotspot blocks ----
    lines.append("## Hotspots by Kernel\n\n")
    lines.append(
        "_Only the top-3 most severe flags per pass are shown. "
        "See individual pass reports for the full list._\n\n"
    )

    for s in summaries:
        has_flags = s.top_fragility or s.top_determinism or s.top_resource
        lines.append(f"### `{s.name}`")
        lines.append(f"  —  combined score **{s.combined_score}/100** [{s.risk}]\n\n")

        rt_str = f"{s.mean_us:.1f} µs" if s.mean_us == s.mean_us else "n/a"
        bw_str = f"{s.bw_GBs:.1f} GB/s" if s.bw_GBs == s.bw_GBs else "n/a"
        lines.append(
            f"- **Bottleneck:** {s.bottleneck}  |  "
            f"**Runtime:** {rt_str}  |  **BW:** {bw_str}\n"
        )
        lines.append(f"- **Next action:** {s.action}\n\n")

        if not has_flags:
            lines.append("_No issues flagged._\n\n")
            continue

        for label, flags in [
            ("Fragility",       s.top_fragility),
            ("Non-Determinism", s.top_determinism),
            ("Resource",        s.top_resource),
        ]:
            if not flags:
                continue
            lines.append(f"**{label} — top flags:**\n\n")
            for f in flags:
                sev  = f.get("severity", "").upper()
                cat  = f.get("category", "")
                loc  = f.get("location", "")
                desc = f.get("description", "")
                lines.append(f"- `[{sev}]` `{cat}` @ `{loc}` — {desc}\n")
            lines.append("\n")

        lines.append("\n")

    # ---- Legend ----
    lines.append("## Score Legend\n\n")
    lines.append(
        "| Score | Meaning |\n"
        "|------:|---------|\n"
        "| 0     | No issues detected in this pass |\n"
        "| 1–24  | Low risk — monitor |\n"
        "| 25–59 | Medium risk — review before next architecture |\n"
        "| 60–100 | High risk — fix before porting |\n\n"
    )
    lines.append(
        "Combined score = 40% Fragility + 35% Non-Determinism + 25% Resource Pressure.\n\n"
    )
    lines.append(
        "**Detailed reports:** "
        "`output/report/portability.md` · "
        "`output/report/determinism.md` · "
        "`output/report/resource_pressure.md` · "
        "`output/report/report.md`\n"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Score bar chart
# ---------------------------------------------------------------------------

def plot_score_chart(summaries: List[KernelSummary], out_path: str) -> None:
    if not summaries:
        return

    names  = [s.name for s in summaries]
    frag   = [s.fragility_score   for s in summaries]
    det    = [s.determinism_score for s in summaries]
    res    = [s.resource_score    for s in summaries]
    risks  = [s.risk              for s in summaries]

    n   = len(names)
    y   = np.arange(n)
    h   = 0.25

    fig, ax = plt.subplots(figsize=(10, max(3, n * 0.55 + 1.5)))

    bars_f = ax.barh(y + h,   frag, h, label="Fragility",        color="#e07b54", alpha=0.85)
    bars_d = ax.barh(y,       det,  h, label="Non-Determinism",  color="#5b8dd9", alpha=0.85)
    bars_r = ax.barh(y - h,   res,  h, label="Resource Pressure",color="#72b96e", alpha=0.85)

    # Risk indicator dots on the right margin
    for i, (risk, row) in enumerate(zip(risks, summaries)):
        colour = _risk_colour(risk)
        ax.text(103, y[i], risk[0],  # H / M / L
                va="center", ha="left", fontsize=7.5, color=colour, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Score (0–100, normalised within this run)")
    ax.set_title("Kernel Analysis Scorecard\n"
                 "(H=High / M=Medium / L=Low risk marker)", fontsize=11)
    ax.axvline(60, color="red",    lw=0.8, ls="--", alpha=0.5, label="High threshold (60)")
    ax.axvline(25, color="orange", lw=0.8, ls="--", alpha=0.5, label="Medium threshold (25)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    rt_df:         Optional[pd.DataFrame] = None,
    frag_json_path: str = _PASS_JSON["fragility"],
    det_json_path:  str = _PASS_JSON["determinism"],
    res_json_path:  str = _PASS_JSON["resource"],
    md_out:         str = os.path.join(REPORT_DIR, "summary.md"),
    chart_out:      str = os.path.join(PLOTS_DIR,  "summary_scores.png"),
) -> List[KernelSummary]:
    if rt_df is None:
        rt_df = _load_runtime()

    frag = _load_pass_json(frag_json_path)
    det  = _load_pass_json(det_json_path)
    res  = _load_pass_json(res_json_path)

    summaries = build_summaries(rt_df, frag, det, res)

    print_console_summary(summaries)
    write_summary_md(summaries, md_out)
    plot_score_chart(summaries, chart_out)

    return summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified CUDA kernel analysis summary report."
    )
    parser.add_argument("--frag-json", default=_PASS_JSON["fragility"],
                        help="Path to portability pass JSON output.")
    parser.add_argument("--det-json",  default=_PASS_JSON["determinism"],
                        help="Path to determinism pass JSON output.")
    parser.add_argument("--res-json",  default=_PASS_JSON["resource"],
                        help="Path to resource pressure pass JSON output.")
    parser.add_argument("--md-out",    default=os.path.join(REPORT_DIR, "summary.md"),
                        help="Output Markdown path.")
    parser.add_argument("--chart-out", default=os.path.join(PLOTS_DIR, "summary_scores.png"),
                        help="Output score bar chart path.")
    args = parser.parse_args()

    main(
        frag_json_path=args.frag_json,
        det_json_path=args.det_json,
        res_json_path=args.res_json,
        md_out=args.md_out,
        chart_out=args.chart_out,
    )
