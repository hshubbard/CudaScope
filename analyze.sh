#!/usr/bin/env bash
# =============================================================================
# analyze.sh  –  CUDA Kernel Micro-Analysis Tool  –  one-shot CLI
#
# Usage:
#   ./analyze.sh [--skip-build] [--skip-bench] [--skip-ptx] [--skip-analyze]
#
# Output folder layout:
#   output/
#     data/    – CSV files  (raw measured numbers)
#     ptx/     – PTX files  (compiler static analysis inputs)
#     plots/   – PNG images (heatmap, summary table)
#     report/  – report.md  (full written analysis)
# =============================================================================

set -euo pipefail

DO_BUILD=1
DO_BENCH=1
DO_PTX=1
DO_ANALYZE=1

for arg in "$@"; do
    case $arg in
        --skip-build)   DO_BUILD=0   ;;
        --skip-bench)   DO_BENCH=0   ;;
        --skip-ptx)     DO_PTX=0     ;;
        --skip-analyze) DO_ANALYZE=0 ;;
        -h|--help)
            echo "Usage: $0 [--skip-build] [--skip-bench] [--skip-ptx] [--skip-analyze]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
banner() {
    echo ""
    echo "========================================"
    echo "  $*"
    echo "========================================"
}

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' not found in PATH.  $2"
        exit 1
    fi
}

find_python() {
    if   command -v python3 &>/dev/null; then echo python3
    elif command -v python  &>/dev/null; then echo python
    else echo ""; fi
}

detect_cc() {
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d '.' || echo "86"
}

# ---------------------------------------------------------------------------
# Step 1: Build
# ---------------------------------------------------------------------------
if [[ $DO_BUILD -eq 1 ]]; then
    banner "Step 1/4  –  Building benchmark binary"
    check_cmd nvcc "Install CUDA toolkit."

    CC=$(detect_cc)
    CC=${CC:-86}
    echo "Detected SM: $CC"

    nvcc -O3 -std=c++17 \
        --allow-unsupported-compiler \
        -gencode arch=compute_${CC},code=sm_${CC} \
        kernels.cu benchmark.cu \
        -o benchmark
    echo "Build succeeded → ./benchmark"
fi

# ---------------------------------------------------------------------------
# Step 2: Run benchmark  →  output/data/runtimes.csv
# ---------------------------------------------------------------------------
if [[ $DO_BENCH -eq 1 ]]; then
    banner "Step 2/4  –  Running benchmark (real GPU execution)"
    [[ -x ./benchmark ]] || { echo "ERROR: ./benchmark not found. Run step 1 first."; exit 1; }
    mkdir -p output/data
    ./benchmark
    echo "Runtime data → output/data/runtimes.csv"
fi

# ---------------------------------------------------------------------------
# Step 3: PTX extraction  →  output/ptx/  +  output/data/ptx_stats.csv
# ---------------------------------------------------------------------------
if [[ $DO_PTX -eq 1 ]]; then
    banner "Step 3/4  –  Extracting PTX + static instruction analysis"
    check_cmd nvcc "Install CUDA toolkit."

    PY=$(find_python)
    [[ -n "$PY" ]] || { echo "ERROR: Python not found in PATH."; exit 1; }

    mkdir -p output/ptx output/data

    CC=$(detect_cc)
    CC=${CC:-86}

    # Full PTX into output/ptx/
    nvcc -O3 -std=c++17 \
        --allow-unsupported-compiler \
        -gencode arch=compute_${CC},code=compute_${CC} \
        --ptx kernels.cu \
        -o output/ptx/kernels.ptx
    echo "  PTX → output/ptx/kernels.ptx"

    echo ""
    echo "  Splitting per-kernel PTX ..."
    $PY - <<'PYEOF'
import re, pathlib

src = pathlib.Path("output/ptx/kernels.ptx").read_text(errors="replace")
pattern = re.compile(
    r'((?:\.visible\s+)?\.entry\s+(\w+)\b.*?^\})',
    re.DOTALL | re.MULTILINE
)

clean_map = {
    "coalesced_add":     "coalesced_add",
    "strided_add":       "strided_add",
    "divergent_compute": "divergent_compute",
    "divergent_add":     "divergent_add",
    "compute_ref":       "compute_ref",
}

pairs = []
for m in pattern.finditer(src):
    mangled  = m.group(2)
    body     = m.group(1)
    clean    = next((v for k, v in clean_map.items() if k in mangled), mangled)
    out_path = f"output/ptx/{clean}.ptx"
    pathlib.Path(out_path).write_text(body)
    print(f"    {clean:25s}  <-  {mangled}")
    pairs.append(f"{out_path}::{clean}")

pathlib.Path("output/ptx/.ptx_pairs").write_text("\n".join(pairs))
PYEOF

    mapfile -t PTX_ARGS < output/ptx/.ptx_pairs
    echo ""
    echo "  Running ptx_parser.py ..."
    $PY ptx_parser.py "${PTX_ARGS[@]}"
    echo "PTX stats → output/data/ptx_stats.csv"
fi

# ---------------------------------------------------------------------------
# Step 4: Analysis  →  output/plots/  +  output/report/
# ---------------------------------------------------------------------------
if [[ $DO_ANALYZE -eq 1 ]]; then
    banner "Step 4/4  –  Metric derivation, bottleneck inference, visualisation"

    PY=$(find_python)
    [[ -n "$PY" ]] || { echo "ERROR: Python not found in PATH."; exit 1; }

    $PY -c "import pandas, numpy, matplotlib, seaborn" 2>/dev/null || {
        echo "Installing required Python packages ..."
        pip install --quiet pandas numpy matplotlib seaborn
    }

    $PY analyze.py
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
banner "Analysis complete"
echo ""
echo "output/"
echo "  data/    <- CSV files (raw numbers)"
echo "  ptx/     <- compiler PTX (static analysis)"
echo "  plots/   <- heatmap + summary table"
echo "  report/  <- report.md"
echo ""
ls -lh output/data/   2>/dev/null || true
ls -lh output/ptx/    2>/dev/null || true
ls -lh output/plots/  2>/dev/null || true
ls -lh output/report/ 2>/dev/null || true
