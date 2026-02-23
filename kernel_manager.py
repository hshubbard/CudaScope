"""
kernel_manager.py
=================
Shared logic for the GUI and CLI tools.

Responsibilities
----------------
- Registry: load/save user_kernels.json
- Codegen: regenerate user_kernels.cuh, user_kernels.cu, user_benchmark.cu
- Pipeline: compile + benchmark + PTX + analyze, streaming log output
"""

import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).parent
REGISTRY_FILE  = BASE_DIR / "user_kernels.json"
USER_CUH       = BASE_DIR / "user_kernels.cuh"
USER_CU        = BASE_DIR / "user_kernels.cu"
USER_BENCH_CU  = BASE_DIR / "user_benchmark.cu"
CONFIG_FILE    = BASE_DIR / "analyzer_config.json"

# ---------------------------------------------------------------------------
# Built-in kernel definitions (mirrored from benchmark.cu / kernels.cu).
# These are always included in the generated benchmark.
# ---------------------------------------------------------------------------
BUILTINS = [
    {
        "name":       "coalesced_add",
        "n_elements": 16 * 1024 * 1024,
        "block_size": 256,
        "iters":      0,
        # params used for the call inside launch_once
        "call_args":  "d_a, d_b, d_c, n",
        "grid_expr":  "grid",
        "decl":       "coalesced_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)",
    },
    {
        "name":       "strided_add",
        "n_elements": 16 * 1024 * 1024 // 32,   # N/STRIDE
        "block_size": 256,
        "iters":      0,
        "call_args":  "d_a, d_b, d_c, n, stride",
        "grid_expr":  "grid2",
        "decl":       "strided_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n, int stride)",
    },
    {
        "name":       "divergent_add",
        "n_elements": 16 * 1024 * 1024,
        "block_size": 256,
        "iters":      0,
        "call_args":  "d_a, d_b, d_c, n",
        "grid_expr":  "grid",
        "decl":       "divergent_add(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)",
    },
    {
        "name":       "divergent_compute",
        "n_elements": 16 * 1024 * 1024,
        "block_size": 256,
        "iters":      2000,
        "call_args":  "d_a, d_c, n, iters",
        "grid_expr":  "grid",
        "decl":       "divergent_compute(const float* __restrict__ a, float* __restrict__ c, int n, int iters)",
    },
    {
        "name":       "compute_ref",
        "n_elements": 16 * 1024 * 1024,
        "block_size": 256,
        "iters":      2000,
        "call_args":  "d_a, d_c, n, iters",
        "grid_expr":  "grid",
        "decl":       "compute_ref(const float* __restrict__ a, float* __restrict__ c, int n, int iters)",
    },
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry() -> List[dict]:
    if not REGISTRY_FILE.exists():
        return []
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(kernels: List[dict]):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(kernels, f, indent=2)


def add_kernel(name: str, code: str, params: str,
               block_size: int = 256, n_elements: int = 16 * 1024 * 1024,
               iters: int = 0,
               grid_x: int = None, grid_y: int = None,
               block_x: int = None, block_y: int = None) -> str:
    """
    Add a user kernel to the registry and regenerate generated files.
    Returns an error string on failure, empty string on success.
    Set grid_x+grid_y for 2D grid launches; leave None for 1D auto mode.
    """
    kernels = load_registry()
    if any(k["name"] == name for k in kernels):
        return f"Kernel '{name}' already exists. Remove it first."
    entry = {
        "name":       name,
        "code":       code,
        "params":     params,
        "block_size": block_size,
        "n_elements": n_elements,
        "iters":      iters,
        "active":     True,
        "added":      datetime.now().isoformat(timespec="seconds"),
    }
    if grid_x is not None:
        entry["grid_x"] = int(grid_x)
    if grid_y is not None:
        entry["grid_y"] = int(grid_y)
    if block_x is not None:
        entry["block_x"] = int(block_x)
    if block_y is not None:
        entry["block_y"] = int(block_y)
    kernels.append(entry)
    save_registry(kernels)
    regenerate_user_files()
    return ""


def remove_kernel(name: str) -> str:
    """Remove a user kernel. Returns error string or empty string."""
    kernels = load_registry()
    before = len(kernels)
    kernels = [k for k in kernels if k["name"] != name]
    if len(kernels) == before:
        return f"Kernel '{name}' not found."
    save_registry(kernels)
    regenerate_user_files()
    return ""


def set_kernel_active(name: str, active: bool) -> str:
    """Set the active flag on a kernel. Returns error string or empty string."""
    kernels = load_registry()
    for k in kernels:
        if k["name"] == name:
            k["active"] = active
            save_registry(kernels)
            regenerate_user_files()
            return ""
    return f"Kernel '{name}' not found."


# ---------------------------------------------------------------------------
# Param extraction
# ---------------------------------------------------------------------------

def extract_params(code: str, name: str) -> str:
    """
    Try to extract the parameter list from a __global__ function signature
    in the kernel source code. Returns empty string if not found.
    """
    pattern = re.compile(
        r'__global__\s+\w+\s+' + re.escape(name) + r'\s*\(([^)]*)\)',
        re.DOTALL
    )
    m = pattern.search(code)
    if m:
        # Collapse whitespace
        return re.sub(r'\s+', ' ', m.group(1).strip())
    return ""


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _make_enum_id(name: str) -> str:
    return "K_" + name.upper()


def regenerate_user_files(include_builtins: bool = True):
    """
    Regenerate user_kernels.cuh, user_kernels.cu, and user_benchmark.cu
    from the current registry + built-ins.
    All registered kernels go into .cuh/.cu; only active ones go into the benchmark.
    """
    kernels = load_registry()
    active  = [k for k in kernels if k.get("active", True)]
    _write_user_cuh(kernels)   # declare all (so .cu compiles cleanly)
    _write_user_cu(kernels)    # define all
    _write_user_benchmark(active, include_builtins=include_builtins)


def _write_user_cuh(kernels: List[dict]):
    lines = [
        "// AUTO-GENERATED by kernel_manager.py — do not edit by hand.\n",
        "#pragma once\n",
        '#include "kernels.cuh"\n',
        "\n",
    ]
    for k in kernels:
        lines.append(f"__global__ void {k['name']}({k['params']});\n")
    USER_CUH.write_text("".join(lines), encoding="utf-8")


def _write_user_cu(kernels: List[dict]):
    lines = [
        "// AUTO-GENERATED by kernel_manager.py — do not edit by hand.\n",
        '#include "user_kernels.cuh"\n',
        "\n",
    ]
    for k in kernels:
        lines.append(f"// ---- {k['name']} (added {k['added']}) ----\n")
        lines.append(k["code"].strip())
        lines.append("\n\n")
    USER_CU.write_text("".join(lines), encoding="utf-8")


def _user_to_bench_entry(k: dict) -> dict:
    iters  = k.get("iters", 0)
    block  = k.get("block_size", 256)
    n      = k.get("n_elements", 16 * 1024 * 1024)
    params = k.get("params", "")
    call_args = _params_to_call_args(params)

    grid_x  = k.get("grid_x")
    grid_y  = k.get("grid_y")
    block_x = k.get("block_x")
    block_y = k.get("block_y")
    is_2d   = (grid_x is not None and grid_y is not None)

    return {
        "name":       k["name"],
        "n_elements": n,
        "block_size": block,
        "iters":      iters,
        "call_args":  call_args,
        "grid_expr":  None,   # filled in _write_user_benchmark
        "decl":       f"{k['name']}({params})",
        "is_2d":      is_2d,
        "grid_x":     grid_x,
        "grid_y":     grid_y,
        "block_x":    block_x if block_x is not None else 32,
        "block_y":    block_y if block_y is not None else 32,
    }


def _params_to_call_args(params: str) -> str:
    """
    Map a user kernel's parameter declaration string to the argument names
    available inside launch_once(): d_a, d_b, d_c, n, iters.

    Rules:
      - const float* (read-only)  → d_a, then d_b  (input arrays)
      - non-const float* (output) → d_c             (output array)
      - int param named iter*     → iters
      - any other int/scalar      → n
    """
    if not params.strip():
        return ""

    in_idx  = 0   # index into [d_a, d_b] for const pointers
    in_ptrs = ["d_a", "d_b"]
    args    = []

    for p in params.split(","):
        p = p.strip()
        if not p:
            continue
        is_ptr   = ("*" in p)
        is_const = p.lstrip().startswith("const")
        if is_ptr:
            if is_const:
                args.append(in_ptrs[in_idx] if in_idx < len(in_ptrs) else "d_a")
                in_idx += 1
            else:
                args.append("d_c")
        else:
            tokens   = p.split()
            raw_name = tokens[-1].lstrip("*&").lower()
            if raw_name.startswith("iter"):
                args.append("iters")
            else:
                args.append("n")

    return ", ".join(args)


def _write_user_benchmark(kernels: List[dict], include_builtins: bool = True):
    """Generate a complete, standalone benchmark .cu for all kernels."""

    builtin_entries = list(BUILTINS) if include_builtins else []
    user_entries    = [_user_to_bench_entry(k) for k in kernels]
    all_kernels     = builtin_entries + user_entries
    total           = len(all_kernels)

    # Assign grid expressions for user kernels.
    # launch_once receives `n` as a parameter, so use that — not n{i}_elem
    # which only exists in main().
    for i, entry in enumerate(all_kernels):
        if entry["grid_expr"] is None:
            if entry.get("is_2d"):
                entry["grid_expr"] = "0"   # ignored; dim3 hardcoded in case
            else:
                block = entry["block_size"]
                entry["grid_expr"] = f"((n + {block} - 1) / {block})"

    # ---------- enum ----------
    enum_entries = "\n".join(
        f"    {_make_enum_id(k['name'])} = {i},"
        for i, k in enumerate(all_kernels)
    )

    # ---------- launch_once cases ----------
    cases = []
    for k in all_kernels:
        kid  = _make_enum_id(k["name"])
        args = k["call_args"]
        grid = k["grid_expr"]
        if k.get("is_2d"):
            gx = k["grid_x"]
            gy = k["grid_y"]
            bx = k["block_x"]
            by = k["block_y"]
            cases.append(
                f"    case {kid}:\n"
                f"        {{ dim3 _g({gx}, {gy}); dim3 _b({bx}, {by}); "
                f"{k['name']}<<<_g, _b>>>({args}); }}\n"
                f"        break;"
            )
        else:
            cases.append(
                f"    case {kid}:\n"
                f"        {k['name']}<<<{grid}, block>>>({args});\n"
                f"        break;"
            )
    cases_str = "\n".join(cases)

    # ---------- n_elements per kernel (declared in main) ----------
    n_elem_decls = "\n    ".join(
        f"int n{i}_elem = {k['n_elements']};"
        for i, k in enumerate(all_kernels)
    )

    # ---------- time_kernel calls ----------
    # For built-ins, pass the pre-computed grid variable (grid or grid2).
    # For user kernels, pass 0 — launch_once computes the grid from n directly.
    n_builtins = len(builtin_entries)
    time_calls = []
    for i, k in enumerate(all_kernels):
        kid        = _make_enum_id(k["name"])
        iters      = k.get("iters", 0)
        # grid arg only matters for built-ins; user kernels ignore it
        grid_arg   = k["grid_expr"] if i < n_builtins else "0"
        time_calls.append(
            f'    printf("Benchmarking {k["name"]} ...\\n");\n'
            f"    float* t{i} = time_kernel({kid}, d_a, d_b, d_c, n{i}_elem, STRIDE, {iters},\n"
            f"                              {grid_arg}, grid2, BLOCK_SIZE, WARMUP_RUNS, BENCH_RUNS);"
        )
    time_calls_str = "\n\n".join(time_calls)

    # ---------- stat vars ----------
    stat_decl    = ", ".join(f"m{i}, s{i}" for i in range(total))
    stat_compute = "\n    ".join(
        f"compute_stats(t{i}, BENCH_RUNS, &m{i}, &s{i});"
        for i in range(total)
    )
    ms_to_us = " ".join(f"m{i}*=1000; s{i}*=1000;" for i in range(total))

    # ---------- printf results ----------
    result_prints = "\n    ".join(
        f'printf("%-26s: mean=%8.2f us  std=%6.2f us\\n", "{k["name"]}", m{i}, s{i});'
        for i, k in enumerate(all_kernels)
    )

    # ---------- CSV rows ----------
    csv_rows = "\n    ".join(
        f'fprintf(f, "{k["name"]},%d,%zu,%d,%d,%.4f,%.4f\\n",'
        f'\n            n{i}_elem, bytes, WARMUP_RUNS, BENCH_RUNS, m{i}, s{i});'
        for i, k in enumerate(all_kernels)
    )

    # ---------- cleanup ----------
    cleanup = " ".join(f"delete[] t{i};" for i in range(total))

    src = f"""\
/*
 * user_benchmark.cu — AUTO-GENERATED by kernel_manager.py
 * Do not edit by hand.
 */
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "user_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#ifdef _WIN32
#  include <direct.h>
#else
#  include <sys/stat.h>
#endif

static const int N            = 16 * 1024 * 1024;
static const int STRIDE       = 32;
static const int BLOCK_SIZE   = 256;
static const int WARMUP_RUNS  = 5;
static const int BENCH_RUNS   = 100;

#define CUDA_CHECK(call)                                                    \\
    do {{                                                                    \\
        cudaError_t _e = (call);                                            \\
        if (_e != cudaSuccess) {{                                            \\
            fprintf(stderr, "CUDA error %s:%d  %s\\n",                      \\
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \\
            exit(EXIT_FAILURE);                                             \\
        }}                                                                   \\
    }} while (0)

enum KernelID {{
{enum_entries}
}};

static void launch_once(KernelID kid,
                        const float* d_a, const float* d_b, float* d_c,
                        int n, int stride, int iters,
                        int grid, int grid2, int block)
{{
    switch (kid) {{
{cases_str}
    default: break;
    }}
}}

static float* time_kernel(KernelID kid,
                           const float* d_a, const float* d_b, float* d_c,
                           int n, int stride, int iters,
                           int grid, int grid2, int block,
                           int warmup, int runs)
{{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i)
        launch_once(kid, d_a, d_b, d_c, n, stride, iters, grid, grid2, block);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* times = new float[runs];
    for (int i = 0; i < runs; ++i) {{
        CUDA_CHECK(cudaEventRecord(start));
        launch_once(kid, d_a, d_b, d_c, n, stride, iters, grid, grid2, block);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }}
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return times;
}}

static void compute_stats(const float* arr, int n,
                           double* mean_out, double* std_out)
{{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += arr[i];
    double mean = sum / n;
    double var  = 0.0;
    for (int i = 0; i < n; ++i) {{ double d = arr[i] - mean; var += d * d; }}
    *mean_out = mean;
    *std_out  = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
}}

static void ensure_output_dirs()
{{
#ifdef _WIN32
    _mkdir("output"); _mkdir("output/data");
#else
    mkdir("output", 0755); mkdir("output/data", 0755);
#endif
}}

int main()
{{
    size_t bytes = (size_t)N * sizeof(float);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    if (!h_a || !h_b) {{ fprintf(stderr, "malloc failed\\n"); return 1; }}
    for (int i = 0; i < N; ++i) {{
        h_a[i] = 1.0f + (float)i * 1e-7f;
        h_b[i] = 1.0f;
    }}

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int n_strided = N / STRIDE;
    int grid      = (N         + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid2     = (n_strided + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device : %s  (SM %d.%d)\\n", prop.name, prop.major, prop.minor);

    // Per-kernel element counts
    {n_elem_decls}

    // ---- Benchmark all kernels ----
{time_calls_str}

    // ---- Stats ----
    double {stat_decl};
    {stat_compute}
    {ms_to_us}

    printf("\\n--- Results ---\\n");
    {result_prints}

    // ---- CSV ----
    ensure_output_dirs();
    FILE* f = fopen("output/data/runtimes.csv", "w");
    if (!f) {{ fprintf(stderr, "Cannot open output/data/runtimes.csv\\n"); return 1; }}
    fprintf(f, "kernel,n_elements,bytes_per_array,warmup_runs,bench_runs,mean_us,std_us\\n");
    {csv_rows}
    fclose(f);
    printf("\\nSaved: output/data/runtimes.csv\\n");

    {cleanup}
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b);
    return 0;
}}
"""
    USER_BENCH_CU.write_text(src, encoding="utf-8")


# ---------------------------------------------------------------------------
# SM detection
# ---------------------------------------------------------------------------

def detect_sm() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        ).strip().splitlines()[0].replace(".", "")
        return out or "86"
    except Exception:
        return "86"


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def find_python() -> str:
    for candidate in ("python3", "python"):
        try:
            subprocess.check_output([candidate, "--version"], stderr=subprocess.STDOUT)
            return candidate
        except Exception:
            pass
    return sys.executable


def run_pipeline(
    skip_build:      bool = False,
    skip_bench:      bool = False,
    skip_ptx:        bool = False,
    skip_analyze:    bool = False,
    include_builtins: bool = True,
    log_cb: Optional[Callable[[str], None]] = None,
):
    """
    Run the full analysis pipeline.
    log_cb(line) is called for each line of output (from any subprocess).
    Blocks until done; intended to be called from a background thread.
    """
    def log(msg: str):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    sm = detect_sm()
    py = find_python()

    # Ensure generated files exist (respecting include_builtins toggle)
    regenerate_user_files(include_builtins=include_builtins)

    # Check there's actually something to run
    active_user = [k for k in load_registry() if k.get("active", True)]
    if not include_builtins and not active_user:
        log("[ERROR] No active kernels to run. Activate at least one kernel or enable built-ins.")
        return 1

    steps = []

    if not skip_build:
        build_sources = []
        if include_builtins:
            build_sources.append("kernels.cu")
        build_sources += ["user_kernels.cu", "user_benchmark.cu"]
        steps.append(("Build", [
            "nvcc", "-O3", "-std=c++17",
            "--allow-unsupported-compiler",
            f"-gencode", f"arch=compute_{sm},code=sm_{sm}",
            *build_sources,
            "-o", "benchmark_user",
        ]))

    if not skip_bench:
        steps.append(("Benchmark", ["./benchmark_user"
                                     if os.name != "nt" else "benchmark_user.exe"]))

    for label, cmd in steps:
        log(f"\n{'='*40}\n  {label}\n{'='*40}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(BASE_DIR),
            )
            for line in proc.stdout:
                log(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                log(f"[ERROR] '{label}' exited with code {proc.returncode}")
                return proc.returncode
        except FileNotFoundError as e:
            log(f"[ERROR] Command not found: {e}")
            return 1

    # PTX extract: nvcc --ptx only accepts one input file at a time.
    # Compile each source separately then concatenate into kernels.ptx.
    if not skip_ptx:
        log(f"\n{'='*40}\n  PTX extract\n{'='*40}")
        ptx_dir = BASE_DIR / "output" / "ptx"
        ptx_dir.mkdir(parents=True, exist_ok=True)

        ptx_sources = []
        if include_builtins:
            ptx_sources.append("kernels.cu")
        # Only include user_kernels.cu if it has actual kernel definitions
        registry = load_registry()
        if registry:
            ptx_sources.append("user_kernels.cu")

        combined_ptx = ""
        for src in ptx_sources:
            out_tmp = ptx_dir / f"{Path(src).stem}.ptx"
            cmd = [
                "nvcc", "-O3", "-std=c++17",
                "--allow-unsupported-compiler",
                f"-gencode", f"arch=compute_{sm},code=compute_{sm}",
                "--ptx", src,
                "-o", str(out_tmp),
            ]
            log(f"  nvcc --ptx {src} ...")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(BASE_DIR),
                )
                for line in proc.stdout:
                    log(line.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    log(f"[ERROR] PTX extract of '{src}' failed (code {proc.returncode})")
                    return proc.returncode
                combined_ptx += out_tmp.read_text(errors="replace") + "\n"
            except FileNotFoundError as e:
                log(f"[ERROR] Command not found: {e}")
                return 1

        combined_path = ptx_dir / "kernels.ptx"
        combined_path.write_text(combined_ptx, encoding="utf-8")
        log(f"  Combined PTX -> {combined_path}")

        log("\n  Splitting PTX and running ptx_parser.py ...")
        _run_ptx_split_and_parse(py, log)

    if not skip_analyze:
        log(f"\n{'='*40}\n  Analyze\n{'='*40}")
        try:
            proc = subprocess.Popen(
                [py, "analyze.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(BASE_DIR),
            )
            for line in proc.stdout:
                log(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                log(f"[ERROR] 'Analyze' exited with code {proc.returncode}")
                return proc.returncode
        except FileNotFoundError as e:
            log(f"[ERROR] Command not found: {e}")
            return 1

        # ---- Static analysis passes (run after PTX is available) ----
        # Guard: if built-ins are disabled and no user kernels are active,
        # there is nothing to analyse — skip all three passes.
        _active_user = [k for k in load_registry() if k.get("active", True)]
        _has_kernels = include_builtins or bool(_active_user)

        if not _has_kernels:
            log("\n[INFO] Skipping static analysis passes — no kernels selected.")
        else:
            # Use the .ptx_pairs file written by _run_ptx_split_and_parse as
            # the authoritative list of PTX files for THIS run.  This avoids
            # analysing stale PTX from a previous run that had a different
            # include_builtins setting.
            _ptx_dir    = BASE_DIR / "output" / "ptx"
            _pairs_file = _ptx_dir / ".ptx_pairs"
            if _pairs_file.exists():
                # Pass individual "path::name" pairs so each script analyses
                # exactly the kernels compiled in this run.
                _ptx_file_args = _pairs_file.read_text(encoding="utf-8").splitlines()
                _ptx_file_args = [p for p in _ptx_file_args if p.strip()]
            else:
                # Fallback: scan the directory (handles skip_ptx=True re-runs)
                _ptx_file_args = ["--ptx-dir", str(_ptx_dir)]

            _report_dir = BASE_DIR / "output" / "report"
            _pass_specs = [
                # (script, label, json_output_path)
                ("portability_pass.py",
                 "Portability Pass",
                 str(_report_dir / "portability.json")),
                ("determinism_pass.py",
                 "Determinism Pass",
                 str(_report_dir / "determinism.json")),
                ("resource_pressure_pass.py",
                 "Resource Pressure Pass",
                 str(_report_dir / "resource.json")),
            ]

            for pass_script, pass_label, json_out in _pass_specs:
                log(f"\n{'='*40}\n  {pass_label}\n{'='*40}")
                try:
                    proc = subprocess.Popen(
                        [py, pass_script, "--json", json_out] + _ptx_file_args,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, cwd=str(BASE_DIR),
                    )
                    for line in proc.stdout:
                        log(line.rstrip())
                    proc.wait()
                    if proc.returncode != 0:
                        log(f"[WARN] {pass_label} exited with code {proc.returncode}")
                except FileNotFoundError as e:
                    log(f"[WARN] {pass_script} not found: {e}")

            # ---- Unified summary report ----
            log(f"\n{'='*40}\n  Summary Report\n{'='*40}")
            try:
                proc = subprocess.Popen(
                    [py, "summary_report.py",
                     "--frag-json", str(_report_dir / "portability.json"),
                     "--det-json",  str(_report_dir / "determinism.json"),
                     "--res-json",  str(_report_dir / "resource.json")],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=str(BASE_DIR),
                )
                for line in proc.stdout:
                    log(line.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    log(f"[WARN] Summary report exited with code {proc.returncode}")
            except FileNotFoundError as e:
                log(f"[WARN] summary_report.py not found: {e}")

    log("\nDone. Outputs in ./output/")
    return 0


def _run_ptx_split_and_parse(py: str, log: Callable[[str], None]):
    """Split monolithic PTX into per-kernel files and run ptx_parser."""
    import pathlib

    # Build clean_map from built-ins + registry
    registry = load_registry()
    clean_map = {k["name"]: k["name"] for k in BUILTINS}
    clean_map.update({k["name"]: k["name"] for k in registry})

    ptx_dir = BASE_DIR / "output" / "ptx"
    ptx_dir.mkdir(parents=True, exist_ok=True)

    src_file = ptx_dir / "kernels.ptx"
    if not src_file.exists():
        log("[WARN] kernels.ptx not found, skipping split")
        return

    import re as _re
    src = src_file.read_text(errors="replace")
    pattern = _re.compile(
        r'((?:\.visible\s+)?\.entry\s+(\w+)\b.*?^\})',
        _re.DOTALL | _re.MULTILINE
    )

    pairs = []
    for m in pattern.finditer(src):
        mangled = m.group(2)
        body    = m.group(1)
        clean   = next((v for k, v in clean_map.items() if k in mangled), mangled)
        out_p   = ptx_dir / f"{clean}.ptx"
        out_p.write_text(body, encoding="utf-8")
        log(f"    {clean:30s} <- {mangled}")
        pairs.append(f"{out_p}::{clean}")

    pairs_file = ptx_dir / ".ptx_pairs"
    pairs_file.write_text("\n".join(pairs), encoding="utf-8")

    if pairs:
        proc = subprocess.Popen(
            [py, "ptx_parser.py"] + pairs,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(BASE_DIR),
        )
        for line in proc.stdout:
            log(line.rstrip())
        proc.wait()
