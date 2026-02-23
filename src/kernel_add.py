"""
kernel_add.py
=============
CLI tool for managing user kernels in the CUDA Kernel Analyzer.

Usage examples
--------------
  # List registered kernels
  python kernel_add.py --list

  # Add a kernel from a .cu file
  python kernel_add.py --name my_kernel --file my_kernel.cu

  # Add and immediately run the full pipeline
  python kernel_add.py --name my_kernel --file my_kernel.cu --run

  # Override launch config
  python kernel_add.py --name my_kernel --file my_kernel.cu --block 128 --n 8388608 --iters 1000

  # Override parameter string (if auto-extraction fails)
  python kernel_add.py --name my_kernel --file my_kernel.cu \\
      --params "const float* __restrict__ a, float* __restrict__ c, int n"

  # Remove a kernel
  python kernel_add.py --remove my_kernel

  # Run the pipeline without adding a kernel
  python kernel_add.py --run

  # Skip steps
  python kernel_add.py --run --skip-build --skip-ptx
"""

import argparse
import sys
from pathlib import Path

import kernel_manager as km


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_kernels():
    registry = km.load_registry()
    if not registry:
        print("No user kernels registered.")
        print("Built-in kernels (always included):")
        for k in km.BUILTINS:
            print(f"  {k['name']}")
        return
    print("Built-in kernels:")
    for k in km.BUILTINS:
        print(f"  {k['name']}")
    print("\nUser kernels:")
    for k in registry:
        print(f"  {k['name']:30s}  block={k['block_size']}  n={k['n_elements']}  "
              f"iters={k['iters']}  added={k['added']}")


def _parse_n(value: str) -> int:
    """Parse n_elements: supports 16M, 8388608, 16*1024*1024 etc."""
    value = value.strip()
    # Handle suffixes
    if value.upper().endswith("M"):
        return int(float(value[:-1]) * 1024 * 1024)
    if value.upper().endswith("K"):
        return int(float(value[:-1]) * 1024)
    # Try eval for expressions like 16*1024*1024
    try:
        return int(eval(value, {"__builtins__": {}}))
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Cannot parse n_elements '{value}'. Use a number, e.g. 16777216 or 16M."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manage user kernels for the CUDA Kernel Analyzer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Actions
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list",   action="store_true",
                        help="List all registered kernels and exit.")
    action.add_argument("--remove", metavar="NAME",
                        help="Remove the named kernel and exit.")

    # Add kernel options
    parser.add_argument("--name",   metavar="NAME",
                        help="Kernel function name (must match __global__ void <name>).")
    parser.add_argument("--file",   metavar="FILE",
                        help="Path to .cu file containing the kernel implementation.")
    parser.add_argument("--params", metavar="PARAMS",
                        help="Parameter declaration string (auto-extracted if omitted).")
    parser.add_argument("--block",  metavar="N", type=int, default=256,
                        help="Block size (default 256).")
    parser.add_argument("--n",      metavar="N", default="16777216",
                        help="Number of elements (default 16777216). Supports 16M.")
    parser.add_argument("--iters",  metavar="N", type=int, default=0,
                        help="Compute iterations (default 0). Set >0 for compute-bound kernels.")

    # Pipeline
    parser.add_argument("--run",          action="store_true",
                        help="Run the full analysis pipeline after adding/removing.")
    parser.add_argument("--skip-build",   action="store_true")
    parser.add_argument("--skip-bench",   action="store_true")
    parser.add_argument("--skip-ptx",     action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")

    args = parser.parse_args()

    # ---- list ----
    if args.list:
        _list_kernels()
        return 0

    # ---- remove ----
    if args.remove:
        err = km.remove_kernel(args.remove)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        print(f"Removed kernel '{args.remove}'.")
        if args.run:
            return _run(args)
        return 0

    # ---- add ----
    if args.name or args.file:
        if not args.name:
            parser.error("--name is required when adding a kernel.")
        if not args.file:
            parser.error("--file is required when adding a kernel.")

        cu_path = Path(args.file)
        if not cu_path.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return 1

        code = cu_path.read_text(encoding="utf-8", errors="replace")

        # Extract or use provided params
        params = args.params
        if not params:
            params = km.extract_params(code, args.name)
            if not params:
                print(
                    f"[WARN] Could not auto-extract parameters for '{args.name}' from {args.file}.\n"
                    f"       Pass --params 'const float* a, int n' explicitly.",
                    file=sys.stderr,
                )
                return 1
            print(f"Auto-extracted params: {params}")

        n_elements = _parse_n(args.n)

        err = km.add_kernel(
            name=args.name,
            code=code,
            params=params,
            block_size=args.block,
            n_elements=n_elements,
            iters=args.iters,
        )
        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1

        print(f"Added kernel '{args.name}'.")
        print(f"  block={args.block}  n={n_elements}  iters={args.iters}")

    # ---- run pipeline ----
    if args.run or (not args.name and not args.file and not args.list and not args.remove):
        if not args.name and not args.file and not args.run:
            parser.print_help()
            return 0
        return _run(args)

    return 0


def _run(args) -> int:
    print("\nRunning analysis pipeline ...")
    rc = km.run_pipeline(
        skip_build   = args.skip_build,
        skip_bench   = args.skip_bench,
        skip_ptx     = args.skip_ptx,
        skip_analyze = args.skip_analyze,
        log_cb       = print,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
