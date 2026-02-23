"""
ptx_parser.py
=============
Static PTX instruction analysis.

PURPOSE
-------
Parse compiled PTX assembly files to count instruction occurrences by
category.  This is STATIC analysis — it operates on the compiler-generated
PTX text, not on a live GPU execution trace.  Instruction counts reflect the
static instruction mix that the compiler emitted; they do NOT account for
dynamic execution counts (loops, branching, divergence).

Every metric produced by this script is clearly tagged as
SOURCE: PTX static analysis.

INSTRUCTION CATEGORIES
----------------------
global_loads    – ld.global.*
global_stores   – st.global.*
shared_loads    – ld.shared.*
shared_stores   – st.shared.*
arithmetic      – add, sub, mul, fma, mad, div (integer and float)
branch          – bra, brx, call, ret (control flow changes)
special         – tex, suld, sust, bar, atom, red (texture, barriers, atomics)
other           – everything else

USAGE
-----
    python ptx_parser.py <kernel1.ptx> [kernel2.ptx ...]
    python ptx_parser.py output/coalesced_add.ptx output/strided_add.ptx output/divergent_add.ptx

Output: output/ptx_stats.csv
"""

import re
import sys
import os
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List

# ---------------------------------------------------------------------------
# Instruction category definitions
# Each entry is (category_name, compiled_regex).
# Categories are tested in order; first match wins.
# ---------------------------------------------------------------------------
CATEGORIES = [
    # ---- Memory ----
    ("global_loads",  re.compile(r'\bld\.global\b')),
    ("global_stores", re.compile(r'\bst\.global\b')),
    ("shared_loads",  re.compile(r'\bld\.shared\b')),
    ("shared_stores", re.compile(r'\bst\.shared\b')),
    # ---- Arithmetic ----
    # fma / mad before add/mul so "fma" doesn't match the "a" in add
    ("arithmetic",    re.compile(r'\b(?:fma|mad|add|sub|mul|div|neg|abs|min|max|rcp|sqrt|rsqrt|sin|cos|lg2|ex2)\b')),
    # ---- Control flow ----
    ("branch",        re.compile(r'\b(?:bra|brx\.idx|call|ret|exit)\b')),
    # ---- Special: barriers, atomics, texture ----
    ("special",       re.compile(r'\b(?:bar|atom|red|tex|suld|sust|vote|shfl|prmt)\b')),
]

@dataclass
class KernelStats:
    kernel_name:     str
    ptx_file:        str
    total_instr:     int
    global_loads:    int
    global_stores:   int
    shared_loads:    int
    shared_stores:   int
    arithmetic:      int
    branch:          int
    special:         int
    other:           int
    # Derived ratios (computed from the counts above — SOURCE: PTX static)
    memory_ratio:    float   # (global_loads + global_stores) / total_instr
    branch_ratio:    float   # branch / total_instr
    arith_ratio:     float   # arithmetic / total_instr

# ---------------------------------------------------------------------------
# PTX parsing utilities
# ---------------------------------------------------------------------------

def strip_ptx_comment(line: str) -> str:
    """Remove // and /* */ style comments from a PTX line."""
    # Single-line comment
    idx = line.find('//')
    if idx != -1:
        line = line[:idx]
    return line.strip()


def is_instruction_line(line: str) -> bool:
    """
    Heuristic: a PTX instruction line starts with optional whitespace,
    an optional predicate (@p0, @!p1 …), then an opcode.
    We reject lines that are directives (.param, .reg, .visible …),
    labels (ending with ':'), or blank.
    """
    if not line:
        return False
    # Labels end with ':'
    if line.endswith(':'):
        return False
    # Directives start with '.'
    if line.startswith('.'):
        return False
    # Pragma / comments
    if line.startswith('//') or line.startswith('/*'):
        return False
    # PTX directives that appear mid-function (e.g. ".loc")
    if re.match(r'\.(loc|pragma|file|section|align|byte|short|word|quad)', line):
        return False
    return True


def categorise(line: str) -> str:
    """Return the category name for a cleaned instruction line."""
    for name, pattern in CATEGORIES:
        if pattern.search(line):
            return name
    return "other"


def parse_ptx_file(path: str) -> Dict[str, int]:
    """
    Parse a single PTX file.
    Returns a dict with category counts + 'total_instr'.
    """
    counts: Dict[str, int] = {name: 0 for name, _ in CATEGORIES}
    counts["other"] = 0
    total = 0

    with open(path, 'r', errors='replace') as fh:
        for raw in fh:
            clean = strip_ptx_comment(raw.strip())
            if not is_instruction_line(clean):
                continue
            total += 1
            cat = categorise(clean)
            counts[cat] += 1

    counts["total_instr"] = total
    return counts


def kernel_name_from_path(path: str) -> str:
    """Derive a short kernel name from the PTX filename."""
    return Path(path).stem


def build_stats(ptx_path: str) -> KernelStats:
    counts = parse_ptx_file(ptx_path)
    t = counts["total_instr"]

    def ratio(k):
        return counts[k] / t if t > 0 else 0.0

    mem_ops = counts["global_loads"] + counts["global_stores"]

    return KernelStats(
        kernel_name   = kernel_name_from_path(ptx_path),
        ptx_file      = ptx_path,
        total_instr   = t,
        global_loads  = counts["global_loads"],
        global_stores = counts["global_stores"],
        shared_loads  = counts["shared_loads"],
        shared_stores = counts["shared_stores"],
        arithmetic    = counts["arithmetic"],
        branch        = counts["branch"],
        special       = counts["special"],
        other         = counts["other"],
        memory_ratio  = mem_ops / t if t > 0 else 0.0,
        branch_ratio  = ratio("branch"),
        arith_ratio   = ratio("arithmetic"),
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str]):
    """
    Usage:
      python ptx_parser.py file1.ptx file2.ptx ...
      python ptx_parser.py file1.ptx::clean_name1 file2.ptx::clean_name2 ...

    Append ::name after a path to override the kernel_name written to CSV.
    Example:
      python ptx_parser.py output/_Z13coalesced_addPKfS0_Pfi.ptx::coalesced_add
    """
    if not argv:
        print("Usage: python ptx_parser.py <file1.ptx[::name]> ...")
        sys.exit(1)

    # Parse optional ::override_name suffixes
    ptx_files: List[tuple[str, str]] = []
    for entry in argv:
        if '::' in entry:
            path, name = entry.split('::', 1)
        else:
            path = entry
            name = kernel_name_from_path(entry)
        ptx_files.append((path, name))

    results: List[KernelStats] = []
    for path, override_name in ptx_files:
        if not os.path.isfile(path):
            print(f"[WARN] File not found, skipping: {path}", file=sys.stderr)
            continue
        stats = build_stats(path)
        stats.kernel_name = override_name   # apply clean name override
        results.append(stats)
        print(f"\n[PTX static] {stats.kernel_name}")
        print(f"  Total instructions : {stats.total_instr}")
        print(f"  Global loads       : {stats.global_loads}")
        print(f"  Global stores      : {stats.global_stores}")
        print(f"  Arithmetic ops     : {stats.arithmetic}")
        print(f"  Branch instr       : {stats.branch}")
        print(f"  Special (bar/atom) : {stats.special}")
        print(f"  Other              : {stats.other}")
        print(f"  Memory ratio       : {stats.memory_ratio:.3f}  [SOURCE: PTX static]")
        print(f"  Branch ratio       : {stats.branch_ratio:.3f}  [SOURCE: PTX static]")
        print(f"  Arithmetic ratio   : {stats.arith_ratio:.3f}  [SOURCE: PTX static]")

    if not results:
        print("No PTX files parsed.")
        return

    os.makedirs("output/data", exist_ok=True)
    out_path = "output/data/ptx_stats.csv"
    fieldnames = list(asdict(results[0]).keys())
    # Exclude ptx_file from CSV — it contains path separators that confuse
    # csv quoting and is not used by analyze.py.
    skip = {"ptx_file"}
    csv_fields = [f for f in fieldnames if f not in skip]
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields,
                                quoting=csv.QUOTE_MINIMAL, extrasaction="ignore",
                                lineterminator='\n')
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])

