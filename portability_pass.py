"""
portability_pass.py
===================
Static analysis pass that detects GPU-architecture-sensitive (fragile) patterns
in CUDA kernels.

INPUTS
------
  kernelIR  – a KernelIR object populated from PTX text and/or CUDA source.
              See KernelIR and the helper constructors at the bottom of this file.
  gpu_config – optional GPUConfig with target-architecture parameters.

OUTPUT
------
  List[FragilityFlag]  – structured issue records with location, category,
                         severity, and description.

PASS STRUCTURE
--------------
  PortabilityPass.run() dispatches five independent sub-checks and aggregates
  their results.  Each sub-check is a pure function that takes the IR and
  config and returns a list of FragilityFlag objects.

  A. _check_warp_assumptions      – hard-coded warp sizes in control flow / math
  B. _check_memory_alignment      – alignment / stride assumptions, vector casts
  C. _check_scheduling            – __syncthreads placement, implicit ordering
  D. _check_arch_instructions     – compute-capability-specific PTX opcodes
  E. _check_undefined_behavior    – overflow, uninitialised vars, OOB accesses

DESIGN NOTES
------------
- Pure static analysis; no GPU simulation.
- Works on PTX text (always available after nvcc --ptx) and optionally on
  raw CUDA source for higher-level checks.
- Produces FragilityFlag objects rather than printing warnings, so callers
  (GUI, report writer, CLI) can format output however they like.
- Integrates with the existing pipeline: kernel_manager.py can call
  run_portability_pass() after the PTX step.
"""

from __future__ import annotations

import re
import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class Category(str, Enum):
    WARP_ASSUMPTION        = "warp_assumption"
    MEMORY_ALIGNMENT       = "memory_alignment"
    SCHEDULING             = "scheduling"
    ARCH_SPECIFIC_INSTR    = "arch_specific_instruction"
    UNDEFINED_BEHAVIOR     = "undefined_behavior"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class FragilityFlag:
    """A single portability issue detected in a kernel."""
    kernel_name:  str
    location:     str           # e.g. "divergent_compute:L42" or "ptx:bar.sync"
    category:     Category
    severity:     Severity
    description:  str
    score:        int = 0       # contribution to fragility score (set by aggregator)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        return d

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper():6s}] [{self.category.value}] "
            f"{self.kernel_name} @ {self.location}\n"
            f"         {self.description}"
        )


@dataclass
class GPUConfig:
    """
    Optional target-architecture constraints.

    If min_compute_capability and/or max_compute_capability are set, the pass
    flags instructions that are not available across that range.
    """
    warp_size:              int  = 32
    # Minimum shared memory per SM in bytes (SM 3.x: 48 KB, SM 8.x: 164 KB)
    shared_mem_per_sm:      int  = 49152
    min_compute_capability: str  = "3.0"   # oldest target, e.g. "3.0"
    max_compute_capability: str  = "9.0"   # newest target, e.g. "9.0"
    # Maximum threads per block (constant across all modern GPUs)
    max_threads_per_block:  int  = 1024

    def cc_tuple(self, cc: str):
        parts = cc.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)

    @property
    def min_cc(self):
        return self.cc_tuple(self.min_compute_capability)

    @property
    def max_cc(self):
        return self.cc_tuple(self.max_compute_capability)


@dataclass
class IRLine:
    """One line of PTX or source with its metadata."""
    lineno:   int
    text:     str       # stripped, comments removed
    raw:      str       # original text


@dataclass
class KernelIR:
    """
    Intermediate representation of a single kernel's code.

    Populate via one of the helper constructors:
      KernelIR.from_ptx_text(name, ptx_text)
      KernelIR.from_source_text(name, cu_text)
      KernelIR.from_ptx_file(name, path)
    """
    name:         str
    ptx_lines:    List[IRLine]   = field(default_factory=list)   # PTX instructions
    source_lines: List[IRLine]   = field(default_factory=list)   # CUDA C++ source
    # Metadata extracted during construction
    has_ptx:      bool = False
    has_source:   bool = False

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ptx_text(cls, name: str, ptx_text: str) -> "KernelIR":
        ir = cls(name=name)
        ir.ptx_lines = _parse_lines(ptx_text)
        ir.has_ptx   = True
        return ir

    @classmethod
    def from_source_text(cls, name: str, cu_text: str) -> "KernelIR":
        ir = cls(name=name)
        ir.source_lines = _parse_lines(cu_text, strip_comments=True)
        ir.has_source   = True
        return ir

    @classmethod
    def from_ptx_file(cls, name: str, path: str) -> "KernelIR":
        text = Path(path).read_text(errors="replace")
        return cls.from_ptx_text(name, text)

    def all_lines(self) -> List[IRLine]:
        """All PTX + source lines."""
        return self.ptx_lines + self.source_lines


# ---------------------------------------------------------------------------
# Line parsing utilities
# ---------------------------------------------------------------------------

def _strip_line_comment(line: str) -> str:
    """Remove // and /* */ style comments."""
    idx = line.find("//")
    if idx != -1:
        line = line[:idx]
    # Remove /* ... */ (single-line only)
    line = re.sub(r'/\*.*?\*/', '', line)
    return line.strip()


def _parse_lines(text: str, strip_comments: bool = True) -> List[IRLine]:
    result = []
    for i, raw in enumerate(text.splitlines(), start=1):
        cleaned = _strip_line_comment(raw.strip()) if strip_comments else raw.strip()
        if cleaned:
            result.append(IRLine(lineno=i, text=cleaned, raw=raw))
    return result


# ---------------------------------------------------------------------------
# Architecture-specific instruction database
# ---------------------------------------------------------------------------
# Maps PTX opcode patterns to (min_cc, max_cc, description).
# min_cc / max_cc use (major, minor) tuples; None = no boundary.
#
# Sources:
#   PTX ISA Reference (NVIDIA):
#     https://docs.nvidia.com/cuda/parallel-thread-execution/
#   CUDA C++ Programming Guide (SM feature table)

_ARCH_INSTRUCTIONS: List[Dict] = [
    # ---- Warp-level shuffle (introduced SM 3.0) ----
    {
        "pattern":  re.compile(r'\bshfl\.(?:up|down|bfly|idx)\b'),
        "min_cc":   (3, 0),
        "max_cc":   (3, 99),   # deprecated form; sync form required from SM 7.0+
        "description": "shfl (non-sync) is deprecated for SM 7.0+; use shfl.sync instead.",
        "severity": Severity.HIGH,
    },
    {
        "pattern":  re.compile(r'\bshfl\.sync\b'),
        "min_cc":   (7, 0),
        "max_cc":   None,
        "description": "shfl.sync requires SM 7.0+; will not compile for older targets.",
        "severity": Severity.MEDIUM,
    },
    # ---- Tensor core (wmma / mma) — SM 7.0+ ----
    {
        "pattern":  re.compile(r'\bwmma\b'),
        "min_cc":   (7, 0),
        "max_cc":   None,
        "description": "WMMA instructions require SM 7.0+ (Volta). Not available on Pascal or older.",
        "severity": Severity.HIGH,
    },
    {
        "pattern":  re.compile(r'\bmma\.sync\b'),
        "min_cc":   (7, 0),
        "max_cc":   None,
        "description": "mma.sync requires SM 7.0+.",
        "severity": Severity.HIGH,
    },
    # ---- cp.async (SM 8.0+, Ampere) ----
    {
        "pattern":  re.compile(r'\bcp\.async\b'),
        "min_cc":   (8, 0),
        "max_cc":   None,
        "description": "cp.async (async copy) requires SM 8.0+ (Ampere). Falls back to ld/st on older SM.",
        "severity": Severity.HIGH,
    },
    # ---- ldmatrix (SM 8.0+) ----
    {
        "pattern":  re.compile(r'\bldmatrix\b'),
        "min_cc":   (8, 0),
        "max_cc":   None,
        "description": "ldmatrix requires SM 8.0+.",
        "severity": Severity.HIGH,
    },
    # ---- bf16 arithmetic (SM 8.0+) ----
    {
        "pattern":  re.compile(r'\b(?:add|mul|fma)\.bf16\b'),
        "min_cc":   (8, 0),
        "max_cc":   None,
        "description": "BF16 arithmetic requires SM 8.0+ (Ampere). Not available on Turing or older.",
        "severity": Severity.HIGH,
    },
    # ---- FP16 arithmetic (SM 5.3+) ----
    {
        "pattern":  re.compile(r'\b(?:add|mul|fma)\.f16\b'),
        "min_cc":   (5, 3),
        "max_cc":   None,
        "description": "FP16 arithmetic requires SM 5.3+. Not available on Maxwell SM 5.0/5.2.",
        "severity": Severity.MEDIUM,
    },
    # ---- atom.cas (present but behavior differences pre-SM 6.0) ----
    {
        "pattern":  re.compile(r'\batom\.global\.cas\b'),
        "min_cc":   None,
        "max_cc":   None,
        "description": "atom.cas semantic differs pre-SM 6.0: 64-bit CAS not available on SM < 3.5.",
        "severity": Severity.LOW,
    },
    # ---- vote.sync (SM 7.0+) ----
    {
        "pattern":  re.compile(r'\bvote\.sync\b'),
        "min_cc":   (7, 0),
        "max_cc":   None,
        "description": "vote.sync requires SM 7.0+; use vote (non-sync) for older targets.",
        "severity": Severity.MEDIUM,
    },
    # ---- bar.warp.sync (SM 7.0+) ----
    {
        "pattern":  re.compile(r'\bbar\.warp\.sync\b'),
        "min_cc":   (7, 0),
        "max_cc":   None,
        "description": "bar.warp.sync requires SM 7.0+ (CUDA 9 cooperative groups).",
        "severity": Severity.MEDIUM,
    },
    # ---- redux / reduce (SM 8.0+) ----
    {
        "pattern":  re.compile(r'\bredux\.sync\b'),
        "min_cc":   (8, 0),
        "max_cc":   None,
        "description": "redux.sync warp reduction requires SM 8.0+.",
        "severity": Severity.HIGH,
    },
    # ---- setmaxnreg (SM 9.0+ / Hopper) ----
    {
        "pattern":  re.compile(r'\bsetmaxnreg\b'),
        "min_cc":   (9, 0),
        "max_cc":   None,
        "description": "setmaxnreg (dynamic register file) requires SM 9.0+ (Hopper).",
        "severity": Severity.HIGH,
    },
    # ---- clusterlaunch / cluster-level barriers (SM 9.0+) ----
    {
        "pattern":  re.compile(r'\bcluster\.arrive\b|\bcluster\.wait\b'),
        "min_cc":   (9, 0),
        "max_cc":   None,
        "description": "Cluster-level barriers require SM 9.0+ (Hopper / CUDA 12).",
        "severity": Severity.HIGH,
    },
    # ---- fence.proxy (SM 7.0+) ----
    {
        "pattern":  re.compile(r'\bfence\.proxy\b'),
        "min_cc":   (7, 0),
        "max_cc":   None,
        "description": "fence.proxy memory ordering requires SM 7.0+.",
        "severity": Severity.MEDIUM,
    },
]

# Hard-coded warp-size constants commonly seen in source and PTX
_WARP_SIZE_LITERALS = [32, 64]   # 64 for AMD / future NVIDIA
_WARP_SIZE_PATTERN  = re.compile(
    r'(?<!\w)(?:' + '|'.join(str(w) for w in _WARP_SIZE_LITERALS) + r')(?!\w)'
)

# Patterns that suggest warp-size arithmetic in PTX
_WARP_ARITH_PTX = re.compile(
    r'\b(?:and|shl|shr|rem|mod)\b.*(?:0x1[Ff]|31|32|63|64)\b'
    r'|\b(?:and|shl|shr|rem|mod)\b.*\b(?:31|32|63|64)\b'
)

# Memory-alignment hints in PTX: vector-width qualifiers
_VECTOR_LOAD_PATTERN  = re.compile(r'\bld(?:\.global|\.shared)?\.v[248]\b')
_VECTOR_STORE_PATTERN = re.compile(r'\bst(?:\.global|\.shared)?\.v[248]\b')

# Stride-related PTX: mul.wide + constant offset patterns
_STRIDE_PTX = re.compile(
    r'\bmul\.(?:lo|hi|wide)\b.*\b(?:stride|STRIDE|step|pitch)\b'
    r'|\bmad\.(?:lo|hi)\b.*\b(?:stride|STRIDE|step|pitch)\b',
    re.IGNORECASE
)

# __syncthreads() in PTX appears as bar.sync or bar.arrive
_BARRIER_PTX    = re.compile(r'\bbar\.(?:sync|arrive|red)\b')
_BARRIER_SOURCE = re.compile(r'\b__syncthreads(?:_count|_and|_or)?\s*\(')

# Shared memory store patterns
_SHARED_STORE = re.compile(r'\bst\.shared\b')
_SHARED_LOAD  = re.compile(r'\bld\.shared\b')

# Integer overflow indicators in PTX
_MUL32_OVERFLOW = re.compile(r'\bmul\.lo\.s32\b|\bmul\.lo\.u32\b')
_ADD_OVERFLOW   = re.compile(r'\badd\.s32\b|\badd\.u32\b')

# Out-of-bounds access: accessing global memory without a guard
_GLOBAL_LOAD  = re.compile(r'\bld\.global\b')
_GLOBAL_STORE = re.compile(r'\bst\.global\b')
_SETP_GUARD   = re.compile(r'\bsetp\b')   # bounds check generates setp in PTX

# CUDA source patterns
_FLOAT4_CAST    = re.compile(r'\(float4\s*\*\)')
_INT4_CAST      = re.compile(r'\(int4\s*\*\)')
_THREADIDX_MOD  = re.compile(r'threadIdx\s*\.\s*[xyz]\s*%\s*(\d+)')
_HARDCODED_32   = re.compile(r'\b32\b')
_HARDCODED_WARP = re.compile(r'\bWARP_SIZE\b|\bwarpSize\b')


# ---------------------------------------------------------------------------
# PortabilityPass
# ---------------------------------------------------------------------------

class PortabilityPass:
    """
    Static analysis pass for detecting GPU-architecture fragility patterns.

    Parameters
    ----------
    kernelIR  : KernelIR  – populated IR for a single kernel.
    gpu_config: GPUConfig – optional target constraints. If None, a default
                            (SM 3.0 → 9.0) config is used.
    """

    def __init__(self, kernelIR: KernelIR, gpu_config: Optional[GPUConfig] = None):
        self.kernelIR   = kernelIR
        self.gpu_config = gpu_config or GPUConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[FragilityFlag]:
        """Run all sub-checks and return aggregated, scored flags."""
        flags: List[FragilityFlag] = []
        flags.extend(_check_warp_assumptions(self.kernelIR, self.gpu_config))
        flags.extend(_check_memory_alignment(self.kernelIR, self.gpu_config))
        flags.extend(_check_scheduling(self.kernelIR, self.gpu_config))
        flags.extend(_check_arch_instructions(self.kernelIR, self.gpu_config))
        flags.extend(_check_undefined_behavior(self.kernelIR, self.gpu_config))
        _assign_scores(flags)
        return flags

    def fragility_score(self, flags: Optional[List[FragilityFlag]] = None) -> int:
        """
        Numeric fragility score for the kernel (higher = more fragile).
        Runs the full pass if flags are not provided.
        """
        if flags is None:
            flags = self.run()
        return sum(f.score for f in flags)

    def heatmap_data(self, flags: Optional[List[FragilityFlag]] = None) -> Dict[str, int]:
        """
        Per-category score breakdown suitable for heatmap visualisation.
        """
        if flags is None:
            flags = self.run()
        totals: Dict[str, int] = {c.value: 0 for c in Category}
        for f in flags:
            totals[f.category.value] += f.score
        return totals


# ---------------------------------------------------------------------------
# A. Warp-size assumption checker
# ---------------------------------------------------------------------------

def _check_warp_assumptions(ir: KernelIR, cfg: GPUConfig) -> List[FragilityFlag]:
    flags = []
    kernel = ir.name

    # ---- PTX: bit-mask operations assuming warpSize == 32 ----
    for line in ir.ptx_lines:
        # Detect `and.b32 %rX, %threadId, 31` (lane ID extraction by masking)
        if re.search(r'\band\.b32\b', line.text) and re.search(r'\b31\b', line.text):
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"ptx:L{line.lineno}",
                category=Category.WARP_ASSUMPTION,
                severity=Severity.HIGH,
                description=(
                    "PTX `and.b32 <reg>, 31` computes lane-ID by masking with 31, "
                    "assuming warpSize == 32. AMD GPUs use warpSize == 64; this mask "
                    "will silently produce wrong lane IDs there."
                ),
            ))

        # Detect shr/shl by 5 (log2(32) = 5) used to divide/multiply by warpSize
        if re.search(r'\bshr\.u32\b|\bshl\.b32\b', line.text) and re.search(r'\b5\b', line.text):
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"ptx:L{line.lineno}",
                category=Category.WARP_ASSUMPTION,
                severity=Severity.MEDIUM,
                description=(
                    "PTX shift by 5 (>> 5 or << 5) used for warp-ID or warp-stride "
                    "arithmetic implies warpSize == 32. Use `warpSize` runtime value "
                    "or PTX `%warpsize` special register instead."
                ),
            ))

        # Detect `rem` / `mod` by 32 (thread-in-warp index)
        if re.search(r'\brem\.u32\b|\brem\.s32\b', line.text) and re.search(r'\b32\b', line.text):
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"ptx:L{line.lineno}",
                category=Category.WARP_ASSUMPTION,
                severity=Severity.MEDIUM,
                description=(
                    "PTX `rem %r, 32` computes intra-warp position assuming warpSize == 32. "
                    "Use `%laneid` special register directly."
                ),
            ))

    # ---- Source: hard-coded integer 32 in arithmetic ----
    warp_modulo = re.compile(r'threadIdx\s*\.\s*[xyz]\s*%\s*32\b')
    warp_div    = re.compile(r'threadIdx\s*\.\s*[xyz]\s*/\s*32\b')
    for line in ir.source_lines:
        if warp_modulo.search(line.text):
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"source:L{line.lineno}",
                category=Category.WARP_ASSUMPTION,
                severity=Severity.HIGH,
                description=(
                    "Hard-coded `threadIdx.x % 32` assumes warpSize == 32. "
                    "Replace with `threadIdx.x % warpSize` (runtime) or use "
                    "`__builtin_ia32_rdtsc` / `__activemask()` patterns."
                ),
            ))
        if warp_div.search(line.text):
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"source:L{line.lineno}",
                category=Category.WARP_ASSUMPTION,
                severity=Severity.HIGH,
                description=(
                    "Hard-coded `threadIdx.x / 32` assumes warpSize == 32. "
                    "Replace with `threadIdx.x / warpSize`."
                ),
            ))

        # Loop bound hard-coded to 32 (e.g. warp reduction loop)
        loop_bound_32 = re.compile(r'\bfor\b.*[<>=]\s*32\b|\b32\b.*;\s*\w+\s*[*+/-]=')
        if loop_bound_32.search(line.text):
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"source:L{line.lineno}",
                category=Category.WARP_ASSUMPTION,
                severity=Severity.LOW,
                description=(
                    "Loop bound of 32 may encode a warp-size assumption "
                    "(e.g., manual warp-level reduction). Verify and replace "
                    "with `warpSize` if applicable."
                ),
            ))

    return flags


# ---------------------------------------------------------------------------
# B. Memory alignment / stride assumption checker
# ---------------------------------------------------------------------------

def _check_memory_alignment(ir: KernelIR, cfg: GPUConfig) -> List[FragilityFlag]:
    flags = []
    kernel = ir.name

    # ---- PTX: vector loads (ld.global.v2, v4) imply 8/16-byte alignment ----
    for line in ir.ptx_lines:
        if _VECTOR_LOAD_PATTERN.search(line.text):
            width = re.search(r'\.v(\d)', line.text)
            w = int(width.group(1)) if width else "?"
            align_bytes = int(w) * 4 if isinstance(w, int) else "?"
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"ptx:L{line.lineno}",
                category=Category.MEMORY_ALIGNMENT,
                severity=Severity.MEDIUM,
                description=(
                    f"Vector load `ld.v{w}` assumes {align_bytes}-byte alignment of the "
                    f"source pointer. Unaligned access triggers undefined behavior "
                    f"or a silent efficiency penalty depending on SM generation."
                ),
            ))

        if _VECTOR_STORE_PATTERN.search(line.text):
            width = re.search(r'\.v(\d)', line.text)
            w = int(width.group(1)) if width else "?"
            align_bytes = int(w) * 4 if isinstance(w, int) else "?"
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"ptx:L{line.lineno}",
                category=Category.MEMORY_ALIGNMENT,
                severity=Severity.MEDIUM,
                description=(
                    f"Vector store `st.v{w}` assumes {align_bytes}-byte alignment. "
                    f"Misaligned stores can cause access-fault on SM 3.x and "
                    f"silent partial writes on some architectures."
                ),
            ))

        # Stride: mul + constant that is likely a stride (not a power of 2)
        stride_const = re.search(r'\bmul\.(?:lo|hi|wide)\.(?:s|u)32\b.*,\s*(\d+)', line.text)
        if stride_const:
            const_val = int(stride_const.group(1))
            if const_val not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) and const_val > 1:
                flags.append(FragilityFlag(
                    kernel_name=kernel,
                    location=f"ptx:L{line.lineno}",
                    category=Category.MEMORY_ALIGNMENT,
                    severity=Severity.LOW,
                    description=(
                        f"PTX multiply by non-power-of-2 constant {const_val} used in "
                        f"address calculation suggests a non-standard stride. "
                        f"Non-coalesced stride patterns degrade performance and may "
                        f"exhibit different cache-line alignment across GPU generations."
                    ),
                ))

    # ---- Source: float4* / int4* casts ----
    for line in ir.source_lines:
        if _FLOAT4_CAST.search(line.text):
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"source:L{line.lineno}",
                category=Category.MEMORY_ALIGNMENT,
                severity=Severity.HIGH,
                description=(
                    "Cast to `float4*` requires 16-byte alignment. "
                    "If the base pointer is not 16-byte aligned (e.g., pointing into "
                    "the middle of an allocation, or an externally provided buffer), "
                    "the load/store will trigger undefined behavior or an access fault."
                ),
            ))
        if _INT4_CAST.search(line.text):
            flags.append(FragilityFlag(
                kernel_name=kernel,
                location=f"source:L{line.lineno}",
                category=Category.MEMORY_ALIGNMENT,
                severity=Severity.HIGH,
                description=(
                    "Cast to `int4*` requires 16-byte alignment. "
                    "Misaligned cast produces undefined behavior per the C++ standard "
                    "and hardware access faults on non-NVLink SM generations."
                ),
            ))

        # Unrolled stride loops: look for [idx + k] with fixed numeric offsets
        fixed_offset = re.compile(r'\[\s*\w+\s*\+\s*(\d{2,})\s*\]')
        offsets = fixed_offset.findall(line.text)
        for off in offsets:
            if int(off) % 4 != 0:
                flags.append(FragilityFlag(
                    kernel_name=kernel,
                    location=f"source:L{line.lineno}",
                    category=Category.MEMORY_ALIGNMENT,
                    severity=Severity.LOW,
                    description=(
                        f"Array access with fixed byte-offset {off} (not a multiple of 4) "
                        f"may produce misaligned float accesses on architectures that "
                        f"require natural alignment."
                    ),
                ))

    return flags


# ---------------------------------------------------------------------------
# C. Thread scheduling / timing assumption checker
# ---------------------------------------------------------------------------

def _check_scheduling(ir: KernelIR, cfg: GPUConfig) -> List[FragilityFlag]:
    flags = []
    kernel = ir.name

    # ---- PTX: check for shared-memory store NOT followed by bar.sync ----
    # Strategy: scan PTX for st.shared lines; if any block of st.shared
    # instructions is not followed (within ~10 instructions) by bar.sync,
    # flag a potential ordering assumption.
    if ir.ptx_lines:
        lines = ir.ptx_lines
        shared_store_indices = [
            i for i, l in enumerate(lines) if _SHARED_STORE.search(l.text)
        ]
        for idx in shared_store_indices:
            window = lines[idx : idx + 15]
            has_barrier = any(_BARRIER_PTX.search(l.text) for l in window)
            if not has_barrier:
                flags.append(FragilityFlag(
                    kernel_name=kernel,
                    location=f"ptx:L{lines[idx].lineno}",
                    category=Category.SCHEDULING,
                    severity=Severity.HIGH,
                    description=(
                        "Shared memory store (`st.shared`) not followed by `bar.sync` "
                        "within the next 15 PTX instructions. "
                        "Without explicit synchronisation, subsequent threads reading "
                        "this location may observe stale values — behavior that "
                        "varies by SM architecture and warp scheduling policy."
                    ),
                ))

    # ---- PTX: bar.sync inside a conditional branch ----
    # bar.sync must be reached by ALL threads in a block; placing it inside
    # a conditional is undefined behavior.
    if ir.ptx_lines:
        in_pred_block = False
        pred_depth    = 0
        for line in ir.ptx_lines:
            # Heuristic: @%p or @!%p prefixed instructions = conditional
            is_predicated = bool(re.match(r'@[!%]?\w+\s', line.text))
            if is_predicated and _BARRIER_PTX.search(line.text):
                flags.append(FragilityFlag(
                    kernel_name=kernel,
                    location=f"ptx:L{line.lineno}",
                    category=Category.SCHEDULING,
                    severity=Severity.HIGH,
                    description=(
                        "Predicated (`@%p`) barrier instruction detected. "
                        "`bar.sync` / `__syncthreads()` must be reached by ALL "
                        "threads in the block; a predicated barrier causes deadlock "
                        "or undefined behavior whenever any thread skips it."
                    ),
                ))

    # ---- Source: __syncthreads inside if/else ----
    if ir.source_lines:
        # Look for syncthreads inside a conditional block (heuristic: preceded
        # by `if (` or `else {` within 5 lines without a closing `}`)
        for i, line in enumerate(ir.source_lines):
            if _BARRIER_SOURCE.search(line.text):
                # Check preceding 5 lines for open conditional
                window_back = ir.source_lines[max(0, i-5):i]
                has_open_if = any(
                    re.search(r'\bif\s*\(', l.text) for l in window_back
                )
                has_close   = any('}' in l.text for l in window_back)
                if has_open_if and not has_close:
                    flags.append(FragilityFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=Category.SCHEDULING,
                        severity=Severity.HIGH,
                        description=(
                            "`__syncthreads()` appears inside a conditional block. "
                            "If any thread in the block takes a different branch, "
                            "this causes deadlock. Restructure so all threads "
                            "unconditionally reach the barrier."
                        ),
                    ))

    # ---- Source: implicit warp ordering (consecutive threadIdx reads without sync) ----
    if ir.source_lines:
        # Heuristic: shared array written with [threadIdx.x] and read with
        # [threadIdx.x - 1] or [threadIdx.x + 1] without intervening syncthreads
        shared_write = re.compile(r'\w+\s*\[\s*threadIdx\s*\.\s*x\s*\]\s*=')
        neighbor_read = re.compile(r'\w+\s*\[\s*threadIdx\s*\.\s*x\s*[+-]\s*\d+\s*\]')
        last_shared_write = -1
        last_sync         = -1
        for i, line in enumerate(ir.source_lines):
            if _BARRIER_SOURCE.search(line.text):
                last_sync = i
            if shared_write.search(line.text):
                last_shared_write = i
            if neighbor_read.search(line.text) and last_shared_write >= 0:
                if last_sync < last_shared_write:
                    flags.append(FragilityFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=Category.SCHEDULING,
                        severity=Severity.MEDIUM,
                        description=(
                            "Shared-memory neighbor access ([threadIdx.x ± k]) after a "
                            "write to the same array, without an intervening "
                            "`__syncthreads()`. Relies on implicit intra-warp ordering "
                            "which is not guaranteed across SM generations."
                        ),
                    ))

    return flags


# ---------------------------------------------------------------------------
# D. Architecture-specific instruction checker
# ---------------------------------------------------------------------------

def _check_arch_instructions(ir: KernelIR, cfg: GPUConfig) -> List[FragilityFlag]:
    flags = []
    kernel = ir.name
    min_cc = cfg.min_cc
    max_cc = cfg.max_cc

    for line in ir.ptx_lines:
        for entry in _ARCH_INSTRUCTIONS:
            if entry["pattern"].search(line.text):
                instr_min = entry.get("min_cc")
                instr_max = entry.get("max_cc")

                # Determine if the instruction falls outside the target CC range
                out_of_range = False
                note_parts   = []

                if instr_min is not None and min_cc < instr_min:
                    out_of_range = True
                    note_parts.append(
                        f"requires SM {instr_min[0]}.{instr_min[1]}+, "
                        f"but target min is SM {min_cc[0]}.{min_cc[1]}"
                    )
                if instr_max is not None and max_cc > instr_max:
                    out_of_range = True
                    note_parts.append(
                        f"deprecated/removed above SM {instr_max[0]}.{instr_max[1]}, "
                        f"but target max is SM {max_cc[0]}.{max_cc[1]}"
                    )

                if out_of_range or instr_min is not None or instr_max is not None:
                    severity = entry["severity"] if out_of_range else Severity.LOW
                    desc = entry["description"]
                    if note_parts:
                        desc += "  [Range conflict: " + "; ".join(note_parts) + "]"
                    flags.append(FragilityFlag(
                        kernel_name=kernel,
                        location=f"ptx:L{line.lineno}",
                        category=Category.ARCH_SPECIFIC_INSTR,
                        severity=severity,
                        description=desc,
                    ))

    # Also check source for CUDA intrinsics that map to arch-specific PTX
    source_intrinsic_checks = [
        (re.compile(r'\b__shfl_(?:up|down|xor)\s*\('),
         Severity.HIGH,
         "Non-sync shuffle `__shfl_*()` is deprecated from CUDA 9 / SM 7.0+. "
         "Use `__shfl_*_sync()` variants with an explicit mask."),
        (re.compile(r'\b__any\s*\(|\b__all\s*\(|\b__ballot\s*\('),
         Severity.HIGH,
         "Non-sync vote `__any/__all/__ballot` deprecated from CUDA 9 / SM 7.0+. "
         "Use `__any_sync/__all_sync/__ballot_sync` with a mask argument."),
        (re.compile(r'\btex1Dfetch\b|\btex2D\b|\btex3D\b'),
         Severity.MEDIUM,
         "Legacy texture fetch API. Behavior and precision differ across SM generations; "
         "prefer `cudaTextureObject_t` (Bindless Textures) for portability."),
        (re.compile(r'\b__ldg\s*\('),
         Severity.LOW,
         "`__ldg()` (read-only data cache) is a hint available on SM 3.5+. "
         "Transparent on older SM but may prevent certain compiler optimisations."),
        (re.compile(r'\b__threadfence_system\s*\('),
         Severity.MEDIUM,
         "`__threadfence_system()` enforces ordering across all devices and host memory. "
         "On pre-SM 2.0 hardware this is not supported."),
    ]
    for line in ir.source_lines:
        for pattern, sev, desc in source_intrinsic_checks:
            if pattern.search(line.text):
                flags.append(FragilityFlag(
                    kernel_name=kernel,
                    location=f"source:L{line.lineno}",
                    category=Category.ARCH_SPECIFIC_INSTR,
                    severity=sev,
                    description=desc,
                ))

    return flags


# ---------------------------------------------------------------------------
# E. Undefined behavior / corner case checker
# ---------------------------------------------------------------------------

def _check_undefined_behavior(ir: KernelIR, cfg: GPUConfig) -> List[FragilityFlag]:
    flags = []
    kernel = ir.name

    # ---- PTX: mul.lo.s32 of two potentially large values (overflow risk) ----
    for line in ir.ptx_lines:
        if _MUL32_OVERFLOW.search(line.text):
            # Heuristic: if the operands include %blockDim or %gridDim variables
            # (which in PTX appear as special registers or are passed as params)
            # the multiplication may overflow int32 for large grids
            if re.search(r'n|N|size|count|stride', line.text, re.IGNORECASE):
                flags.append(FragilityFlag(
                    kernel_name=kernel,
                    location=f"ptx:L{line.lineno}",
                    category=Category.UNDEFINED_BEHAVIOR,
                    severity=Severity.MEDIUM,
                    description=(
                        "32-bit integer multiply (`mul.lo.s32`) with a likely large "
                        "operand (size/count/stride variable). Overflows silently for "
                        "N > 2^31, producing wrong indices on large datasets. "
                        "Use `mul.wide.s32` (produces 64-bit) or cast to `ptrdiff_t` "
                        "before multiplying."
                    ),
                ))

    # ---- Source: int arithmetic in index expressions ----
    if ir.source_lines:
        large_index = re.compile(
            r'\b(?:blockIdx\s*\.\s*[xyz])\s*\*\s*blockDim\s*\.\s*[xyz]'
        )
        for line in ir.source_lines:
            if large_index.search(line.text):
                # Check if the expression is explicitly cast to long/size_t
                if not re.search(r'\((?:long|int64_t|size_t|ptrdiff_t)\)', line.text):
                    flags.append(FragilityFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=Category.UNDEFINED_BEHAVIOR,
                        severity=Severity.MEDIUM,
                        description=(
                            "`blockIdx.x * blockDim.x` computed in 32-bit int. "
                            "For grids with > 2^31 total threads (e.g., N > 2 billion), "
                            "this overflows. Cast to `(long long)blockIdx.x * blockDim.x`."
                        ),
                    ))

    # ---- PTX: global load without a setp guard in the same basic block ----
    # Heuristic: scan for ld.global not preceded (within 5 lines) by setp or @
    if ir.ptx_lines:
        lines = ir.ptx_lines
        for i, line in enumerate(lines):
            if _GLOBAL_LOAD.search(line.text):
                window = lines[max(0, i-6):i]
                has_guard = any(
                    _SETP_GUARD.search(l.text) or re.match(r'@[!%]?\w+\s+bra\b', l.text)
                    for l in window
                )
                if not has_guard:
                    flags.append(FragilityFlag(
                        kernel_name=kernel,
                        location=f"ptx:L{line.lineno}",
                        category=Category.UNDEFINED_BEHAVIOR,
                        severity=Severity.LOW,
                        description=(
                            "Global memory load without a visible bounds-check guard "
                            "in the preceding PTX. If thread count does not evenly divide "
                            "the array size, out-of-bounds threads may load garbage or "
                            "fault. Verify that an `if (idx < n)` guard is present."
                        ),
                    ))

    # ---- Source: uninitialized shared memory usage ----
    if ir.source_lines:
        # Detect: `__shared__ <type> <name>[...]` followed by read of <name>
        # before any write to <name>
        smem_decl = re.compile(r'__shared__\s+\w+\s+(\w+)\s*\[')
        declared_smem: Dict[str, int] = {}
        for i, line in enumerate(ir.source_lines):
            m = smem_decl.search(line.text)
            if m:
                declared_smem[m.group(1)] = i

        for var, decl_line_idx in declared_smem.items():
            # Look for first use of var after declaration
            escaped = re.escape(var)
            first_write = None
            first_read  = None
            for j in range(decl_line_idx + 1, len(ir.source_lines)):
                l = ir.source_lines[j]
                write_pat = re.compile(rf'\b{escaped}\s*\[.*\]\s*=')
                read_pat  = re.compile(rf'=\s*{escaped}\s*\[')
                if write_pat.search(l.text) and first_write is None:
                    first_write = j
                if read_pat.search(l.text) and first_read is None:
                    first_read = j
                if first_write is not None and first_read is not None:
                    break
            if first_read is not None and (first_write is None or first_read < first_write):
                flags.append(FragilityFlag(
                    kernel_name=kernel,
                    location=f"source:L{ir.source_lines[first_read].lineno}",
                    category=Category.UNDEFINED_BEHAVIOR,
                    severity=Severity.HIGH,
                    description=(
                        f"Shared memory array `{var}` is read before it is written. "
                        f"Uninitialized shared memory contains arbitrary values; "
                        f"results will differ across GPU architectures and runs."
                    ),
                ))

    return flags


# ---------------------------------------------------------------------------
# Score assignment
# ---------------------------------------------------------------------------

_SEVERITY_SCORE = {
    Severity.HIGH:   10,
    Severity.MEDIUM:  5,
    Severity.LOW:     2,
}


def _assign_scores(flags: List[FragilityFlag]):
    for f in flags:
        f.score = _SEVERITY_SCORE.get(f.severity, 1)


# ---------------------------------------------------------------------------
# Pipeline integration helpers
# ---------------------------------------------------------------------------

def run_portability_pass(
    kernel_name: str,
    ptx_path:    Optional[str] = None,
    source_text: Optional[str] = None,
    gpu_config:  Optional[GPUConfig] = None,
) -> List[FragilityFlag]:
    """
    Convenience wrapper: build KernelIR from file paths / text and run the pass.

    Parameters
    ----------
    kernel_name : str  – human-readable kernel name
    ptx_path    : str  – path to .ptx file (optional but recommended)
    source_text : str  – CUDA C++ source text for this kernel (optional)
    gpu_config  : GPUConfig – target architecture constraints
    """
    ir = KernelIR(name=kernel_name)
    if ptx_path and os.path.isfile(ptx_path):
        ptx_text    = Path(ptx_path).read_text(errors="replace")
        ir.ptx_lines = _parse_lines(ptx_text)
        ir.has_ptx   = True
    if source_text:
        ir.source_lines = _parse_lines(source_text, strip_comments=True)
        ir.has_source   = True

    return PortabilityPass(ir, gpu_config).run()


def run_portability_pass_on_directory(
    ptx_dir:     str,
    gpu_config:  Optional[GPUConfig] = None,
) -> Dict[str, List[FragilityFlag]]:
    """
    Run the pass on all .ptx files found in ptx_dir.
    Returns a dict mapping kernel_name -> list of FragilityFlag.
    """
    results: Dict[str, List[FragilityFlag]] = {}
    for ptx_file in sorted(Path(ptx_dir).glob("*.ptx")):
        name  = ptx_file.stem
        flags = run_portability_pass(name, str(ptx_file), gpu_config=gpu_config)
        results[name] = flags
    return results


def write_portability_report(
    results:    Dict[str, List[FragilityFlag]],
    out_path:   str = "output/report/portability.md",
):
    """Write a Markdown portability report from a results dict."""
    lines = []
    lines.append("# CUDA Kernel Portability Report\n\n")
    lines.append(
        "Static analysis for GPU-architecture-sensitive patterns.\n"
        "_Generated by portability_pass.py_\n\n"
    )
    lines.append("## Summary\n\n")
    lines.append("| Kernel | Score | High | Medium | Low |\n")
    lines.append("|--------|------:|-----:|-------:|----:|\n")

    all_flags: List[FragilityFlag] = []
    for kernel_name, flags in results.items():
        all_flags.extend(flags)
        score  = sum(f.score for f in flags)
        highs  = sum(1 for f in flags if f.severity == Severity.HIGH)
        meds   = sum(1 for f in flags if f.severity == Severity.MEDIUM)
        lows   = sum(1 for f in flags if f.severity == Severity.LOW)
        lines.append(f"| `{kernel_name}` | {score} | {highs} | {meds} | {lows} |\n")
    lines.append("\n")

    lines.append("## Detailed Findings\n\n")
    for kernel_name, flags in results.items():
        if not flags:
            lines.append(f"### `{kernel_name}` — no issues found\n\n")
            continue
        score = sum(f.score for f in flags)
        lines.append(f"### `{kernel_name}`  (fragility score: {score})\n\n")
        for f in sorted(flags, key=lambda x: -x.score):
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(f.severity.value, "⚪")
            lines.append(
                f"- **[{f.severity.value.upper()}]** `{f.category.value}` "
                f"@ `{f.location}`\n"
                f"  {f.description}\n\n"
            )
        lines.append("\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    print(f"Saved portability report: {out_path}")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _cli_main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="CUDA kernel portability / fragility static analysis pass."
    )
    parser.add_argument(
        "ptx_files", nargs="*",
        help="PTX files to analyse (can use ::name suffix to set kernel name, same as ptx_parser.py)."
    )
    parser.add_argument(
        "--ptx-dir", default=None,
        help="Analyse all .ptx files in this directory (alternative to listing files)."
    )
    parser.add_argument(
        "--source", default=None,
        help="Path to CUDA .cu source file for source-level checks (optional)."
    )
    parser.add_argument(
        "--min-cc", default="3.0",
        help="Minimum target compute capability, e.g. 3.0 (default: 3.0)."
    )
    parser.add_argument(
        "--max-cc", default="9.0",
        help="Maximum target compute capability, e.g. 9.0 (default: 9.0)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Write Markdown report to this path (default: output/report/portability.md)."
    )
    parser.add_argument(
        "--json", default=None,
        help="Write JSON flags to this path."
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Print only kernel names and fragility scores."
    )
    args = parser.parse_args()

    cfg = GPUConfig(
        min_compute_capability=args.min_cc,
        max_compute_capability=args.max_cc,
    )

    source_text: Optional[str] = None
    if args.source and os.path.isfile(args.source):
        source_text = Path(args.source).read_text(errors="replace")

    results: Dict[str, List[FragilityFlag]] = {}

    if args.ptx_dir:
        results = run_portability_pass_on_directory(args.ptx_dir, gpu_config=cfg)
    elif args.ptx_files:
        for entry in args.ptx_files:
            if "::" in entry:
                path, name = entry.split("::", 1)
            else:
                path = entry
                name = Path(entry).stem
            flags = run_portability_pass(name, ptx_path=path,
                                         source_text=source_text, gpu_config=cfg)
            results[name] = flags
    else:
        # Default: scan output/ptx/ directory
        default_dir = os.path.join("output", "ptx")
        if os.path.isdir(default_dir):
            results = run_portability_pass_on_directory(default_dir, gpu_config=cfg)
        else:
            parser.print_help()
            sys.exit(0)

    # ---- Print to stdout ----
    sep = "=" * 70
    print(f"\n{sep}")
    print("  PORTABILITY PASS — FRAGILITY FLAGS")
    print(sep)

    total_flags = sum(len(f) for f in results.values())
    if total_flags == 0:
        print("  No fragility issues detected.")
    else:
        for kernel_name, flags in results.items():
            score = sum(f.score for f in flags)
            if args.score_only:
                print(f"  {kernel_name:40s}  score={score}")
                continue
            print(f"\n  Kernel: {kernel_name}  (fragility score: {score})")
            print(f"  {'-' * 50}")
            if not flags:
                print("    No issues found.")
            for f in sorted(flags, key=lambda x: -x.score):
                print(f"    {f}")
    print(f"\n{sep}")

    # ---- Write report ----
    out_path = args.output or os.path.join("output", "report", "portability.md")
    write_portability_report(results, out_path)

    # ---- Write JSON ----
    if args.json:
        json_out = {k: [f.as_dict() for f in v] for k, v in results.items()}
        Path(args.json).write_text(json.dumps(json_out, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    _cli_main()
