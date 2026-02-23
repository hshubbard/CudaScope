"""
determinism_pass.py
===================
Static analysis pass that detects patterns that can produce non-deterministic
or irreproducible results across CUDA kernel runs.

INPUTS
------
  kernelIR  – a KernelIR object (from portability_pass.py).
              Accepts PTX lines, CUDA source lines, or both.

OUTPUT
------
  List[DeterminismFlag]  – structured issue records.

PASS STRUCTURE
--------------
  DeterminismPass.run() dispatches three independent sub-checks:

  A. _check_race_conditions       – shared/global memory races, missing atomics
  B. _check_order_sensitive_ops   – FP reductions, non-associative accumulation
  C. _check_timing_dependencies   – implicit thread ordering, barrier placement

DESIGN NOTES
------------
- Pure static analysis; no GPU execution required.
- Reuses KernelIR / IRLine / _parse_lines / _SEVERITY_SCORE from portability_pass.
- DeterminismFlag mirrors FragilityFlag so both can be consumed by the same
  report writer or GUI table.
- Each sub-check is a standalone function for easy unit testing.
"""

from __future__ import annotations

import re
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

# Reuse shared IR primitives from portability_pass
from portability_pass import (
    KernelIR, IRLine, GPUConfig,
    _parse_lines,
    run_portability_pass,          # re-exported for convenience in callers
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DetSeverity(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class DetCategory(str, Enum):
    RACE_CONDITION          = "race_condition"
    ORDER_SENSITIVE_REDUCE  = "order_sensitive_reduction"
    TIMING_DEPENDENCY       = "timing_dependency"


# ---------------------------------------------------------------------------
# Flag dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeterminismFlag:
    """A single non-determinism issue detected in a kernel."""
    kernel_name: str
    location:    str          # e.g. "ptx:L42" or "source:L17"
    category:    DetCategory
    severity:    DetSeverity
    description: str
    score:       int = 0      # numeric contribution; assigned by _assign_scores

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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SEVERITY_SCORE: Dict[DetSeverity, int] = {
    DetSeverity.HIGH:   10,
    DetSeverity.MEDIUM:  5,
    DetSeverity.LOW:     2,
}


def _assign_scores(flags: List[DeterminismFlag]) -> None:
    for f in flags:
        f.score = _SEVERITY_SCORE.get(f.severity, 1)


# ---------------------------------------------------------------------------
# PTX / source patterns
# ---------------------------------------------------------------------------

# ---- Atomic operations ----
_ATOM_PTX       = re.compile(r'\batom\.(?:global|shared)\b')
_RED_PTX        = re.compile(r'\bred\.(?:global|shared)\b')

# ---- Memory access without atomics ----
_ST_GLOBAL      = re.compile(r'\bst\.global\b')
_LD_GLOBAL      = re.compile(r'\bld\.global\b')
_ST_SHARED      = re.compile(r'\bst\.shared\b')
_LD_SHARED      = re.compile(r'\bld\.shared\b')

# ---- Barrier ----
_BARRIER_PTX    = re.compile(r'\bbar\.(?:sync|arrive|red)\b')
_BARRIER_SRC    = re.compile(r'\b__syncthreads(?:_count|_and|_or)?\s*\(')

# ---- FP reductions in PTX ----
# fma / add used in a loop suggest order-sensitive accumulation
_FMA_PTX        = re.compile(r'\bfma\.(?:rn|rz|rm|rp)?\.f(?:32|64)\b')
_FADD_PTX       = re.compile(r'\badd\.(?:rn|rz|rm|rp)?\.f(?:32|64)\b')
_FMUL_PTX       = re.compile(r'\bmul\.(?:rn|rz|rm|rp)?\.f(?:32|64)\b')

# ---- Warp-shuffle reductions (order-dependent) ----
_SHFL_PTX       = re.compile(r'\bshfl\.(?:sync\.)?(?:down|up|bfly|xor)\b')

# ---- Source-level patterns ----
_SHARED_DECL    = re.compile(r'__shared__\s+\w+\s+(\w+)\s*\[')
_ATOMIC_SRC     = re.compile(
    r'\b(?:atomicAdd|atomicSub|atomicExch|atomicMin|atomicMax|'
    r'atomicAnd|atomicOr|atomicXor|atomicCAS|atomicInc|atomicDec)\s*\('
)
_FP_ACCUM_SRC   = re.compile(
    r'(?:float|double)\s+\w+\s*=\s*0'          # zero-init accumulator
    r'|(?:\w+)\s*\+=\s*\w+\s*\[',              # += array[...]
)
_REDUCTION_SRC  = re.compile(
    r'for\s*\(.*;\s*\w+\s*[<>]=?\s*\w+.*\)\s*\{'
    r'|for\s*\(.*;\s*\w+\s*[<>]=?\s*\d+.*\)\s*\{'
)
_THREADIDX_COND = re.compile(r'if\s*\(\s*threadIdx\s*\.\s*[xyz]')
_BLOCKIDX_COND  = re.compile(r'if\s*\(\s*blockIdx\s*\.\s*[xyz]')

# PTX: predicated store to shared without surrounding barrier
_PRED_ST_SHARED = re.compile(r'@[!%]?\w+\s+st\.shared\b')
# PTX: predicated load from shared without surrounding barrier
_PRED_LD_SHARED = re.compile(r'@[!%]?\w+\s+ld\.shared\b')

# PTX: loop-carried dependency patterns (phi-like via ld then st to same addr)
# We approximate by looking for back-to-back ld then st to the same base register
_LOOP_CARRIED   = re.compile(
    r'\bld\.(?:global|shared).*\[(%\w+)\+?\d*\].*\n'
    r'(?:.*\n){0,3}'
    r'\bst\.(?:global|shared).*\[(%\w+)\+?\d*\]',
)


# ---------------------------------------------------------------------------
# A. Race condition checker
# ---------------------------------------------------------------------------

def _check_race_conditions(ir: KernelIR, cfg: Optional[GPUConfig]) -> List[DeterminismFlag]:
    flags: List[DeterminismFlag] = []
    kernel = ir.name

    # ---- PTX: global store not guarded by an atomic ----
    # Heuristic: if we see st.global AND atom.global in the same kernel,
    # the non-atomic store is suspicious (one path atomic, one raw).
    has_atom_global = any(_ATOM_PTX.search(l.text) for l in ir.ptx_lines
                          if 'global' in l.text)
    has_raw_st_global = any(_ST_GLOBAL.search(l.text) for l in ir.ptx_lines)

    if has_atom_global and has_raw_st_global:
        # Find the raw st.global lines and flag them
        for line in ir.ptx_lines:
            if _ST_GLOBAL.search(line.text) and not _ATOM_PTX.search(line.text):
                flags.append(DeterminismFlag(
                    kernel_name=kernel,
                    location=f"ptx:L{line.lineno}",
                    category=DetCategory.RACE_CONDITION,
                    severity=DetSeverity.HIGH,
                    description=(
                        "Non-atomic global store (`st.global`) co-exists with atomic "
                        "operations in the same kernel. Multiple threads writing the "
                        "same address via `st.global` without atomics produces a data "
                        "race; results depend on warp scheduling order."
                    ),
                ))

    # ---- PTX: shared memory store without a subsequent barrier ----
    # Already caught by portability_pass (scheduling check), but here we
    # specifically flag it as a RACE issue, not just a scheduling issue.
    if ir.ptx_lines:
        lines = ir.ptx_lines
        shared_store_idxs = [
            i for i, l in enumerate(lines) if _ST_SHARED.search(l.text)
        ]
        for idx in shared_store_idxs:
            window = lines[idx: idx + 20]
            has_barrier = any(_BARRIER_PTX.search(l.text) for l in window)
            has_ld_after = any(
                _LD_SHARED.search(l.text) for l in lines[idx + 1: idx + 20]
            )
            if has_ld_after and not has_barrier:
                flags.append(DeterminismFlag(
                    kernel_name=kernel,
                    location=f"ptx:L{lines[idx].lineno}",
                    category=DetCategory.RACE_CONDITION,
                    severity=DetSeverity.HIGH,
                    description=(
                        "Shared memory write (`st.shared`) followed by a read "
                        "(`ld.shared`) from another thread without an intervening "
                        "`bar.sync`. Without the barrier, the reading thread may "
                        "observe a stale value — a data race whose outcome depends "
                        "on warp scheduling."
                    ),
                ))

    # ---- PTX: red.global (reduction) — always non-deterministic for FP ----
    for line in ir.ptx_lines:
        if _RED_PTX.search(line.text) and re.search(r'\.f(?:32|64)\b', line.text):
            flags.append(DeterminismFlag(
                kernel_name=kernel,
                location=f"ptx:L{line.lineno}",
                category=DetCategory.RACE_CONDITION,
                severity=DetSeverity.MEDIUM,
                description=(
                    "Floating-point `red.global` (hardware reduction) is inherently "
                    "non-associative: the accumulation order depends on which threads "
                    "arrive first, varying run-to-run. Results will differ at the "
                    "ULP level across runs and GPU generations."
                ),
            ))

    # ---- Source: non-atomic write to a shared variable indexed by a loop ----
    if ir.source_lines:
        # Detect: shared array written inside a loop body without atomicAdd
        in_loop        = False
        loop_depth     = 0
        smem_vars      = set()
        for line in ir.source_lines:
            m = _SHARED_DECL.search(line.text)
            if m:
                smem_vars.add(m.group(1))
            if re.search(r'\bfor\s*\(', line.text):
                in_loop    = True
                loop_depth += 1
            if in_loop and '{' in line.text:
                loop_depth += line.text.count('{') - line.text.count('}')
                if loop_depth <= 0:
                    in_loop    = False
                    loop_depth = 0
            if in_loop:
                for var in smem_vars:
                    if re.search(rf'\b{re.escape(var)}\s*\[.*\]\s*[+\-\*]?=', line.text):
                        if not _ATOMIC_SRC.search(line.text):
                            flags.append(DeterminismFlag(
                                kernel_name=kernel,
                                location=f"source:L{line.lineno}",
                                category=DetCategory.RACE_CONDITION,
                                severity=DetSeverity.HIGH,
                                description=(
                                    f"Non-atomic write to shared memory array `{var}` "
                                    f"inside a loop. If multiple threads update the "
                                    f"same index without atomics, the result is a data "
                                    f"race."
                                ),
                            ))

        # ---- Source: global pointer written by multiple threads without atomics ----
        out_ptr_write = re.compile(
            r'(?:c|out|result|output)\s*\[\s*(?:blockIdx|threadIdx|idx)\b.*\]\s*[+\-\*]?='
        )
        has_atomic = any(_ATOMIC_SRC.search(l.text) for l in ir.source_lines)
        for line in ir.source_lines:
            if out_ptr_write.search(line.text) and not _ATOMIC_SRC.search(line.text):
                if has_atomic:
                    # Mixed atomic/non-atomic in the same kernel
                    flags.append(DeterminismFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=DetCategory.RACE_CONDITION,
                        severity=DetSeverity.MEDIUM,
                        description=(
                            "Non-atomic output write (`c[idx] =`) co-exists with "
                            "atomic operations in this kernel. If threads can alias "
                            "the same output index, the non-atomic path creates a race."
                        ),
                    ))

    return flags


# ---------------------------------------------------------------------------
# B. Order-sensitive reduction checker
# ---------------------------------------------------------------------------

def _check_order_sensitive_ops(ir: KernelIR, cfg: Optional[GPUConfig]) -> List[DeterminismFlag]:
    flags: List[DeterminismFlag] = []
    kernel = ir.name

    # ---- PTX: FP fma/add in what appears to be a reduction loop ----
    # Heuristic: if we see a sequence of fma/fadd instructions accumulating
    # into the same destination register, flag it.
    if ir.ptx_lines:
        # Track destination registers for fma/add
        fp_accum: Dict[str, int] = {}   # reg -> count of times it's the dest
        for line in ir.ptx_lines:
            if _FMA_PTX.search(line.text) or _FADD_PTX.search(line.text):
                # PTX format: fma.rn.f32 %dest, %a, %b, %c
                m = re.match(r'\s*(?:fma|add)\S*\s+(%\w+)', line.text)
                if m:
                    reg = m.group(1)
                    fp_accum[reg] = fp_accum.get(reg, 0) + 1

        # Any register accumulated >2 times indicates a reduction loop
        for reg, count in fp_accum.items():
            if count > 2:
                flags.append(DeterminismFlag(
                    kernel_name=kernel,
                    location=f"ptx:reg:{reg}",
                    category=DetCategory.ORDER_SENSITIVE_REDUCE,
                    severity=DetSeverity.MEDIUM,
                    description=(
                        f"Register `{reg}` is the destination of {count} floating-point "
                        f"fma/add instructions — consistent with a loop reduction. "
                        f"Floating-point addition is non-associative: changing thread "
                        f"scheduling or warp count changes the accumulation order "
                        f"and produces bit-different results across runs."
                    ),
                ))

    # ---- PTX: warp-shuffle reduction (shfl.down or shfl.bfly) ----
    shfl_lines = [l for l in ir.ptx_lines if _SHFL_PTX.search(l.text)]
    if shfl_lines:
        # Check if the shuffles feed into an FP add (butterfly reduction pattern)
        shfl_regs = set()
        for l in shfl_lines:
            m = re.match(r'\s*shfl\S*\s+(%\w+)[|,]', l.text)
            if m:
                shfl_regs.add(m.group(1))

        for line in ir.ptx_lines:
            if (_FADD_PTX.search(line.text) or _FMA_PTX.search(line.text)):
                for reg in shfl_regs:
                    if reg in line.text:
                        flags.append(DeterminismFlag(
                            kernel_name=kernel,
                            location=f"ptx:L{line.lineno}",
                            category=DetCategory.ORDER_SENSITIVE_REDUCE,
                            severity=DetSeverity.MEDIUM,
                            description=(
                                "Warp-shuffle value fed into a floating-point accumulation. "
                                "Warp-level reductions via `shfl.down` / `shfl.bfly` are "
                                "deterministic within a single warp but the cross-warp "
                                "aggregation order (typically via atomics or shared memory) "
                                "is not guaranteed, yielding run-to-run FP variation."
                            ),
                        ))
                        break

    # ---- Source: parallel reduction patterns ----
    if ir.source_lines:
        # Pattern 1: FP accumulator updated inside a thread-indexed loop
        fp_accum_src = re.compile(
            r'(?:float|double)\s+(\w+)\s*=\s*(?:0\.0f?|0)'
        )
        accum_vars: set = set()
        for line in ir.source_lines:
            m = fp_accum_src.search(line.text)
            if m:
                accum_vars.add(m.group(1))

        for line in ir.source_lines:
            for var in accum_vars:
                # var += something involving a thread-indexed array
                if re.search(rf'\b{re.escape(var)}\s*\+=\s*\w+\s*\[', line.text):
                    flags.append(DeterminismFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=DetCategory.ORDER_SENSITIVE_REDUCE,
                        severity=DetSeverity.LOW,
                        description=(
                            f"Floating-point accumulator `{var}` updated with `+=` "
                            f"over an array. Summation order depends on thread/loop "
                            f"scheduling. For reproducible results use a deterministic "
                            f"reduction algorithm (e.g., pairwise summation, Kahan "
                            f"compensated sum, or fixed-order sequential reduction)."
                        ),
                    ))

        # Pattern 2: atomicAdd on a float — globally non-deterministic order
        atomic_fp = re.compile(r'\batomicAdd\s*\(\s*\w+\s*,\s*(?:float|double|\w+)')
        for line in ir.source_lines:
            if atomic_fp.search(line.text):
                flags.append(DeterminismFlag(
                    kernel_name=kernel,
                    location=f"source:L{line.lineno}",
                    category=DetCategory.ORDER_SENSITIVE_REDUCE,
                    severity=DetSeverity.MEDIUM,
                    description=(
                        "`atomicAdd` on a floating-point variable produces non-deterministic "
                        "results: the hardware serialises the atomic but does not enforce "
                        "an ordering across threads, so the summation order varies per run. "
                        "Results are correct in expectation but not bit-reproducible."
                    ),
                ))

    return flags


# ---------------------------------------------------------------------------
# C. Timing / scheduling dependency checker
# ---------------------------------------------------------------------------

def _check_timing_dependencies(ir: KernelIR, cfg: Optional[GPUConfig]) -> List[DeterminismFlag]:
    flags: List[DeterminismFlag] = []
    kernel = ir.name

    # ---- PTX: missing __syncthreads() before reading a value written by another thread ----
    # Already covered by race check (st.shared / ld.shared without bar.sync).
    # Here we focus on TIMING dependencies: control flow that assumes a specific
    # thread ordering rather than a data race on a single location.

    # ---- PTX: conditional branch depending on shared memory read without barrier ----
    # Pattern: ld.shared -> setp -> bra  (without bar.sync before ld.shared)
    if ir.ptx_lines:
        lines = ir.ptx_lines
        for i, line in enumerate(lines):
            if _LD_SHARED.search(line.text):
                # Check if a setp / bra follows within 4 lines
                window_fwd = lines[i + 1: i + 5]
                is_cond_branch = any(
                    re.search(r'\bsetp\b|\bbra\b', l.text) for l in window_fwd
                )
                # Check for barrier in the 10 lines BEFORE this load
                window_back = lines[max(0, i - 10): i]
                has_barrier_before = any(_BARRIER_PTX.search(l.text) for l in window_back)

                if is_cond_branch and not has_barrier_before:
                    flags.append(DeterminismFlag(
                        kernel_name=kernel,
                        location=f"ptx:L{line.lineno}",
                        category=DetCategory.TIMING_DEPENDENCY,
                        severity=DetSeverity.HIGH,
                        description=(
                            "Conditional branch driven by a shared-memory load "
                            "(`ld.shared` -> `setp` -> `bra`) without a preceding "
                            "`bar.sync`. If the shared memory was written by another "
                            "thread, the branch outcome depends on when that write "
                            "completes — a timing-dependent control-flow race."
                        ),
                    ))

    # ---- PTX: bar.sync only in ONE branch of a divergent warp ----
    # Detected by: predicated bar.sync (already flagged by portability_pass as
    # a scheduling issue). Here we re-flag as a TIMING issue.
    for line in ir.ptx_lines:
        if re.match(r'@[!%]?\w+\s', line.text) and _BARRIER_PTX.search(line.text):
            flags.append(DeterminismFlag(
                kernel_name=kernel,
                location=f"ptx:L{line.lineno}",
                category=DetCategory.TIMING_DEPENDENCY,
                severity=DetSeverity.HIGH,
                description=(
                    "Predicated (`@%p`) barrier instruction. `bar.sync` must be "
                    "reached unconditionally by all threads in the block. A "
                    "predicated barrier means some threads skip it, causing deadlock "
                    "or an implicit timing dependency where the barrier fires only "
                    "when a scheduling-dependent condition is met."
                ),
            ))

    # ---- Source: threadIdx-conditional containing a barrier ----
    if ir.source_lines:
        for i, line in enumerate(ir.source_lines):
            if _BARRIER_SRC.search(line.text):
                window_back = ir.source_lines[max(0, i - 6): i]
                has_threadidx_if = any(
                    _THREADIDX_COND.search(l.text) for l in window_back
                )
                has_close = any('}' in l.text for l in window_back)
                if has_threadidx_if and not has_close:
                    flags.append(DeterminismFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=DetCategory.TIMING_DEPENDENCY,
                        severity=DetSeverity.HIGH,
                        description=(
                            "`__syncthreads()` inside a `threadIdx`-conditional block. "
                            "Threads that skip the condition miss the barrier, causing "
                            "deadlock or a timing race where completion depends on "
                            "which threads the scheduler activates."
                        ),
                    ))

    # ---- Source: implicit thread ordering via sequential index access ----
    # Pattern: sdata[threadIdx.x] read immediately after sdata[threadIdx.x-1] write
    # without syncthreads (neighbor dependency)
    if ir.source_lines:
        shared_write_pat = re.compile(r'(\w+)\s*\[\s*threadIdx\s*\.\s*x\s*\]\s*=')
        neighbor_read_pat = re.compile(r'(\w+)\s*\[\s*threadIdx\s*\.\s*x\s*[-+]\s*\d+\s*\]')
        last_smem_write: Dict[str, int] = {}   # var -> line index of last write
        last_sync_idx = -1

        for i, line in enumerate(ir.source_lines):
            if _BARRIER_SRC.search(line.text):
                last_sync_idx = i
            m_w = shared_write_pat.search(line.text)
            if m_w:
                last_smem_write[m_w.group(1)] = i
            m_r = neighbor_read_pat.search(line.text)
            if m_r:
                var = m_r.group(1)
                if var in last_smem_write and last_smem_write[var] > last_sync_idx:
                    flags.append(DeterminismFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=DetCategory.TIMING_DEPENDENCY,
                        severity=DetSeverity.MEDIUM,
                        description=(
                            f"Thread reads `{var}[threadIdx.x ± k]` — a neighbor's "
                            f"element — after `{var}[threadIdx.x]` was written, with no "
                            f"`__syncthreads()` in between. Intra-warp ordering is not "
                            f"guaranteed across SM generations; result depends on which "
                            f"warp half executes first."
                        ),
                    ))

    # ---- Source: blockIdx-gated reduction output without fence ----
    # Pattern: if (blockIdx.x == 0) { ... atomicAdd(global, local_sum); }
    # The final aggregation step implicitly assumes all blocks have finished.
    if ir.source_lines:
        for i, line in enumerate(ir.source_lines):
            if _BLOCKIDX_COND.search(line.text):
                # Check next 10 lines for atomic on a global pointer
                window = ir.source_lines[i: i + 10]
                has_atomic = any(_ATOMIC_SRC.search(l.text) for l in window)
                if has_atomic:
                    flags.append(DeterminismFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=DetCategory.TIMING_DEPENDENCY,
                        severity=DetSeverity.MEDIUM,
                        description=(
                            "`if (blockIdx.x == 0)` gate before a global atomic "
                            "reduction. This pattern assumes all other blocks have "
                            "already written their partial results — an implicit "
                            "inter-block ordering dependency. CUDA does not guarantee "
                            "block execution order within a single kernel launch; "
                            "use cooperative groups or a second kernel launch for safe "
                            "inter-block aggregation."
                        ),
                    ))

    return flags


# ---------------------------------------------------------------------------
# DeterminismPass
# ---------------------------------------------------------------------------

class DeterminismPass:
    """
    Static analysis pass for detecting non-determinism patterns.

    Parameters
    ----------
    kernelIR  : KernelIR     – populated IR for a single kernel.
    gpu_config: GPUConfig    – optional; currently used for warp size context.
    """

    def __init__(self, kernelIR: KernelIR, gpu_config: Optional[GPUConfig] = None):
        self.kernelIR   = kernelIR
        self.gpu_config = gpu_config or GPUConfig()

    def run(self) -> List[DeterminismFlag]:
        """Run all sub-checks; return scored, aggregated flags."""
        flags: List[DeterminismFlag] = []
        flags.extend(_check_race_conditions(self.kernelIR, self.gpu_config))
        flags.extend(_check_order_sensitive_ops(self.kernelIR, self.gpu_config))
        flags.extend(_check_timing_dependencies(self.kernelIR, self.gpu_config))
        _assign_scores(flags)
        # Deduplicate by (location, category, severity) — multiple checks
        # can independently detect the same PTX line for the same reason.
        seen = set()
        unique: List[DeterminismFlag] = []
        for f in flags:
            key = (f.location, f.category, f.severity)
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def nondeterminism_score(self, flags: Optional[List[DeterminismFlag]] = None) -> int:
        if flags is None:
            flags = self.run()
        return sum(f.score for f in flags)

    def category_breakdown(self, flags: Optional[List[DeterminismFlag]] = None) -> Dict[str, int]:
        if flags is None:
            flags = self.run()
        totals: Dict[str, int] = {c.value: 0 for c in DetCategory}
        for f in flags:
            totals[f.category.value] += f.score
        return totals


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def run_determinism_pass(
    kernel_name: str,
    ptx_path:    Optional[str] = None,
    source_text: Optional[str] = None,
    gpu_config:  Optional[GPUConfig] = None,
) -> List[DeterminismFlag]:
    ir = KernelIR(name=kernel_name)
    if ptx_path and os.path.isfile(ptx_path):
        ir.ptx_lines = _parse_lines(Path(ptx_path).read_text(errors="replace"))
        ir.has_ptx   = True
    if source_text:
        ir.source_lines = _parse_lines(source_text, strip_comments=True)
        ir.has_source   = True
    return DeterminismPass(ir, gpu_config).run()


def run_determinism_pass_on_directory(
    ptx_dir:    str,
    gpu_config: Optional[GPUConfig] = None,
) -> Dict[str, List[DeterminismFlag]]:
    results: Dict[str, List[DeterminismFlag]] = {}
    for ptx_file in sorted(Path(ptx_dir).glob("*.ptx")):
        name  = ptx_file.stem
        flags = run_determinism_pass(name, str(ptx_file), gpu_config=gpu_config)
        results[name] = flags
    return results


def write_determinism_report(
    results:  Dict[str, List[DeterminismFlag]],
    out_path: str = "output/report/determinism.md",
) -> None:
    lines = []
    lines.append("# CUDA Kernel Determinism Report\n\n")
    lines.append("_Static analysis for non-determinism and race-condition patterns._\n")
    lines.append("_Generated by determinism_pass.py_\n\n")

    lines.append("## Summary\n\n")
    lines.append("| Kernel | Score | High | Medium | Low |\n")
    lines.append("|--------|------:|-----:|-------:|----:|\n")
    for name, flags in results.items():
        score = sum(f.score for f in flags)
        h = sum(1 for f in flags if f.severity == DetSeverity.HIGH)
        m = sum(1 for f in flags if f.severity == DetSeverity.MEDIUM)
        l = sum(1 for f in flags if f.severity == DetSeverity.LOW)
        lines.append(f"| `{name}` | {score} | {h} | {m} | {l} |\n")
    lines.append("\n")

    lines.append("## Detailed Findings\n\n")
    for name, flags in results.items():
        if not flags:
            lines.append(f"### `{name}` — no issues found\n\n")
            continue
        score = sum(f.score for f in flags)
        lines.append(f"### `{name}`  (non-determinism score: {score})\n\n")
        for f in sorted(flags, key=lambda x: -x.score):
            lines.append(
                f"- **[{f.severity.value.upper()}]** `{f.category.value}` "
                f"@ `{f.location}`\n"
                f"  {f.description}\n\n"
            )
        lines.append("\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    print(f"Saved determinism report: {out_path}")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="CUDA kernel determinism / race-condition static analysis pass."
    )
    parser.add_argument("ptx_files", nargs="*",
                        help="PTX files (optional ::name suffix like ptx_parser.py).")
    parser.add_argument("--ptx-dir",    default=None,
                        help="Analyse all .ptx files in this directory.")
    parser.add_argument("--source",     default=None,
                        help="CUDA .cu source file for source-level checks.")
    parser.add_argument("--output",     default=None,
                        help="Markdown report path (default: output/report/determinism.md).")
    parser.add_argument("--json",       default=None,
                        help="Write JSON flags to this path.")
    parser.add_argument("--score-only", action="store_true",
                        help="Print kernel names and scores only.")
    args = parser.parse_args()

    source_text: Optional[str] = None
    if args.source and os.path.isfile(args.source):
        source_text = Path(args.source).read_text(errors="replace")

    results: Dict[str, List[DeterminismFlag]] = {}

    if args.ptx_dir:
        results = run_determinism_pass_on_directory(args.ptx_dir)
    elif args.ptx_files:
        for entry in args.ptx_files:
            path, name = (entry.split("::", 1) if "::" in entry
                          else (entry, Path(entry).stem))
            results[name] = run_determinism_pass(name, ptx_path=path,
                                                 source_text=source_text)
    else:
        default_dir = os.path.join("output", "ptx")
        if os.path.isdir(default_dir):
            results = run_determinism_pass_on_directory(default_dir)
        else:
            parser.print_help()
            sys.exit(0)

    sep = "=" * 70
    print(f"\n{sep}")
    print("  DETERMINISM PASS — NON-DETERMINISM FLAGS")
    print(sep)

    total = sum(len(f) for f in results.values())
    if total == 0:
        print("  No non-determinism issues detected.")
    else:
        for name, flags in results.items():
            score = sum(f.score for f in flags)
            if args.score_only:
                print(f"  {name:40s}  score={score}")
                continue
            print(f"\n  Kernel: {name}  (non-determinism score: {score})")
            print(f"  {'-' * 50}")
            if not flags:
                print("    No issues found.")
            for f in sorted(flags, key=lambda x: -x.score):
                print(f"    {f}")
    print(f"\n{sep}")

    out_path = args.output or os.path.join("output", "report", "determinism.md")
    write_determinism_report(results, out_path)

    if args.json:
        data = {k: [f.as_dict() for f in v] for k, v in results.items()}
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    _cli_main()
