"""
resource_pressure_pass.py
=========================
Static analysis pass that detects patterns where a kernel aggressively
consumes SM resources, reduces occupancy, or monopolises the GPU.

INPUTS
------
  kernelIR  – a KernelIR object (from portability_pass.py).
  gpu_config – optional SMConfig with SM resource limits.

OUTPUT
------
  List[ResourceFlag]  – structured resource-pressure issue records.

PASS STRUCTURE
--------------
  ResourcePressurePass.run() dispatches four independent sub-checks:

  A. _check_shared_memory    – smem allocation near/over SM limits
  B. _check_register_pressure – per-thread register usage, spill hints
  C. _check_occupancy        – thread/block patterns that reduce occupancy
  D. _check_sm_monopolization – long sequential warp serialization patterns

DESIGN NOTES
------------
- Pure static analysis.
- Reuses KernelIR / IRLine / GPUConfig / _parse_lines from portability_pass.
- ResourceFlag mirrors the flag design of the other two passes.
- Occupancy estimates are based on CUDA occupancy formulas (SM resource
  limits documented in the CUDA Programming Guide, Appendix H).
"""

from __future__ import annotations

import re
import json
import math
import os
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from portability_pass import (
    KernelIR, IRLine, GPUConfig, _parse_lines,
)


# ---------------------------------------------------------------------------
# SM resource limits per architecture generation
# Used for occupancy calculation.  Values from CUDA Programming Guide.
# ---------------------------------------------------------------------------

@dataclass
class SMConfig:
    """
    Resource limits for a single SM on the target GPU.

    Defaults represent a conservative mid-range target (SM 7.0 / Volta).
    Override for more specific analysis.
    """
    # Threads
    max_threads_per_sm:    int = 2048    # V100: 2048, A100: 2048, H100: 2048
    max_threads_per_block: int = 1024
    max_warps_per_sm:      int = 64      # V100: 64,  A100: 64,   H100: 64

    # Shared memory (bytes)
    shared_mem_per_sm:     int = 98304   # 96 KB (V100 default)
    shared_mem_per_block:  int = 49152   # 48 KB max per block on most SM

    # Registers
    registers_per_sm:      int = 65536   # 64 K regs (V100, A100, H100)
    max_regs_per_thread:   int = 255
    # nvcc default register cap (--maxrregcount); 0 = uncapped
    default_reg_cap:       int = 0

    # Blocks
    max_blocks_per_sm:     int = 32      # SM 8.0+; 16 on older SM

    # Warp size (always 32 on NVIDIA)
    warp_size:             int = 32

    def warps_from_block(self, block_size: int) -> int:
        return math.ceil(block_size / self.warp_size)

    def max_blocks_by_threads(self, block_size: int) -> int:
        if block_size == 0:
            return 0
        return min(
            self.max_blocks_per_sm,
            self.max_threads_per_sm // block_size,
        )

    def max_blocks_by_smem(self, smem_per_block: int) -> int:
        if smem_per_block == 0:
            return self.max_blocks_per_sm
        return min(self.max_blocks_per_sm,
                   self.shared_mem_per_sm // smem_per_block)

    def max_blocks_by_regs(self, block_size: int, regs_per_thread: int) -> int:
        if regs_per_thread == 0:
            return self.max_blocks_per_sm
        regs_per_block = block_size * regs_per_thread
        # Round up regs_per_block to next multiple of 256 (hardware granularity)
        regs_per_block = math.ceil(regs_per_block / 256) * 256
        if regs_per_block == 0:
            return self.max_blocks_per_sm
        return min(self.max_blocks_per_sm,
                   self.registers_per_sm // regs_per_block)

    def theoretical_occupancy(
        self,
        block_size:       int,
        smem_per_block:   int = 0,
        regs_per_thread:  int = 0,
    ) -> float:
        """
        Fraction of maximum warps that can reside on one SM simultaneously.
        Returns a value in [0.0, 1.0].
        """
        active_by_threads = self.max_blocks_by_threads(block_size)
        active_by_smem    = self.max_blocks_by_smem(smem_per_block)
        active_by_regs    = self.max_blocks_by_regs(block_size, regs_per_thread)
        active_blocks     = min(active_by_threads, active_by_smem, active_by_regs)
        active_warps      = active_blocks * self.warps_from_block(block_size)
        return min(active_warps / self.max_warps_per_sm, 1.0)


# Default SM config (conservative: Volta / SM 7.0)
_DEFAULT_SM = SMConfig()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ResSeverity(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class ResCategory(str, Enum):
    SHARED_MEMORY    = "shared_memory_pressure"
    REGISTER_PRESSURE = "register_pressure"
    OCCUPANCY        = "occupancy_problem"
    SM_MONOPOLIZATION = "sm_monopolization"


# ---------------------------------------------------------------------------
# Flag dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResourceFlag:
    """A single resource-pressure issue detected in a kernel."""
    kernel_name: str
    location:    str
    category:    ResCategory
    severity:    ResSeverity
    description: str
    score:       int = 0

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

_SEVERITY_SCORE: Dict[ResSeverity, int] = {
    ResSeverity.HIGH:   10,
    ResSeverity.MEDIUM:  5,
    ResSeverity.LOW:     2,
}


def _assign_scores(flags: List[ResourceFlag]) -> None:
    for f in flags:
        f.score = _SEVERITY_SCORE.get(f.severity, 1)


# ---------------------------------------------------------------------------
# PTX / source patterns
# ---------------------------------------------------------------------------

# ---- Shared memory ----
_SHARED_DECL_PTX  = re.compile(r'\.shared\s+\.align\s+\d+\s+\.b8\s+\w+\s*\[(\d+)\]')
_SHARED_DECL_SRC  = re.compile(r'__shared__\s+(\w+)\s+(\w+)\s*\[([^\]]+)\]')
_TYPE_SIZES       = {"char": 1, "short": 2, "int": 4, "float": 4,
                     "double": 8, "long": 8, "uint": 4, "int4": 16,
                     "float4": 16, "float2": 8, "int2": 8, "half": 2}

# ---- Register usage hints in PTX ----
_REG_DECL_PTX     = re.compile(r'\.reg\s+\.(b|u|s|f)(\d+)\s+(%\w+)<(\d+)>')
_PRAGMA_MAXREG    = re.compile(r'#pragma\s+nounroll|maxrregcount\s*=\s*(\d+)')
_SPILL_PTX        = re.compile(r'\bld\.local\b|\bst\.local\b')  # local = register spill

# ---- Block / grid size hints in source ----
_LAUNCH_BOUNDS    = re.compile(
    r'__launch_bounds__\s*\(\s*(\d+)(?:\s*,\s*(\d+))?\s*\)'
)
_BLOCK_DIM_SRC    = re.compile(r'<<<\s*\w+\s*,\s*(\d+)\s*>>>')
_BLOCK_DIM2_SRC   = re.compile(r'dim3\s+\w+\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*(?:,\s*(\d+))?)?\s*\)')

# ---- Divergence / serialization proxies ----
_BRANCH_PTX       = re.compile(r'\bbra\b|\bbra\.uni\b')
_DIVERGE_SRC      = re.compile(r'threadIdx\s*\.\s*[xyz]\s*%\s*\d+')
_SYNC_PTX         = re.compile(r'\bbar\.sync\b')
_LONG_LOOP_SRC    = re.compile(
    r'for\s*\(.*;\s*\w+\s*<\s*(\d+)\s*;'
    r'|while\s*\('
)
# PTX: many consecutive non-barrier instructions — proxy for long sequential warp execution
_INSTR_PTX        = re.compile(r'^\s+(?!bar|bra|ret|exit|\.)[\w.]+')

# ---- Atomic operations (potential bottleneck / serialization) ----
_ATOM_GLOBAL_PTX  = re.compile(r'\batom\.global\b')
_ATOM_SRC         = re.compile(r'\batomic\w+\s*\(')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_shared_mem_ptx(ptx_lines: List[IRLine]) -> int:
    """Sum `.shared .alignN .b8 name[bytes]` directives in PTX."""
    total = 0
    for line in ptx_lines:
        m = _SHARED_DECL_PTX.search(line.text)
        if m:
            total += int(m.group(1))
    return total


def _estimate_shared_mem_src(source_lines: List[IRLine]) -> Tuple[int, List[str]]:
    """
    Estimate shared memory from __shared__ declarations.
    Returns (total_bytes, list_of_descriptions).
    """
    total = 0
    descs = []
    for line in source_lines:
        m = _SHARED_DECL_SRC.search(line.text)
        if m:
            typ, name, count_expr = m.group(1), m.group(2), m.group(3)
            try:
                count = int(count_expr)
            except ValueError:
                # Non-constant size (template param / #define) — flag separately
                descs.append(
                    f"Dynamic shared memory `{name}[{count_expr}]` — cannot "
                    f"determine size statically; verify it fits in SM limits."
                )
                continue
            elem_size = _TYPE_SIZES.get(typ, 4)
            nbytes = count * elem_size
            total += nbytes
            descs.append(f"`{name}[{count}]` ({nbytes} bytes)")
    return total, descs


def _count_registers_ptx(ptx_lines: List[IRLine]) -> int:
    """
    Sum all register declarations in PTX (.reg .f32 %r<N> counts N registers).
    Returns total register count (not per-thread — PTX declares one set per thread).
    """
    total = 0
    for line in ptx_lines:
        m = _REG_DECL_PTX.search(line.text)
        if m:
            total += int(m.group(4))
    return total


def _extract_block_size_src(source_lines: List[IRLine]) -> Optional[int]:
    """Try to extract a static block size from <<< >>> or dim3 in source."""
    for line in source_lines:
        m = _BLOCK_DIM_SRC.search(line.text)
        if m:
            return int(m.group(1))
        m = _LAUNCH_BOUNDS.search(line.text)
        if m:
            return int(m.group(1))
    return None


def _count_consecutive_non_barrier(ptx_lines: List[IRLine]) -> int:
    """Return the longest run of instruction lines between two bar.sync calls."""
    max_run = 0
    current = 0
    for line in ptx_lines:
        if _SYNC_PTX.search(line.text):
            max_run = max(max_run, current)
            current = 0
        elif _INSTR_PTX.match(line.raw):
            current += 1
    return max(max_run, current)


# ---------------------------------------------------------------------------
# A. Shared memory pressure checker
# ---------------------------------------------------------------------------

def _check_shared_memory(
    ir: KernelIR, sm: SMConfig
) -> List[ResourceFlag]:
    flags: List[ResourceFlag] = []
    kernel = ir.name

    # ---- Estimate from PTX directive ----
    ptx_smem = _estimate_shared_mem_ptx(ir.ptx_lines)

    # ---- Estimate from source declarations ----
    src_smem, src_descs = _estimate_shared_mem_src(ir.source_lines)

    # Flag dynamic (unknown-size) shared arrays
    for desc in src_descs:
        if "Dynamic" in desc:
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="source:smem_decl",
                category=ResCategory.SHARED_MEMORY,
                severity=ResSeverity.MEDIUM,
                description=desc,
            ))

    # Use the larger of the two estimates (PTX is post-compiler; src is pre-compiler)
    smem_estimate = max(ptx_smem, src_smem)
    if smem_estimate == 0:
        return flags  # No shared memory used

    # ---- Check against per-block limit ----
    if smem_estimate > sm.shared_mem_per_block:
        flags.append(ResourceFlag(
            kernel_name=kernel,
            location="ptx:smem_total",
            category=ResCategory.SHARED_MEMORY,
            severity=ResSeverity.HIGH,
            description=(
                f"Estimated shared memory per block ({smem_estimate:,} bytes) "
                f"exceeds the default per-block limit "
                f"({sm.shared_mem_per_block:,} bytes). "
                f"Kernel will fail to launch unless the limit is raised via "
                f"`cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`."
            ),
        ))
    elif smem_estimate > sm.shared_mem_per_block * 0.75:
        flags.append(ResourceFlag(
            kernel_name=kernel,
            location="ptx:smem_total",
            category=ResCategory.SHARED_MEMORY,
            severity=ResSeverity.MEDIUM,
            description=(
                f"Shared memory per block ({smem_estimate:,} bytes) is "
                f"{100*smem_estimate//sm.shared_mem_per_block}% of the "
                f"per-block limit ({sm.shared_mem_per_block:,} bytes). "
                f"This limits the number of concurrent blocks per SM and "
                f"reduces occupancy."
            ),
        ))
    elif smem_estimate > sm.shared_mem_per_block * 0.5:
        flags.append(ResourceFlag(
            kernel_name=kernel,
            location="ptx:smem_total",
            category=ResCategory.SHARED_MEMORY,
            severity=ResSeverity.LOW,
            description=(
                f"Shared memory per block ({smem_estimate:,} bytes) is "
                f"{100*smem_estimate//sm.shared_mem_per_block}% of the "
                f"per-block limit. Monitor occupancy if block count per SM "
                f"drops below 2."
            ),
        ))

    # ---- Check how many blocks fit per SM given shared memory ----
    max_blocks_smem = sm.max_blocks_by_smem(smem_estimate)
    if max_blocks_smem < 2:
        flags.append(ResourceFlag(
            kernel_name=kernel,
            location="ptx:smem_total",
            category=ResCategory.OCCUPANCY,
            severity=ResSeverity.HIGH,
            description=(
                f"Shared memory usage ({smem_estimate:,} bytes/block) limits "
                f"the SM to {max_blocks_smem} concurrent block(s). "
                f"With fewer than 2 blocks per SM, the GPU cannot hide latency "
                f"by switching between blocks, severely reducing throughput."
            ),
        ))

    return flags


# ---------------------------------------------------------------------------
# B. Register pressure checker
# ---------------------------------------------------------------------------

def _check_register_pressure(
    ir: KernelIR, sm: SMConfig
) -> List[ResourceFlag]:
    flags: List[ResourceFlag] = []
    kernel = ir.name

    # ---- Spill detection: ld.local / st.local in PTX ----
    spill_loads  = [l for l in ir.ptx_lines if re.search(r'\bld\.local\b', l.text)]
    spill_stores = [l for l in ir.ptx_lines if re.search(r'\bst\.local\b', l.text)]
    if spill_loads or spill_stores:
        n_spill = len(spill_loads) + len(spill_stores)
        sev = ResSeverity.HIGH if n_spill > 10 else ResSeverity.MEDIUM
        first_loc = f"ptx:L{spill_loads[0].lineno}" if spill_loads else f"ptx:L{spill_stores[0].lineno}"
        flags.append(ResourceFlag(
            kernel_name=kernel,
            location=first_loc,
            category=ResCategory.REGISTER_PRESSURE,
            severity=sev,
            description=(
                f"Register spill detected: {len(spill_stores)} `st.local` and "
                f"{len(spill_loads)} `ld.local` instructions. "
                f"Local memory is off-chip (same latency as global memory). "
                f"Spilling typically indicates register usage exceeds the SM "
                f"register file capacity; reduce live variables or add "
                f"`__launch_bounds__` to cap register usage."
            ),
        ))

    # ---- Total register count from PTX declarations ----
    total_regs = _count_registers_ptx(ir.ptx_lines)
    if total_regs > 0:
        if total_regs > sm.max_regs_per_thread:
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="ptx:reg_decl",
                category=ResCategory.REGISTER_PRESSURE,
                severity=ResSeverity.HIGH,
                description=(
                    f"PTX declares {total_regs} registers per thread, which exceeds "
                    f"the hardware maximum of {sm.max_regs_per_thread} per thread. "
                    f"The compiler must spill the excess; add `__launch_bounds__` or "
                    f"refactor the kernel to reduce live values."
                ),
            ))
        elif total_regs > 64:
            # > 64 regs/thread caps occupancy to 50% on most SM generations
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="ptx:reg_decl",
                category=ResCategory.REGISTER_PRESSURE,
                severity=ResSeverity.MEDIUM,
                description=(
                    f"PTX declares {total_regs} registers per thread. "
                    f"On SM 7.0–9.0 (64 K regs / SM), more than 64 regs/thread "
                    f"limits concurrent warps to <= 32 (50% occupancy). "
                    f"Consider `__launch_bounds__(block_size, min_blocks)` to give "
                    f"the compiler a register budget."
                ),
            ))
        elif total_regs > 32:
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="ptx:reg_decl",
                category=ResCategory.REGISTER_PRESSURE,
                severity=ResSeverity.LOW,
                description=(
                    f"PTX declares {total_regs} registers per thread. "
                    f"This is within bounds but will limit occupancy to "
                    f"~{100 * sm.registers_per_sm // (total_regs * sm.max_threads_per_sm * sm.warp_size // sm.max_warps_per_sm) // sm.warp_size}% "
                    f"on a 64K-register SM. Monitor with Nsight Compute."
                ),
            ))

    # ---- __launch_bounds__ absent when high register count ----
    has_launch_bounds = any(
        _LAUNCH_BOUNDS.search(l.text) for l in ir.source_lines
    )
    if total_regs > 32 and not has_launch_bounds and ir.has_source:
        flags.append(ResourceFlag(
            kernel_name=kernel,
            location="source:launch_bounds",
            category=ResCategory.REGISTER_PRESSURE,
            severity=ResSeverity.LOW,
            description=(
                f"`__launch_bounds__` annotation is absent. With {total_regs} "
                f"PTX registers per thread, adding "
                f"`__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)` lets the compiler "
                f"limit register allocation to guarantee a minimum occupancy level."
            ),
        ))

    return flags


# ---------------------------------------------------------------------------
# C. Occupancy checker
# ---------------------------------------------------------------------------

def _check_occupancy(
    ir: KernelIR, sm: SMConfig
) -> List[ResourceFlag]:
    flags: List[ResourceFlag] = []
    kernel = ir.name

    # Gather available sizing hints
    block_size  = _extract_block_size_src(ir.source_lines)
    total_regs  = _count_registers_ptx(ir.ptx_lines)
    src_smem, _ = _estimate_shared_mem_src(ir.source_lines)
    ptx_smem    = _estimate_shared_mem_ptx(ir.ptx_lines)
    smem_est    = max(src_smem, ptx_smem)

    # ---- Block size sanity checks (source-only) ----
    if block_size is not None:
        if block_size > sm.max_threads_per_block:
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="source:launch_config",
                category=ResCategory.OCCUPANCY,
                severity=ResSeverity.HIGH,
                description=(
                    f"Block size {block_size} exceeds the hardware maximum of "
                    f"{sm.max_threads_per_block} threads per block. Kernel will "
                    f"fail to launch."
                ),
            ))
        elif block_size < 64:
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="source:launch_config",
                category=ResCategory.OCCUPANCY,
                severity=ResSeverity.HIGH,
                description=(
                    f"Block size of {block_size} threads is very small. "
                    f"Each block uses {math.ceil(block_size/sm.warp_size)} warp(s); "
                    f"the SM needs {sm.max_warps_per_sm} warps for full occupancy. "
                    f"Small blocks prevent the scheduler from hiding latency. "
                    f"Typical effective block sizes are 128–512."
                ),
            ))
        elif block_size % sm.warp_size != 0:
            partial = block_size % sm.warp_size
            wasted  = sm.warp_size - partial
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="source:launch_config",
                category=ResCategory.OCCUPANCY,
                severity=ResSeverity.MEDIUM,
                description=(
                    f"Block size {block_size} is not a multiple of warp size "
                    f"({sm.warp_size}). The last warp has only {partial} active "
                    f"threads; {wasted} thread slots are wasted per block."
                ),
            ))

        # ---- Full occupancy estimate ----
        if total_regs > 0 or smem_est > 0:
            occ = sm.theoretical_occupancy(block_size, smem_est, total_regs)
            if occ < 0.25:
                flags.append(ResourceFlag(
                    kernel_name=kernel,
                    location="ptx:occupancy",
                    category=ResCategory.OCCUPANCY,
                    severity=ResSeverity.HIGH,
                    description=(
                        f"Estimated theoretical occupancy is {occ:.0%} "
                        f"(block={block_size}, smem={smem_est} B, regs={total_regs}). "
                        f"Below 25% occupancy severely limits the SM's ability to "
                        f"hide memory and arithmetic latency via warp switching."
                    ),
                ))
            elif occ < 0.50:
                flags.append(ResourceFlag(
                    kernel_name=kernel,
                    location="ptx:occupancy",
                    category=ResCategory.OCCUPANCY,
                    severity=ResSeverity.MEDIUM,
                    description=(
                        f"Estimated theoretical occupancy is {occ:.0%} "
                        f"(block={block_size}, smem={smem_est} B, regs={total_regs}). "
                        f"Consider reducing shared memory or register usage to "
                        f"allow more concurrent blocks per SM."
                    ),
                ))

    # ---- Warp non-multiple warning (when no explicit launch config found) ----
    # Check PTX for .maxntid directive (embedded block size hint)
    for line in ir.ptx_lines:
        m = re.search(r'\.maxntid\s+(\d+)', line.text)
        if m:
            hint_bs = int(m.group(1))
            if hint_bs % sm.warp_size != 0:
                flags.append(ResourceFlag(
                    kernel_name=kernel,
                    location=f"ptx:L{line.lineno}",
                    category=ResCategory.OCCUPANCY,
                    severity=ResSeverity.MEDIUM,
                    description=(
                        f"PTX `.maxntid {hint_bs}` (embedded block size hint) is not "
                        f"a multiple of warp size {sm.warp_size}. "
                        f"The last partial warp wastes "
                        f"{sm.warp_size - hint_bs % sm.warp_size} thread slots."
                    ),
                ))

    return flags


# ---------------------------------------------------------------------------
# D. SM monopolization checker
# ---------------------------------------------------------------------------

def _check_sm_monopolization(
    ir: KernelIR, sm: SMConfig
) -> List[ResourceFlag]:
    flags: List[ResourceFlag] = []
    kernel = ir.name

    # ---- Long sequential instruction sequences between barriers ----
    max_run = _count_consecutive_non_barrier(ir.ptx_lines)
    if max_run > 500:
        flags.append(ResourceFlag(
            kernel_name=kernel,
            location="ptx:long_sequence",
            category=ResCategory.SM_MONOPOLIZATION,
            severity=ResSeverity.HIGH,
            description=(
                f"Longest contiguous PTX instruction sequence without a barrier: "
                f"{max_run} instructions. Very long uninterrupted sequences prevent "
                f"the warp scheduler from interleaving other warps, reducing the "
                f"SM's ability to hide instruction latency and starving concurrent "
                f"kernels or warps."
            ),
        ))
    elif max_run > 200:
        flags.append(ResourceFlag(
            kernel_name=kernel,
            location="ptx:long_sequence",
            category=ResCategory.SM_MONOPOLIZATION,
            severity=ResSeverity.LOW,
            description=(
                f"Longest contiguous PTX instruction sequence: {max_run} instructions. "
                f"This is a latency-hiding concern for kernels that share the SM "
                f"with other work (e.g., via CUDA streams or MPS)."
            ),
        ))

    # ---- High global atomic density — contention point on SM ----
    atom_count = sum(1 for l in ir.ptx_lines if _ATOM_GLOBAL_PTX.search(l.text))
    total_instr = sum(1 for l in ir.ptx_lines
                      if not l.text.startswith('.') and not l.text.endswith(':'))
    if total_instr > 0:
        atom_ratio = atom_count / total_instr
        if atom_ratio > 0.10:
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="ptx:atom_density",
                category=ResCategory.SM_MONOPOLIZATION,
                severity=ResSeverity.HIGH,
                description=(
                    f"{atom_count} global atomic instructions out of {total_instr} "
                    f"total ({atom_ratio:.0%}). Heavy atomic use serialises memory "
                    f"transactions across all threads accessing the same address, "
                    f"effectively monopolising the L2 / DRAM interconnect and "
                    f"starving other warps."
                ),
            ))
        elif atom_ratio > 0.04:
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="ptx:atom_density",
                category=ResCategory.SM_MONOPOLIZATION,
                severity=ResSeverity.MEDIUM,
                description=(
                    f"{atom_count} global atomic instructions ({atom_ratio:.0%} of total). "
                    f"Moderate atomic density may cause memory-bus contention on "
                    f"high-thread-count launches. Consider warp-level pre-reduction "
                    f"before the global atomic to reduce contention."
                ),
            ))

    # ---- Source: divergent per-thread branch covering many iterations ----
    if ir.source_lines:
        for line in ir.source_lines:
            m = _LONG_LOOP_SRC.search(line.text)
            if m and m.group(1) and int(m.group(1)) > 1000:
                flags.append(ResourceFlag(
                    kernel_name=kernel,
                    location=f"source:L{line.lineno}",
                    category=ResCategory.SM_MONOPOLIZATION,
                    severity=ResSeverity.MEDIUM,
                    description=(
                        f"Loop with large static bound ({m.group(1)} iterations) detected. "
                        f"If different threads execute different loop counts (data-dependent "
                        f"exit), the warp serialises; threads that finish early remain idle "
                        f"until the slowest thread exits, monopolising the warp slot."
                    ),
                ))

        # ---- Source: threadIdx-divergent loops ----
        for i, line in enumerate(ir.source_lines):
            if _DIVERGE_SRC.search(line.text):
                # Check if this is inside a loop (look back a few lines)
                window = ir.source_lines[max(0, i - 5): i + 1]
                in_loop = any(_LONG_LOOP_SRC.search(l.text) for l in window)
                if in_loop:
                    flags.append(ResourceFlag(
                        kernel_name=kernel,
                        location=f"source:L{line.lineno}",
                        category=ResCategory.SM_MONOPOLIZATION,
                        severity=ResSeverity.HIGH,
                        description=(
                            "Thread-divergent expression (`threadIdx.x % N`) inside a "
                            "loop. Each warp iteration serialises the two sub-groups, "
                            "cutting effective throughput in half and extending SM "
                            "occupancy time for this warp at the expense of others."
                        ),
                    ))

    # ---- PTX: high branch density (proxy for divergence) ----
    branch_count = sum(1 for l in ir.ptx_lines if _BRANCH_PTX.search(l.text))
    if total_instr > 0:
        branch_ratio = branch_count / total_instr
        if branch_ratio > 0.15:
            flags.append(ResourceFlag(
                kernel_name=kernel,
                location="ptx:branch_density",
                category=ResCategory.SM_MONOPOLIZATION,
                severity=ResSeverity.MEDIUM,
                description=(
                    f"Branch instruction density {branch_ratio:.0%} "
                    f"({branch_count}/{total_instr}). High branch density in PTX "
                    f"indicates frequent control flow changes; if warps diverge at "
                    f"these branches, the SM serialises both paths, reducing the "
                    f"effective thread count and leaving other warps starved."
                ),
            ))

    return flags


# ---------------------------------------------------------------------------
# ResourcePressurePass
# ---------------------------------------------------------------------------

class ResourcePressurePass:
    """
    Static analysis pass for detecting SM resource pressure patterns.

    Parameters
    ----------
    kernelIR  : KernelIR   – populated IR for a single kernel.
    gpu_config: GPUConfig  – used for warp size; ignored in favour of SMConfig.
    sm_config : SMConfig   – SM resource limits. Defaults to Volta (SM 7.0).
    """

    def __init__(
        self,
        kernelIR:   KernelIR,
        gpu_config: Optional[GPUConfig] = None,
        sm_config:  Optional[SMConfig]  = None,
    ):
        self.kernelIR   = kernelIR
        self.gpu_config = gpu_config or GPUConfig()
        self.sm         = sm_config or _DEFAULT_SM

    def run(self) -> List[ResourceFlag]:
        flags: List[ResourceFlag] = []
        flags.extend(_check_shared_memory(self.kernelIR, self.sm))
        flags.extend(_check_register_pressure(self.kernelIR, self.sm))
        flags.extend(_check_occupancy(self.kernelIR, self.sm))
        flags.extend(_check_sm_monopolization(self.kernelIR, self.sm))
        _assign_scores(flags)
        # Deduplicate by (location, category)
        seen: set = set()
        unique: List[ResourceFlag] = []
        for f in flags:
            key = (f.location, f.category)
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def resource_score(self, flags: Optional[List[ResourceFlag]] = None) -> int:
        if flags is None:
            flags = self.run()
        return sum(f.score for f in flags)

    def category_breakdown(self, flags: Optional[List[ResourceFlag]] = None) -> Dict[str, int]:
        if flags is None:
            flags = self.run()
        totals: Dict[str, int] = {c.value: 0 for c in ResCategory}
        for f in flags:
            totals[f.category.value] += f.score
        return totals

    def occupancy_summary(self, block_size: Optional[int] = None) -> Dict[str, Any]:
        """Return a dict with occupancy metrics for reporting."""
        bs = block_size or _extract_block_size_src(self.kernelIR.source_lines)
        if bs is None:
            return {"block_size": None, "occupancy": None, "note": "block size unknown"}
        src_smem, _ = _estimate_shared_mem_src(self.kernelIR.source_lines)
        ptx_smem    = _estimate_shared_mem_ptx(self.kernelIR.ptx_lines)
        smem        = max(src_smem, ptx_smem)
        regs        = _count_registers_ptx(self.kernelIR.ptx_lines)
        occ         = self.sm.theoretical_occupancy(bs, smem, regs)
        return {
            "block_size":        bs,
            "smem_bytes":        smem,
            "regs_per_thread":   regs,
            "theoretical_occ":   round(occ, 3),
            "active_warps_est":  int(occ * self.sm.max_warps_per_sm),
            "max_warps_per_sm":  self.sm.max_warps_per_sm,
        }


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def run_resource_pass(
    kernel_name: str,
    ptx_path:    Optional[str]    = None,
    source_text: Optional[str]    = None,
    sm_config:   Optional[SMConfig] = None,
) -> List[ResourceFlag]:
    ir = KernelIR(name=kernel_name)
    if ptx_path and os.path.isfile(ptx_path):
        ir.ptx_lines = _parse_lines(Path(ptx_path).read_text(errors="replace"))
        ir.has_ptx   = True
    if source_text:
        ir.source_lines = _parse_lines(source_text, strip_comments=True)
        ir.has_source   = True
    return ResourcePressurePass(ir, sm_config=sm_config).run()


def run_resource_pass_on_directory(
    ptx_dir:   str,
    sm_config: Optional[SMConfig] = None,
) -> Dict[str, List[ResourceFlag]]:
    results: Dict[str, List[ResourceFlag]] = {}
    for ptx_file in sorted(Path(ptx_dir).glob("*.ptx")):
        name  = ptx_file.stem
        flags = run_resource_pass(name, str(ptx_file), sm_config=sm_config)
        results[name] = flags
    return results


def write_resource_report(
    results:  Dict[str, List[ResourceFlag]],
    out_path: str = "output/report/resource_pressure.md",
) -> None:
    lines = []
    lines.append("# CUDA Kernel Resource Pressure Report\n\n")
    lines.append("_Static analysis for SM resource consumption and occupancy._\n")
    lines.append("_Generated by resource_pressure_pass.py_\n\n")

    lines.append("## Summary\n\n")
    lines.append("| Kernel | Score | High | Medium | Low |\n")
    lines.append("|--------|------:|-----:|-------:|----:|\n")
    for name, flags in results.items():
        score = sum(f.score for f in flags)
        h = sum(1 for f in flags if f.severity == ResSeverity.HIGH)
        m = sum(1 for f in flags if f.severity == ResSeverity.MEDIUM)
        l = sum(1 for f in flags if f.severity == ResSeverity.LOW)
        lines.append(f"| `{name}` | {score} | {h} | {m} | {l} |\n")
    lines.append("\n")

    lines.append("## Detailed Findings\n\n")
    for name, flags in results.items():
        if not flags:
            lines.append(f"### `{name}` — no issues found\n\n")
            continue
        score = sum(f.score for f in flags)
        lines.append(f"### `{name}`  (resource score: {score})\n\n")
        for f in sorted(flags, key=lambda x: -x.score):
            lines.append(
                f"- **[{f.severity.value.upper()}]** `{f.category.value}` "
                f"@ `{f.location}`\n"
                f"  {f.description}\n\n"
            )
        lines.append("\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    print(f"Saved resource pressure report: {out_path}")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="CUDA kernel resource pressure / occupancy static analysis pass."
    )
    parser.add_argument("ptx_files", nargs="*",
                        help="PTX files (optional ::name suffix).")
    parser.add_argument("--ptx-dir",        default=None,
                        help="Analyse all .ptx files in this directory.")
    parser.add_argument("--source",         default=None,
                        help="CUDA .cu source file for source-level checks.")
    parser.add_argument("--output",         default=None,
                        help="Markdown report path.")
    parser.add_argument("--json",           default=None,
                        help="Write JSON flags to this path.")
    parser.add_argument("--score-only",     action="store_true",
                        help="Print kernel names and scores only.")
    parser.add_argument("--max-threads-sm", type=int, default=2048,
                        help="Max threads per SM (default 2048).")
    parser.add_argument("--shared-mem-sm",  type=int, default=98304,
                        help="Shared memory per SM in bytes (default 98304 = 96 KB).")
    parser.add_argument("--regs-sm",        type=int, default=65536,
                        help="Register file size per SM (default 65536).")
    args = parser.parse_args()

    sm = SMConfig(
        max_threads_per_sm=args.max_threads_sm,
        shared_mem_per_sm=args.shared_mem_sm,
        registers_per_sm=args.regs_sm,
    )

    source_text: Optional[str] = None
    if args.source and os.path.isfile(args.source):
        source_text = Path(args.source).read_text(errors="replace")

    results: Dict[str, List[ResourceFlag]] = {}

    if args.ptx_dir:
        results = run_resource_pass_on_directory(args.ptx_dir, sm_config=sm)
    elif args.ptx_files:
        for entry in args.ptx_files:
            path, name = (entry.split("::", 1) if "::" in entry
                          else (entry, Path(entry).stem))
            results[name] = run_resource_pass(name, ptx_path=path,
                                              source_text=source_text, sm_config=sm)
    else:
        default_dir = os.path.join("output", "ptx")
        if os.path.isdir(default_dir):
            results = run_resource_pass_on_directory(default_dir, sm_config=sm)
        else:
            parser.print_help()
            sys.exit(0)

    sep = "=" * 70
    print(f"\n{sep}")
    print("  RESOURCE PRESSURE PASS — FLAGS")
    print(sep)

    total = sum(len(f) for f in results.values())
    if total == 0:
        print("  No resource pressure issues detected.")
    else:
        for name, flags in results.items():
            score = sum(f.score for f in flags)
            if args.score_only:
                print(f"  {name:40s}  score={score}")
                continue
            print(f"\n  Kernel: {name}  (resource score: {score})")
            print(f"  {'-' * 50}")
            if not flags:
                print("    No issues found.")
            for f in sorted(flags, key=lambda x: -x.score):
                print(f"    {f}")
    print(f"\n{sep}")

    out_path = args.output or os.path.join("output", "report", "resource_pressure.md")
    write_resource_report(results, out_path)

    if args.json:
        data = {k: [f.as_dict() for f in v] for k, v in results.items()}
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.json}")


if __name__ == "__main__":
    _cli_main()
