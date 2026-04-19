# CUTLASS Group GEMM Developer Guide

This document provides a comprehensive introduction to CUTLASS and explains how CUTLASS 2.0 implements Group GEMM, which is a critical operation for MoE (Mixture of Experts) models in SGLang.

## Table of Contents

- [What is CUTLASS?](#what-is-cutlass)
- [What is GEMM?](#what-is-gemm)
- [What is Group GEMM?](#what-is-group-gemm)
- [CUTLASS 2.0 Group GEMM Implementation](#cutlass-20-group-gemm-implementation)
  - [Overview](#overview)
  - [Step 1: Problem Description Array](#step-1-problem-description-array)
  - [Step 2: Tile Flattening](#step-2-tile-flattening)
  - [Step 3: Prefix Sum Mapping](#step-3-prefix-sum-mapping)
  - [Step 4: Kernel Execution](#step-4-kernel-execution)
- [Scheduling Modes](#scheduling-modes)
- [Code Example](#code-example)
- [Connection to SGLang](#connection-to-sglang)

## What is CUTLASS?

[CUTLASS](https://github.com/NVIDIA/cutlass) (CUDA Templates for Linear Algebra Subroutines) is an open-source C++ template library from NVIDIA for writing high-performance matrix multiplication (GEMM) kernels on GPUs.

You can think of it as:
- **cuBLAS** is a "black-box" library — you call a function and it computes matrix multiply, but you cannot modify the internals.
- **CUTLASS** is a "white-box" library — it exposes every layer of a high-performance GEMM kernel as C++ templates, allowing you to customize data types, tile sizes, pipeline depth, epilogue operations, and more.

Analogy: cuBLAS is like buying a pre-built computer. CUTLASS gives you high-quality components to assemble your own, where you can swap the GPU, adjust memory, and tune the configuration.

## What is GEMM?

GEMM (General Matrix Multiply) is the standard matrix multiplication operation:

```
C = α × A × B + β × C
```

Where A is an M×K matrix, B is a K×N matrix, and C is an M×N matrix. This is the most fundamental computation in deep learning — fully-connected layers, attention mechanisms, and convolutions all boil down to GEMM at the lowest level.

### How GPU GEMM Works (Tiling)

A single GEMM on GPU is computed using **tiling**: the output matrix C (M×N) is divided into small blocks (tiles), and each CUDA threadblock computes one tile.

```
Output matrix C (M×N):
┌──────────┬──────────┬──────────┬──────────┐
│  tile 0  │  tile 1  │  tile 2  │  tile 3  │  ← each tile computed by
│ (128×128)│ (128×128)│ (128×128)│ (128×128)│    one CUDA threadblock
├──────────┼──────────┼──────────┼──────────┤
│  tile 4  │  tile 5  │  tile 6  │  tile 7  │
│ (128×128)│ (128×128)│ (128×128)│ (128×128)│
└──────────┴──────────┴──────────┴──────────┘
```

Each threadblock:
1. Loads a tile of A and B from global memory into shared memory
2. Performs warp-level matrix multiply-accumulate (MMA) using Tensor Cores
3. Writes the result tile back to global memory

## What is Group GEMM?

Group GEMM executes **multiple independent matrix multiplications** simultaneously:

```
For i = 0, 1, ..., G-1:
    C[i] = α × A[i] × B[i] + β × C[i]
```

Each sub-problem can have **different M, N, K dimensions**.

### Why Do We Need Group GEMM?

The most important use case is **MoE (Mixture of Experts) models** (e.g., Mixtral, DeepSeek-V2/V3):

- An MoE model has multiple experts, each being an independent fully-connected layer (= one GEMM)
- A router assigns different tokens to different experts; each expert may process a different number of tokens
- You need to execute multiple **differently-sized** GEMMs simultaneously

If you execute these small GEMMs one by one (serially), GPU utilization is low because each individual GEMM may be too small to fill the GPU. Group GEMM **batches all these small GEMMs into a single kernel launch**, maximizing GPU utilization.

```
Serial execution (wasteful):          Group GEMM (efficient):
┌─────────┐                           ┌─────────────────────────┐
│ Expert 0 │ ← GPU underutilized      │ Expert 0 │ Expert 1 │...│
├─────────┤                           │  tiles   │  tiles   │   │
│ Expert 1 │ ← GPU underutilized      │          │          │   │
├─────────┤                           │    All experts run   │   │
│ Expert 2 │ ← GPU underutilized      │    in ONE kernel     │   │
└─────────┘                           └─────────────────────────┘
  3 kernel launches                     1 kernel launch, GPU fully utilized
```

## CUTLASS 2.0 Group GEMM Implementation

### Overview

CUTLASS 2.0 implements Group GEMM via a **host-side problem visitor** + **device-side grouped kernel** approach. The core challenge is: in a single GPU kernel, how do thousands of threadblocks know which sub-problem to work on and which tile to compute?

The solution has four key steps:

### Step 1: Problem Description Array

On the host side, construct an array describing all sub-problems. Each entry contains:
- Pointers to A, B, C, D matrices
- M, N, K dimensions
- Leading dimensions (lda, ldb, ldc, ldd)

This array is copied to device memory before kernel launch.

```cpp
// Each sub-problem is described by:
struct GemmProblem {
    half_t* ptr_A;     // pointer to matrix A
    half_t* ptr_B;     // pointer to matrix B
    half_t* ptr_C;     // pointer to matrix C (input/output)
    half_t* ptr_D;     // pointer to matrix D (output)
    int M, N, K;       // problem dimensions
    int lda, ldb;      // leading dimensions
    int ldc, ldd;
};
// Array of G such structs is prepared on host, copied to device
```

### Step 2: Tile Flattening

All sub-problems' tiles are **flattened into a single 1D tile list**:

```
Sub-problem 0 (M=256, N=256, tile=128×128) → 4 tiles: [tile0, tile1, tile2, tile3]
Sub-problem 1 (M=128, N=256, tile=128×128) → 2 tiles: [tile4, tile5]
Sub-problem 2 (M=384, N=256, tile=128×128) → 6 tiles: [tile6, ..., tile11]

Flattened global tile list:
[tile0, tile1, tile2, tile3, tile4, tile5, tile6, tile7, tile8, tile9, tile10, tile11]
 ←── sub-problem 0 ──→  ←─ sub-prob 1 ─→  ←────── sub-problem 2 ──────→
```

### Step 3: Prefix Sum Mapping

The host pre-computes the tile count for each sub-problem, then builds a prefix sum:

```
Sub-problem:       0    1    2
Tile count:        4    2    6
Prefix sum:   [0,  4,   6,  12]
```

Given a global tile index, a binary search on the prefix sum array determines which sub-problem it belongs to:

```
tile index 3  → in [0, 4)  → sub-problem 0, local tile index 3
tile index 5  → in [4, 6)  → sub-problem 1, local tile index 5-4=1
tile index 9  → in [6, 12) → sub-problem 2, local tile index 9-6=3
```

### Step 4: Kernel Execution

Each threadblock runs the following logic:

```
1. Obtain global tile index (via blockIdx or atomic operation)
2. Binary search the prefix sum array → find sub-problem index i
3. Load sub-problem i's info from the problem description array:
   - A[i], B[i], C[i] pointers
   - M[i], N[i], K[i] dimensions
4. Compute local tile coordinates (row, col) within sub-problem i
5. Execute standard GEMM tile computation:
   a. Load data from global memory → shared memory
   b. Warp-level MMA (Matrix Multiply-Accumulate) via Tensor Cores
   c. Write result back to global memory
```

## Scheduling Modes

CUTLASS 2.0 provides two scheduling strategies:

### kHostPrecompute (Host Pre-computation)

```
Host pre-computes: threadblock 0 → tile 0, threadblock 1 → tile 1, ...
Each threadblock looks up its assignment directly.
```

- **Pros**: Simple, minimal device-side overhead
- **Cons**: If some sub-problems finish faster than others, workload may be imbalanced

### kDeviceOnly (Device-side Dynamic Scheduling)

```
A global atomic counter is maintained.
After completing a tile, each threadblock atomically increments
the counter to get the next tile to work on.
Similar to "work stealing" in parallel programming.
```

- **Pros**: Excellent load balancing — faster threadblocks automatically pick up more work
- **Cons**: Small overhead from atomic operations

Choose `kDeviceOnly` when sub-problem sizes vary significantly (common in MoE), and `kHostPrecompute` when sub-problems are roughly equal in size.

## Code Example

Below is a complete CUTLASS 2.0 Group GEMM example:

```cpp
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

// Step 1: Define the GEMM type via template parameters
//
// This specifies every aspect of the computation:
// - Data types (FP16 inputs, FP32 accumulator)
// - Matrix layouts (A=RowMajor, B=ColumnMajor)
// - Tile sizes for threadblock (128×128×32), warp (64×64×32), instruction (16×8×16)
// - Target architecture (SM80 = A100 GPU)
// - Pipeline depth (4 stages for latency hiding)
// - Scheduling mode (kDeviceOnly for dynamic load balancing)
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t,                          // ElementA: FP16 input
    cutlass::layout::RowMajor,                // LayoutA
    cutlass::ComplexTransform::kNone,         // TransformA (no complex transform)
    8,                                        // AlignmentA (vector width)
    cutlass::half_t,                          // ElementB: FP16 input
    cutlass::layout::ColumnMajor,             // LayoutB
    cutlass::ComplexTransform::kNone,         // TransformB
    8,                                        // AlignmentB
    cutlass::half_t,                          // ElementC: FP16 output
    cutlass::layout::RowMajor,               // LayoutC
    float,                                    // ElementAccumulator: FP32 for precision
    cutlass::arch::OpClassTensorCore,         // Use Tensor Cores
    cutlass::arch::Sm80,                      // Target A100 GPU
    cutlass::gemm::GemmShape<128, 128, 32>,  // ThreadblockShape (M, N, K per block)
    cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape (M, N, K per warp)
    cutlass::gemm::GemmShape<16, 8, 16>,     // InstructionShape (MMA instruction)
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, float, float>,    // Epilogue: D = alpha*AB + beta*C
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, // Swizzle
    4,                                        // Stages (pipeline depth)
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly  // Dynamic scheduling
>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// Step 2: Prepare arguments and launch
void run_group_gemm(
    int problem_count,                    // Number of sub-problems (e.g., num_experts)
    cutlass::gemm::GemmCoord* problem_sizes,  // Array of (M,N,K) per sub-problem
    cutlass::half_t** ptr_A,              // Array of A pointers per sub-problem
    cutlass::half_t** ptr_B,              // Array of B pointers per sub-problem
    cutlass::half_t** ptr_C,              // Array of C pointers per sub-problem
    cutlass::half_t** ptr_D,              // Array of D pointers per sub-problem
    int64_t* lda, int64_t* ldb,
    int64_t* ldc, int64_t* ldd)
{
    // Construct arguments:
    // - problem_sizes: (M[i], N[i], K[i]) for each sub-problem
    // - problem_count: total number of sub-problems G
    // - threadblock_count: how many threadblocks to launch (tunable)
    // - {alpha, beta}: scalar parameters for D = alpha*A*B + beta*C
    // - ptr_A..ptr_D: per-problem matrix pointers
    // - lda..ldd: per-problem leading dimensions
    typename GemmGrouped::Arguments args(
        problem_sizes,
        problem_count,
        512,                 // threadblock_count (tunable parameter)
        {1.0f, 0.0f},       // alpha=1.0, beta=0.0 → D = A * B
        ptr_A, ptr_B, ptr_C, ptr_D,
        lda, ldb, ldc, ldd
    );

    GemmGrouped gemm_op;

    // Step 3: Initialize
    // This is where prefix sums are computed on the host,
    // the problem description array is copied to device, etc.
    gemm_op.initialize(args);

    // Step 4: Launch the kernel
    // All sub-problems are computed in a single kernel launch.
    gemm_op();
}
```

## Connection to SGLang

SGLang uses Group GEMM extensively in its MoE (Mixture of Experts) implementation. While the core concept is the same as CUTLASS 2.0, SGLang's implementation uses **CUTLASS 3.x** APIs targeting Hopper (SM90) and Blackwell (SM100) GPUs.

### Key Files

| File | Description |
|------|-------------|
| `sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu` | FP8 blockwise grouped GEMM for MoE using CUTLASS 3.x |
| `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` | INT4×FP8 mixed-precision grouped GEMM |
| `sgl-kernel/csrc/moe/cutlass_moe_helper.cu` | Helper kernel for computing per-expert pointer offsets |
| `python/sglang/srt/layers/moe/cutlass_moe.py` | Python interface for CUTLASS-based fused MoE |
| `python/sglang/srt/layers/moe/cutlass_moe_params.py` | Parameter dataclass for CUTLASS MoE operations |

### How SGLang Uses Group GEMM for MoE

In SGLang's MoE implementation, a fused MoE layer performs two grouped GEMMs with a SiLU activation in between:

```
GEMM 1: hidden_states × W1 (gate + up projection)
         ↓
    SiLU activation
         ↓
GEMM 2: activated_states × W2 (down projection)
```

Each GEMM is a Group GEMM where each expert processes a different number of tokens. The `CutlassMoEParams` class (in `cutlass_moe_params.py`) manages the per-expert problem descriptions:

```python
@dataclass
class CutlassMoEParams:
    # Per-expert strides for the GEMM operations
    ab_strides_13: torch.Tensor  # [num_experts] — activation/weight strides for GEMM 1
    ab_strides_2: torch.Tensor   # [num_experts] — activation/weight strides for GEMM 2

    # Per-expert problem sizes: (M, N, K) where M varies per expert
    problem_sizes1: torch.Tensor  # [num_experts, 3] — (M_i, 2*N, K) for GEMM 1
    problem_sizes2: torch.Tensor  # [num_experts, 3] — (M_i, K, N) for GEMM 2

    # Offsets marking where each expert's tokens begin
    expert_offsets: torch.Tensor  # [num_experts + 1]

    # Per-expert pointers to input/output/scale tensors
    a_ptrs: torch.Tensor          # [num_experts] — input activation pointers
    b_ptrs: torch.Tensor          # [num_experts] — weight pointers
    out_ptrs: torch.Tensor        # [num_experts] — output pointers
    a_scales_ptrs: torch.Tensor   # [num_experts] — activation scale pointers
    b_scales_ptrs: torch.Tensor   # [num_experts] — weight scale pointers
```

The `cutlass_moe_helper.cu` kernel computes per-expert pointer offsets on the GPU — this is the equivalent of the "problem description array" from CUTLASS 2.0, but computed on-device for efficiency.

### CUTLASS 2.0 vs 3.x in SGLang

| Aspect | CUTLASS 2.0 | CUTLASS 3.x (used in SGLang) |
|--------|-------------|------------------------------|
| Problem shape | `GemmCoord` (M, N, K) | `GroupProblemShape` with CuTe layouts |
| Scheduling | `kDeviceOnly` / `kHostPrecompute` | `PersistentScheduler` with group array support |
| Data movement | Shared memory + warp MMA | TMA (Tensor Memory Accelerator) + warpgroup MMA |
| Architecture | SM80 (A100) | SM90 (H100) / SM100 (B200) |
| API style | Monolithic template | Modular Collective + Kernel builder pattern |

The fundamental **concept** is the same: flatten all sub-problems' tiles into a unified space, map each threadblock to a specific sub-problem and tile coordinate, and execute the GEMM tile computation. The difference is in the low-level implementation details that leverage newer hardware features.
