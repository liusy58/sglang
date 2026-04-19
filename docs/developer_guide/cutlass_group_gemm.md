# CUTLASS Group GEMM 开发者指南

本文档全面介绍 CUTLASS 及其 Group GEMM 实现，详细解析 SGLang 中相关代码的工作原理，并对比 CUTLASS 1.0 与 2.0 的核心区别。

## 目录

- [什么是 CUTLASS？](#什么是-cutlass)
- [什么是 GEMM？](#什么是-gemm)
- [什么是 Group GEMM？](#什么是-group-gemm)
- [CUTLASS 2.0 Group GEMM 实现原理](#cutlass-20-group-gemm-实现原理)
  - [总览](#总览)
  - [第一步：问题描述数组](#第一步问题描述数组)
  - [第二步：Tile 展平](#第二步tile-展平)
  - [第三步：前缀和映射](#第三步前缀和映射)
  - [第四步：Kernel 执行](#第四步kernel-执行)
- [调度模式](#调度模式)
- [CUTLASS 2.0 代码示例](#cutlass-20-代码示例)
- [SGLang 中的代码详解](#sglang-中的代码详解)
  - [整体架构与文件关系](#整体架构与文件关系)
  - [Python 层：cutlass_moe_params.py 详解](#python-层cutlass_moe_paramspy-详解)
  - [Python 层：cutlass_moe.py 详解](#python-层cutlass_moepy-详解)
  - [CUDA 层：prepare_moe_input.cu 详解](#cuda-层prepare_moe_inputcu-详解)
  - [CUDA 层：cutlass_moe_helper.cu 详解](#cuda-层cutlass_moe_helpercu-详解)
  - [CUDA 层：fp8_blockwise_moe_kernel.cu 详解](#cuda-层fp8_blockwise_moe_kernelcu-详解)
- [CUTLASS 1.0 vs 2.0 对比](#cutlass-10-vs-20-对比)
- [CUTLASS 2.0 vs 3.x 对比（SGLang 使用的版本）](#cutlass-20-vs-3x-对比sglang-使用的版本)

---

## 什么是 CUTLASS？

[CUTLASS](https://github.com/NVIDIA/cutlass)（CUDA Templates for Linear Algebra Subroutines）是 NVIDIA 开源的 C++ 模板库，用于在 GPU 上编写高性能矩阵乘法（GEMM）kernel。

可以这样理解：
- **cuBLAS** 是一个"黑盒"库——你调用一个函数就能计算矩阵乘法，但你不能修改内部实现。
- **CUTLASS** 是一个"白盒"库——它把高性能 GEMM kernel 的每一层都暴露为 C++ 模板，允许你自定义数据类型、tile 大小、流水线深度、epilogue 操作等。

类比：cuBLAS 好比买一台品牌整机，CUTLASS 给你提供高质量的零件来自由组装，你可以换显卡、调内存、改配置。

## 什么是 GEMM？

GEMM（General Matrix Multiply，通用矩阵乘法）是标准的矩阵乘法运算：

```
C = α × A × B + β × C
```

其中 A 是 M×K 矩阵，B 是 K×N 矩阵，C 是 M×N 矩阵。这是深度学习中最基础的计算——全连接层、注意力机制、卷积最终都归结为 GEMM。

### GPU GEMM 的工作原理（Tiling）

GPU 上的单个 GEMM 通过 **分块（tiling）** 来计算：输出矩阵 C（M×N）被划分为小块（tile），每个 CUDA threadblock 计算一个 tile。

```
输出矩阵 C (M×N):
┌──────────┬──────────┬──────────┬──────────┐
│  tile 0  │  tile 1  │  tile 2  │  tile 3  │  ← 每个 tile 由
│ (128×128)│ (128×128)│ (128×128)│ (128×128)│    一个 threadblock 计算
├──────────┼──────────┼──────────┼──────────┤
│  tile 4  │  tile 5  │  tile 6  │  tile 7  │
│ (128×128)│ (128×128)│ (128×128)│ (128×128)│
└──────────┴──────────┴──────────┴──────────┘
```

每个 threadblock 的工作流程：
1. 从 Global Memory 加载 A 和 B 的 tile 到 Shared Memory
2. 通过 Tensor Core 执行 warp 级的 MMA（Matrix Multiply-Accumulate）
3. 将结果 tile 写回 Global Memory

## 什么是 Group GEMM？

Group GEMM 同时执行 **多个独立的矩阵乘法**：

```
对于 i = 0, 1, ..., G-1：
    C[i] = α × A[i] × B[i] + β × C[i]
```

每个子问题可以有 **不同的 M、N、K 维度**。

### 为什么需要 Group GEMM？

最重要的应用场景是 **MoE（Mixture of Experts，混合专家）模型**（例如 Mixtral、DeepSeek-V2/V3）：

- MoE 模型有多个专家（expert），每个专家是一个独立的全连接层（= 一个 GEMM）
- 路由器（router）将不同的 token 分配给不同的专家；每个专家可能处理不同数量的 token
- 你需要同时执行多个 **不同大小** 的 GEMM

如果逐个执行这些小 GEMM（串行），GPU 利用率很低，因为每个单独的 GEMM 可能太小，无法充分利用 GPU。Group GEMM 将所有小 GEMM **打包到一次 kernel 启动中**，最大化 GPU 利用率。

```
串行执行（浪费）：                    Group GEMM（高效）：
┌─────────┐                           ┌─────────────────────────┐
│ Expert 0 │ ← GPU 利用率低            │ Expert 0 │ Expert 1 │...│
├─────────┤                           │  tiles   │  tiles   │   │
│ Expert 1 │ ← GPU 利用率低            │          │          │   │
├─────────┤                           │   所有专家在一个      │   │
│ Expert 2 │ ← GPU 利用率低            │   kernel 中运行      │   │
└─────────┘                           └─────────────────────────┘
  3 次 kernel 启动                      1 次 kernel 启动，GPU 满载
```

## CUTLASS 2.0 Group GEMM 实现原理

### 总览

CUTLASS 2.0 通过 **主机端问题访问器 + 设备端分组 kernel** 的方式实现 Group GEMM。核心挑战是：在一个 GPU kernel 中，数千个 threadblock 如何知道自己应该处理哪个子问题的哪个 tile？

解决方案分四个关键步骤：

### 第一步：问题描述数组

在主机端构造一个数组来描述所有子问题，每个条目包含：
- A、B、C、D 矩阵的指针
- M、N、K 维度
- Leading dimensions（lda、ldb、ldc、ldd）

这个数组在 kernel 启动前被拷贝到设备内存。

```cpp
// 每个子问题的描述：
struct GemmProblem {
    half_t* ptr_A;     // 矩阵 A 的指针
    half_t* ptr_B;     // 矩阵 B 的指针
    half_t* ptr_C;     // 矩阵 C 的指针（输入/输出）
    half_t* ptr_D;     // 矩阵 D 的指针（输出）
    int M, N, K;       // 问题维度
    int lda, ldb;      // leading dimensions
    int ldc, ldd;
};
// 在主机端准备 G 个这样的结构体数组，拷贝到设备
```

### 第二步：Tile 展平

所有子问题的 tile 被 **展平为一个一维 tile 列表**：

```
子问题 0 (M=256, N=256, tile=128×128) → 4 个 tile：[tile0, tile1, tile2, tile3]
子问题 1 (M=128, N=256, tile=128×128) → 2 个 tile：[tile4, tile5]
子问题 2 (M=384, N=256, tile=128×128) → 6 个 tile：[tile6, ..., tile11]

展平后的全局 tile 列表：
[tile0, tile1, tile2, tile3, tile4, tile5, tile6, tile7, tile8, tile9, tile10, tile11]
 ←──── 子问题 0 ────→  ←── 子问题 1 ──→  ←────────── 子问题 2 ──────────→
```

### 第三步：前缀和映射

主机预计算每个子问题的 tile 数量，然后构建前缀和：

```
子问题：           0    1    2
tile 数量：        4    2    6
前缀和：      [0,  4,   6,  12]
```

给定一个全局 tile 索引，通过二分查找前缀和数组来确定它属于哪个子问题：

```
tile 索引 3  → 在 [0, 4)  → 子问题 0，本地 tile 索引 3
tile 索引 5  → 在 [4, 6)  → 子问题 1，本地 tile 索引 5-4=1
tile 索引 9  → 在 [6, 12) → 子问题 2，本地 tile 索引 9-6=3
```

### 第四步：Kernel 执行

每个 threadblock 执行以下逻辑：

```
1. 获取全局 tile 索引（通过 blockIdx 或原子操作）
2. 二分查找前缀和数组 → 找到子问题索引 i
3. 从问题描述数组加载子问题 i 的信息：
   - A[i]、B[i]、C[i] 指针
   - M[i]、N[i]、K[i] 维度
4. 计算子问题 i 内的本地 tile 坐标（行，列）
5. 执行标准 GEMM tile 计算：
   a. 从 Global Memory 加载数据 → Shared Memory
   b. 通过 Tensor Core 执行 warp 级 MMA
   c. 将结果写回 Global Memory
```

## 调度模式

CUTLASS 2.0 提供两种调度策略：

### kHostPrecompute（主机预计算）

```
主机预计算：threadblock 0 → tile 0，threadblock 1 → tile 1，...
每个 threadblock 直接查找自己的任务分配。
```

- **优点**：简单，设备端开销极小
- **缺点**：如果某些子问题比其他的更快完成，工作负载可能不均衡

### kDeviceOnly（设备端动态调度）

```
维护一个全局原子计数器。
每个 threadblock 完成一个 tile 后，原子递增计数器来获取下一个要处理的 tile。
类似于并行编程中的"工作窃取"。
```

- **优点**：出色的负载均衡——更快的 threadblock 自动承担更多工作
- **缺点**：原子操作带来少量开销

当子问题大小差异显著时（在 MoE 中很常见）选择 `kDeviceOnly`，当子问题大小大致相等时选择 `kHostPrecompute`。

## CUTLASS 2.0 代码示例

以下是一个完整的 CUTLASS 2.0 Group GEMM 示例：

```cpp
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

// 第一步：通过模板参数定义 GEMM 类型
//
// 这指定了计算的每个方面：
// - 数据类型（FP16 输入，FP32 累加器）
// - 矩阵布局（A=行主序，B=列主序）
// - threadblock 的 tile 大小（128×128×32），warp（64×64×32），指令（16×8×16）
// - 目标架构（SM80 = A100 GPU）
// - 流水线深度（4 stage，用于延迟隐藏）
// - 调度模式（kDeviceOnly 用于动态负载均衡）
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::half_t,                          // ElementA：FP16 输入
    cutlass::layout::RowMajor,                // LayoutA：行主序
    cutlass::ComplexTransform::kNone,         // TransformA（无复数变换）
    8,                                        // AlignmentA（向量宽度）
    cutlass::half_t,                          // ElementB：FP16 输入
    cutlass::layout::ColumnMajor,             // LayoutB：列主序
    cutlass::ComplexTransform::kNone,         // TransformB
    8,                                        // AlignmentB
    cutlass::half_t,                          // ElementC：FP16 输出
    cutlass::layout::RowMajor,               // LayoutC
    float,                                    // ElementAccumulator：FP32 精度累加
    cutlass::arch::OpClassTensorCore,         // 使用 Tensor Core
    cutlass::arch::Sm80,                      // 目标 A100 GPU
    cutlass::gemm::GemmShape<128, 128, 32>,  // ThreadblockShape（每个 block 的 M,N,K）
    cutlass::gemm::GemmShape<64, 64, 32>,    // WarpShape（每个 warp 的 M,N,K）
    cutlass::gemm::GemmShape<16, 8, 16>,     // InstructionShape（MMA 指令尺寸）
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 8, float, float>,    // Epilogue：D = alpha*AB + beta*C
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, // Swizzle
    4,                                        // Stages（流水线深度）
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly  // 动态调度
>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

// 第二步：准备参数并启动
void run_group_gemm(
    int problem_count,                    // 子问题数量（如专家数）
    cutlass::gemm::GemmCoord* problem_sizes,  // 每个子问题的 (M,N,K)
    cutlass::half_t** ptr_A,              // 每个子问题的 A 指针数组
    cutlass::half_t** ptr_B,              // 每个子问题的 B 指针数组
    cutlass::half_t** ptr_C,              // 每个子问题的 C 指针数组
    cutlass::half_t** ptr_D,              // 每个子问题的 D 指针数组
    int64_t* lda, int64_t* ldb,
    int64_t* ldc, int64_t* ldd)
{
    // 构造参数：
    // - problem_sizes：每个子问题的 (M[i], N[i], K[i])
    // - problem_count：子问题总数 G
    // - threadblock_count：启动多少个 threadblock（可调）
    // - {alpha, beta}：D = alpha*A*B + beta*C 的标量参数
    // - ptr_A..ptr_D：各子问题的矩阵指针
    // - lda..ldd：各子问题的 leading dimensions
    typename GemmGrouped::Arguments args(
        problem_sizes,
        problem_count,
        512,                 // threadblock_count（可调参数）
        {1.0f, 0.0f},       // alpha=1.0, beta=0.0 → D = A * B
        ptr_A, ptr_B, ptr_C, ptr_D,
        lda, ldb, ldc, ldd
    );

    GemmGrouped gemm_op;

    // 第三步：初始化
    // 在这里计算前缀和、将问题描述数组拷贝到设备等
    gemm_op.initialize(args);

    // 第四步：启动 kernel
    // 所有子问题在一次 kernel 启动中完成计算
    gemm_op();
}
```

---

## SGLang 中的代码详解

SGLang 在其 MoE（混合专家）实现中大量使用 Group GEMM。虽然核心概念与 CUTLASS 2.0 相同，但 SGLang 使用 **CUTLASS 3.x** API，目标是 Hopper（SM90）和 Blackwell（SM100）GPU。

### 整体架构与文件关系

SGLang 的 CUTLASS MoE 实现分为 Python 层和 CUDA 层，整体数据流如下：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Python 层（调度与编排）                              │
│                                                                      │
│  cutlass_moe_params.py ──→ 预分配所有 GPU tensor（指针、strides 等）     │
│         │                                                            │
│         ▼                                                            │
│  cutlass_moe.py ──→ 编排整个 MoE 前向过程：                             │
│    1. prepare_moe_input()  → 计算路由、排序 token                      │
│    2. 量化输入为 FP8                                                   │
│    3. fp8_blockwise_scaled_grouped_mm()  → 第一个 Group GEMM           │
│    4. silu_and_mul()                     → 激活函数                     │
│    5. 量化中间结果为 FP8                                               │
│    6. fp8_blockwise_scaled_grouped_mm()  → 第二个 Group GEMM           │
│    7. apply_shuffle_mul_sum()            → 加权求和、还原 token 顺序     │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ 调用
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    CUDA 层（实际计算）                                  │
│                                                                      │
│  prepare_moe_input.cu   → 统计各专家 token 数、计算偏移量               │
│  cutlass_moe_helper.cu  → 计算各专家的指针偏移（问题描述数组）            │
│  fp8_blockwise_moe_kernel.cu → CUTLASS 3.x Group GEMM kernel         │
└──────────────────────────────────────────────────────────────────────┘
```

### 关键文件一览

| 文件路径 | 作用 |
|---------|------|
| `python/sglang/srt/layers/moe/cutlass_moe_params.py` | 参数数据类，预分配 GPU tensor |
| `python/sglang/srt/layers/moe/cutlass_moe.py` | Python 层 MoE 前向编排逻辑 |
| `sgl-kernel/csrc/moe/prepare_moe_input.cu` | 统计每个专家的 token 数，计算偏移 |
| `sgl-kernel/csrc/moe/cutlass_moe_helper.cu` | 计算各专家矩阵指针偏移 |
| `sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu` | FP8 分块量化 Group GEMM kernel |

---

### Python 层：cutlass_moe_params.py 详解

`CutlassMoEParams` 是一个数据类，在模型初始化时 **一次性预分配** 所有 Group GEMM 需要的 GPU tensor，避免每次前向传播时重复分配内存。

```python
# 文件：python/sglang/srt/layers/moe/cutlass_moe_params.py

@dataclass
class CutlassMoEParams:
    """CUTLASS MoE 操作的参数容器。"""

    # ═══════════════ 类型 ═══════════════
    cutlass_moe_type: CutlassMoEType  # BlockscaledFP8 或 BlockscaledFP4

    # ═══════════════ 步长（Strides）═══════════════
    # 这些张量描述每个专家的矩阵内存布局
    # GEMM 1 的步长：输入 [m, k] × 权重 [e, k, 2n] → 输出 [m, 2n]
    ab_strides_13: torch.Tensor  # [e]，每个值 = k（输入隐藏维度）
    c_strides_13: torch.Tensor   # [e]，每个值 = 2*n（中间层宽度×2，因为 gate+up）
    # GEMM 2 的步长：输入 [m, n] × 权重 [e, n, k] → 输出 [m, k]
    ab_strides_2: torch.Tensor   # [e]，每个值 = n（中间层宽度）
    c_strides_2: torch.Tensor    # [e]，每个值 = k（输出隐藏维度）

    # ═══════════════ 问题描述 ═══════════════
    # 类似 CUTLASS 2.0 中的问题描述数组
    problem_sizes1: torch.Tensor  # [e, 3]，GEMM 1 的 (M_i, 2*N, K)
    problem_sizes2: torch.Tensor  # [e, 3]，GEMM 2 的 (M_i, K, N)
    # 其中 M_i 是分配给专家 i 的 token 数（每次前向动态变化）

    # ═══════════════ 偏移量 ═══════════════
    # 标记每个专家的 token 从哪里开始
    # tokens 被排序后连续存放：[expert0 的 tokens | expert1 的 tokens | ...]
    expert_offsets: torch.Tensor  # [e+1]，expert_offsets[i+1]-expert_offsets[i] = 专家 i 的 token 数

    # ═══════════════ 指针数组（Pointer Arrays）═══════════════
    # 这些对应 CUTLASS 2.0 中的 ptr_A[], ptr_B[] 等指针数组
    # 每个元素存储的是指向对应专家数据的 GPU 内存地址
    a_ptrs: torch.Tensor          # [e]，int64，各专家输入激活值的指针
    b_ptrs: torch.Tensor          # [e]，int64，各专家权重的指针
    out_ptrs: torch.Tensor        # [e]，int64，各专家输出的指针
    a_scales_ptrs: torch.Tensor   # [e]，int64，各专家激活值 scale 的指针
    b_scales_ptrs: torch.Tensor   # [e]，int64，各专家权重 scale 的指针
```

**要点**：
- `ab_strides` 在初始化时是固定的（因为隐藏维度不变），但 `problem_sizes` 中的 M 维度在每次前向传播时动态更新
- `expert_offsets` 是一个前缀和数组，形状为 `[e+1]`，与 CUTLASS 2.0 的前缀和概念完全对应

---

### Python 层：cutlass_moe.py 详解

`cutlass_fused_experts_fp8()` 是核心的前向编排函数，它将整个 MoE 层的计算组织为一系列步骤：

```python
# 文件：python/sglang/srt/layers/moe/cutlass_moe.py

def cutlass_fused_experts_fp8(
    a,              # [m, k]，输入 token（BF16/FP16）
    w1_q, w2_q,     # [e, k, 2n] 和 [e, n, k]，FP8 量化的权重
    w1_scale, w2_scale,  # 权重的 per-block scale
    topk_weights,   # [m, topk]，路由权重
    topk_ids,       # [m, topk]，选中的专家 ID
    ...
) -> torch.Tensor:  # [m, k]，输出

    # ========= 阶段 1：路由准备 =========
    # prepare_moe_input 是一个 CUDA kernel，它做了三件事：
    #   (a) 统计每个专家被分配了多少 token → 填入 problem_sizes1/2
    #   (b) 计算 expert_offsets（前缀和）
    #   (c) 生成排序映射 a_map 和 c_map
    prepare_moe_input(
        topk_ids,           # 输入：每个 token 选择的专家 ID
        expert_offsets,     # 输出：前缀和数组
        problem_sizes1,     # 输出：GEMM 1 的 [M_i, 2*N, K]
        problem_sizes2,     # 输出：GEMM 2 的 [M_i, K, N]
        a_map,              # 输出：输入排序映射（将 token 按专家分组）
        c_map,              # 输出：输出逆排序映射（恢复原始顺序）
        num_experts, n, k,
    )

    # ========= 阶段 2：量化 + 排序 =========
    # 将 BF16 输入量化为 FP8，并按 a_map 重排 token 顺序
    # 排序后，同一个专家的 token 在内存中连续存放
    a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)
    rep_a_q = shuffle_rows(a_q, a_map, (m * topk, k))      # 按专家分组排列
    rep_a1_scales = shuffle_rows(a1_scale, a_map, ...)

    # ========= 阶段 3：Group GEMM 1（gate + up projection）=========
    # 这就是核心的 Group GEMM 调用
    # 对每个专家 i：C1[i] = rep_a_q[专家i的tokens] × w1_q[i]
    # 输出 c1 shape = [m*topk, 2*n]
    fp8_blockwise_scaled_grouped_mm(
        c1,                 # 输出
        a_ptrs, b_ptrs, out_ptrs,           # 指针数组（per-expert）
        a_scales_ptrs, b_scales_ptrs,       # scale 指针数组
        rep_a_q, w1_q,                      # 实际数据张量
        rep_a1_scales, w1_scale,            # scale 张量
        a1_strides, a1_strides, c1_strides, # 步长
        a_sf_layout, w_sf_layout,           # blockscale 布局
        problem_sizes1,                     # 问题描述
        expert_offsets[:-1],                # 专家偏移
        workspace,                          # 工作空间
    )

    # ========= 阶段 4：SiLU 激活 =========
    # c1 的 shape = [m*topk, 2*n]
    # 其中前 n 列是 gate，后 n 列是 up
    # silu_and_mul 计算：intermediate = SiLU(gate) * up
    # intermediate shape = [m*topk, n]
    silu_and_mul(c1, intermediate)

    # ========= 阶段 5：量化中间结果 =========
    intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(intermediate, 128)

    # ========= 阶段 6：Group GEMM 2（down projection）=========
    # 对每个专家 i：C2[i] = intermediate_q[专家i的tokens] × w2_q[i]
    # 输出 c2 shape = [m*topk, k]
    fp8_blockwise_scaled_grouped_mm(
        c2,                 # 输出
        ...,                # 同上的指针和参数
        problem_sizes2,     # 第二个 GEMM 的问题描述
        expert_offsets[:-1],
        workspace,
    )

    # ========= 阶段 7：加权求和 + 还原顺序 =========
    # c2 目前按专家分组排列，需要通过 c_map 还原为原始 token 顺序
    # 同时乘以路由权重 topk_weights 并对 topk 维度求和
    apply_shuffle_mul_sum(c2, output, c_map, topk_weights)
    return output  # [m, k]
```

**数据流图示**：

```
输入 a [m, k] (BF16)
    │
    ▼ 量化 + 按专家排序
rep_a_q [m*topk, k] (FP8)  ──→  排序后 token 连续排列：
                                  [expert0的tokens | expert1的tokens | ...]
    │
    ▼ Group GEMM 1：rep_a_q × W1
c1 [m*topk, 2*n] (BF16)
    │
    ▼ SiLU(gate) × up
intermediate [m*topk, n] (BF16)
    │
    ▼ 量化
intermediate_q [m*topk, n] (FP8)
    │
    ▼ Group GEMM 2：intermediate_q × W2
c2 [m*topk, k] (BF16)
    │
    ▼ 逆排序 + 路由权重加权 + topk 求和
output [m, k] (BF16)
```

---

### CUDA 层：prepare_moe_input.cu 详解

`prepare_moe_input` 是路由准备阶段的 CUDA kernel，它的作用是将路由结果转换为 Group GEMM 需要的格式。

```cpp
// 文件：sgl-kernel/csrc/moe/prepare_moe_input.cu

// ═══ Kernel 1：统计每个专家的 token 数 ═══
__global__ void compute_problem_sizes(
    const int* topk_ids,       // [m * topk]，每个位置的专家 ID
    int32_t* problem_sizes1,   // 输出：[e, 3]，GEMM 1 的 (M, 2N, K)
    int32_t* problem_sizes2,   // 输出：[e, 3]，GEMM 2 的 (M, K, N)
    int32_t* atomic_buffer,    // 原子计数器缓冲区
    int64_t topk_length,       // = m * topk
    int64_t n, int64_t k)
{
    int expert_id = blockIdx.x;  // 每个 block 负责一个专家

    // 线程并行统计：遍历所有 token，累计分配给当前专家的数量
    int occurrences = 0;
    for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
        occurrences += (topk_ids[i] == expert_id);
    }
    atomicAdd(&atomic_buffer[expert_id], occurrences);
    __syncthreads();

    // 线程 0 写入最终结果
    if (threadIdx.x == 0) {
        int M = atomic_buffer[expert_id];  // 该专家的 token 数
        problem_sizes1[expert_id * 3 + 0] = M;
        problem_sizes1[expert_id * 3 + 1] = 2 * n;  // gate+up 输出宽度
        problem_sizes1[expert_id * 3 + 2] = k;       // 输入隐藏维度
        problem_sizes2[expert_id * 3 + 0] = M;
        problem_sizes2[expert_id * 3 + 1] = k;       // 输出隐藏维度
        problem_sizes2[expert_id * 3 + 2] = n;       // 中间层宽度
    }
}

// ═══ Kernel 2：计算专家偏移量（前缀和）═══
// 这对应 CUTLASS 2.0 中 Step 3 的前缀和计算
__global__ void compute_expert_offsets(
    const int32_t* problem_sizes1,
    int32_t* expert_offsets,    // 输出：[e+1] 的前缀和
    int32_t* atomic_buffer,
    int64_t num_experts)
{
    int32_t tot_offset = 0;
    expert_offsets[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
        atomic_buffer[i] = tot_offset;       // 当前专家的起始位置
        tot_offset += problem_sizes1[i * 3]; // 累加该专家的 token 数
        expert_offsets[i + 1] = tot_offset;  // 下一个专家的起始位置
    }
    // 结果示例：
    // 如果 3 个专家分别有 10, 5, 8 个 token
    // expert_offsets = [0, 10, 15, 23]
}
```

**要点**：`expert_offsets` 的形状为 `[e+1]`，其中 `expert_offsets[i]` 表示第 i 个专家的 token 在排序后数组中的起始位置。这直接对应 CUTLASS 2.0 中用于 tile-to-problem 映射的前缀和数组。

---

### CUDA 层：cutlass_moe_helper.cu 详解

`get_group_gemm_starts` 是关键的指针偏移计算 kernel。它的作用是根据 `expert_offsets` 和问题描述，为每个专家计算出实际的矩阵起始地址——**这就是 CUTLASS 2.0 中"问题描述数组"的设备端计算版本**。

```cpp
// 文件：sgl-kernel/csrc/moe/cutlass_moe_helper.cu

// 这个 kernel 只用 1 个 block 启动，线程数 = 专家数（通常 8~256）
// 每个线程负责一个专家的指针计算
template <typename ElementAB, typename ElementC, typename ElementAccumulator,
          typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
__global__ void get_group_gemm_starts(
    int32_t* expert_offsets,       // [e]，每个专家的 token 起始偏移
    ElementAB** a_offsets,         // 输出：每个专家的 A 矩阵指针
    ElementAB** b_offsets,         // 输出：每个专家的 B 矩阵指针
    ElementC**  out_offsets,       // 输出：每个专家的输出矩阵指针
    ElementAccumulator** a_scales_offsets,  // 输出：A 的 scale 指针
    ElementAccumulator** b_scales_offsets,  // 输出：B 的 scale 指针
    ElementAB* a_base,             // A 数据的基地址（所有专家共享）
    ElementAB* b_base,             // B 数据的基地址
    ElementC*  out_base,           // 输出的基地址
    ElementAccumulator* a_scales_base,  // A scale 的基地址
    ElementAccumulator* b_scales_base,  // B scale 的基地址
    LayoutSFA* layout_sfa_base,    // blockscale 布局的基地址
    LayoutSFB* layout_sfb_base,
    int* problem_sizes,            // [e, 3]，每个专家的 (M, N, K)
    int* problem_sizes_transpose,  // 转置时使用
    bool transpose)
{
    int64_t expert_id = threadIdx.x;

    // 读取当前专家的问题维度
    int m = problem_sizes[expert_id * 3];
    int n = problem_sizes[expert_id * 3 + 1];
    int k = problem_sizes[expert_id * 3 + 2];

    // 读取当前专家的 token 起始偏移
    int64_t expert_offset = expert_offsets[expert_id];

    // ═══ 核心逻辑：计算各矩阵的偏移量 ═══
    //
    // A 矩阵（激活值）：所有专家的 token 排列在同一个连续数组中
    //   A = [expert0的tokens | expert1的tokens | ...]
    //   专家 i 的起始位置 = expert_offset * k（因为每个 token 有 k 个元素）
    //
    // B 矩阵（权重）：每个专家有独立的权重矩阵
    //   B = [expert0的权重 | expert1的权重 | ...]
    //   专家 i 的起始位置 = expert_id * k * n
    //
    // 输出矩阵：排列方式与 A 相同
    //   Out = [expert0的输出 | expert1的输出 | ...]
    //   专家 i 的起始位置 = expert_offset * n

    if (!transpose) {
        // 正常情况：A=[tokens, k], B=[e, k, n]
        a_offsets[expert_id] = a_base + expert_offset * k;
        b_offsets[expert_id] = b_base + expert_id * k * n;
        a_scales_offsets[expert_id] = a_scales_base + expert_offset * k / 128;
        b_scales_offsets[expert_id] = b_scales_base + expert_id * k * n / 128 / 128;
    } else {
        // 转置情况：用于小 batch size 优化
        a_offsets[expert_id] = a_base + expert_id * k * n;
        b_offsets[expert_id] = b_base + expert_offset * k;
        a_scales_offsets[expert_id] = a_scales_base + expert_id * k * n / 128 / 128;
        b_scales_offsets[expert_id] = b_scales_base + expert_offset * k / 128;
    }
    out_offsets[expert_id] = out_base + expert_offset * n;

    // 计算 blockscale 布局（CUTLASS 3.x 特有，用于 FP8 块量化）
    *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(m, n, k, 1));
    *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(m, n, k, 1));
}
```

**图解指针偏移计算**：

```
假设 3 个专家，expert_offsets = [0, 10, 15, 23]，k=4096, n=11008

A 矩阵（排序后的 token）：连续存放
┌──────────────────┬──────────────────┬──────────────────┐
│  Expert 0 的 token │  Expert 1 的 token │  Expert 2 的 token │
│  (10 × 4096)      │  (5 × 4096)       │  (8 × 4096)       │
└──────────────────┴──────────────────┴──────────────────┘
a_offsets[0] = a_base + 0*4096
a_offsets[1] = a_base + 10*4096
a_offsets[2] = a_base + 15*4096

B 矩阵（权重）：每个专家独立存放
┌──────────────────┬──────────────────┬──────────────────┐
│  Expert 0 的权重   │  Expert 1 的权重   │  Expert 2 的权重   │
│  (4096 × 11008)   │  (4096 × 11008)   │  (4096 × 11008)   │
└──────────────────┴──────────────────┴──────────────────┘
b_offsets[0] = b_base + 0*4096*11008
b_offsets[1] = b_base + 1*4096*11008
b_offsets[2] = b_base + 2*4096*11008
```

---

### CUDA 层：fp8_blockwise_moe_kernel.cu 详解

这是实际执行 Group GEMM 的 CUTLASS 3.x kernel。它使用 CUTLASS 的 **Collective Builder** 模式来组装 GEMM 操作。

#### 核心模板结构

```cpp
// 文件：sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu

// 问题形状使用 CUTLASS 3.x 的 GroupProblemShape
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

template <typename OutType, typename ScheduleConfig, typename LayoutD>
void launch_sm90_fp8_blockwise_scaled_group_mm(...) {
    // ═══ 类型定义 ═══
    using ElementA = cutlass::float_e4m3_t;  // FP8 E4M3 格式
    using ElementB = cutlass::float_e4m3_t;
    using ElementD = OutType;                 // BF16 或 FP16 输出
    using ElementAccumulator = float;         // FP32 累加

    // ═══ Epilogue（后处理）构建 ═══
    // 这里使用 Identity epilogue，即不做额外的 alpha/beta 操作
    // 直接输出 GEMM 结果（反量化后的值）
    using CustomEVTIdentity = cutlass::epilogue::fusion::Sm90EVT<
        Sm90Compute<Identity, ElementD, ElementAccumulator, round_to_nearest>,
        Sm90AccFetch>;

    using CollectiveEpilogue = CollectiveBuilder<
        Sm90, OpClassTensorOp,
        MmaTileShape, ClusterShape,
        EpilogueTileAuto,
        float, float,           // 累加器类型
        void,                   // C = void，不读入矩阵 C
        LayoutC*, AlignmentC,
        ElementD, LayoutC*, AlignmentC,
        EpilogueSchedule,
        CustomEVTIdentity       // 使用自定义 epilogue
    >::CollectiveOp;

    // ═══ Mainloop（主循环）构建 ═══
    // 关键区别于 CUTLASS 2.0：
    //   - 布局使用指针数组（LayoutA*），而非单一布局
    //   - 额外携带 scale factor 布局（LayoutSFA*, LayoutSFB*）
    //   - 使用 TMA 而非手动 shared memory 加载
    using CollectiveMainloop = CollectiveBuilder<
        Sm90, OpClassTensorOp,
        ElementA,
        tuple<LayoutA*, LayoutSFA*>,  // A 的布局 + scale 布局
        AlignmentA,
        ElementB,
        tuple<LayoutB*, LayoutSFB*>,  // B 的布局 + scale 布局
        AlignmentB,
        float,                        // 累加器类型
        MmaTileShape,                 // tile 形状，如 <128, 128, 128>
        ClusterShape,                 // cluster 形状，如 <1, 2, 1>
        StageCountAutoCarveout<...>,  // 自动计算流水线 stage 数
        KernelSchedule                // 调度策略
    >::CollectiveOp;

    // ═══ 组装最终 kernel ═══
    using GemmKernel = GemmUniversal<ProblemShape, CollectiveMainloop,
                                      CollectiveEpilogue, void>;
    using Gemm = GemmUniversalAdapter<GemmKernel>;
```

#### Mainloop 参数构造

```cpp
    // ═══ 构造 Mainloop 参数 ═══
    // 注意：所有参数都是"指针的数组"或"布局的数组"，
    // 因为每个专家有不同的 A/B 起始地址
    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<const ElementA**>(a_ptrs.data_ptr()),     // A 指针数组
        static_cast<StrideA*>(stride_a.data_ptr()),           // A 步长数组
        static_cast<const ElementB**>(b_ptrs.data_ptr()),     // B 指针数组
        static_cast<StrideB*>(stride_b.data_ptr()),           // B 步长数组
        static_cast<const float**>(a_scales_ptrs.data_ptr()), // A scale 指针数组
        reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),  // A scale 布局数组
        static_cast<const float**>(b_scales_ptrs.data_ptr()), // B scale 指针数组
        reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),  // B scale 布局数组
    };
```

#### 最终参数组装与启动

```cpp
    // ═══ 最终参数 ═══
    typename GemmKernel::Arguments args{
        GemmUniversalMode::kGrouped,  // 关键：指定为 Grouped 模式
        {num_experts,                 // 子问题数量（= 专家数）
         problem_sizes_as_shapes,     // 每个专家的 (M, N, K)
         nullptr},                    // host_problem_shapes（不需要）
        mainloop_args,
        epilogue_args,
        hw_info                       // SM 数量等硬件信息
    };

    Gemm gemm_op;
    gemm_op.initialize(args, workspace, stream);  // 初始化
    gemm_op.run(stream);                           // 启动 kernel！
```

#### 基于问题规模的调度策略选择

SGLang 根据矩阵大小自动选择最优的 tile 和 cluster 配置：

```cpp
// SM90（H100/H200）上的三种配置：

// 配置 1：小 M（≤2048 tokens）→ 转置矩阵 + Pingpong 调度
// 小 batch 时 M 小 N 大，转置后 M 和 N 互换，让更长的维度成为行维度
struct MmaConfigSmallM {
    using MmaTileShape = Shape<_128, _32, _128>;   // 窄 N tile
    using ClusterShape = Shape<_2, _1, _1>;        // 2 SM 协作
    using KernelSchedule = ...PingpongFP8Blockwise; // Pingpong 调度
};

// 配置 2：H20 + 大 K → Pingpong 调度
struct MmaConfigH20LargeK {
    using MmaTileShape = Shape<_64, _128, _128>;
    using ClusterShape = Shape<_2, _1, _1>;
    using KernelSchedule = ...PingpongFP8Blockwise;
};

// 配置 3：通用（大 M + H100/H800）→ Cooperative 调度
struct MmaConfigHx00AndH20SmallK {
    using MmaTileShape = Shape<_128, _128, _128>;  // 大方块 tile
    using ClusterShape = Shape<_1, _2, _1>;        // 列方向 2 SM 协作
    using KernelSchedule = ...CooperativeFP8Blockwise;
};
```

---

## CUTLASS 1.0 vs 2.0 对比

CUTLASS 1.0 和 2.0 是两个完全不同的架构设计。以下是核心区别：

### 架构设计

| 方面 | CUTLASS 1.0 | CUTLASS 2.0 |
|------|-------------|-------------|
| **设计哲学** | 手写的高性能 GEMM 模板 | 可组合的组件化 GEMM 框架 |
| **代码结构** | 扁平的模板特化 | 分层的 Threadblock → Warp → Instruction |
| **Group GEMM** | ❌ 不支持 | ✅ 原生支持（`GemmGrouped`） |
| **Batch GEMM** | ✅ 基础支持 | ✅ 完整支持 |
| **目标架构** | SM50~SM75（Volta 及以前） | SM75~SM80（Turing、Ampere） |

### 数据类型支持

| 方面 | CUTLASS 1.0 | CUTLASS 2.0 |
|------|-------------|-------------|
| **FP16** | ✅ | ✅ |
| **BF16** | ❌ | ✅ |
| **TF32** | ❌ | ✅（A100 Tensor Core） |
| **INT8/INT4** | ❌ | ✅ |
| **FP8** | ❌ | ❌（需要 3.x） |

### Tensor Core 支持

| 方面 | CUTLASS 1.0 | CUTLASS 2.0 |
|------|-------------|-------------|
| **Volta (SM70)** | `mma.sync` | `mma.sync`（统一接口） |
| **Turing (SM75)** | 有限支持 | 完整 `mma.sync` |
| **Ampere (SM80)** | ❌ | ✅ `mma.sync` + `ldmatrix` |
| **指令抽象** | 手动嵌入 PTX | `MmaTensorOp` 模板自动选择 |

### 流水线（Software Pipelining）

```
CUTLASS 1.0：仅支持 2-stage 双缓冲
   Shared Memory：[Buffer A] [Buffer B]
   Stage 0: 加载 Buffer A ←→ 计算 Buffer B
   Stage 1: 加载 Buffer B ←→ 计算 Buffer A

CUTLASS 2.0：支持多 stage（2~6）异步流水线
   Shared Memory：[Buffer 0] [Buffer 1] [Buffer 2] [Buffer 3]
   利用 cp.async（异步拷贝指令）实现更深的流水线
   更好地隐藏 Global Memory → Shared Memory 的延迟
```

### Epilogue 设计

| 方面 | CUTLASS 1.0 | CUTLASS 2.0 |
|------|-------------|-------------|
| **可定制性** | 固定的 alpha*AB + beta*C | 完全可定制的 Epilogue Functor |
| **融合操作** | ❌ | ✅ 可融合 bias、激活函数、量化等 |
| **向量化** | 固定宽度 | 可配置的向量宽度（Alignment） |

### Group GEMM 支持——最关键的区别

```
CUTLASS 1.0：
  ❌ 没有 Group GEMM 支持
  ❌ 只能通过 Batch GEMM 模拟（要求所有子问题 M, N, K 相同）
  ❌ MoE 只能串行启动多个 kernel，或者 pad 到相同大小后用 Batch GEMM

  for (int i = 0; i < num_experts; i++) {
      // 逐个启动 kernel，GPU 利用率低
      cutlass_gemm(A[i], B[i], C[i]);
  }

CUTLASS 2.0：
  ✅ 原生 GemmGrouped 支持
  ✅ 每个子问题可以有不同的 M, N, K
  ✅ 所有子问题在一次 kernel 启动中完成
  ✅ 支持 kDeviceOnly 动态调度，自动负载均衡

  // 一次调用，所有专家并行计算
  GemmGrouped gemm_op;
  gemm_op.initialize(args);  // 包含所有子问题的描述
  gemm_op();                  // 单次 kernel 启动
```

### 对 MoE 的实际影响

| 场景 | CUTLASS 1.0 方案 | CUTLASS 2.0 方案 |
|------|-----------------|-----------------|
| **DeepSeek-V3（256 个专家）** | 256 次 kernel 启动 | 1 次 kernel 启动 |
| **不同专家 token 数差异大** | 必须 pad 到最大值，浪费计算 | 每个专家用实际 M 值，无浪费 |
| **kernel 启动开销** | 256 × ~5μs = ~1.3ms | 1 × ~5μs = ~5μs |
| **GPU 利用率** | 低（每次 kernel 太小） | 高（所有 tile 统一调度） |

### 总结

CUTLASS 1.0 本质上只是一个"可定制的单 GEMM 模板库"，而 CUTLASS 2.0 是一个"可组合的 GEMM 框架"。2.0 的关键创新有三点：
1. **Group GEMM**：支持不同大小的子问题统一调度
2. **多 stage 异步流水线**：利用 `cp.async` 指令深度隐藏延迟
3. **可组合的组件设计**：Mainloop、Epilogue、Scheduler 等组件可以独立替换

这三点使得 CUTLASS 2.0 成为 MoE 推理的基础设施。而 SGLang 使用的 CUTLASS 3.x 则在 2.0 基础上进一步利用了 Hopper/Blackwell 的硬件特性（TMA、warpgroup MMA 等）。

---

## CUTLASS 2.0 vs 3.x 对比（SGLang 使用的版本）

SGLang 实际使用的是 CUTLASS 3.x。以下是 2.0 和 3.x 的关键区别：

| 方面 | CUTLASS 2.0 | CUTLASS 3.x（SGLang 使用） |
|------|-------------|---------------------------|
| **布局描述** | `GemmCoord` (M, N, K) | CuTe `Layout` + `GroupProblemShape` |
| **调度策略** | `kDeviceOnly` / `kHostPrecompute` | `PersistentScheduler` + warpgroup |
| **数据搬运** | Shared Memory + warp MMA | TMA（Tensor Memory Accelerator）硬件加速 |
| **计算单元** | warp 级 MMA（32 线程） | warpgroup 级 MMA（128 线程） |
| **目标架构** | SM80（A100） | SM90（H100）/ SM100（B200） |
| **API 风格** | 单一模板 `DefaultGemmGrouped` | 模块化 `CollectiveBuilder` 组装 |
| **FP8 支持** | ❌ | ✅ 原生 FP8 E4M3/E5M2 |
| **Block 量化** | ❌ | ✅ per-block scale factor |
| **Cluster 支持** | ❌ | ✅ SM cluster（多 SM 协作） |

底层 **概念** 是相同的：将所有子问题的 tile 展平到统一空间，将每个 threadblock 映射到特定的子问题和 tile 坐标，然后执行 GEMM tile 计算。区别在于利用新硬件特性的底层实现细节。
