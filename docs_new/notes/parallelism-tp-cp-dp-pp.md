# SGLang 并行维度笔记：TP / CP / DP / PP 的关系与切分

## 一、核心概念：`tp_size` 是 world size，不是真实 TP

SGLang 里最容易踩的坑：命令行参数 `--tp-size`（`tp_size`）**不是真实的张量并行大小**，
而是 **每个 DP 副本 × 每个 PP stage 内部的 GPU 总数**。代码注释多处强调：

> "The tp_size is the world size, not the real tensor parallel size"

CP、真实 attn-TP、attn-DP 等都是从 `tp_size` 这块"大蛋糕"里再切分出来的。

## 二、通项公式

### 1. 总 GPU 数（world size）

来源：`python/sglang/srt/ray/engine.py:108`

```
总 GPU 数 = pp_size × dp_size × tp_size        （常规）
          = pp_size × tp_size                  （enable_dp_attention，DP 折叠进 TP）
```

### 2. `tp_size` 内部再分解（attention 侧）

来源：`python/sglang/srt/distributed/parallel_state.py:2211`、`python/sglang/srt/disaggregation/common/conn.py:1476`

```
tp_size = attn_dp_size × attn_cp_size × attn_tp_size(真实)

attn_tp_size(真实) = tp_size / attn_dp_size / attn_cp_size
```

### 3. 记忆口诀

> 外层相乘 `PP × DP × TP = 总卡数`；`tp_size` 是"每副本卡数"这块蛋糕，
> CP 和真实 TP 都从这块蛋糕里再切，所以必须整除 `tp_size`。

## 三、参数速查表

| 维度 | 参数 | 含义 |
|------|------|------|
| PP | `pp_size` | 流水线级数 |
| DP | `dp_size` | 数据并行副本数 |
| "TP"(world 内) | `tp_size` | 每个 DP 副本 × 每个 PP stage 内 GPU 数（**非真实 TP**） |
| Attention CP | `attn_cp_size`（`--attention-context-parallel-size`） | 从 `tp_size` 切出的上下文并行 |
| Decode CP | `dcp_size`（`--decode-context-parallel-size`） | 解码阶段 CP（平台受限：CUDA / HIP） |
| Prefill CP | `--enable-prefill-cp` + `--cp-strategy`(zigzag/interleave) | prefill 阶段 CP |
| 真实 attn-TP | 派生量 | `tp_size / attn_dp_size / attn_cp_size` |
| MoE DP | `moe_dp_size` | MoE 数据并行 |

## 四、整除约束（各种 assert 的来源）

来源：`python/sglang/srt/server_args.py:5401`、`5404`、`7423` 等

```
tp_size % attn_cp_size             == 0
tp_size % (dp_size × attn_cp_size) == 0
tp_size % nnodes                   == 0     # 每节点 GPU 数整齐
tp_size % moe_dp_size              == 0
ep_size × moe_dp_size             <= tp_size
```

其他限制：

- `attn_cp_size != moe_dp_size` 时，要求 `moe_dp_size == 1`。
- CP（`attn_cp_size>1` 或 `moe_dp_size>1`）下不支持 `enable_aiter_allreduce_fusion`。
- MoE 的 CP 路径不支持 PP（要求 `pp_size == 1`）。

## 五、TP 与 CP 能否设不同值？

**可以，但两者不正交**。`attn_cp_size` 不必等于 `tp_size`，但必须整除 `tp_size`
（以及 `dp_size × attn_cp_size` 整除 `tp_size`）。

例子：

- `tp_size=8, attn_cp_size=2` ✅ → 真实 attn-TP = 8/2 = 4
- `tp_size=8, attn_cp_size=1`（默认）✅ → 纯 TP，无 CP
- `tp_size=8, attn_cp_size=3` ❌ → 8 不能被 3 整除

## 六、案例：`tp=cp=16` 怎么切？

代入公式：`attn_tp_size = 16 / 16 / 1 = 1` → 真实注意力 TP = 1，16 张卡全部做序列维度的上下文并行。

代码走 `attn_cp_size == tp_size` 捷径分支（`python/sglang/srt/distributed/parallel_state.py:2217`）：

```python
if attn_cp_size == tensor_model_parallel_size:
    _ATTN_CP = _TP        # CP 组直接 = TP 组
```

分组结果（单副本、无 DP/PP）：

| 组 | ranks |
|----|-------|
| TP group | `[g0 … g15]` |
| attn_cp group | `[g0 … g15]`（同 TP 组） |
| attn_tp group | 每卡自成一组（`attn_tp_size=1`） |

数据切分方式：

- **Attention**：权重**不切分**（每卡持完整 QKV/O 权重），16 张卡沿**序列长度**切分，
  每卡处理 1/16 的 token，attention 计算时经 CP 组交换 K/V。
- **MoE / FFN**：仍按 `tp_size=16`（或 `ep_size`/`moe_dp_size`）在 16 卡上做张量/专家并行；
  CP 与 MoE 分组桥接见 `python/sglang/srt/distributed/parallel_state.py:2291`
  （`attn_cp_size > moe_dp_size` 时 CP ranks 在进 MoE 前先 all-gather 交换 token）。

**适用场景**：`attn_tp=1` 意味着单卡要放下完整注意力权重，因此这种配置多用于**长序列**——
用 CP 分摊超长上下文的 KV，而非分摊权重。

## 七、关键源码位置

| 内容 | 文件:行 |
|------|---------|
| world size 公式 | `python/sglang/srt/ray/engine.py:108` |
| `attn_tp_size` 派生 & 分组构造 | `python/sglang/srt/distributed/parallel_state.py:2209-2268` |
| CP↔MoE 桥接分组 | `python/sglang/srt/distributed/parallel_state.py:2291` |
| 整除校验 | `python/sglang/srt/server_args.py:5397-5433` |
| 参数定义（`attn_cp_size`/`dcp_size`） | `python/sglang/srt/server_args.py:934-975` |
