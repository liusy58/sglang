# SGLang 中的 Context Parallel（上下文并行）

## 一、什么是 Context Parallel (CP)

Context Parallel 是一种 **沿序列维度（token 维度）切分** 的并行方式，专门用来解决 **长序列** 场景下单卡放不下、或单卡 attention 计算太慢的问题。

### 与其他并行方式的对比

| 并行方式 | 切分维度 | 解决的问题 |
| --- | --- | --- |
| TP (Tensor Parallel) | 权重矩阵 / head 维度 | 单卡放不下权重、计算分摊 |
| DP (Data Parallel) | 不同的 request / batch | 吞吐 |
| PP (Pipeline Parallel) | 模型的层 | 单卡放不下整个模型 |
| **CP (Context Parallel)** | **同一条序列的 token** | **单条序列太长**（KV cache 太大、attention O(N²) 太慢） |

### 核心难点：attention

attention 中每个 query token 需要看到它前面 **所有** 的 key/value token。如果把序列切到不同卡上，每张卡只有自己那部分 KV，就必须通过通信把别人持有的 KV 拿过来（或把 query / 部分结果发出去）才能算出完整的 attention。

而 MLP / MoE 部分是逐 token 独立的，所以天然可并行，不需要跨卡通信。

### 负载均衡问题（zigzag 的由来）

在 causal attention 下，序列后半部分的 token 要看的 KV 比前半部分多。如果简单地把序列前一半给 rank0、后一半给 rank1，rank1 的计算量会远大于 rank0，负载不均。

解决办法是 **zigzag（之字形）切分**：把序列切成 `2 * cp_size` 个 block，每个 rank 各拿一个"靠前"的 block 和一个对称"靠后"的 block，使每张卡的计算量大致相等。

---

## 二、SGLang 是如何实现 Context Parallel 的

SGLang 的 CP 主要面向 **prefill 阶段的长序列**，目前主要用在 MLA（DeepSeek V2/V3）以及 DSA（DeepSeek V3.2）这类模型上。

### 1. 启动参数（用户视角）

- `--attention-context-parallel-size` / `--attn-cp-size`：CP 的并行度（从 TP 里划分出来）。`server_args.py` 中要求 `tp_size % (dp_size * attn_cp_size) == 0`。
- `--enable-prefill-context-parallel`：开启通用 prefill CP，配合 `--prefill-cp-mode`（默认 `in-seq-split`，即 zigzag）。
- `--enable-dsa-prefill-context-parallel` + `--dsa-prefill-cp-mode`：DeepSeek V3.2 (DSA/NSA) 专用，支持：
  - `round-robin-split`：按 `token_idx % cp_size` 分发，支持多 batch prefill、fused MoE、FP8 KV cache；
  - `in-seq-split`：zigzag 切分。

### 2. 通信组的建立（`distributed/parallel_state.py`）

在 `initialize_model_parallel` 中，从 TP 组里再切出一个 `_ATTN_CP` 通信组（`get_attn_cp_group()`）。

例如 8 卡，`attn_cp_size=2`、`attn_tp_size=4` 时：

- attention CP 组：`[g0,g4], [g1,g5], [g2,g6], [g3,g7]`（CP 伙伴跨步排列）。

> 细节：当 `moe_dp_size < attn_cp_size` 时，`_MOE_DP` 直接复用 `_ATTN_CP` 组——因为 MoE 之前需要把 CP 各 rank 的 token 重新聚合，复用已有的 DP allgather/scatter 即可。

### 3. 核心数据流（以 zigzag / in-seq-split 为例）

实现集中在 `python/sglang/srt/layers/utils/cp_utils.py`：

**(a) 切分输入** — `cp_split_and_rebuild_data` / `cp_split_and_rebuild_position` / `cp_round_robin_input_ids`

把 input_ids、positions 按 CP 元数据切成 `2*cp_size` 个 block，每张卡按 `zigzag_index` 取出属于自己的（prev + next）两块，重排成 `[所有 prev 块, 所有 next 块]` 的布局。

**(b) 准备元数据** — `prepare_context_parallel_metadata(...)`

生成 `ContextParallelMetadata`，包含：

- `split_list` / `zigzag_index` / `cp_reverse_index`：切分和复原索引；
- 每条序列 prev/next 两半各自的 `cu_seqlens_q`、`kv_len`（KV 长度，已把 radix-cache 命中的 prefix 长度算进去）、`max_seqlen_q` 等，供 FlashAttention 的变长接口使用；
- `per_rank_actual_token` / `max_rank_len`：用于 allgather 的对齐 padding。

它对每条序列独立切成 `cp_segment_num = 2*cp_size` 个 block，前 `L % cp_segment_num` 个 block 多分 1 个 token，保证均匀。

**(c) attention 计算** — `cp_attn_forward_extend(...)`

把本卡的 q 按 `total_q_prev_tokens` 切成 prev/next 两半，分别用对应的 `cu_seqlens_q` 和 `kv_len` 调用一次后端 attention（如 FA3），再 concat。期间需要拿到完整 KV：

- `cp_allgather_and_save_kv_cache` / `cp_all_gather_reorganized_into_tensor_kv_cache`：通过 `_ATTN_CP` 组做 **async allgather**，把各 rank 的 KV cache（必要时含 index_k、hidden_states）聚合，先 padding 到统一 `max_rank_len` 再通信，通信完去掉 padding 并按真实 token 数重组。

**(d) 结果复原** — `cp_all_gather_rerange_output` / `cp_reverse_index`

算完后把各卡 zigzag 布局的输出 allgather 回来，按 `cp_reverse_index` 还原成原始 token 顺序，交给后续 MLP / MoE / 采样。

### 4. 两种切分模式

- **`in-seq-split`（zigzag，默认）**：序列内对称切块，负载均衡好，是 MLA prefill CP 的主路径。
- **`round-robin-split`**：按 `token_idx % cp_size` 轮转分发（`input_ids[cp_rank::cp_size]`），DSA(V3.2) 专用，能兼容多 batch prefill、fused MoE 和 FP8 KV cache。

### 5. 关键文件清单

| 文件 | 作用 |
| --- | --- |
| `distributed/parallel_state.py` | 建立 `_ATTN_CP` / `_ATTN_TP` / `_MOE_DP` 通信组 |
| `layers/dp_attention.py` | `get_attention_cp_group/rank/size`、`attn_cp_all_gather_into_tensor` 等基础通信封装 |
| `layers/utils/cp_utils.py` | CP 的主体逻辑（切分、元数据、allgather、attention 调度） |
| `layers/communicator_dsa_cp.py`、`layers/attention/dsa/utils.py` | DSA(V3.2) 专用 CP |
| `layers/attention/flashattention_backend.py`、`models/deepseek_v2.py` / `deepseek_v4.py` | 模型 / 后端里实际调用 CP 的地方 |
| `test/registered/cp/`、`test/registered/kernels/test_mla_cp_fa3_parity.py`、`test_cp_prefix_len_fa3_parity.py` | 测试参考 |

---

## 三、一句话总结

SGLang 的 CP 是「prefill 阶段沿 token 维度切分」，用 zigzag（或 round-robin）做负载均衡切块，每张卡只算自己那部分 query，通过 `_ATTN_CP` 组 allgather 把完整 KV cache 聚合起来完成 attention，算完再按反向索引还原 token 顺序送入 MLP / MoE。它主要服务于长序列的 MLA（DeepSeek V2/V3）和 DSA（V3.2）模型。
