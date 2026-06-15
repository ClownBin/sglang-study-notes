# MiniMax-M3 推理实现调用路径分析

本文基于当前仓库静态代码梳理 MiniMax-M3 在 SGLang 中的推理/生成调用链。这里的“推理”指模型 inference/generation 执行链路；`reasoning_parser="minimax-m3"` 只属于 OpenAI 响应解析层，文末单独说明。

## 1. 关键结论

MiniMax-M3 在本仓库的核心实现分为三层：

- 模型层：`python/sglang/srt/models/minimax_m3.py`
  - 文本模型入口：`MiniMaxM3SparseForCausalLM`
  - 语言骨干：`MiniMaxM3Model`
  - 单层：`MiniMaxM3DecoderLayer`
  - attention：`MiniMaxM3Attention`
  - MoE/MLP：`MiniMaxM3MoE` / `MiniMaxM3MLP`
- attention 后端层：`python/sglang/srt/layers/attention/minimax_sparse_backend.py`
  - `MiniMaxSparseAttnBackend` 负责稀疏层 prefill/decode。
  - `MiniMaxHybridAttnBackend` 按 layer id 在 dense backend 与 sparse backend 间分发。
- 稀疏算子和 KV cache 层：
  - 稀疏主流程：`python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py`
  - Triton prefill/decode kernels：`prefill/*`、`decode/*`
  - KV pool：`python/sglang/srt/mem_cache/memory_pool.py::MiniMaxSparseKVPool`
  - 可选 JIT 优化：`python/sglang/jit_kernel/minimax_qknorm_rope.py`、`minimax_store_kv_index.py`、`minimax_decode_topk.py`

MiniMax-M3 的核心特殊点是“dense/sparse 混合 attention”。稀疏层除了普通 Q/K/V 分支，还会额外构造 index Q/K/(V) 分支。index 分支先选出 top-k KV blocks，再由主 attention 只对这些 block 做 sparse attention。

## 2. 从请求到模型 forward 的通用调用链

HTTP `/generate` 入口：

1. `python/sglang/srt/entrypoints/http_server.py:795`
   - `generate_request(obj, request)` 接收请求。
   - 流式请求迭代 `_global_state.tokenizer_manager.generate_request(...)`。

2. `python/sglang/srt/managers/tokenizer_manager.py:575`
   - `TokenizerManager.generate_request(...)`
   - 归一化请求参数，tokenize。
   - 单请求走 `_tokenize_one_request` 后 `_send_one_request(tokenized_obj)`，等待 detokenizer/worker 返回。

3. `python/sglang/srt/managers/scheduler.py:1322`
   - `init_request_dispatcher()` 将 `TokenizedGenerateReqInput` 分发到 `handle_generate_request`。

4. `python/sglang/srt/managers/scheduler.py:1961`
   - `handle_generate_request(...)`
   - 把 tokenized request 转成调度内部的 `Req`，进入 waiting/running batch。

5. `python/sglang/srt/managers/scheduler.py:1471`
   - `event_loop_normal()` 循环收请求，调用 `get_next_batch_to_run()`，再调用 `run_batch(batch)`。

6. `python/sglang/srt/managers/scheduler.py:3055`
   - `run_batch(batch)`
   - generation 路径下调用 `self.model_worker.forward_batch_generation(batch)`。

7. `python/sglang/srt/managers/tp_worker.py:466`
   - `TpModelWorker.forward_batch_generation(...)`
   - `ForwardBatch.init_new(batch, self.model_runner)` 将 `ScheduleBatch` 转成 GPU tensor 为主的 `ForwardBatch`。
   - 调用 `self.model_runner.forward(forward_batch)`。
   - 非 prefill-only 时再调用 `self.model_runner.sample(...)` 采样下一 token。

8. `python/sglang/srt/model_executor/model_runner.py:3434`
   - `ModelRunner.forward(...)`
   - 内部调用 `_forward_raw(...)`。

9. `python/sglang/srt/model_executor/model_runner.py:3525`
   - `_forward_raw(...)`
   - 建立 `ForwardContext(attn_backend=self.attn_backend)`。
   - 按 `forward_batch.forward_mode` 分发到：
     - decode：`forward_decode(...)`
     - extend/prefill：`forward_extend(...)`
     - idle：`forward_idle(...)`

10. `python/sglang/srt/model_executor/model_runner.py:3229` / `3279`
    - `forward_decode(...)` 或 `forward_extend(...)`
    - 初始化 attention metadata：`self.attn_backend.init_forward_metadata(forward_batch)`。
    - 调用 `self.model.forward(input_ids, positions, forward_batch, ...)`。

## 3. 模型类解析和加载路径

MiniMax-M3 模型类不是硬编码在请求链路里，而是通过 HF config 的 `architectures` 解析：

1. `python/sglang/srt/models/registry.py:94`
   - `import_model_classes(...)` 扫描 `sglang.srt.models` 下每个模块的 `EntryClass`。
   - `python/sglang/srt/models/minimax_m3.py:1923` 注册 `MiniMaxM3SparseForCausalLM`。
   - `python/sglang/srt/models/minimax_m3_vl.py:395` 注册 `MiniMaxM3SparseForConditionalGeneration`。

2. `python/sglang/srt/model_loader/utils.py:195`
   - `get_model_architecture(model_config)` 从 `model_config.hf_config.architectures` 中解析模型类。
   - 对 MiniMax-M3 文本模型通常解析到 `MiniMaxM3SparseForCausalLM`。
   - 对 VL 模型解析到 `MiniMaxM3SparseForConditionalGeneration`。

3. `python/sglang/srt/model_loader/__init__.py:23`
   - `get_model(...)` 选择 loader 并调用 `loader.load_model(...)`。

4. `python/sglang/srt/model_loader/loader.py:719`
   - `DefaultModelLoader.load_model(...)`
   - `_initialize_model(...)` 构建模型对象。
   - `load_weights_and_postprocess(...)` 调用模型自身 `load_weights(...)`。

5. `python/sglang/srt/models/minimax_m3.py:1747`
   - `MiniMaxM3SparseForCausalLM.load_weights(...)`
   - 合并 checkpoint 中分开的 `.q_proj/.k_proj/.v_proj` 到 `qkv_proj`。
   - 稀疏模型额外合并 `.index_q_proj/.index_k_proj/.index_v_proj` 到 `index_qkv_proj`。
   - MoE checkpoint 的 `block_sparse_moe` 名称映射到本实现的 `mlp`。
   - 最后调用 `build_minimax_fused_qkv_index(self)`，为稀疏层尝试构建主 QKV + index QKV 单 GEMM 融合。

## 4. MiniMax-M3 专用初始化：server args、KV pool、attention backend

### 4.1 server args 默认调整

`python/sglang/srt/server_args.py:1968` 检测架构为：

- `MiniMaxM3SparseForCausalLM`
- `MiniMaxM3SparseForConditionalGeneration`

随后做模型特定调整：

- 自动读取 checkpoint quantization；如果是 MXFP8 且用户未显式关闭 quantization，则设置 `self.quantization`。
- ROCm 上默认 `attention_backend="triton"`；MXFP8 的 MoE runner 默认走 `triton`。
- SM100/Blackwell 上默认 `attention_backend="fa4"`，`page_size=128`，MXFP8 MoE runner 可走 `deep_gemm`。
- bf16 权重下避免 `deep_gemm` MoE runner，默认/强制改为 `triton`。

### 4.2 KV pool 初始化

`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:649`：

- 如果 `is_minimax_sparse(self.model_config.hf_config)` 为真，则创建 `MiniMaxSparseKVPool`。
- 稀疏配置来自 `get_minimax_sparse_attention_config(...)`。
- dense/sparse layer ids 来自 `get_minimax_sparse_layer_ids(...)`。
- `sparse_disable_index_value` 决定哪些 sparse layer 只需要 index K，不需要 index V。

`python/sglang/srt/configs/model_config.py:130` 判断 MiniMax sparse 架构；`python/sglang/srt/configs/model_config.py:138` 读取 sparse config；`python/sglang/srt/configs/model_config.py:152` 根据 `sparse_attention_freq` 切 dense/sparse 层。

### 4.3 attention backend 包装

`python/sglang/srt/model_executor/model_runner.py:2363` 初始化 attention backend。

`python/sglang/srt/model_executor/model_runner.py:2430`：

- `_get_attention_backend_from_str(...)` 先构建普通 dense backend。
- 再调用 `attn_backend_wrapper(self, full_attention_backend)`。

`python/sglang/srt/layers/attention/attention_registry.py:246`：

- 如果 `is_minimax_sparse(runner.model_config.hf_config)`，包装为：
  - dense backend：原 attention backend，例如 fa4/triton/flashinfer 等
  - sparse backend：`MiniMaxSparseAttnBackend`
  - hybrid backend：`MiniMaxHybridAttnBackend(dense, sparse, sparse_layer_ids)`

最终模型内部 `RadixAttention.forward(...)` 通过当前 `ForwardContext` 获取这个 hybrid backend。

## 5. 模型 forward 主路径

### 5.1 顶层 CausalLM

`python/sglang/srt/models/minimax_m3.py:1715`

`MiniMaxM3SparseForCausalLM.forward(...)`：

1. 调用 `self.model(input_ids, positions, forward_batch, ...)`，进入 `MiniMaxM3Model`。
2. 如果是 PP 最后一 rank，调用 `self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch, ...)`。
3. 输出 `LogitsProcessorOutput`，由 `TpModelWorker` 后续采样。

### 5.2 语言骨干

`python/sglang/srt/models/minimax_m3.py:1551`

`MiniMaxM3Model.forward(...)`：

1. 第一 PP rank 做 token embedding。
2. 遍历 `[start_layer, end_layer)`：
   - `layer(positions, forward_batch, hidden_states, residual, ...)`
3. 最后一 PP rank 做最终 norm。
4. 返回 hidden states。

### 5.3 Decoder layer

`python/sglang/srt/models/minimax_m3.py:1424`

`MiniMaxM3DecoderLayer.forward(...)`：

1. `LayerCommunicator.prepare_attn...` 做 attention 前的 norm/residual/通信准备。
2. 调 `self.self_attn(...)`。
3. `LayerCommunicator.prepare_mlp(...)` 做 MLP/MoE 前处理。
4. 调 `self.mlp(...)`：
   - MoE 层：`MiniMaxM3MoE`
   - dense MLP 层：`MiniMaxM3MLP`
5. 根据通信策略做 all-reduce/reduce-scatter/postprocess。

### 5.4 Attention 层初始化

`python/sglang/srt/models/minimax_m3.py:491`

`MiniMaxM3Attention` 支持两种模式：

- dense 层：
  - 只有普通 `qkv_proj` + `o_proj`。
- sparse 层：
  - 普通 `qkv_proj` + `o_proj`。
  - 额外 `index_qkv_proj` + 可选 `index_o_proj`。
  - index 分支用于 block 选择和可选 index value 输出。

是否 sparse 由 `MiniMaxM3DecoderLayer.__init__` 在 `python/sglang/srt/models/minimax_m3.py:1332` 读取 `config.sparse_attention_config` 后决定。

## 6. Attention forward 详细调用路径

### 6.1 `MiniMaxM3Attention.forward`

`python/sglang/srt/models/minimax_m3.py:1293`

`forward(...)` 只是两段式：

1. `forward_prepare(...)`
2. `forward_core(...)`

### 6.2 `forward_prepare`

`python/sglang/srt/models/minimax_m3.py:1122`

主要工作：

1. 如果 `self._fused_qkv_index` 已构建，执行一次 `fused_qkv_index_proj(hidden_states)`，产出：
   - main Q/K/V
   - index Q/K/(V)
2. 否则执行普通 `self.qkv_proj(hidden_states)`，稀疏层再单独执行 `self.index_qkv_proj(hidden_states)`。
3. 对 Q/K 做 QK norm + RoPE：
   - CUDA 条件满足时走 `minimax_qknorm_rope` JIT 融合。
   - ROCm 条件满足时走 `sglang.jit_kernel.minimax_m3.qk_norm_rope`。
   - 否则走普通 `_qk_norm` + `rotary_emb(...)`。
4. sparse 层还会对 index Q/K 做 norm + RoPE。
5. ROCm 下如果满足条件，`_sparse_qk_index_norm_rope_cache(...)` 可以在同一个 kernel 中完成 main Q/K、index Q/K 的 norm+rope，并把 sparse KV 写入 cache。
6. 返回 `inner_state`，供 `forward_core(...)` 使用。

### 6.3 `forward_core`

`python/sglang/srt/models/minimax_m3.py:1251`

dense 层：

1. `attn_output = self.attn(q, k, v, forward_batch)`
2. 可选 output gate。
3. `output, _ = self.o_proj(attn_output)`

sparse 层：

1. reshape main Q/K/V 为 `[tokens, heads, dim]`。
2. reshape index Q/K/(V)。
3. 调 `self.attn(q, k, v, forward_batch, idx_q=..., idx_k=..., idx_v=...)`。
4. `self.attn` 返回 `(idx_o, attn_output)`。
5. main 输出走 `self.o_proj(attn_output)`。
6. 如果 index value 没被禁用，`idx_o` 走 `self.index_o_proj(idx_o)`，最后 `output + idx_output`。

## 7. `RadixAttention` 到 hybrid backend

`python/sglang/srt/layers/radix_attention.py:109`

普通非 piecewise graph 路径：

```text
RadixAttention.forward(...)
  -> get_attn_backend().forward(q, k, v, self, forward_batch, save_kv_cache, **kwargs)
```

这里 `get_attn_backend()` 来自 `ForwardContext(attn_backend=...)`，MiniMax-M3 会拿到 `MiniMaxHybridAttnBackend`。

piecewise CUDA graph 路径：

- dense attention 走 `unified_attention_with_output(...)`。
- MiniMax-M3 sparse attention 有 `idx_q`，会走 `unified_sparse_attention_with_output(...)`，这是一个双输出 split op：同时写 main attention output 和 index output，再回到外层 `o_proj/index_o_proj`。

对应代码在 `python/sglang/srt/layers/radix_attention.py:127` 和 `python/sglang/srt/layers/radix_attention.py:283`。

## 8. Hybrid backend 的 dense/sparse 分发

`python/sglang/srt/layers/attention/minimax_sparse_backend.py:526`

`MiniMaxHybridAttnBackend.forward(...)`：

- 如果 `layer.layer_id in self.sparse_layer_ids`：
  - 调 `self.sparse.forward(...)`
- 否则：
  - 调 `self.dense.forward(...)`

`forward_extend(...)` 和 `forward_decode(...)` 也做同样分发：

- `python/sglang/srt/layers/attention/minimax_sparse_backend.py:606`
- `python/sglang/srt/layers/attention/minimax_sparse_backend.py:625`

## 9. Sparse backend: prefill 路径

入口：`python/sglang/srt/layers/attention/minimax_sparse_backend.py:286`

`MiniMaxSparseAttnBackend.forward_extend(...)`：

1. 判断当前 layer 是否禁用 index value：
   - `disable_value = layer.layer_id in self.disable_value_layer_ids`
2. 如果 KV 没被前面的 fused norm/rope/cache kernel 写过：
   - 调 `self.kv_pool.set_fused_kv_index_buffer(...)`
   - 写 main K/V + index K/(V)
3. 取 cache：
   - `k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)`
   - `idx_k_cache` 或 `idx_k_cache, idx_v_cache`
4. 根据 `extend_seq_lens` 构造 `cu_seqlens`、`seq_lens`、`prefix_lens`。
5. 调 `minimax_sparse_prefill(...)`。
6. 输出 reshape 成外层期望的二维输出：
   - `idx_o.reshape(original_num_tokens, -1)`
   - `o.reshape(original_num_tokens, -1)`

底层 `minimax_sparse_prefill` 在 `python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py:15`：

1. `flash_prefill_with_topk_index(...)`
   - index Q/K/(V) 计算每个 query block 的 top-k sparse block。
   - 同时可产生 index value 分支输出 `idx_o`。
2. 如果 index heads 多于 KV heads，则 `topk_index_reduce(...)` 合并 head group。
3. main sparse attention：
   - Blackwell + MSA 可用时：`msa_sparse_prefill_main(...)`
   - 否则：`flash_prefill_with_gqa_share_sparse(...)`

Triton main sparse prefill kernel 在 `python/sglang/srt/layers/attention/minimax_sparse_ops/prefill/topk_sparse.py:261`。

## 10. Sparse backend: decode 路径

入口：`python/sglang/srt/layers/attention/minimax_sparse_backend.py:434`

`MiniMaxSparseAttnBackend.forward_decode(...)`：

1. 调 `self.kv_pool.set_fused_kv_index_buffer(...)` 写当前 token 的 main K/V + index K/(V)。
2. 取 main cache 与 index cache。
3. 可能启用 dense sparse decode：
   - 条件：`self.use_dense_sparse_decode and k_cache.shape[1] == 1`
   - 此时 indexer 直接输出 dense backend 可用的 page table。
4. 可能使用 MSA decode：
   - `_use_msa_decode` 为真且没有 dense sparse decode。
   - MSA decode metadata 在 `init_forward_metadata_out_graph(...)` 预先准备。
5. 调 `minimax_sparse_decode(...)`。

底层 `minimax_sparse_decode` 在 `python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py:136`：

1. `flash_decode_with_topk_idx(...)`
   - index Q/K/(V) 计算 sparse block score。
   - 可用 `minimax_decode_topk` JIT radix-select 选 top-k。
   - dense sparse decode 时可用 `minimax_decode_topk_page_table` 直接输出 page table。
2. 如果 dense main attention 被启用：
   - 调 `dense_main_attn_fn(q, topk_idx, real_seq_lens)`。
3. 否则：
   - 必要时 `topk_index_reduce(...)`。
   - MSA 可用时走 `msa_sparse_decode_main(...)`。
   - 否则走 `flash_decode_with_gqa_share_sparse(...)`。

Triton main sparse decode kernel 在 `python/sglang/srt/layers/attention/minimax_sparse_ops/decode/topk_sparse.py:302`。

## 11. KV cache 实现

`python/sglang/srt/mem_cache/memory_pool.py:2794`

`MiniMaxSparseKVPool` 没有直接继承普通单一 KV pool 的存储方式，而是组合多个 sub-pool：

- `main_pool: MHATokenToKVPool`
  - dense layer 和 sparse layer 的 main K/V 都在这里。
- `index_kv_pool: MHATokenToKVPool`
  - sparse layer 的 index K/V。
  - 只给没有禁用 index value 的 sparse layer 分配。
- `index_k_pool: MHATokenToKOnlyPool`
  - sparse layer 的 index K。
  - 给 `sparse_disable_index_value` 指定的 sparse layer 使用，省掉 index V 内存。

写入路径：

- main K/V：`set_kv_buffer(...)`，见 `python/sglang/srt/mem_cache/memory_pool.py:2955`
- index K/V：`set_index_kv_buffer(...)`，见 `python/sglang/srt/mem_cache/memory_pool.py:2974`
- index K only：`set_index_k_buffer(...)`，见 `python/sglang/srt/mem_cache/memory_pool.py:3000`
- sparse 层统一写入：`set_fused_kv_index_buffer(...)`，见 `python/sglang/srt/mem_cache/memory_pool.py:3046`

`set_fused_kv_index_buffer(...)` 会尝试使用 `python/sglang/jit_kernel/minimax_store_kv_index.py:31` 的 `store_kv_index(...)`。条件不满足时 fallback 到 main KV 和 index KV/K 的分开写入。

## 12. MoE/MLP 路径

`MiniMaxM3DecoderLayer` 根据 `moe_layer_freq` 决定这一层的 MLP 是 MoE 还是 dense MLP：

- `python/sglang/srt/models/minimax_m3.py:1360`

MoE 路径：

- `python/sglang/srt/models/minimax_m3.py:307`
- `MiniMaxM3MoE.forward(...)`
  - DeepEP backend 启用时走 `forward_deepep(...)`
  - 否则走 `forward_normal(...)`

`forward_normal(...)`：

1. 可选 shared experts。
2. `router_logits = self._compute_router_logits(hidden_states)`
3. `topk_output = self.topk(hidden_states, router_logits)`
4. `final_hidden_states = self.experts(hidden_states, topk_output)`
5. 必要时 TP all-reduce。

dense MLP 路径：

- `python/sglang/srt/models/minimax_m3.py:241`
- `MiniMaxM3MLP.forward(...)`
  - `gate_up_proj`
  - activation：`silu` 或 `swigluoai`
  - `down_proj`

## 13. 多模态 MiniMax-M3 VL 旁路

VL 模型入口是 `python/sglang/srt/models/minimax_m3_vl.py:58` 的 `MiniMaxM3SparseForConditionalGeneration`。

它做两件事：

- 构造 vision tower：`MiniMaxVLVisionModel`
- 构造语言骨干：复用 `MiniMaxM3Model`

见 `python/sglang/srt/models/minimax_m3_vl.py:98` 和 `python/sglang/srt/models/minimax_m3_vl.py:111`。

因此，多模态请求多了图像/视频 processor 与 vision tower/projector 路径，但语言模型 token 进入 `MiniMaxM3Model` 后，dense/sparse attention、KV pool、MoE 这条推理核心路径与文本模型一致。

多模态 processor 对应：

- `python/sglang/srt/multimodal/processors/minimax_m3_vl.py:231`
- `MiniMaxM3VLProcessor.models = [MiniMaxM3SparseForConditionalGeneration]`

## 14. reasoning parser 与模型推理的区别

如果服务使用 OpenAI chat API，并设置 `--reasoning-parser auto` 或 `minimax-m3`，解析逻辑在：

- `python/sglang/srt/entrypoints/openai/serving_chat.py:1680`
- `python/sglang/srt/parser/reasoning_parser.py:503`
- `python/sglang/srt/parser/reasoning_parser.py:1102`

`MiniMaxM3Detector` 只识别 `<mm:think>` / `</mm:think>`，把模型输出拆成 `reasoning_content` 和普通 content。它不参与模型 forward、attention backend、KV cache 或 MoE 算子执行。

## 15. 总调用链速查

```text
HTTP /generate
  -> http_server.generate_request
  -> TokenizerManager.generate_request
  -> Scheduler.handle_generate_request
  -> Scheduler.event_loop_normal / event_loop_overlap
  -> Scheduler.run_batch
  -> TpModelWorker.forward_batch_generation
  -> ForwardBatch.init_new
  -> ModelRunner.forward
  -> ModelRunner._forward_raw
  -> ModelRunner.forward_extend / forward_decode
  -> self.model.forward
  -> MiniMaxM3SparseForCausalLM.forward
  -> MiniMaxM3Model.forward
  -> MiniMaxM3DecoderLayer.forward
  -> MiniMaxM3Attention.forward
  -> MiniMaxM3Attention.forward_prepare
  -> MiniMaxM3Attention.forward_core
  -> RadixAttention.forward
  -> MiniMaxHybridAttnBackend.forward
       dense layer:
         -> dense backend forward_extend / forward_decode
       sparse layer:
         -> MiniMaxSparseAttnBackend.forward_extend / forward_decode
         -> MiniMaxSparseKVPool.set_fused_kv_index_buffer
         -> minimax_sparse_prefill / minimax_sparse_decode
         -> flash_*_with_topk_idx
         -> topk_index_reduce
         -> MSA main sparse attention 或 Triton gqa_share_sparse attention
  -> o_proj / index_o_proj
  -> MLP/MoE
  -> final norm
  -> LogitsProcessor
  -> ModelRunner.sample
  -> detokenizer / response
```

## 16. 主要文件清单

- `python/sglang/srt/models/minimax_m3.py`
  - MiniMax-M3 文本模型、attention、MoE、weight loading。
- `python/sglang/srt/models/minimax_m3_vl.py`
  - MiniMax-M3 VL 入口，复用 `MiniMaxM3Model` 语言骨干。
- `python/sglang/srt/multimodal/processors/minimax_m3_vl.py`
  - MiniMax-M3 VL 多模态输入处理。
- `python/sglang/srt/configs/model_config.py`
  - 判断 MiniMax sparse 架构，读取 sparse attention config。
- `python/sglang/srt/server_args.py`
  - MiniMax-M3 的硬件/backend/quantization 默认调整。
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
  - 为 MiniMax-M3 创建 `MiniMaxSparseKVPool`。
- `python/sglang/srt/layers/attention/attention_registry.py`
  - 为 MiniMax-M3 包装 hybrid attention backend。
- `python/sglang/srt/layers/radix_attention.py`
  - 模型层到当前 attention backend 的统一入口；包含 sparse 双输出 split op。
- `python/sglang/srt/layers/attention/minimax_sparse_backend.py`
  - MiniMax sparse/hybrid attention backend。
- `python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py`
  - sparse prefill/decode 高层算子编排。
- `python/sglang/srt/layers/attention/minimax_sparse_ops/prefill/*`
  - prefill indexer 与 main sparse attention Triton kernels。
- `python/sglang/srt/layers/attention/minimax_sparse_ops/decode/*`
  - decode indexer/top-k 与 main sparse attention Triton kernels。
- `python/sglang/srt/layers/attention/minimax_sparse_ops/msa.py`
  - Blackwell MSA/fmha_sm100 可选 fast path。
- `python/sglang/srt/mem_cache/memory_pool.py`
  - `MiniMaxSparseKVPool`、index K-only pool、fused KV/index store。
- `python/sglang/jit_kernel/minimax_qknorm_rope.py`
  - CUDA fused QK norm + RoPE。
- `python/sglang/jit_kernel/minimax_m3/qk_norm_rope.py`
  - ROCm MiniMax-M3 fused QK/index norm + RoPE。
- `python/sglang/jit_kernel/minimax_store_kv_index.py`
  - CUDA fused main KV + index K/V cache store。
- `python/sglang/jit_kernel/minimax_decode_topk.py`
  - decode top-k radix-select / page-table transform。
