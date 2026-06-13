---
title: "Ascend NPU 与 sgl-kernel-npu 桥接链路"
description: "DeepSeek 在 Ascend NPU 场景下，SGLang 主仓如何接入 sgl-kernel-npu 的 DeepEP/FuseEP 与自定义 kernel。"
icon: "cpu"
---

# Ascend NPU 与 sgl-kernel-npu 桥接链路

本文解释 DeepSeek 在 Ascend NPU 场景下，SGLang 主仓如何接到 `sgl-kernel-npu` 的 DeepEP/FuseEP 与自定义 kernel。

## 1. 总体结构

```text
sglang/
  -> hardware_backend/npu
  -> torch_npu
  -> torch.ops.npu.*
  -> deep_ep Python package

sgl-kernel-npu/
  -> python/deep_ep
  -> python/sgl_kernel_npu
  -> csrc/deepep
  -> csrc/mla_preprocess
  -> csrc/batch_matmul_transpose
```

`sglang/` 负责“什么时候调用”，`sgl-kernel-npu/` 负责“底层怎么高效执行”。

## 2. NPU 默认参数

NPU 后端默认设置在：

```text
sglang/python/sglang/srt/hardware_backend/npu/utils.py
```

关键默认值：

```python
args.attention_backend = "ascend"
args.prefill_attention_backend = "ascend"
args.decode_attention_backend = "ascend"
if args.page_size is None:
    args.page_size = 128
```

这意味着 Ascend 上 DeepSeek attention 不会走 CUDA 的 flashinfer/flashmla/triton 路径，而是进入 `AscendAttnBackend`。

## 3. DeepSeek MLA 到 Ascend attention backend

SGLang 主仓路径：

```text
DeepseekV2AttentionMLA.forward
  -> dispatch_attn_forward_method
  -> AttnForwardMethod.MLA_NPU / DSA_NPU
  -> hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
  -> AscendAttnBackend.forward_extend / forward_decode
  -> torch_npu / torch.ops.npu
```

核心文件：

- `sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
- `sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
- `sglang/python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`

### MLA prepare 阶段

`deepseek_v2_attention_mla_npu.py` 负责把 DeepSeek MLA 的张量准备成 NPU backend 可消费的形式：

```text
forward_mla_prepare_npu
  -> fused qkv split
  -> RMSNorm
  -> q_b_proj
  -> q_nope / q_pe / k_nope / k_pe
  -> optional DSA topk
```

如果打开 MLA preprocess：

```text
SGLANG_NPU_USE_MLAPO=1
  -> NPUFusedMLAPreprocess
  -> torch.ops.npu.mla_preprocess
```

如果同时启用 FIA NZ KV cache 布局，需要设置：

```bash
export SGLANG_USE_FIA_NZ=1
```

代码中要求 `SGLANG_USE_FIA_NZ=1` 必须和 `SGLANG_NPU_USE_MLAPO=1` 搭配使用。

### MLA core 阶段

`forward_mla_core_npu` 的核心是：

```text
attn_mqa(...)
  -> AscendAttnBackend
  -> attention output
  -> torch.ops.npu.batch_matmul_transpose(attn_output, w_vc, ...)
  -> o_proj
```

这里的 `batch_matmul_transpose` 就来自 `sgl-kernel-npu` 的 PyTorch 扩展。

## 4. AscendAttnBackend 做了什么

文件：

```text
sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py
```

关键职责：

- 根据 `req_pool_indices` 与 `seq_lens` 构建 `block_tables`；
- 为 prefill/decode 初始化 attention metadata；
- 在非 MLA 场景写 KV cache 并调用 `torch_npu.npu_fused_infer_attention_score`；
- 在 MLA 场景处理 latent KV、rope、prefix cache、sparse/topk 等；
- DSA sparse 场景会进入 `forward_sparse`。

KV cache 的 page/block table 是理解 decode 性能的关键。Decode 阶段每个请求只追加一个 token，但要通过 block table 快速索引历史 KV。

## 5. sgl-kernel-npu 的 PyTorch op 注册

自定义 op 注册在：

```text
sgl-kernel-npu/csrc/pytorch_extensions.cpp
```

可以看到类似结构：

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("mla_preprocess(...)", ...);
    m.def("batch_matmul_transpose(...)", ...);
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("mla_preprocess", &sglang::npu_kernel::mla_preprocess);
    m.impl("batch_matmul_transpose", &sglang::npu_kernel::batch_matmul_transpose);
}
```

对应 kernel：

```text
torch.ops.npu.mla_preprocess
  -> csrc/mla_preprocess/op_host/mla_preprocess.cpp
  -> csrc/mla_preprocess/op_kernel/mla_preprocess_kernel.cpp

torch.ops.npu.batch_matmul_transpose
  -> csrc/batch_matmul_transpose/op_host/batch_matmul_transpose.cpp
  -> csrc/batch_matmul_transpose/op_kernel/batch_matmul_transpose_kernel.cpp
```

## 6. DeepEP/FuseEP 链路

DeepSeek MoE 在专家并行或 A2A 场景中，通信和专家计算会成为核心瓶颈。Ascend 上的 FuseEP 路径试图把 dispatch、GEMM、combine 融合。

SGLang 侧入口：

```text
sglang/python/sglang/srt/hardware_backend/npu/moe/fuseep.py
```

关键链路：

```text
FusedMoE.forward
  -> forward_fuseep(layer, hidden_states, topk_output)
  -> DeepEPBuffer.set_dispatch_mode_as_low_latency
  -> DeepEPBuffer.get_deepep_buffer(...)
  -> buf.fused_deep_moe(...)
```

`forward_fuseep` 传入的主要数据：

- `hidden_states`
- `topk_idx`
- `topk_weights`
- `gmm1_permuted_weight`
- `gmm1_permuted_weight_scale`
- `gmm2_weight`
- `gmm2_weight_scale`
- `num_max_dispatch_tokens_per_rank`
- `num_experts`
- `fuse_mode`

NPU FuseEP mode 定义在 `hardware_backend/npu/utils.py`：

```python
class FusedMoEMode(IntEnum):
    FUSED_DEEP_MOE = 1
    DISPATCH_FFN_COMBINE = 2
```

默认环境变量：

```text
SGLANG_NPU_FUSED_MOE_MODE=1
```

## 7. deep_ep Python 到 C++/ACL

`sgl-kernel-npu` 侧链路：

```text
python/deep_ep/deep_ep/buffer.py::Buffer.fused_deep_moe
  -> deep_ep_cpp.Buffer.fused_deep_moe
  -> csrc/deepep/pybind_extension.cpp
  -> csrc/deepep/deep_ep.cpp::Buffer::fused_deep_moe
  -> EXEC_NPU_CMD(aclnnFusedDeepMoe, ...)
```

`Buffer.__init__` 会拿到 HCCL group name，并创建 C++ runtime：

```python
moe_all_to_all_group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
self.runtime = deep_ep_cpp.Buffer(
    rank,
    group_size,
    num_nvl_bytes,
    num_rdma_bytes,
    low_latency_mode,
    moe_all_to_all_group_name,
)
```

C++ 侧最终会调用 Ascend ACLNN：

```cpp
EXEC_NPU_CMD(
    aclnnFusedDeepMoe,
    x,
    expert_ids,
    gmm1_permuted_weight,
    gmm2_weight,
    ...
    output,
    ep_recv_count
);
```

当 `fuse_mode=2` 时会走 `aclnnDispatchFFNCombine`，适合另一种 dispatch/FFN/combine 融合形式。

## 8. NPU 场景建议优先看的链路

如果你的目标是理解 DeepSeek 在 Ascend 上为什么快或哪里慢，建议按这个顺序读：

1. `sglang/python/sglang/srt/hardware_backend/npu/utils.py`
   - 看 NPU 默认参数如何改写 server args。
2. `sglang/python/sglang/srt/models/deepseek_v2.py`
   - 看 DeepSeek attention/MoE 如何选择 forward method。
3. `sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`
   - 看 MLA prepare/core 如何映射到 NPU。
4. `sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`
   - 看 attention backend 如何处理 prefill/decode。
5. `sglang/python/sglang/srt/hardware_backend/npu/moe/fuseep.py`
   - 看 MoE 如何进入 FuseEP。
6. `sgl-kernel-npu/python/deep_ep/deep_ep/buffer.py`
   - 看 Python API 到 C++ runtime。
7. `sgl-kernel-npu/csrc/deepep/deep_ep.cpp`
   - 看最终 ACLNN op 调用。
8. `sgl-kernel-npu/csrc/pytorch_extensions.cpp`
   - 看 `torch.ops.npu.*` 是如何注册的。

## 9. 调试时常见观察点

| 目标 | 观察点 |
| --- | --- |
| attention backend 是否走 Ascend | `server_args.attention_backend`、日志中的 backend 名称 |
| MLA preprocess 是否开启 | `SGLANG_NPU_USE_MLAPO` |
| FIA NZ KV cache 是否开启 | `SGLANG_USE_FIA_NZ`，必须配合 `SGLANG_NPU_USE_MLAPO` |
| FuseEP 是否开启 | CLI `--moe-a2a-backend ascend_fuseep` |
| FuseEP mode | `SGLANG_NPU_FUSED_MOE_MODE`，1 为 `FUSED_DEEP_MOE`，2 为 `DISPATCH_FFN_COMBINE` |
| DeepEP buffer token 上限 | `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` |
| HCCL 通信异常 | HCCL env、rank/world size、device group |
| KV cache/page 异常 | `page_size`、`block_tables`、`seq_lens`、`req_pool_indices` |
