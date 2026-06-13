---
title: "两仓职责与最佳切入点"
description: "SGLang 主仓与 sgl-kernel-npu 的职责边界、推荐阅读顺序和 DeepSeek 场景源码地图。"
icon: "map"
---

# 两仓职责与最佳切入点

## 1. 代码仓分工

| 仓库 | 主要职责 | DeepSeek 场景里重点关注 |
| --- | --- | --- |
| `sglang/` | serving runtime、API server、调度器、模型加载、模型执行、DeepSeek 模型结构、硬件后端选择 | 启动入口、请求链路、prefill/decode、MLA、MoE、DP/TP/EP、NPU 后端调用 |
| `sgl-kernel-npu/` | Ascend NPU kernel 库，提供 DeepEP-Ascend 与 SGLang-Kernel-NPU PyTorch 扩展 | `deep_ep.Buffer`、FuseEP、MLA preprocess、`batch_matmul_transpose`、C++/ACL kernel |

从学习角度看，`sglang/` 是主线，`sgl-kernel-npu/` 是 Ascend NPU 场景下的底层加速插件。先把 `sglang/` 的请求和模型执行链路跑通，再回头看 NPU kernel，会清楚很多。

## 2. 先读哪些文件

### 第一层：启动和进程拓扑

| 文件 | 读法 |
| --- | --- |
| `sglang/python/sglang/cli/serve.py` | 看 `serve(args, extra_argv)` 如何选择 LLM serving 并调用 `run_server` |
| `sglang/python/sglang/launch_server.py` | 看 `prepare_server_args` 和 `run_server`，这是旧入口兼容层 |
| `sglang/python/sglang/srt/entrypoints/http_server.py` | 看 `launch_server`，理解 HTTP server 与 SRT engine 的组合 |
| `sglang/python/sglang/srt/entrypoints/engine.py` | 看 `Engine._launch_subprocesses`，这是进程结构的核心入口 |

### 第二层：请求调度和模型调用

| 文件 | 读法 |
| --- | --- |
| `sglang/python/sglang/srt/managers/tokenizer_manager.py` | 看 `generate_request`、`_tokenize_one_request`、`_send_one_request` |
| `sglang/python/sglang/srt/managers/scheduler.py` | 看 `event_loop_normal`、`get_next_batch_to_run`、`run_batch` |
| `sglang/python/sglang/srt/managers/tp_worker.py` | 看 `forward_batch_generation` 如何把 batch 交给 `ModelRunner` |
| `sglang/python/sglang/srt/model_executor/forward_batch_info.py` | 看 `ForwardBatch.init_new` 如何把调度 batch 转成模型输入 |
| `sglang/python/sglang/srt/model_executor/model_runner.py` | 看 `load_model`、`forward_extend`、`forward_decode`、`sample` |

### 第三层：DeepSeek 模型结构

| 文件 | 读法 |
| --- | --- |
| `sglang/python/sglang/srt/models/deepseek_v2.py` | DeepSeek V2/V3/V3.2 主实现；重点看 `DeepseekV2ForCausalLM`、`DeepseekV2Model`、`DeepseekV2DecoderLayer`、`DeepseekV2AttentionMLA`、`DeepseekV2MoE` |
| `sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/` | MLA/MHA/DSA 不同 forward 方法 |
| `sglang/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` | 权重加载后的 MLA absorb 权重处理 |
| `sglang/python/sglang/srt/layers/moe/` | MoE runner、token dispatcher、DeepEP 接入 |

### 第四层：Ascend NPU 适配

| 文件 | 读法 |
| --- | --- |
| `sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` | Ascend attention backend 的 prefill/decode 路径 |
| `sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | DeepSeek MLA 在 NPU 上的 prepare/core 实现 |
| `sglang/python/sglang/srt/hardware_backend/npu/moe/fuseep.py` | `--moe-a2a-backend ascend_fuseep` 后如何进入 FuseEP |
| `sglang/python/sglang/srt/hardware_backend/npu/utils.py` | NPU 默认参数与 FuseEP mode |

### 第五层：NPU kernel 仓

| 文件 | 读法 |
| --- | --- |
| `sgl-kernel-npu/python/deep_ep/deep_ep/buffer.py` | Python 层 `Buffer`，DeepEP/FuseEP API 入口 |
| `sgl-kernel-npu/csrc/deepep/pybind_extension.cpp` | `deep_ep_cpp.Buffer` 的 pybind 暴露 |
| `sgl-kernel-npu/csrc/deepep/deep_ep.cpp` | C++ 层调用 `aclnnFusedDeepMoe` / `aclnnDispatchFFNCombine` |
| `sgl-kernel-npu/csrc/pytorch_extensions.cpp` | `torch.ops.npu.*` 自定义 op 注册 |
| `sgl-kernel-npu/csrc/mla_preprocess/` | MLA preprocess host/kernel 实现 |
| `sgl-kernel-npu/csrc/batch_matmul_transpose/` | MLA decode 后处理用到的 batch matmul transpose |

## 3. 最佳学习路径

### 路径 A：先理解服务怎么启动

```text
sglang serve
  -> sglang.cli.main
  -> sglang.cli.serve.serve
  -> sglang.launch_server.run_server
  -> sglang.srt.entrypoints.http_server.launch_server
  -> Engine._launch_subprocesses
```

读到这里，你应该能回答三个问题：

- 哪些东西在主进程？
- Scheduler 为什么是子进程？
- `tp/dp/pp/ep` 参数什么时候开始影响进程数量？

### 路径 B：再理解一次请求怎么推理

```text
HTTP /v1/chat/completions
  -> OpenAI serving adapter
  -> TokenizerManager.generate_request
  -> Scheduler.handle_generate_request
  -> Scheduler.get_next_batch_to_run
  -> Scheduler.run_batch
  -> TpModelWorker.forward_batch_generation
  -> ModelRunner.forward
  -> DeepseekV3ForCausalLM.forward
  -> sampler
  -> DetokenizerManager
  -> HTTP response
```

读到这里，你应该能区分：

- prefill 与 decode 的输入张量差异；
- KV cache 是谁分配、谁写入、谁读取；
- batch result 何时进入采样、何时返回给 detokenizer。

### 路径 C：最后理解 DeepSeek 特有优化

```text
DeepSeek
  -> MLA attention
  -> MoE routing
  -> optional DSA sparse attention for V3.2
  -> optional DeepEP/FuseEP for expert parallel
  -> hardware backend specific kernels
```

DeepSeek 不只是一个普通 Transformer。学习时要特别关注：

- MLA：`kv_b_proj` 权重加载后会拆成 `w_kc` 和 `w_vc`，decode 时减少 KV cache 体积与计算。
- MoE：gate/topk 选专家，token dispatcher 负责跨卡专家通信，底层可走 DeepEP 或 Ascend FuseEP。
- DSA：DeepSeek V3.2 可能走稀疏 attention path，forward method 会切到 DSA 相关实现。
- NPU：Ascend 上会替换 attention backend，并通过 `torch_npu` 和 `torch.ops.npu` 接到 `sgl-kernel-npu`。
