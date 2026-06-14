---
title: "源码地图"
description: "NPU/GPU 能力差异分析对应的核心源码路径、继续追踪命令和验证记录。"
icon: "map"
---

# 源码地图

分析时间：`2026-06-13 19:38:25 CST`

## 入口与参数

| 路径 | 作用 |
| --- | --- |
| `sglang/python/sglang/srt/server_args.py` | Server 参数枚举和平台后端处理，包含 attention、sampling、quantization、MoE、LoRA、HiCache、PD 等参数 |
| `sglang/python/sglang/srt/hardware_backend/npu/utils.py` | NPU 默认参数注入、NPU backend 初始化、NPU format cast |
| `sglang/python/sglang/srt/layers/attention/attention_registry.py` | attention backend 注册中心，能直接看 GPU 多后端与 NPU `ascend` 后端的分叉 |

## NPU 上层支持文档

| 路径 | 作用 |
| --- | --- |
| `sglang/docs_new/docs/hardware-platforms/ascend-npus/ascend_npu.mdx` | Ascend NPU 安装、组件版本、运行环境 |
| `sglang/docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx` | NPU 支持特性矩阵，包含 LoRA、kernel backends、speculative、EP、HiCache、PD、多模态等 |
| `sglang/docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx` | NPU 支持模型清单 |
| `sglang/docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quantization.mdx` | NPU 量化说明，包含 ModelSlim、GGUF、MoE 量化等 |
| `sglang/docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx` | NPU 优化建议，包含 DeepEP、Runtime queue、NPU graph 等 |

## GPU 上层支持文档

| 路径 | 作用 |
| --- | --- |
| `sglang/docs_new/docs/get-started/quickstart.mdx` | 快速开始，说明 NVIDIA CUDA GPU 前置条件 |
| `sglang/docs_new/docs/get-started/install.mdx` | 安装文档，说明 NVIDIA GPU 路径、CUDA 镜像、FlashInfer 默认 backend |
| `sglang/docs_new/docs/hardware-platforms/nvidia-gpus.mdx` | NVIDIA GPU 平台入口 |
| `sglang/docs/basic_usage/deepseek_v3.md` | DeepSeek V3 GPU 使用说明，包含 CUDA Graph、Torch.compile、DeepGEMM 等提示 |
| `sglang/docs/basic_usage/deepseek_v32.md` | DeepSeek V3.2 GPU/DSA/HiSparse 等相关能力说明 |

## NPU 后端源码

| 路径 | 作用 |
| --- | --- |
| `sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` | Ascend attention backend，核心调用 `npu_fused_infer_attention_score` |
| `sglang/python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py` | NPU MLA preprocess 调用封装 |
| `sglang/python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | DeepSeek MLA NPU 模块，包含 batch matmul transpose 等调用 |
| `sglang/python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py` | NPU MoE 量化和 grouped matmul 主路径 |
| `sglang/python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py` | NPU 量化 linear 主路径 |
| `sglang/python/sglang/srt/hardware_backend/npu/moe/topk.py` | NPU MoE top-k gating |
| `sglang/python/sglang/srt/hardware_backend/npu/moe/fuseep.py` | NPU FuseEP/DeepEP 集成 |
| `sglang/python/sglang/srt/hardware_backend/npu/allocator_npu.py` | NPU KV page allocator |
| `sglang/python/sglang/srt/hardware_backend/npu/graph_runner/` | NPU graph runner、EAGLE graph runner、VIT graph runner |
| `sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py` | NPU GDN/linear attention 后端 |
| `sglang/python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py` | NPU hybrid linear attention 与 Mamba2 相关后端 |

## sgl-kernel-npu 算子源码

| 路径 | 作用 |
| --- | --- |
| `sgl-kernel-npu/csrc/pytorch_extensions.cpp` | NPU custom ops 的 PyTorch 注册入口 |
| `sgl-kernel-npu/include/sgl_kenel_npu_ops.h` | NPU custom ops 声明 |
| `sgl-kernel-npu/csrc/mla_preprocess/README.md` | MLA preprocess 算子能力、输入输出、cache mode、约束 |
| `sgl-kernel-npu/csrc/batch_matmul_transpose/README.md` | batch matmul transpose 算子说明 |
| `sgl-kernel-npu/csrc/build_tree/README.md` | speculative decoding tree build 算子说明 |
| `sgl-kernel-npu/csrc/deepep/` | DeepEP-Ascend C++/CANN 侧实现 |
| `sgl-kernel-npu/python/deep_ep/README_CN.md` | DeepEP-Ascend Python 包说明 |
| `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/attention/` | NPU decode attention、attention sinks Python wrapper |
| `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/norm/` | NPU RMSNorm、QKV split、RoPE 相关 wrapper |
| `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/activation/` | NPU SwiGLU/quant activation wrapper |
| `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/` | NPU linear attention/GDN 相关 wrapper |
| `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/mamba/` | NPU Mamba/causal conv1d wrapper |
| `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/mem_cache/` | NPU KV cache allocator wrapper |

## GPU 后端源码

| 路径 | 作用 |
| --- | --- |
| `sglang/python/sglang/srt/layers/attention/flashinfer_backend.py` | GPU FlashInfer attention backend |
| `sglang/python/sglang/srt/layers/attention/flashinfer_mla_backend.py` | GPU FlashInfer MLA backend |
| `sglang/python/sglang/srt/layers/attention/triton_backend.py` | GPU Triton attention backend |
| `sglang/python/sglang/srt/layers/attention/flashattention_backend.py` | FlashAttention v3/v4 backend |
| `sglang/python/sglang/srt/layers/attention/flashmla_backend.py` | FlashMLA backend |
| `sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py` | CUTLASS MLA backend |
| `sglang/python/sglang/srt/layers/attention/trtllm_mha_backend.py` | TensorRT-LLM MHA backend |
| `sglang/python/sglang/srt/layers/attention/trtllm_mla_backend.py` | TensorRT-LLM MLA backend |
| `sglang/python/sglang/srt/layers/quantization/` | GPU/通用量化实现集合，包含 FP8、FP4、AWQ、GPTQ、Marlin、ModelOpt、W8A8 等 |

## 继续追踪命令

查 NPU 自定义算子注册：

```bash
rg -n "TORCH_LIBRARY|m\\.def|m\\.impl" sgl-kernel-npu/csrc/pytorch_extensions.cpp
```

查 NPU 侧 `torch.ops.npu` 调用点：

```bash
rg -n "torch\\.ops\\.npu|sgl_kernel_npu" \
  sglang/python/sglang/srt/hardware_backend/npu \
  sglang/python/sglang/srt/layers \
  sglang/python/sglang/srt/speculative \
  sglang/python/sglang/srt/models
```

查 GPU/NPU attention backend 注册：

```bash
rg -n "register_attention_backend|create_.*backend" \
  sglang/python/sglang/srt/layers/attention/attention_registry.py
```

查 server args 中上层能力枚举：

```bash
rg -n "ATTENTION_BACKEND_CHOICES|QUANTIZATION_CHOICES|MOE_RUNNER_BACKEND_CHOICES|LORA_BACKEND_CHOICES|MAMBA_BACKEND_CHOICES" \
  sglang/python/sglang/srt/server_args.py
```

查 NPU support feature matrix：

```bash
rg -n "LoRA|Kernel Backends|Speculative|Expert parallelism|Hierarchical cache|PD disaggregation|Special for GPU|Planned" \
  sglang/docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx
```

## 本次验证记录

已执行的本地检查：

- `sed` 读取 `server_args.py` 的 `_handle_npu_backends`、attention/sampling/quant/MoE/LoRA/Mamba backend choices。
- `sed` 读取 `hardware_backend/npu/utils.py` 的 `set_default_server_args` 和 `init_npu_backend`。
- `sed` 读取 Ascend NPU support feature matrix 中 LoRA、kernel backends、speculative、expert parallelism、hierarchical cache、PD disaggregation 相关段落。
- `sed` 读取 GPU quickstart/install 中 NVIDIA CUDA、FlashInfer 默认 backend 和 Triton/PyTorch fallback 提示。
- `find` 和 `rg` 枚举 `sgl-kernel-npu/csrc`、`sgl-kernel-npu/python/sgl_kernel_npu`、NPU `torch.ops.npu` 调用点。
- `git -C sglang status --short` 与 `git -C sgl-kernel-npu status --short` 均无输出，两个子仓在分析时没有未提交变更。

未执行的验证：

- 未在真实 NVIDIA GPU 或 Ascend NPU 上启动 `launch_server`。
- 未运行 benchmark 或 kernel 单测。
- 未验证不同硬件型号 A2/A3/A5 的实际性能差异。
