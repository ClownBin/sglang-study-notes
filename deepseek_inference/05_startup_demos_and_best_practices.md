---
title: "DeepSeek 启动 Demo 与参数实践"
description: "SGLang 不同启动入口的 demo，以及 DeepSeek 场景下 CLI 参数最佳实践。"
icon: "terminal"
---

# DeepSeek 启动 Demo 与参数实践

本文给出当前项目中常见入口的启动方式，并解释 DeepSeek 场景下优先考虑的参数。示例命令需要根据机器 GPU/NPU 数量、显存、模型路径和网络环境调整。

## 1. 推荐入口：sglang serve

`sglang serve` 是当前推荐 CLI 入口。

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --trust-remote-code \
  --mem-fraction-static 0.9
```

参数说明：

| 参数 | 建议 |
| --- | --- |
| `--model-path` | 使用本地模型目录或 HF repo id |
| `--tp-size 8` | 单机 8 卡常见起点；DeepSeek 大模型通常需要 TP；项目文档中也常见短写 `--tp` |
| `--trust-remote-code` | DeepSeek HF config/自定义代码场景常用 |
| `--mem-fraction-static 0.9` | 给 KV cache 和运行时预留显存；OOM 时下调 |
| `--host 0.0.0.0` | 容器或远程访问常用 |
| `--port 30000` | SGLang 默认常用端口 |

注意：官方 DeepSeek V3/R1 权重已经是 FP8 的场景，不要额外加 `--quantization fp8`，否则可能走错量化路径。

## 2. 兼容入口：python -m sglang.launch_server

旧文档和 benchmark 中经常使用这个入口，仍然可用。

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --trust-remote-code \
  --mem-fraction-static 0.9
```

它最终也会进入同一个 `run_server(server_args)`，因此学习链路时可以把它和 `sglang serve` 视为同一条主线。

## 3. OpenAI-compatible 请求 Demo

启动服务后，可以用 OpenAI-compatible API 调用：

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3-0324",
    "messages": [
      {"role": "user", "content": "用三句话解释 SGLang 的 prefill 和 decode。"}
    ],
    "temperature": 0,
    "max_tokens": 128,
    "stream": true
  }'
```

## 4. Python offline Engine Demo

适合本地脚本、benchmark、调试模型 forward 链路。

```python
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs

server_args = ServerArgs(
    model_path="deepseek-ai/DeepSeek-V3-0324",
    tp_size=8,
    trust_remote_code=True,
    mem_fraction_static=0.9,
)

llm = sgl.Engine(**dataclasses.asdict(server_args))

outputs = llm.generate(
    ["请解释 DeepSeek MLA 的核心思想。"],
    sampling_params={
        "temperature": 0,
        "max_new_tokens": 128,
    },
)

print(outputs[0]["text"])
llm.shutdown()
```

## 5. 高吞吐：TP + DP attention

高并发、大 batch、吞吐优先时，可以启用 DP attention。它适合 batch 足够大的场景，不一定适合极低延迟的小 batch。

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --max-running-requests 128 \
  --cuda-graph-max-bs-decode 128
```

实践建议：

- 请求量低或追求单请求低延迟时，先用纯 TP。
- 请求量高、batch 能堆起来时，再打开 `--enable-dp-attention`。
- `--max-running-requests` 不要盲目拉太高，先结合显存和 P99 延迟逐步调。
- `--cuda-graph-max-bs-decode` 通常与预期 decode batch size 对齐；`--cuda-graph-max-bs` 是兼容旧写法的别名。

## 6. 低延迟：纯 TP 起步

小 batch、交互式应用、低延迟优先时，建议先用纯 TP。

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --trust-remote-code \
  --mem-fraction-static 0.88 \
  --max-running-requests 32 \
  --stream-interval 1
```

调参思路：

- OOM：优先下调 `--mem-fraction-static` 或 `--max-running-requests`。
- 首 token 慢：关注 prompt 长度、chunked prefill、prefill backend。
- decode 慢：关注 attention backend、cuda graph batch size、KV cache dtype。

## 7. DeepSeek V3.2 / DSA 示例

DeepSeek V3.2 引入 DSA 相关路径，backend 选择更重要。

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --trust-remote-code \
  --reasoning-parser deepseek-v3 \
  --tool-call-parser deepseekv31 \
  --chat-template ./examples/chat_template/tool_chat_template_deepseekv32.jinja
```

如果要显式指定 DSA backend，可结合硬件选择，例如：

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --tp-size 8 \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --trust-remote-code
```

## 8. MTP / speculative decoding 示例

DeepSeek 的 MTP/EAGLE 路径适合进一步提升 decode 吞吐，但会增加调参复杂度。

```bash
SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --max-running-requests 96 \
  --cuda-graph-max-bs-decode 128
```

实践建议：

- 先跑通非 speculative，再开启 EAGLE。
- batch 较大时，适当提高 `--max-running-requests` 和 cuda graph batch size。
- 如果收益不稳定，优先检查 draft token 接受率和请求长度分布。

## 9. 多机 TP 示例

两台 8 卡机器跑 `--tp-size 16` 的示例：

Node 0：

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr 10.0.0.1:5000 \
  --trust-remote-code
```

Node 1：

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr 10.0.0.1:5000 \
  --trust-remote-code
```

多机启动时优先确认：

- 两台机器模型路径一致；
- master 地址和端口互通；
- NCCL/HCCL 网络配置正确；
- 两边 `--tp-size`、`--nnodes`、`--dist-init-addr` 完全一致；
- 只有 `--node-rank` 不同。

## 10. PD disaggregation 示例

Prefill/decode 分离适合长 prompt 或吞吐优化场景。下面是结构示例。

Prefill node：

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --disaggregation-mode prefill \
  --host 0.0.0.0 \
  --port 30001 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --dist-init-addr 10.0.0.1:6000 \
  --disaggregation-bootstrap-port 8998 \
  --trust-remote-code \
  --mem-fraction-static 0.9
```

Decode node：

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --disaggregation-mode decode \
  --host 0.0.0.0 \
  --port 30002 \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention \
  --dist-init-addr 10.0.0.2:6000 \
  --trust-remote-code \
  --mem-fraction-static 0.9
```

Router：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.0.0.1:30001 8998 \
  --decode http://10.0.0.2:30002 \
  --host 0.0.0.0 \
  --port 30000
```

## 11. Docker 示例

适合快速验证 CUDA 环境：

```bash
docker run --gpus all \
  --shm-size 32g \
  --ipc=host \
  --network=host \
  --privileged \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  sglang serve \
    --model-path deepseek-ai/DeepSeek-V3 \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 8 \
    --trust-remote-code
```

## 12. Ascend NPU / FuseEP 示例

此示例用于理解参数形态，实际要按 CANN、torch-npu、机器型号和 `sgl-kernel-npu` 编译结果调整。

先准备 NPU kernel 包：

```bash
cd sgl-kernel-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh -a deepep
pip install output/deep_ep*.whl
```

启动 SGLang：

```bash
cd ../sglang
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_BUFFSIZE=1024
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_USE_FIA_NZ=1
export SGLANG_NPU_FUSED_MOE_MODE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

sglang serve \
  --model-path deepseek-ai/DeepSeek-V3 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 8 \
  --trust-remote-code \
  --device npu \
  --attention-backend ascend \
  --moe-a2a-backend ascend_fuseep \
  --mem-fraction-static 0.9
```

参数含义：

| 参数/环境变量 | 作用 |
| --- | --- |
| `--attention-backend ascend` | 走 Ascend attention backend |
| `--device npu` | 显式指定 NPU 设备 |
| `--moe-a2a-backend ascend_fuseep` | MoE 走 Ascend FuseEP |
| `SGLANG_NPU_USE_MLAPO=1` | 开启 NPU MLA preprocess |
| `SGLANG_USE_FIA_NZ=1` | 开启 FIA NZ KV cache 布局，需配合 MLAPO |
| `SGLANG_NPU_FUSED_MOE_MODE=1` | 走 `FUSED_DEEP_MOE` |
| `SGLANG_NPU_FUSED_MOE_MODE=2` | 走 `DISPATCH_FFN_COMBINE` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | DeepEP buffer 的 dispatch token 上限 |

## 13. 参数选择速查

| 目标 | 优先参数 |
| --- | --- |
| 跑通单机 | `--tp-size <num_gpus>`、`--trust-remote-code` |
| 降低 OOM 风险 | 下调 `--mem-fraction-static`、`--max-running-requests` |
| 高吞吐 | `--enable-dp-attention`、`--dp-size`、提高 `--max-running-requests` |
| 低延迟 | 纯 TP、较小 `--max-running-requests`、合适 `--stream-interval` |
| 长 prompt | 调整 `--chunked-prefill-size`、关注 prefill backend |
| Decode 吞吐 | `--cuda-graph-max-bs-decode`、KV cache dtype、attention backend |
| DeepSeek V3.2 | 关注 DSA backend、reasoning/tool parser |
| Ascend NPU | `--attention-backend ascend`、`--moe-a2a-backend ascend_fuseep`、NPU env |
