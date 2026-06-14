# FCB Study 文档目录

这里用于保存个人学习、源码分析、推理链路梳理、启动 Demo 和最佳实践类输出。

## 作者与仓库

| 项 | 信息 |
| --- | --- |
| 作者仓库 | [ClownBin/sglang-study-notes.git](https://github.com/ClownBin/sglang-study-notes.git) |
| 联系方式 | [chaobin1993@126.com](mailto:chaobin1993@126.com) |

## Mintlify 本地预览

本目录已改造为 Mintlify 文档站，可从仓库根目录直接启动：

```bash
nvm use
mint dev
```

也可以使用 npm 脚本：

```bash
npm run dev
```

默认访问地址：`http://localhost:3000`。

说明：Mintlify CLI 不支持 Node 25+；本目录已提供 `.nvmrc`，并且 `npm run dev` 会优先使用本机已安装的 Node 24.13.0。

## 目录规范

- 每个独立主题使用一个场景目录：`<scenario>/`。
- 场景目录使用 `lower_snake_case` 命名。
- Mintlify 左侧导航采用 `主题分组 -> 场景目录 -> 文档页` 的层级展示。
- 每个场景目录优先包含 `README.md` 作为个人文档入口索引，同时补充 `index.md` 作为 Mintlify 页面入口。
- 长文档按主题拆分为编号文件，例如 `01_xxx.md`、`02_xxx.md`。
- 默认使用 Markdown；只有明确面向 docs site、Mintlify 或现有 `.mdx` 文档站时才使用 MDX。

## 当前场景

| 分组 | 场景目录 | 内容 |
| --- | --- | --- |
| 推理与缓存 | [deepseek_inference](deepseek_inference/README.md) | SGLang 与 sgl-kernel-npu 两仓分析，DeepSeek 启动加载、一次推理链路、NPU kernel 桥接、启动 Demo 和参数实践 |
| 推理与缓存 | [multi_level_prefix_cache](multi_level_prefix_cache/README.md) | SGLang 多级 PrefixCache/HiCache 架构，L1/L2/L3 缓存链路、核心代码和调用链 |
| 硬件与算子 | [npu_gpu_feature_diff_20260613_193825](npu_gpu_feature_diff_20260613_193825/README.md) | 2026-06-13 19:38:25 CST 归档，比较 SGLang Ascend NPU 与 NVIDIA CUDA GPU 的上层能力和底层算子能力差异 |
| 模型与资料 | [deepseek_qwen_model_architecture_20260613_203749](deepseek_qwen_model_architecture_20260613_203749/README.md) | 2026-06-13 20:37:49 CST 归档，以 DeepSeek-V4-Pro 与 Qwen3.6-35B-A3B 为例分析模型特征、架构图和工程映射 |
| 模型与资料 | [sglang_community_resources](sglang_community_resources/README.md) | SGLang 社区群、官方资料与 Ascend NPU 相关入口归档 |

## 推荐新增方式

新增场景时，建议使用下面结构：

```text
<scenario>/
├── README.md
├── 01_xxx.md
├── 02_xxx.md
└── 03_xxx.md
```

示例：

```text
deepseek_inference/
npu_kernel_bridge/
startup_demos/
```
