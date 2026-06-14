---
title: "FCB Study"
description: "个人学习、源码分析、推理链路梳理、启动 Demo 和最佳实践文档站。"
icon: "book-open"
---

# FCB Study 文档目录

这里用于保存个人学习、源码分析、推理链路梳理、启动 Demo 和最佳实践类输出。

## 作者与仓库

| 项 | 信息 |
| --- | --- |
| 作者仓库 | [ClownBin/sglang-study-notes.git](https://github.com/ClownBin/sglang-study-notes.git) |
| 联系方式 | [chaobin1993@126.com](mailto:chaobin1993@126.com) |

## 场景入口

左侧导航按 `主题分组 → 场景目录 → 文档页` 展示；下面的卡片保留中文说明，方便快速进入。

<CardGroup cols={3}>
  <Card title="DeepSeek 推理链路" icon="route" href="deepseek_inference/index">
    `deepseek_inference`：启动加载、一次推理、NPU kernel 桥接、Demo 与参数实践。
  </Card>
  <Card title="多级 PrefixCache" icon="layers" href="multi_level_prefix_cache/index">
    `multi_level_prefix_cache`：HiCache 架构、L1/L2/L3 缓存、调用链和核心代码。
  </Card>
  <Card title="NPU/GPU 能力差异" icon="cpu" href="npu_gpu_feature_diff_20260613_193825/index">
    `npu_gpu_feature_diff_20260613_193825`：上层特性与底层算子能力差异。
  </Card>
  <Card title="DeepSeek/Qwen 模型架构" icon="network" href="deepseek_qwen_model_architecture_20260613_203749/index">
    `deepseek_qwen_model_architecture_20260613_203749`：模型架构解构和对比。
  </Card>
  <Card title="社区与官方资料" icon="messages-square" href="sglang_community_resources/index">
    `sglang_community_resources`：社区群、官方文档、Roadmap、论文和 NPU 资料入口。
  </Card>
</CardGroup>

## 学习路径

```mermaid
flowchart LR
    A["先看总览"] --> B["推理链路"]
    B --> C["PrefixCache"]
    C --> D["硬件与算子"]
    D --> E["模型架构"]
    E --> F["社区与资料"]
```

## 导航分组

| 主题分组 | 场景目录 | 适合解决的问题 |
| --- | --- | --- |
| 推理与缓存 | `deepseek_inference`、`multi_level_prefix_cache` | 服务启动、模型加载、一次请求生命周期、缓存复用 |
| 硬件与算子 | `npu_gpu_feature_diff_20260613_193825` | NPU/GPU 特性边界、底层 kernel 和算子能力 |
| 模型与资料 | `deepseek_qwen_model_architecture_20260613_203749`、`sglang_community_resources` | 最新模型结构、官方资料入口、社区协作渠道 |

## 目录规范

- 每个独立主题使用一个场景目录：`<scenario>/`。
- 场景目录使用 `lower_snake_case` 命名。
- 每个场景目录保留 `README.md` 作为个人文档索引，同时补充 `index.md` 作为 Mintlify 页面入口。
- 长文档按主题拆分为编号文件，例如 `01_xxx.md`、`02_xxx.md`。
- 默认使用 Markdown；只有明确面向 docs site、Mintlify 或现有 `.mdx` 文档站时才使用 MDX。

## 本地启动

在当前项目根路径运行：

```bash
nvm use
mint dev
```

也可以使用 npm 脚本：

```bash
npm run dev
```

默认访问地址是 `http://localhost:3000`。

说明：Mintlify CLI 不支持 Node 25+；本目录已提供 `.nvmrc`，并且 `npm run dev` 会优先使用本机已安装的 Node 24.13.0。
