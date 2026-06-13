---
title: "FCB Study"
description: "个人学习、源码分析、推理链路梳理、启动 Demo 和最佳实践文档站。"
icon: "book-open"
---

# FCB Study 文档目录

这里用于保存个人学习、源码分析、推理链路梳理、启动 Demo 和最佳实践类输出。

## 当前场景

| 场景 | 内容 |
| --- | --- |
| [DeepSeek 推理链路](deepseek_inference/index) | SGLang 与 sgl-kernel-npu 两仓分析，DeepSeek 启动加载、一次推理链路、NPU kernel 桥接、启动 Demo 和参数实践 |
| [多级 PrefixCache](multi_level_prefix_cache/index) | SGLang 多级 PrefixCache / HiCache 架构，L1/L2/L3 缓存链路、核心代码和调用链 |
| [社区与资料](sglang_community_resources/index) | SGLang 社区群、官方资料与 Ascend NPU 相关入口归档 |

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
