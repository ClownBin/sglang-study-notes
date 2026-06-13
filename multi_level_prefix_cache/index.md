---
title: "SGLang 多级 PrefixCache"
description: "SGLang HiCache / Hierarchical Cache 的 L1 GPU、L2 Host、L3 Storage 架构和请求调用链。"
icon: "layers"
---

# SGLang 多级 PrefixCache 学习入口

本目录归档 SGLang 多级 PrefixCache 架构分析。这里的“多级 PrefixCache”主要对应源码和启动参数中的 **HiCache / Hierarchical Cache**：在 RadixAttention 的 GPU PrefixCache 基础上，把可复用 KV Cache 扩展到 Host 内存和远端/文件存储。

## 核心结论

- SGLang 的基础 PrefixCache 是 `RadixCache`，多级版本是 `HiRadixCache`。
- 多级缓存分为三层：L1 GPU KV、L2 Host KV、L3 Storage KV。
- L1/L2 的命中元数据维护在 HiRadixTree 节点上，L3 不常驻树内，按请求的 page hash 实时查询存储后端。
- 读路径分两段：入队前对 L3 做异步 prefetch，真正调度 prefill 时把 L2 命中的 KV load back 到 GPU。
- 写路径由请求结束或阶段性缓存触发，先把 GPU KV 写入 RadixTree，再按策略备份到 Host 和 Storage。
- 调度器只感知 `BasePrefixCache` 抽象；多级缓存的异步传输、后台线程、存储后端由 `HiCacheController` 收口。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [多级 PrefixCache 架构](multi_level_prefix_cache/01_architecture) | 多级 PrefixCache 的总体架构、组件职责、启动配置和架构图 |
| [关键调用链](multi_level_prefix_cache/02_call_chain) | 启动选型、请求命中、L3 预取、Host 回填、写入和淘汰调用链 |
| [核心代码](multi_level_prefix_cache/03_core_code) | 核心源码位置、关键代码块示例和阅读切入点 |

## 建议阅读顺序

1. 先读 [多级 PrefixCache 架构](multi_level_prefix_cache/01_architecture)，建立 L1/L2/L3、HiRadixTree、HiCacheController 的关系。
2. 再读 [关键调用链](multi_level_prefix_cache/02_call_chain)，按一次请求的生命周期串起来。
3. 最后读 [核心代码](multi_level_prefix_cache/03_core_code)，对照源码做断点或日志验证。
