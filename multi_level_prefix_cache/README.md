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
| [01_architecture.md](01_architecture.md) | 多级 PrefixCache 的总体架构、组件职责、启动配置和架构图 |
| [02_call_chain.md](02_call_chain.md) | 启动选型、请求命中、L3 预取、Host 回填、写入和淘汰调用链 |
| [03_core_code.md](03_core_code.md) | 核心源码位置、关键代码块示例和阅读切入点 |

## 建议阅读顺序

1. 先读 [01_architecture.md](01_architecture.md)，建立 L1/L2/L3、HiRadixTree、HiCacheController 的关系。
2. 再读 [02_call_chain.md](02_call_chain.md)，按一次请求的生命周期串起来。
3. 最后读 [03_core_code.md](03_core_code.md)，对照源码做断点或日志验证。

## 源码快速入口

| 模块 | 源码路径 | 重点 |
| --- | --- | --- |
| 缓存后端选择 | `sglang/python/sglang/srt/mem_cache/registry.py` | `--enable-hierarchical-cache` 如何选到 `HiRadixCache` |
| 多级 RadixCache | `sglang/python/sglang/srt/mem_cache/hiradix_cache.py` | match、prefetch、load_back、insert、evict |
| 基础 RadixTree | `sglang/python/sglang/srt/mem_cache/radix_cache.py` | `TreeNode.value`、`host_value`、`hash_value` |
| 传输控制器 | `sglang/python/sglang/srt/managers/cache_controller.py` | Host/GPU 传输队列、L3 prefetch/backup 线程 |
| 调度器入口 | `sglang/python/sglang/srt/managers/scheduler.py` | 入队前 L3 prefetch、batch 前 start loading、事件轮询 |
| Prefill 装配 | `sglang/python/sglang/srt/managers/schedule_policy.py` | Host 命中时调用 `init_load_back` |
| 启动参数 | `sglang/python/sglang/srt/server_args.py` | HiCache CLI 参数、默认值和合法选项 |
