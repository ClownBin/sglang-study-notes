---
title: "SGLang 多级 PrefixCache 核心代码"
description: "HiRadixCache、HiCacheController、Scheduler 和 Storage backend 的核心源码入口与关键代码块。"
icon: "code"
---

# SGLang 多级 PrefixCache 核心代码

本篇按“从入口到执行”的顺序列核心代码。代码片段只保留关键行，便于对照源码继续深入。

## 1. CLI 参数入口

路径：`sglang/python/sglang/srt/server_args.py:6504`

```python
parser.add_argument(
    "--enable-hierarchical-cache",
    action="store_true",
    help="Enable hierarchical cache",
)
parser.add_argument(
    "--hicache-ratio",
    type=float,
    default=ServerArgs.hicache_ratio,
)
parser.add_argument(
    "--hicache-write-policy",
    type=str,
    choices=["write_back", "write_through", "write_through_selective"],
    default=ServerArgs.hicache_write_policy,
)
parser.add_argument(
    "--hicache-storage-backend",
    type=str,
    choices=["file", "mooncake", "hf3fs", "nixl", "aibrix", "dynamic", "eic", "simm"],
    default=ServerArgs.hicache_storage_backend,
)
```

阅读重点：

- 不传 `--enable-hierarchical-cache` 时，不会进入 HiCache。
- 只启用 hierarchical cache 时是 L1 + L2。
- 再传 `--hicache-storage-backend` 时才启用 L3 Storage。

## 2. 缓存后端选择

路径：`sglang/python/sglang/srt/mem_cache/registry.py:77`

```python
def default_radix_cache_factory(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    server_args = ctx.server_args
    params = ctx.params

    if ctx.enable_hierarchical_cache:
        if ctx.is_hybrid_ssm or ctx.is_hybrid_swa:
            return _create_unified_radix_cache(ctx, server_args, params)
        else:
            from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

            cache = HiRadixCache(params=params, server_args=server_args)
        ctx.tp_worker.register_hicache_layer_transfer_counter(
            cache.cache_controller.layer_done_counter
        )
        return cache
```

阅读重点：

- 多级 PrefixCache 的默认实现类是 `HiRadixCache`。
- hybrid SWA/SSM 不是直接返回 `HiRadixCache`，而是通过 `UnifiedRadixCache` 接入。
- `layer_done_counter` 用于多层 KV 传输进度同步。

## 3. HiCacheController 初始化

路径：`sglang/python/sglang/srt/managers/cache_controller.py:248`

```python
class HiCacheController:
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
```

阅读重点：

- Controller 同时持有 device pool 和 host pool。
- `write_queue`、`load_queue`、`prefetch_queue`、`backup_queue` 都在 Controller 内部收口。
- 如果 `storage_backend` 不为空，初始化阶段会 attach L3 后端并启动相关后台线程。

## 4. 本地 L1/L2 Prefix 匹配

路径：`sglang/python/sglang/srt/mem_cache/hiradix_cache.py:1438`

```python
def match_prefix(self, params: MatchPrefixParams):
    key = params.key
    key, _ = key.maybe_to_bigram_view(self.is_eagle)
    key = key.page_aligned(self.page_size)

    value, last_node = self._match_prefix_helper(self.root_node, key)
    if value:
        value = torch.cat(value)
    else:
        value = self._empty_match_result.device_indices

    host_hit_length = 0
    last_host_node = last_node
    while last_node.evicted:
        host_hit_length += len(last_node.host_value)
        last_node = last_node.parent
    while not last_host_node.backuped:
        last_host_node = last_host_node.parent

    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_host_node,
        best_match_node=last_host_node,
        host_hit_length=host_hit_length,
    )
```

阅读重点：

- `device_indices` 是 L1 GPU 命中。
- `while last_node.evicted` 统计的是 L2 Host 命中长度。
- `last_host_node` 是后续 L3 prefetch 的起点，也是 Host load back 的候选节点。

## 5. 请求入队前触发 L3 预取

路径：`sglang/python/sglang/srt/managers/scheduler.py:2156`

```python
def _prefetch_kvcache(self, req: Req):
    if self.enable_hicache_storage:
        req.init_next_round_input(self.tree_cache, cow_mamba=False)
        last_host_node = req.last_host_node
        if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
            last_hash = last_host_node.get_last_hash_value()
            matched_len = len(req.prefix_indices) + req.host_hit_length
            new_input_tokens = req.full_untruncated_fill_ids[matched_len:]

            self.tree_cache.prefetch_from_storage(
                req.rid,
                last_host_node,
                new_input_tokens,
                last_hash,
                prefix_keys,
            )
```

阅读重点：

- L3 prefetch 在请求进入 waiting queue 前触发。
- `matched_len` 会跳过 L1 和 L2 已命中的前缀，只查询后续未命中的 token page。
- `last_hash` 是增量计算 page hash 的起点。

## 6. HiRadixCache 发起 Storage prefetch

路径：`sglang/python/sglang/srt/mem_cache/hiradix_cache.py:1471`

```python
def prefetch_from_storage(
    self,
    req_id: str,
    last_host_node: TreeNode,
    new_input_tokens: List[int],
    last_hash: Optional[str] = None,
    prefix_keys: Optional[List[str]] = None,
):
    prefetch_key = RadixKey(
        new_input_tokens,
        extra_key=last_host_node.key.extra_key,
        is_bigram=self.is_eagle,
    )
    prefetch_key = prefetch_key.page_aligned(self.page_size)
    prefetch_length = len(prefetch_key)
    if (
        not self.enable_storage
        or prefetch_length < self.prefetch_threshold
        or self.cache_controller.prefetch_rate_limited()
    ):
        return

    last_host_node.protect_host()
    host_indices = self.cache_controller.mem_pool_host.alloc(prefetch_length)
    operation = self.cache_controller.prefetch(
        req_id,
        host_indices,
        prefetch_key,
        last_hash,
        prefix_keys,
        **self._get_extra_pools(),
    )
```

阅读重点：

- 预取前先按 page 对齐。
- 不满足阈值、未启用 storage、触发限流时直接跳过。
- 预取目标不是 GPU，而是 Host KV pool。

## 7. Storage 命中查询与拉取

路径：`sglang/python/sglang/srt/managers/cache_controller.py:1034`

```python
def _storage_hit_query(self, operation) -> tuple[list[str], int]:
    last_hash = operation.last_hash
    tokens_to_fetch = operation.token_ids
    hash_value = []

    for start in range(0, len(tokens_to_fetch), self.page_size * STORAGE_BATCH_SIZE):
        batch_tokens = tokens_to_fetch[start:end]
        batch_hashes = []
        for i in range(0, len(batch_tokens), self.page_size):
            last_hash = self.get_hash_str(
                batch_tokens[i : i + self.page_size], last_hash
            )
            batch_hashes.append(last_hash)
        hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
        hash_value.extend(batch_hashes[:hit_page_num])
        storage_query_count += hit_page_num * self.page_size
        if hit_page_num < len(batch_hashes):
            break

    return hash_value, storage_query_count
```

路径：`sglang/python/sglang/srt/managers/cache_controller.py:977`

```python
def _page_transfer(self, operation):
    for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
        batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
        batch_host_indices = operation.host_indices[
            i * self.page_size : (i + len(batch_hashes)) * self.page_size
        ]

        extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
        self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
        if operation.completed_tokens != prev_completed_tokens + len(batch_hashes) * self.page_size:
            operation.mark_terminate()
            break
```

阅读重点：

- `batch_exists` 只要遇到未命中的 page 就停止连续前缀查询。
- `batch_get` 或 zero-copy get 会把 KV page 写入 Host。
- 完成度记录在 `operation.completed_tokens`。

## 8. L3 prefetch 结束后插入 Host 树

路径：`sglang/python/sglang/srt/mem_cache/hiradix_cache.py:1365`

```python
def check_prefetch_progress(self, req_id: str) -> bool:
    if req_id not in self.ongoing_prefetch:
        return True

    last_host_node, prefetch_key, host_indices, operation = self.ongoing_prefetch[
        req_id
    ]

    if not self.can_terminate_prefetch(operation):
        return False

    completed_tokens, hash_value = self.cache_controller.terminate_prefetch(operation)
    completed_tokens_tensor = torch.tensor(completed_tokens, dtype=torch.int)
    self._all_reduce_attn_groups(
        completed_tokens_tensor, torch.distributed.ReduceOp.MIN
    )
    min_completed_tokens = completed_tokens_tensor.item()
    fetched_key = prefetch_key[:min_completed_tokens]
    written_indices = host_indices[:min_completed_tokens]
    matched_length = self._insert_helper_host(
        last_host_node,
        fetched_key,
        written_indices,
        hash_value[: min_completed_tokens // self.page_size],
    )
```

阅读重点：

- `can_terminate_prefetch` 由 `best_effort`、`wait_complete`、`timeout` 策略决定。
- 多卡取 `MIN`，避免部分 rank 看到更长前缀。
- `_insert_helper_host` 把 L3 拉回来的数据作为 Host-only 节点插入 HiRadixTree。

## 9. Host 命中回填 GPU

路径：`sglang/python/sglang/srt/managers/schedule_policy.py:931`

```python
if req.needs_host_load_back():
    new_indices, req.last_node = self.tree_cache.init_load_back(
        InitLoadBackParams(
            best_match_node=req.best_match_node,
            host_hit_length=req.host_hit_length,
            req=req,
        )
    )
    req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
    req.set_extend_input_len(
        len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
    )
    req.cache_protected_len = len(req.prefix_indices)
```

路径：`sglang/python/sglang/srt/mem_cache/hiradix_cache.py:1141`

```python
def load_back(self, node: TreeNode, mem_quota: Optional[int] = None):
    last_hit_node = node
    nodes_to_load = []
    while node.evicted:
        assert node.backuped
        nodes_to_load.insert(0, node)
        node = node.parent

    host_indices = torch.cat([n.host_value for n in nodes_to_load])
    if len(host_indices) < self.load_back_threshold:
        self.dec_lock_ref(ancester_node)
        return None

    device_indices = self.cache_controller.load(
        host_indices=host_indices,
        node_id=last_hit_node.id,
        **self._get_extra_pools(),
    )
    if device_indices is None:
        self.evict(EvictParams(num_tokens=len(host_indices)))
        device_indices = self.cache_controller.load(...)

    for node in nodes_to_load:
        node.value = device_indices[offset : offset + len(node.host_value)].clone()
        self._record_store_event(node, medium=StorageMedium.GPU)

    return device_indices
```

阅读重点：

- Host 命中的节点必须先变回 L1 GPU KV，才能参与 attention。
- 成功后会扩展 `req.prefix_indices`，减少 `extend_input_len`。
- GPU 空间不足时会先 evict 再重试 load。

## 10. GPU 写入树并触发备份

路径：`sglang/python/sglang/srt/mem_cache/hiradix_cache.py:1630`

```python
def insert(self, params: InsertParams) -> InsertResult:
    key = params.key
    value = params.value
    chunked = params.chunked

    key, value = key.maybe_to_bigram_view(self.is_eagle, value)
    key = key.page_aligned(self.page_size)
    value = value[: len(key)]

    while len(key) > 0 and child_key in node.children.keys():
        node = node.children[child_key]
        prefix_len = node.key.match(key, page_size=self.page_size)

        if prefix_len == len(node.key):
            if node.evicted:
                node.value = value[:prefix_len].clone()
                self.evictable_size_ += len(node.value)
            else:
                self._inc_hit_count(node, chunked)

    if len(key):
        new_node = TreeNode(priority=priority)
        new_node.value = value.clone()
        if self.enable_storage or self.enable_kv_cache_events:
            new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
        self._record_store_event(new_node)

        if self.cache_controller.write_policy != "write_back":
            self._inc_hit_count(new_node, chunked)
```

阅读重点：

- `insert` 写入的是 GPU KV indices，也就是 L1。
- 如果节点之前已经被 L1 evict，但 Host 仍有备份，重新计算后会把 `value` 补回来。
- 启用 Storage 时会计算 `hash_value`，后续 L3 backup/query 都依赖它。
- 非 `write_back` 策略会在 insert 后增加命中计数并可能触发 Host 备份。

## 11. Host 到 Storage 的备份线程

路径：`sglang/python/sglang/srt/managers/cache_controller.py:1207`

```python
def _page_backup(self, operation):
    prefix_keys = operation.prefix_keys
    for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
        batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
        batch_host_indices = operation.host_indices[
            i * self.page_size : (i + len(batch_hashes)) * self.page_size
        ]
        extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
        success = self.page_set_func(batch_hashes, batch_host_indices, extra_info)
        if not success:
            break
        operation.completed_tokens += self.page_size * len(batch_hashes)

def backup_thread_func(self):
    while not self.storage_stop_event.is_set():
        operation = self.backup_queue.get(block=True, timeout=1)
        if not self.backup_skip:
            self._page_backup(operation)
        self.ack_backup_queue.put(operation)
```

阅读重点：

- L3 写入是后台线程消费 `backup_queue`。
- 写入粒度是 page hash。
- 支持不同后端的 `page_set_func`，因此 Storage 接口可替换。

## 12. 最小调试观察点

如果要在本地加日志或断点，建议从这些点开始：

| 目的 | 断点位置 |
| --- | --- |
| 确认是否启用 HiCache | `registry.py:77` |
| 看 L1/L2 命中长度 | `hiradix_cache.py:1438` |
| 看是否触发 L3 prefetch | `scheduler.py:2156`、`hiradix_cache.py:1471` |
| 看 Storage 命中 page 数 | `cache_controller.py:1034` |
| 看 L3 拉回 Host 的 token 数 | `hiradix_cache.py:1365` |
| 看 Host 回填 GPU | `schedule_policy.py:931`、`hiradix_cache.py:1141` |
| 看 GPU 写树和 Host 备份 | `hiradix_cache.py:1630` |
| 看 L3 backup | `cache_controller.py:1207` |
