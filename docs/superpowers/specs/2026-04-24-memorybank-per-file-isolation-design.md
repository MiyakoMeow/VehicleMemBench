# MemoryBank 每文件对象隔离

## 目标

消除 MemoryBank 测试阶段最后一个共享对象 `shared_state`，实现 50 个测试组之间完全的对象隔离。

## 背景

### 当前架构

MemoryBank 遵循 memory system 插件协议：`init_test_state` → `build_test_client`(xN) → `close_test_state`。

当前 `init_test_state` 计算全局 `reference_date`（所有 history 文件最大时间戳+1天），通过 `shared_state` dict 传递给每个 `build_test_client` 调用。这是唯一跨测试组共享的可变对象。

### 对比其他系统

| 系统 | init_test_state | build_test_client | shared_state |
|------|----------------|-------------------|--------------|
| mem0/memos/supermemory/memobase | return None | 创建新客户端 | 无 |
| lightmem | 创建共享客户端 | switch_user | 共享客户端（sequential） |
| **memorybank（当前）** | 计算全局 reference_date | 从 shared_state 读取 | reference_date dict |
| **memorybank（目标）** | return None | 独立计算 reference_date | 无 |

## 设计

### 改动范围

仅修改 `evaluation/memorysystems/memorybank.py`。

### 具体改动

#### 1. `init_test_state` → no-op

删除 `_compute_reference_date` 调用，改为仅做参数校验后返回 None。

```python
def init_test_state(args, file_numbers, user_id_prefix):
    del file_numbers, user_id_prefix
    validate_test_args(args)
    return None
```

#### 2. `build_test_client` → 每文件独立计算 reference_date

忽略 `shared_state` 参数，当 client.reference_date 未设置时，从对应单个 history 文件计算。

```python
def build_test_client(args, file_num: int, user_id_prefix: str, shared_state: Any):
    del shared_state
    client = _build_client(args)
    if not client.reference_date:
        history_dir = os.path.abspath(args.history_dir)
        client.reference_date = _compute_reference_date(history_dir, str(file_num))
    uid = f"{user_id_prefix}_{file_num}"
    client._get_or_create_index(uid)
    return _MemoryBankTestWrapper(client, uid)
```

复用现有 `_compute_reference_date(history_dir, str(file_num))` —— 传入单个文件号作为 file_range，仅处理该文件的 history。

#### 3. `close_test_state` → 不变

已经是 no-op，无需修改。

#### 4. 不修改的部分

- `_build_client`、`MemoryBankClient`、`run_add`
- `_MemoryBankTestWrapper`、`format_search_results`、`is_test_sequential`
- `memorysystem_evaluation.py`（已正确处理 init_test_state 返回 None）

### reference_date 语义变化

| | 改动前 | 改动后 |
|---|--------|--------|
| 来源 | 所有 history 文件全局最大时间戳 | 每个 history 文件各自最大时间戳 |
| 值 | 所有测试组相同 | 每个测试组可能不同 |
| 遗忘基准 | 统一基准日 | 每用户独立基准日 |

当环境变量 `MEMORYBANK_REFERENCE_DATE` 已设置时，行为不变（所有测试组使用该固定值）。

## 约束

- 不考虑内存/IO/连接开销
- 优先可维护性和执行效率
- 仅修改 memorybank 相关代码
