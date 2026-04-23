# MemoryBank 每文件对象隔离 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 MemoryBank 测试阶段的 shared_state，实现 50 个测试组间完全的对象隔离。

**Architecture:** 每个测试组（文件）在 `build_test_client` 中独立计算 reference_date，`init_test_state`/`close_test_state` 变为 no-op。仅修改 `memorybank.py`。

**Tech Stack:** Python 3, 无新依赖

---

### Task 1: 修改 init_test_state

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:483-491`

- [ ] **Step 1: 替换 init_test_state 实现**

将 `init_test_state` 从计算全局 reference_date 改为 no-op：

```python
def init_test_state(args, file_numbers, user_id_prefix):
    del file_numbers, user_id_prefix
    validate_test_args(args)
    return None
```

原代码（`memorybank.py:483-491`）：
```python
def init_test_state(args, file_numbers, user_id_prefix):
    del file_numbers, user_id_prefix
    validate_test_args(args)
    reference_date = _resolve_reference_date()
    if not reference_date:
        reference_date = _compute_reference_date(
            os.path.abspath(args.history_dir), args.file_range
        )
    return {"reference_date": reference_date}
```

- [ ] **Step 2: 验证语法**

Run: `cd /root/Codes/VehicleMemBench && uv run python -c "from evaluation.memorysystems.memorybank import init_test_state; print('OK')"`

Expected: `OK`

---

### Task 2: 修改 build_test_client

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:494-500`

- [ ] **Step 1: 替换 build_test_client 实现**

将 `build_test_client` 改为独立计算 reference_date，忽略 shared_state：

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

原代码（`memorybank.py:494-500`）：
```python
def build_test_client(args, file_num: int, user_id_prefix: str, shared_state: Any):
    client = _build_client(args)
    if not client.reference_date:
        client.reference_date = shared_state["reference_date"]
    uid = f"{user_id_prefix}_{file_num}"
    client._get_or_create_index(uid)
    return _MemoryBankTestWrapper(client, uid)
```

- [ ] **Step 2: 验证语法和导入**

Run: `cd /root/Codes/VehicleMemBench && uv run python -c "from evaluation.memorysystems.memorybank import build_test_client; print('OK')"`

Expected: `OK`

---

### Task 3: 端到端验证

- [ ] **Step 1: 验证 memorysystem_evaluation.py 正确调用新接口**

Run: `cd /root/Codes/VehicleMemBench && uv run python -c "
from evaluation.memorysystems import get_system_module
mod = get_system_module('memorybank')
print('init_test_state returns:', mod.init_test_state.__module__)
print('is_test_sequential:', mod.is_test_sequential())
print('OK')
"`

Expected: 输出模块名 + `False` + `OK`

- [ ] **Step 2: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "fix(memorybank): eliminate shared_state, compute reference_date per-file"
```
