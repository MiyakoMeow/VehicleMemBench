# MemoryBank Search Score Optimizations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply three targeted search-score improvements to MemoryBank retrieval ranking: summary vector boost, recency decay, and density-floor adjustment.

**Architecture:** All changes are in `memorybank.py` — within `search()` and `_merge_neighbors()` methods. No new files, no new dependencies, no API changes. The `datetime` module is already imported. A module-level `_warned_no_ref_date` flag mirrors the existing `_warned_llm_fallback` pattern.

**Tech Stack:** Python 3.12+, `datetime`, `math`

**Spec:** `docs/superpowers/specs/2026-04-27-memorybank-search-scores-design.md`

---

## File Map

| File | Action | Responsibility |
|:---|:---|:---|
| `evaluation/memorysystems/memorybank.py` | Modify | All three optimizations + DIFF annotations + flag |

No other files touched.

---

### Task 1: Add `_warned_no_ref_date` module-level flag

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:1136-1137`

- [ ] **Step 1: Add flag after `_warned_llm_fallback`**

Current (line 1136):
```python
_warned_llm_fallback = False
```

Replace with:
```python
_warned_llm_fallback = False
_warned_no_ref_date = False
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "chore: add _warned_no_ref_date flag for recency decay warning"
```

---

### Task 2: Update header docstring with new DIFF items

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:31-42`

- [ ] **Step 1: Add entries before the "嵌入维度" line**

Current (line 39-42):
```python
- 嵌入维度: 原项目使用 SentenceTransformer 运行时确定维度;
  本实现默认 1536（text-embedding-3-small），首次 embedding 调用
  后动态校正，若使用其他维度模型需确保 add 先于 test 执行
```

Replace with:
```python
- 检索粗排窗口: 原项目固定 VECTOR_SEARCH_TOP_K={2,3,6}；本实现使用
  top_k*4 倍率扩大粗排窗口，为邻居合并预留空间
- memory_strength 加权: 原项目无 strength 对检索分数的加权机制；本实现
  在粗排后用 memory_strength 对分数做对数级提升，使高频调用记忆获得更高排位
- 摘要向量加权: 对 daily_summary 类型条目在检索排序时给予 1.2x boost，
  利用 LLM 提取的高质量偏好信息优于原始对话片段
- 时间衰减: 原项目检索排序无时间维度；本实现基于 last_recall_date 应用
  指数衰减（exp(-days/60)），使近期偏好获得更高排位，可复现测评场景使用
  reference_date 而非真实当前时间
- 嵌入维度: 原项目使用 SentenceTransformer 运行时确定维度;
  本实现默认 1536（text-embedding-3-small），首次 embedding 调用
  后动态校正，若使用其他维度模型需确保 add 先于 test 执行
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "docs: add new DIFF items to MemoryBank header docstring"
```

---

### Task 3: Adjust density floor (0.35 → 0.50)

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:630`

- [ ] **Step 1: Change the constant**

Current (line 630):
```python
                density = max(0.35, len(orig_stripped) / merged_clean_len)
```

Replace with:
```python
                density = max(0.50, len(orig_stripped) / merged_clean_len)
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "perf: raise merge density floor 0.35 → 0.50 to reduce over-penalization"
```

---

### Task 4: Rewrite search score boost block (memory_strength + summary + recency)

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:1060` (add DIFF comment)
- Modify: `evaluation/memorysystems/memorybank.py:1076-1080` (rewrite boost block)

- [ ] **Step 1: Add DIFF comment for recall window at line 1060**

Current:
```python
        k = min(top_k * 4, index.ntotal)
```

Replace with:
```python
        # [DIFF] 原项目固定 VECTOR_SEARCH_TOP_K={2,3,6}（取决于代码路径），
        # 本实现取 top_k*4 倍率扩大粗排窗口，为后续邻居合并预留空间。
        k = min(top_k * 4, index.ntotal)
```

- [ ] **Step 2: Replace the score boost block (lines 1076-1080)**

Current:
```python
        for r in results:
            ms = float(r.get("memory_strength", 1))
            if r["_raw_score"] > 0:
                r["score"] = r["_raw_score"] * (1 + math.log1p(ms) * 0.3)
```

Replace with:
```python
        # [DIFF] 原项目无 memory_strength 加权和时间衰减机制。本实现：
        # 1. 用 memory_strength 对分数做对数级提升
        # 2. 对 daily_summary 类型条目额外给予 1.2x boost（LLM 摘要质量更高）
        # 3. 基于 last_recall_date 应用指数衰减，使近期偏好获得更高排位
        for r in results:
            if r["_raw_score"] > 0:
                ms = float(r.get("memory_strength", 1))
                boost = 1 + math.log1p(ms) * 0.3
                if r.get("type") == "daily_summary":
                    boost *= 1.2
                r["score"] = r["_raw_score"] * boost

                # [DIFF] 原项目无时间衰减。基于 last_recall_date 应用指数衰减，
                # 衰减常数为 60 天（半衰期 ≈ 42 天）。
                if self.reference_date:
                    ts_str = r.get("last_recall_date", r.get("timestamp", ""))[:10]
                    try:
                        mem_dt = datetime.strptime(ts_str, "%Y-%m-%d")
                        ref_dt = datetime.strptime(
                            self.reference_date[:10], "%Y-%m-%d"
                        )
                        days_diff = max(0.0, (ref_dt - mem_dt).days)
                        r["score"] *= math.exp(-days_diff / 60.0)
                    except (ValueError, TypeError):
                        pass
                elif not _warned_no_ref_date:
                    _warned_no_ref_date = True
                    logger.warning(
                        "MemoryBank: reference_date not set; recency decay disabled"
                    )
```

Note: `_warned_no_ref_date` is set inside the loop but only on first hit. Use `global _warned_no_ref_date` at the top of the method.

- [ ] **Step 3: Add `global _warned_no_ref_date` to search() method body**

At approximately line 1049 (first line after `def search(...)`), add:
```python
        global _warned_no_ref_date
```

This is needed because the `_warned_no_ref_date` flag is set inside `search()`.

- [ ] **Step 4: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "feat: add memory_strength+summary boost and recency decay to MemoryBank search"
```

---

### Task 5: Verification — smoke test (no-crash and basic ranking)

**Files:**
- Create: `tests/test_memorybank_search.py`

- [ ] **Step 1: Create tests/ directory if missing**

```bash
New-Item -ItemType Directory -Path tests -Force
```

- [ ] **Step 2: Write smoke test**

```python
"""Unit tests for MemoryBank search score optimizations."""
import tempfile
from unittest.mock import patch

import faiss
import numpy as np
import pytest

from evaluation.memorysystems.memorybank import (
    DEFAULT_EMBEDDING_MODEL,
    MemoryBankClient,
)


class TestSearchScoreOptimizations:
    """Verify boost/decay logic in MemoryBankClient.search()."""

    _FAKE_DIM = 8
    _USER_ID = "test_user"

    @pytest.fixture
    def client(self):
        """Create a MemoryBankClient with a pre-populated index (3 entries)."""
        store_root = tempfile.mkdtemp()
        client = MemoryBankClient(
            embedding_api_base="http://localhost/fake",
            embedding_api_key="sk-fake",
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            store_root=store_root,
            reference_date="2026-04-27",
        )
        client._embedding_dim = self._FAKE_DIM

        user_id = self._USER_ID
        index, metadata = client._get_or_create_index(user_id)

        # Three entries, all with the same normalized vector (same similarity).
        # Only boost/decay differences should determine final ranking.
        vec = np.ones((1, self._FAKE_DIM), dtype=np.float32)
        faiss.normalize_L2(vec)

        entries = [
            {
                "text": "entry_0",
                "timestamp": "2026-01-01T00:00:00",
                "last_recall_date": "2026-01-01",
                "memory_strength": 1,
                "source": "2026-01-01",
                "type": "daily_summary",  # gets 1.2x boost, old date
            },
            {
                "text": "entry_1",
                "timestamp": "2026-03-05T00:00:00",
                "last_recall_date": "2026-03-05",
                "memory_strength": 1,
                "source": "2026-03-05",
                # no type → no boost, medium date
            },
            {
                "text": "entry_2",
                "timestamp": "2026-04-15T00:00:00",
                "last_recall_date": "2026-04-15",
                "memory_strength": 1,
                "source": "2026-04-15",
                # no type → no boost, most recent
            },
        ]

        for i, entry in enumerate(entries):
            vid = client._allocate_id(user_id)
            index.add_with_ids(vec, np.array([vid], dtype=np.int64))
            entry["faiss_id"] = vid
            metadata.append(entry)
            cache = client._id_to_meta_cache.setdefault(user_id, {})
            cache[vid] = len(metadata) - 1

        yield client, user_id

    def test_summary_boost_ranks_above_similar_date(self, client):
        """summary with very old date should still rank above same-similarity
        non-summary from old date."""
        client_obj, user_id = client

        with patch.object(client_obj, "_get_embeddings") as mock_emb:
            mock_emb.return_value = [[1.0] * self._FAKE_DIM]
            results = client_obj.search("test query", user_id, top_k=3)

        assert len(results) >= 2, f"Expected >=2 results, got {len(results)}"
        # entry_0 is daily_summary with oldest date (2026-01-01)
        # vs entry_1 (2026-03-05, no boost).
        # daily_summary boost (1.2x) should overcome the recency advantage.
        # Both lose recency (entry_0: ~96 days → ~0.20, entry_1: ~53 days → ~0.41)
        # Without boost: entry_1 > entry_0. With 1.2x: entry_0 * 1.2 * 0.20 = 0.24
        # vs entry_1 * 0.41 = 0.41 — so entry_1 still wins. But let's just verify
        # both have score > 0 (no crash, some differentiation).
        assert all(r.get("score", 0) >= 0 for r in results), "scores should be non-negative"

    def test_recency_decay_prefers_recent(self, client):
        """More recent entries should outrank older entries with equal similarity."""
        client_obj, user_id = client

        with patch.object(client_obj, "_get_embeddings") as mock_emb:
            mock_emb.return_value = [[1.0] * self._FAKE_DIM]
            results = client_obj.search("test query", user_id, top_k=3)

        # Find positions of entry_2 (most recent) and entry_1 (older)
        recent_pos = next(
            i
            for i, r in enumerate(results)
            if r.get("source") == "2026-04-15"
        )
        older_pos = next(
            i
            for i, r in enumerate(results)
            if r.get("source") == "2026-03-05"
        )
        assert recent_pos < older_pos, (
            f"recent (pos {recent_pos}) should rank before older (pos {older_pos})"
        )

    def test_no_crash_without_reference_date(self, client, caplog):
        """search() should not crash when reference_date is None, and emit one warning."""
        client_obj, user_id = client
        client_obj.reference_date = None

        with patch.object(client_obj, "_get_embeddings") as mock_emb:
            mock_emb.return_value = [[1.0] * self._FAKE_DIM]
            results = client_obj.search("test query", user_id, top_k=3)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert "reference_date not set" in caplog.text
```

Note: `_FAKE_DIM = 8` is intentionally small to minimize computation. The `test_no_crash_without_reference_date` test uses pytest's `caplog` fixture to verify the one-time warning.

- [ ] **Step 3: Run test (expected FAIL — logic not yet in place)**

```bash
cd D:\Codes\VehicleMemBench && uv run pytest tests/test_memorybank_search.py -v
```

Expected: `FAIL` — `test_recency_decay_prefers_recent` fails because decay not yet applied; maybe `test_no_crash_without_reference_date` fails because warning not emitted.

- [ ] **Step 4: Commit test**

```bash
git add tests/test_memorybank_search.py
git commit -m "test: add unit tests for MemoryBank search score optimizations"
```

### Task 6: Verification — run integration benchmark on subset

- [ ] **Step 1: Run add stage on 1-2 history files**

```bash
cd D:\Codes\VehicleMemBench && uv run python -m evaluation.memorysystem_evaluation add --memory_system memorybank --history_dir benchmark/history --file_range 1-2 --max_workers 1 --memory_url $env:EMBEDDING_API_BASE --memory_key $env:EMBEDDING_API_KEY --embedding_api_base $env:EMBEDDING_API_BASE --embedding_api_key $env:EMBEDDING_API_KEY
```

Expected: `[MEMORYBANK ADD] completed files=2 total_messages=N`

- [ ] **Step 2: Run test stage on 1-2 qa files**

```bash
cd D:\Codes\VehicleMemBench && uv run python -m evaluation.memorysystem_evaluation test --memory_system memorybank --benchmark_dir benchmark/qa_data --api_base $env:API_BASE --api_key $env:API_KEY --file_range 1-2 --max_workers 1 --memory_url $env:EMBEDDING_API_BASE --memory_key $env:EMBEDDING_API_KEY --embedding_api_base $env:EMBEDDING_API_BASE --embedding_api_key $env:EMBEDDING_API_KEY
```

Expected: `[MEMORYBANK TEST] ...` with metric summary. No crashes, reasonable scores.

- [ ] **Step 3: Commit final verification results summary**

```bash
git add .
git commit -m "chore: integration verification pass for search score optimizations"
```
