# MemoryBank Search Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve MemoryBank retrieval quality on VehicleMemBench by removing non-original scoring modifiers, expanding the FAISS coarse-recall window, and preserving relevance order in output formatting.

**Architecture:** Three focused edits inside `evaluation/memorysystems/memorybank.py`: (1) simplify `search()` scoring to pure vector similarity, (2) multiply initial FAISS `k` by 4 then truncate back to `top_k` after merge, (3) remove date-based reordering in `format_search_results()`.

**Tech Stack:** Python 3.12, FAISS (faiss-cpu), NumPy, existing `memorybank.py` module.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `evaluation/memorysystems/memorybank.py` | Core MemoryBank system — contains `search()` and `format_search_results()` to modify. |
| `scripts/verify_memorybank_changes.py` | Standalone verification script: mocks embedding API, builds a tiny FAISS index, asserts that scores are raw similarities and formatting preserves relevance order. |

---

### Task 1: Remove recency decay and strength boost from `search()` scoring

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:1059-1095`

- [ ] **Step 1: Replace the score-adjustment loop with raw similarity assignment**

Replace this block (lines 1060-1095):

```python
        results: List[dict] = []
        # [DIFF] 原项目仅用 FAISS L2 距离排序。本实现改进为三项融合：
        # 1) 向量相似度 (cosine/IP)  2) 时效性衰减  3) memory_strength 加权。
        ref_dt = None
        if self.reference_date:
            try:
                ref_dt = datetime.strptime(self.reference_date[:10], "%Y-%m-%d")
            except ValueError:
                pass

        for score, faiss_id in zip(scores[0], indices[0]):
            meta_idx = id_to_meta.get(int(faiss_id))
            if meta_idx is None:
                continue
            meta = dict(metadata[meta_idx])
            adjusted = float(score)

            # 时效性衰减：时间常数 30 天 (exp(-t/30))，约 21 天半衰期，加权下限 0.3
            if ref_dt:
                date_str = meta.get("last_recall_date", meta.get("timestamp", ""))[:10]
                try:
                    mem_dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    mem_dt = None
                if mem_dt:
                    days_ago = max(0, (ref_dt - mem_dt).days)
                    recency = max(0.3, math.exp(-days_ago / 30))
                    adjusted *= recency

            # memory_strength 加权：高频访问记忆适度加权
            strength = meta.get("memory_strength", 1)
            adjusted *= 1 + 0.15 * math.log1p(strength)

            meta["score"] = adjusted
            meta["_raw_score"] = float(score)
            meta["_meta_idx"] = meta_idx
            results.append(meta)
```

With:

```python
        results: List[dict] = []
        for score, faiss_id in zip(scores[0], indices[0]):
            meta_idx = id_to_meta.get(int(faiss_id))
            if meta_idx is None:
                continue
            meta = dict(metadata[meta_idx])
            meta["score"] = float(score)
            meta["_raw_score"] = float(score)
            meta["_meta_idx"] = meta_idx
            results.append(meta)
```

- [ ] **Step 2: Remove unused `ref_dt` variable and related imports if now unused**

`ref_dt` was only used in the recency-decay block above. `datetime` and `math` are still used elsewhere in the file (e.g. `_forgetting_retention`, `_compute_reference_date`), so do **not** remove imports.

- [ ] **Step 3: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "fix(memorybank): remove non-original recency decay and strength boost in search"
```

---

### Task 2: Expand FAISS coarse-recall window and truncate after merge

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:1054` and `1110`

- [ ] **Step 1: Increase initial FAISS search `k`**

At line 1054, change:
```python
        k = min(top_k, index.ntotal)
```
To:
```python
        k = min(top_k * 4, index.ntotal)
```

**Side effect note**: This also expands the `memory_strength` / `last_recall_date` update loop (lines 1097–1104) to cover up to `4×top_k` candidates. This is acceptable because the benchmark disables forgetting by default (`enable_forgetting=False`), and the extra candidates are still the most relevant ones retrieved by FAISS. If forgetting were enabled, this would slightly increase the recall-boost surface, which is consistent with the intended behavior of strengthening retrieved memories.

- [ ] **Step 2: Truncate merged results back to requested `top_k`**

At line 1109-1110, change:
```python
        merged = self._merge_neighbors(results, user_id)
```
To:
```python
        merged = self._merge_neighbors(results, user_id)
        merged = merged[:top_k]
```

- [ ] **Step 3: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "feat(memorybank): expand FAISS coarse-recall window 4x and truncate after merge"
```

---

### Task 3: Preserve relevance order in `format_search_results()`

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:1388-1391`

- [ ] **Step 1: Remove date-based sorting**

Replace:
```python
    sorted_results = sorted(
        non_overall,
        key=lambda r: (r.get("source") or "").removeprefix("summary_"),
    )
```
With:
```python
    sorted_results = non_overall
```

- [ ] **Step 2: Update docstring**

Change the function docstring from:
```python
    """将检索结果格式化为带编号的文本，按日期分组并标注记忆强度。"""
```
To:
```python
    """将检索结果格式化为带编号的文本，按相关性顺序分组并标注记忆强度。"""
```

- [ ] **Step 3: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "fix(memorybank): preserve relevance order in format_search_results"
```

---

### Task 4: Write and run verification script

**Files:**
- Create: `scripts/verify_memorybank_changes.py`

- [ ] **Step 1: Create the verification script**

Create `scripts/verify_memorybank_changes.py` with the following content:

```python
"""Quick verification that MemoryBank search returns raw cosine scores
and format_search_results preserves relevance order."""
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import faiss
import numpy as np
from evaluation.memorysystems.memorybank import (
    MemoryBankClient,
    format_search_results,
)


def test_raw_scores_and_relevance_order():
    client = MemoryBankClient(
        embedding_api_base="http://dummy",
        embedding_api_key="dummy",
        enable_summary=False,
        enable_forgetting=False,
        reference_date="2025-04-27",
    )
    # Bypass real embedding calls
    client._get_embeddings = lambda texts: [[0.9, 0.1]]

    user_id = "test_user"
    index, metadata = client._get_or_create_index(user_id)
    dim = 2
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    client._indices[user_id] = index
    client._metadata[user_id] = []

    # Vector A: lower similarity to query (0.5, 0.5), recent date
    vec_a = np.array([[0.5, 0.5]], dtype=np.float32)
    faiss.normalize_L2(vec_a)
    index.add_with_ids(vec_a, np.array([0], dtype=np.int64))
    client._metadata[user_id].append({
        "text": "A: recent low similarity",
        "timestamp": "2025-04-26T00:00:00",
        "memory_strength": 1,
        "last_recall_date": "2025-04-26",
        "faiss_id": 0,
        "source": "2025-04-26",
    })

    # Vector B: higher similarity to query (0.9, 0.1), old date
    vec_b = np.array([[0.9, 0.1]], dtype=np.float32)
    faiss.normalize_L2(vec_b)
    index.add_with_ids(vec_b, np.array([1], dtype=np.int64))
    client._metadata[user_id].append({
        "text": "B: old high similarity",
        "timestamp": "2025-03-01T00:00:00",
        "memory_strength": 10,
        "last_recall_date": "2025-03-01",
        "faiss_id": 1,
        "source": "2025-03-01",
    })

    # Query vector identical to B
    q = np.array([[0.9, 0.1]], dtype=np.float32)
    faiss.normalize_L2(q)

    # Direct FAISS search to get ground-truth raw scores
    raw_scores, raw_indices = index.search(q, 2)
    print("Raw FAISS scores:", raw_scores[0])
    print("Raw FAISS indices:", raw_indices[0])

    # Client search
    results = client.search(query="dummy", user_id=user_id, top_k=2)
    returned_scores = [r["score"] for r in results]
    print("Returned scores:", returned_scores)

    # Assert scores are raw cosine similarities (within float tolerance)
    # FAISS returns scores sorted descending; raw_scores[0][0] is highest (B)
    assert abs(returned_scores[0] - raw_scores[0][0]) < 1e-5, (
        f"Expected first score {raw_scores[0][0]}, got {returned_scores[0]}"
    )
    assert abs(returned_scores[1] - raw_scores[0][1]) < 1e-5, (
        f"Expected second score {raw_scores[0][1]}, got {returned_scores[1]}"
    )
    print("PASS: scores are raw cosine similarities")

    # Assert most relevant (B) comes first in formatted output
    formatted, count = format_search_results(results)
    print("Formatted output:\n", formatted)
    lines = formatted.split("\n\n")
    assert "B: old high similarity" in lines[0], (
        "Expected most relevant item (B) first in formatted output"
    )
    print("PASS: format_search_results preserves relevance order")

    # Assert top_k truncation works with expanded recall
    results_k1 = client.search(query="dummy", user_id=user_id, top_k=1)
    assert len(results_k1) == 1, f"Expected 1 result for top_k=1, got {len(results_k1)}"
    assert results_k1[0]["text"] == "B: old high similarity", (
        "Expected B to be the single truncated result"
    )
    print("PASS: top_k truncation works correctly")

    # Clean up test user store directory
    import shutil
    store_dir = os.path.join(client._store_root, f"user_{user_id}")
    if os.path.isdir(store_dir):
        shutil.rmtree(store_dir)

    print("\nAll verification tests passed.")


if __name__ == "__main__":
    test_raw_scores_and_relevance_order()
```

- [ ] **Step 2: Run the verification script**

```bash
uv run python scripts/verify_memorybank_changes.py
```

Expected output (approximate, exact float values may vary slightly):
```
Raw FAISS scores: [1.0 0.7808688]
Raw FAISS indices: [1 0]
Returned scores: [1.0, 0.7808688282966614]
PASS: scores are raw cosine similarities
Formatted output:
 ...
PASS: format_search_results preserves relevance order
PASS: top_k truncation works correctly

All verification tests passed.
```

- [ ] **Step 3: Commit the verification script**

```bash
git add scripts/verify_memorybank_changes.py
git commit -m "test: add MemoryBank search verification script"
```

---

### Task 5: Run benchmark integration test (optional but recommended)

**Files:**
- No file changes.

- [ ] **Step 1: Run add stage on a small file range**

Requires `EMBEDDING_API_BASE` and `EMBEDDING_API_KEY` environment variables.

```bash
uv run python evaluation/memorysystem_evaluation.py add \
  --memory_system memorybank \
  --history_dir benchmark/history \
  --file_range "1-5" \
  --embedding_api_base "$EMBEDDING_API_BASE" \
  --embedding_api_key "$EMBEDDING_API_KEY"
```

- [ ] **Step 2: Run test stage on the same range**

Requires `LLM_API_BASE` and `LLM_API_KEY` as well.

```bash
uv run python evaluation/memorysystem_evaluation.py test \
  --memory_system memorybank \
  --benchmark_dir benchmark/qa_data \
  --file_range "1-5" \
  --api_base "$LLM_API_BASE" \
  --api_key "$LLM_API_KEY" \
  --model gpt-4o-mini \
  --max_workers 1 \
  --embedding_api_base "$EMBEDDING_API_BASE" \
  --embedding_api_key "$EMBEDDING_API_KEY"
```

- [ ] **Step 3: Check the generated report**

Look at `memory_system_log/memorysystem_eval_memorybank_*/report.txt` and confirm:
- No errors or skipped tasks.
- `exact_match_rate` is non-zero (indicates memories are being retrieved).

---

## Spec Coverage Checklist

| Spec Requirement | Plan Task |
|---|---|
| Remove recency decay and strength boost from `search()` | Task 1 |
| Expand FAISS coarse-recall `k` to `top_k * 4` | Task 2 |
| Truncate merged results back to `top_k` | Task 2 |
| Preserve relevance order in `format_search_results()` | Task 3 |
| Verify changes with automated test | Task 4 |
| Validate with benchmark integration test | Task 5 |

## Placeholder Scan

- No "TBD", "TODO", or "implement later" strings.
- No vague "add error handling" or "write tests for the above" steps.
- Every code block contains the exact code to write/replace.
- Every run command contains the exact shell command.

## Type Consistency Check

- `meta["score"]` remains `float` in all tasks.
- `top_k` remains `int`.
- `merged` remains `List[dict]`.
- `format_search_results` signature and return type unchanged.
