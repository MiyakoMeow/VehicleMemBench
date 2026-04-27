# MemoryBank Search Optimization Design

## Date

2026-04-27

## Background & Motivation

`evaluation/memorysystems/memorybank.py` is an adapted port of the MemoryBank-SiliconFriend memory system for the VehicleMemBench benchmark. The port fixes several bugs from the original implementation (operator precedence in forgetting curve, broken neighbor-merge logic due to `source=memory_id`, score pollution across merged results, etc.).

However, the search phase introduced two behaviors **not present in the original paper/code**:

1. **Recency decay penalty** (`exp(-days_ago / 30)`, floored at 0.3) multiplicatively down-weights older memories during retrieval.
2. **Memory-strength boost** (`1 + 0.15 * log1p(strength)`) multiplicatively up-weights frequently recalled memories.

Additionally, two structural limitations hurt recall on VehicleMemBench's long English dialog histories:

3. **Small initial FAISS recall**: `k = min(top_k, ntotal)` uses the caller's `top_k` (default 5) directly as the FAISS search limit. With ~1 000+ chunks per history file, a relevant memory ranked 6th is permanently lost before re-ranking/merging can recover it.
4. **Date-based formatting reordering**: `format_search_results` sorts results by `source` date string, destroying the relevance order produced by `_merge_neighbors`. The LLM therefore sees chronologically grouped but not relevance-ranked context.

VehicleMemBench contains many `preference_conflict` and `conditional_constraint` tasks where the critical preference may have been stated weeks earlier. The recency penalty pushes these memories out of `top_k`, and the small FAISS recall window prevents the re-ranking stage from ever seeing them.

## Goal

Improve MemoryBank retrieval quality on VehicleMemBench with **minimal, non-invasive changes**:
- No new dependencies.
- No new configuration surface.
- No API changes (same `search` / `add` / `format_search_results` signatures).
- No index rebuild required.
- Keep all original bug fixes intact.

## Chosen Approach: Minimal Invasive Fix (Plan A)

### Change 1: Remove non-original scoring modifiers in `search()`

**Location**: `memorybank.py` inside the `search` method, around the score-adjustment loop.

**Current behavior**:
```python
adjusted = float(score)
# recency decay
if ref_dt:
    ...
    recency = max(0.3, math.exp(-days_ago / 30))
    adjusted *= recency
# strength boost
strength = meta.get("memory_strength", 1)
adjusted *= 1 + 0.15 * math.log1p(strength)
meta["score"] = adjusted
```

**New behavior**: `meta["score"] = float(score)` (pure vector similarity, exactly as the original MemoryBank retrieval did).

**Rationale**: The original paper's retrieval relies on vector similarity + neighbor merge. The extra modifiers were added during porting but are not grounded in the original design. Removing them restores cross-system fairness and improves recall for older-but-critical preferences.

### Change 2: Expand FAISS coarse-recall window

**Location**: `memorybank.py` in `search()`, before `index.search()`.

**Current**:
```python
k = min(top_k, index.ntotal)
```

**New**:
```python
k = min(top_k * 4, index.ntotal)
```

And after `_merge_neighbors`:
```python
merged = self._merge_neighbors(results, user_id)
merged = merged[:top_k]
```

**Rationale**: Standard IR pattern — retrieve a larger candidate pool from the vector index, let the existing re-ranking/merge/dedup logic operate on it, then truncate back to the requested `top_k`. This directly compensates for the long English histories (500+ chunks per file) without changing any merge logic.

### Change 3: Preserve relevance order in `format_search_results()`

**Location**: `memorybank.py` in `format_search_results()`.

**Current**:
```python
sorted_results = sorted(
    non_overall,
    key=lambda r: (r.get("source") or "").removeprefix("summary_"),
)
```

**New**: Remove the `sorted` call; iterate `non_overall` in the order it was returned (already sorted by `score` descending after `_merge_neighbors`).

**Rationale**: The LLM should see the most relevant memory first. Date grouping is a presentation convenience that harms task accuracy when the key fact is not in the most recent date group.

### Unchanged Components

- `_merge_neighbors` logic (bidirectional collection + deque trimming + subset dedup).
- Forgetting curve formula and `_forget_at_ingestion`.
- Daily/overall summary & personality generation pipelines.
- Embedding API calls, L2 normalization, `IndexFlatIP` usage.
- Chunking strategy (`add()` two-line pairing with date prefix).
- `save_index` / `load` serialization format.
- All bug fixes documented in the file header (`[DIFF]` / `[BUGFIX]` annotations).

## Risk & Compatibility

| Risk | Mitigation |
|------|------------|
| Index compatibility | No format change; index files remain valid. |
| Behavioral drift vs. original paper | We are *removing* drift (recency/strength modifiers), moving closer to the original. |
| LLM prompt context length | `top_k` truncation still respects the caller's limit; only internal candidates expand. |
| Performance | `index.search` with `k=20` instead of `k=5` is negligible for `IndexFlatIP` with <10k vectors. |
| Regression on tasks that *do* need recent info | Vector similarity already captures recency when the query mentions current context; explicit decay was over-penalizing. |

## Testing Plan

1. Run the existing `memorysystem_evaluation` test suite on a representative file range (e.g., `1-10`) before the change, recording `exact_match_rate`, `state_f1_positive`, and `change_accuracy`.
2. Apply the three changes.
3. Re-run the identical evaluation command.
4. Compare metrics. Expect improvement in `preference_conflict` and `conditional_constraint` reasoning types, with no significant regression in `error_correction` or `simple_preference`.
5. If metrics improve, run the full `1-50` range and update the benchmark report.

## Success Criteria

- `exact_match_rate` on the full benchmark does not decrease.
- At least one of `state_f1_positive`, `change_accuracy`, or `exact_match_rate` shows measurable improvement on the `preference_conflict` reasoning type.
- No new warnings or errors during `add` / `test` stages.
- Code review passes with no new configuration variables or external dependencies introduced.
