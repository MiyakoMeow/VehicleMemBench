# MemoryBank Search Score Optimizations

**Date**: 2026-04-27
**Status**: approved
**Scope**: `evaluation/memorysystems/memorybank.py` — `search()` and `_merge_neighbors()` methods only

---

## 1. Motivation

Three targeted improvements to MemoryBank's retrieval ranking, drawn from comparison with the original MemoryBank-SiliconFriend implementation:

1. **Summary vectors are underweighted**: daily_summary entries (LLM-extracted vehicle preferences) have much higher information density than raw conversation snippets, yet they receive no ranking boost.
2. **No recency signal**: vehicle preferences naturally decay in relevance over time, but the current score ignores temporal distance entirely.
3. **Merge density penalty is too aggressive**: floor is 0.35 (65% score reduction minimum), penalizing reasonable neighbor merges excessively.

All changes are additive, with no API contract changes and no new dependencies.

---

## 2. Design

All score transformations (boost, decay) are guarded by the existing `if r["_raw_score"] > 0:` check. Negative-IP results (dissimilar vectors, rare in the top-20 recall window) pass through unchanged. This preserves the current guard semantics.

### 2.1 Combined Boost (memory_strength + summary)

**Location**: `search()`, approximately line 1076–1079, inside the raw-score-to-final-score loop.

**Change**:

```python
# Before
if r["_raw_score"] > 0:
    r["score"] = r["_raw_score"] * (1 + math.log1p(ms) * 0.3)

# After
if r["_raw_score"] > 0:
    boost = 1 + math.log1p(ms) * 0.3
    if r.get("type") == "daily_summary":
        boost *= 1.2
    r["score"] = r["_raw_score"] * boost
```

**Rationale**: daily_summary entries are LLM-distilled vehicle preferences. A 1.2x multiplier gives them a modest edge over equally-similar raw conversation snippets. The boost is combined with memory_strength boost into a single multiplicative factor, keeping the loop at O(1) per result. The `_raw_score > 0` guard is preserved: negative scores (dissimilar vectors) are left unchanged.

### 2.2 Recency Decay

**Location**: `search()`, immediately after the combined boost (2.1), still inside the `_raw_score > 0` guard.

**Change**:

```python
if self.reference_date:
    ts_str = r.get("last_recall_date", r.get("timestamp", ""))[:10]
    try:
        mem_dt = datetime.strptime(ts_str, "%Y-%m-%d")
        ref_dt = datetime.strptime(self.reference_date[:10], "%Y-%m-%d")
        days_diff = max(0.0, (ref_dt - mem_dt).days)
        r["score"] *= math.exp(-days_diff / 60.0)
    except (ValueError, TypeError):
        pass
```

**Rationale**: Exponential decay with 60-day constant (half-life ≈ 42 days). A preference recalled 30 days ago retains ~61% of its score; 90 days ago ~22%; 180 days ago ~5%. Uses `last_recall_date` (updated on each search recall) rather than `timestamp` (creation date), so frequently-accessed memories decay slower. Decay is only applied to positive scores (inside the guard) to avoid inverting negative scores toward zero.

**Edge case — reference_date is None**: Emit a one-time `WARNING` log message (using a file-level `_warned_no_ref_date` flag, consistent with the existing `_warned_llm_fallback` pattern) and skip decay entirely. No crash, no exception.

### 2.3 Density Floor Adjustment

**Location**: `_merge_neighbors()`, approximately line 630.

**Change**:

```python
# Before
density = max(0.35, len(orig_stripped) / merged_clean_len)

# After
density = max(0.50, len(orig_stripped) / merged_clean_len)
```

**Rationale**: Merging 2–3 semantically-related neighbors into a single result is normal and desirable behavior. A 0.35 floor reduces the merged result's score to at most 35% of the original hit's score, which is too harsh. Raising to 0.50 means the worst-case penalty is a 50% reduction.

---

## 3. DIFF Annotation Additions

Three missing `[DIFF]` markers to add alongside the code changes:

| Line (approx) | Content |
|:---|:---|
| ~1060 | `k = min(top_k * 4, ...)` — original uses fixed `VECTOR_SEARCH_TOP_K={2,3,6}`; this impl uses 4x to widen recall window for neighbor merging |
| ~1076 | memory_strength score boost — original has no strength-weighted re-ranking |
| ~1076 | summary boost and recency decay — both are new additions |

---

## 4. Non-Goals

- Query expansion (extra LLM call overhead, out of scope)
- Date pre-filtering (low benefit for current benchmark time span)
- Changing the merge algorithm itself (already well-tested)
- Adding new configuration knobs — parameters are hardcoded to keep surface area small

---

## 5. Verification

- Run existing `memorysystem_evaluation` with `memorybank` on a subset (e.g. `--file_range 1-5`) and compare Exact Match Rate / Field F1 before and after.
- Manually inspect a few `search()` result score orderings to sanity-check that daily_summary entries rank higher than noise.
- Confirm no crash when `reference_date=None` (log warning appears once).
