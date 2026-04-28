# MemoryBank Prompt & Credential Fix

**Date**: 2026-04-28
**Status**: approved
**Scope**: `evaluation/memorysystem_evaluation.py`, `evaluation/memorysystems/memorybank.py`, `scripts/base_test.sh`

---

## 1. Motivation

Two root causes identified from MemoryBank integration evaluation (5% exact match, 85% of tasks had zero tool calls):

1. **System prompt too passive** — model retrieves correct preferences via `search_memory` then asks the user for permission instead of executing tool calls.
2. **Credential model fractured** — MemoryBank used separate `LLM_API_BASE`/`LLM_API_KEY` env vars with fallback logic for summary generation, diverging from other modules. This made summary generation silently fail, degrading retrieval quality.

---

## 2. Design

### 2.1 System Prompt Hardening

**File**: `evaluation/memorysystem_evaluation.py`, `get_search_memory_schema()` inner function area (~line 158-164).

**Current Rules**:
```
Rules:
1. Call search_memory first when the request depends on user preferences or history.
2. Use list_module_tools(module_name=...) to discover vehicle tools.
3. Call vehicle tools to satisfy the query.
4. Avoid unnecessary parameter changes if exact values are unavailable.
5. Do not repeatedly query the same memory information or invoke the same vehicle tool
   in consecutive steps unless new evidence requires it.
```

**New Rules**:
```
Rules:
1. Call search_memory first when the request depends on user preferences or history.
2. Use list_module_tools(module_name=...) to discover vehicle tools.
3. Call vehicle tools to satisfy the query.
4. Never ask the user for permission, clarification, or confirmation about vehicle
   operations. If search_memory returns relevant preferences, act on them immediately.
   You are an autonomous agent — execute, do not chat.
5. Avoid unnecessary parameter changes if exact values are unavailable.
6. Do not repeatedly query the same memory information or invoke the same vehicle tool
   in consecutive steps unless new evidence requires it.
```

**Rationale**: A single hard rule against user interaction breaks the "chat-first" behavior observed in glm-4.5-air. The phrase "autonomous agent — execute, do not chat" reinforces the role distinction.

### 2.2 Unified LLM/Embedding Credential Model

#### 2.2.1 CLI Additions

**File**: `evaluation/memorysystem_evaluation.py`.

Three touch-points:

**(a)** `_build_cli_parser()` add subparser: Add two new arguments:
```python
add_parser.add_argument("--api_base", type=str, default=None, help="LLM API base URL (for summary generation)")
add_parser.add_argument("--api_key", type=str, default=None, help="LLM API key")
```
`--model` already exists on the add subcommand.

**(b)** `memorysystem_add()` signature: Add `api_base` and `api_key` parameters and include them in `args` namespace:
```python
def memorysystem_add(
    *, ...,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    ...
):
    args = argparse.Namespace(
        ...,
        api_base=api_base,
        api_key=api_key,
        model=model,
        ...
    )
```

**(c)** `__main__` dispatch block: Update the `memorysystem_add(...)` call to forward `cli_args.api_base` and `cli_args.api_key` (currently only `cli_args.model` is forwarded):
```python
memorysystem_add(
    ...,
    api_base=cli_args.api_base,
    api_key=cli_args.api_key,
    model=cli_args.model,
    ...
)
```

#### 2.2.2 MemoryBank Credential Cleanup

**File**: `evaluation/memorysystems/memorybank.py`.

**Remove**:
- `_warned_llm_fallback` module flag (near `_warned_llm_fallback = False`)
- `_resolve_llm_credentials()` function (entire function body)

**Modify** `_build_client()` — replace the `_resolve_llm_credentials(...)` call with direct `args` access:
```python
def _build_client(args, user_id: str = "") -> MemoryBankClient:
    api_key = require_value(...)      # embedding — unchanged
    api_base = require_value(...)     # embedding — unchanged
    enable_summary = _resolve_enable_summary()
    enable_forgetting = _resolve_enable_forgetting()
    seed = _resolve_seed()
    reference_date = _resolve_reference_date()

    # LLM credentials come from the same backbone API as evaluation
    llm_api_base = getattr(args, "api_base", None)
    llm_api_key = getattr(args, "api_key", None)
    llm_model = getattr(args, "model", None) or os.getenv("LLM_MODEL", "gpt-4o-mini")

    client = MemoryBankClient(
        embedding_api_base=api_base,
        embedding_api_key=api_key,
        embedding_model=...,
        enable_forgetting=enable_forgetting,
        enable_summary=enable_summary,
        seed=seed,
        reference_date=reference_date,
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        store_root=_resolve_store_root(args),
    )
    if enable_summary and client._llm_client is None:
        logger.warning(
            "MemoryBank: summaries enabled but no LLM credentials available "
            "(pass --api_base/--api_key to the add command); "
            "daily/overall summaries and personality analysis will NOT be generated"
        )
    return client
```

No fallback to embedding credentials. LLM and embedding are now separate, non-overlapping configuration domains.

#### 2.2.3 Shell Script Update

**File**: `scripts/base_test.sh`, memorybank add section (~line 154) and test section (~line 165).

Add `--api_base "$API_BASE" --api_key "$API_KEY" --model "$MODEL"` to both the add and test invocations.

---

## 3. Non-Goals

- Changing the `--memory_url`/`--memory_key` generic args (kept as embedding fallback per existing pattern)
- Modifying other memory system modules (mem0, memos, etc.)
- Changing the summary generation prompts or logic
- Adding structured key-value memory (separate future work)

---

## 4. Verification

1. Run `add` stage with `--api_base`/`--api_key`/`--model` set — summary generation should succeed (no "LLM_API_BASE not set" warning)
2. Run `test` stage on file_range 1-2 — exact match rate should improve from ~5% baseline
3. Run `add` with `--api_base`/`--api_key` omitted — faiss indexes build normally, but no `daily_summary`/`overall_summary`/`personality` vectors are created (existing behavior: the `enable_summary and client._llm_client is None` gate skips all LLM-dependent generation). One WARNING logged.
4. `_resolve_llm_credentials` function and `_warned_llm_fallback` flag no longer exist in codebase
