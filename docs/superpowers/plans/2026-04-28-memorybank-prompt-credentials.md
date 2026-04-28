# MemoryBank Prompt & Credential Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden system prompt to enforce autonomous tool execution (fix 85% zero-tool-call rate), and unify LLM credentials to use backbone model API (remove separate LLM_API_* env vars with fallback logic).

**Architecture:** Three files modified. Prompt change is a single rule insertion in `memorysystem_evaluation.py`. Credential change removes `_warned_llm_fallback` + `_resolve_llm_credentials()` from `memorybank.py`, wiring LLM through `args.api_base`/`args.api_key`/`args.model` that already exist in the test CLI and are now added to the add CLI. Shell script adds the new CLI args to the add invocation.

**Tech Stack:** Python 3.12+, argparse, bash

**Spec:** `docs/superpowers/specs/2026-04-28-memorybank-prompt-credentials-design.md`

---

## File Map

| File | Action | Responsibility |
|:---|:---|:---|
| `evaluation/memorysystem_evaluation.py` | Modify | Prompt hardening, add CLI args, signature, dispatch |
| `evaluation/memorysystems/memorybank.py` | Modify | Remove credential fallback, direct args access |
| `scripts/base_test.sh` | Modify | Pass `--api_base`/`--api_key`/`--model` to add |

---

### Task 1: System prompt hardening

**Files:**
- Modify: `evaluation/memorysystem_evaluation.py:158-164`

- [ ] **Step 1: Insert autonomous-agent rule**

Current (lines 158-164):
```python
Rules:
1. Call search_memory first when the request depends on user preferences or history.
2. Use list_module_tools(module_name=...) to discover vehicle tools.
3. Call vehicle tools to satisfy the query.
4. Avoid unnecessary parameter changes if exact values are unavailable.
5. Do not repeatedly query the same memory information or invoke the same vehicle tool in consecutive steps unless new evidence requires it.
```

Replace with:
```python
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

- [ ] **Step 2: Verify file parses**

```bash
cd D:\Codes\VehicleMemBench\evaluation && uv run python -c "import memorysystem_evaluation"
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add evaluation/memorysystem_evaluation.py
git commit -m "fix: harden system prompt — enforce autonomous tool execution"
```

---

### Task 2: Add `--api_base`/`--api_key` to add CLI and wire through

**Files:**
- Modify: `evaluation/memorysystem_evaluation.py:919` (add CLI args)
- Modify: `evaluation/memorysystem_evaluation.py:603-634` (memorysystem_add signature + args)
- Modify: `evaluation/memorysystem_evaluation.py:970-985` (__main__ dispatch)

- [ ] **Step 1: Add CLI args to add_parser**

After `--store_root` (line 919), insert:
```python
    add_parser.add_argument("--api_base", type=str, default=None, help="LLM API base URL (for summary generation)")
    add_parser.add_argument("--api_key", type=str, default=None, help="LLM API key")
```

- [ ] **Step 2: Add params to memorysystem_add()**

Current signature (lines 603-618):
```python
def memorysystem_add(
    *,
    memory_system: str,
    history_dir: str,
    file_range: Optional[str] = None,
    max_workers: int = 1,
    memory_url: Optional[str] = None,
    memory_key: Optional[str] = None,
    enable_graph: bool = False,
    model: str = "gpt-4o-mini",
    device: str = "cpu",
    embedding_api_base: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    store_root: Optional[str] = None,
) -> None:
```

Replace with:
```python
def memorysystem_add(
    *,
    memory_system: str,
    history_dir: str,
    file_range: Optional[str] = None,
    max_workers: int = 1,
    memory_url: Optional[str] = None,
    memory_key: Optional[str] = None,
    enable_graph: bool = False,
    model: str = "gpt-4o-mini",
    device: str = "cpu",
    embedding_api_base: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
    store_root: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
```

And in the `args = argparse.Namespace(...)` block (lines 620-634), add:
```python
        api_base=api_base,
        api_key=api_key,
```
after `model=model,`.

- [ ] **Step 3: Wire through __main__ dispatch**

Current (lines 970-985):
```python
    if cli_args.command == "add":
        memorysystem_add(
            memory_system=cli_args.memory_system,
            history_dir=cli_args.history_dir,
            file_range=cli_args.file_range,
            max_workers=cli_args.max_workers,
            memory_url=cli_args.memory_url,
            memory_key=cli_args.memory_key,
            enable_graph=cli_args.enable_graph,
            model=cli_args.model,
            device=cli_args.device,
            embedding_api_base=cli_args.embedding_api_base,
            embedding_api_key=cli_args.embedding_api_key,
            embedding_model=cli_args.embedding_model,
            store_root=cli_args.store_root,
        )
```

Add after `store_root=cli_args.store_root,`:
```python
            api_base=cli_args.api_base,
            api_key=cli_args.api_key,
```

- [ ] **Step 4: Verify**

```bash
cd D:\Codes\VehicleMemBench\evaluation && uv run python -c "import memorysystem_evaluation"
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add evaluation/memorysystem_evaluation.py
git commit -m "feat: add --api_base/--api_key to memorysystem add CLI for LLM"
```

---

### Task 3: MemoryBank credential cleanup

**Files:**
- Modify: `evaluation/memorysystems/memorybank.py:1176` (remove `_warned_llm_fallback`)
- Modify: `evaluation/memorysystems/memorybank.py:1180-1212` (remove `_resolve_llm_credentials`)
- Modify: `evaluation/memorysystems/memorybank.py:1231-1254` (rewrite LLM creds in `_build_client`)

- [ ] **Step 1: Remove `_warned_llm_fallback` flag**

Current (line 1176-1177):
```python
_warned_llm_fallback = False
_warned_no_ref_date = False
```

Replace with:
```python
_warned_no_ref_date = False
```

- [ ] **Step 2: Remove `_resolve_llm_credentials` function**

Current (lines 1180-1212): entire function body from `def _resolve_llm_credentials(` through `return explicit_base or api_base, explicit_key or api_key`.

Delete all 33 lines (including the blank line before the function).

- [ ] **Step 3: Rewrite LLM credential access in `_build_client`**

Current (lines 1231-1254):
```python
    llm_api_base, llm_api_key = _resolve_llm_credentials(args, api_base, api_key)
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    client = MemoryBankClient(
        embedding_api_base=api_base,
        embedding_api_key=api_key,
        embedding_model=getattr(args, "embedding_model", None)
        or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
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
            "(set LLM_API_BASE/LLM_API_KEY or provide embedding API fallback); "
            "daily/overall summaries and personality analysis will NOT be generated"
        )
    return client
```

Replace with:
```python
    llm_api_base = getattr(args, "api_base", None)
    llm_api_key = getattr(args, "api_key", None)
    llm_model = getattr(args, "model", None) or os.getenv("LLM_MODEL", "gpt-4o-mini")

    client = MemoryBankClient(
        embedding_api_base=api_base,
        embedding_api_key=api_key,
        embedding_model=getattr(args, "embedding_model", None)
        or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
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

Note: `Tuple` import at line 53 may become unused after removing `_resolve_llm_credentials`. Check — it's still used by `_parse_speaker`, `_get_or_create_index`, and `format_search_results`. Safe to keep.

- [ ] **Step 4: Verify file parses**

```bash
uv run python -c "import evaluation.memorysystems.memorybank"
```

Expected: no errors.

- [ ] **Step 5: Run existing unit tests**

```bash
uv run pytest tests/test_memorybank_search.py -v
```

Expected: 3/3 pass.

- [ ] **Step 6: Commit**

```bash
git add evaluation/memorysystems/memorybank.py
git commit -m "refactor: unify LLM credentials — use backbone API, remove fallback"
```

---

### Task 4: Shell script update + integration verification

**Files:**
- Modify: `scripts/base_test.sh:153-162`

- [ ] **Step 1: Add LLM args to add invocation**

Current (lines 153-162):
```bash
  echo "=== Running memorybank add stage (store_root=${MB_STORE_ROOT}) ==="
  uv run memorysystem_evaluation.py add \
    --memory_system memorybank \
    --history_dir "$HISTORY_DIR" \
    --file_range "$FILE_RANGE" \
    --max_workers 3 \
    --store_root "$MB_STORE_ROOT" \
    --embedding_api_base "$EMBEDDING_API_BASE" \
    --embedding_api_key "$EMBEDDING_API_KEY" \
    --embedding_model "$EMBEDDING_MODEL"
```

Replace with:
```bash
  echo "=== Running memorybank add stage (store_root=${MB_STORE_ROOT}) ==="
  uv run memorysystem_evaluation.py add \
    --memory_system memorybank \
    --history_dir "$HISTORY_DIR" \
    --file_range "$FILE_RANGE" \
    --max_workers 3 \
    --api_base "$API_BASE" --api_key "$API_KEY" --model "$MODEL" \
    --store_root "$MB_STORE_ROOT" \
    --embedding_api_base "$EMBEDDING_API_BASE" \
    --embedding_api_key "$EMBEDDING_API_KEY" \
    --embedding_model "$EMBEDDING_MODEL"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/base_test.sh
git commit -m "chore: pass --api_base/--api_key/--model to memorybank add invocation"
```

- [ ] **Step 3: Integration verification — run add with summaries enabled**

```bash
$env:EMBEDDING_API_BASE = "https://openrouter.ai/api/v1"
$env:EMBEDDING_API_KEY = $env:OPENROUTER_API_KEY
$env:API_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"
$env:API_KEY = $env:ZHIPU_API_KEY
$env:MEMORYBANK_ENABLE_SUMMARY = "true"
cd D:\Codes\VehicleMemBench\evaluation
uv run python memorysystem_evaluation.py add --memory_system memorybank --history_dir ../benchmark/history --file_range 1-1 --max_workers 1 --api_base $env:API_BASE --api_key $env:API_KEY --model "glm-4.5-air" --embedding_api_base $env:EMBEDDING_API_BASE --embedding_api_key $env:EMBEDDING_API_KEY --embedding_model "baai/bge-m3"
```

Expected: summary generation succeeds (no "LLM_API_BASE not set" warning, daily_summary vectors created).

- [ ] **Step 4: Run test on file_range 1-1**

```bash
uv run python memorysystem_evaluation.py test --memory_system memorybank --benchmark_dir ../benchmark/qa_data --file_range 1-1 --max_workers 1 --reflect_num 10 --api_base $env:API_BASE --api_key $env:API_KEY --model "glm-4.5-air" --prefix "mb_prompt_fix" --embedding_api_base $env:EMBEDDING_API_BASE --embedding_api_key $env:EMBEDDING_API_KEY --embedding_model "baai/bge-m3"
```

Expected: improved metrics over baseline 5% exact match, no crashes.

- [ ] **Step 5: Commit verification**

```bash
git add .
git commit -m "chore: integration verification pass for prompt+credential fixes"
```
