#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPT="${ROOT_DIR}/evaluation/memorysystem_evaluation.py"
HISTORY_DIR="${ROOT_DIR}/benchmark/history"
BENCHMARK_DIR="${ROOT_DIR}/benchmark/qa_data"
OUTPUT_DIR="${ROOT_DIR}/memory_system_log"

: "${LLM_API_BASE:?Set LLM_API_BASE}"
: "${LLM_API_KEY:?Set LLM_API_KEY}"
: "${LLM_MODEL:?Set LLM_MODEL}"
: "${EMBEDDING_API_BASE:=$LLM_API_BASE}"
: "${EMBEDDING_API_KEY:=$LLM_API_KEY}"
: "${MEM0_API_KEY:?Set MEM0_API_KEY}"
: "${MEMOS_API_URL:?Set MEMOS_API_URL}"
: "${MEMOS_API_KEY:?Set MEMOS_API_KEY}"
: "${LIGHTMEM_API_KEY:?Set LIGHTMEM_API_KEY}"
: "${LIGHTMEM_API_BASE:?Set LIGHTMEM_API_BASE}"
: "${LIGHTMEM_MODEL:?Set LIGHTMEM_MODEL}"
: "${SUPERMEMORY_API_KEY:?Set SUPERMEMORY_API_KEY}"
: "${MEMOBASE_API_KEY:?Set MEMOBASE_API_KEY}"
: "${MEMOBASE_API_URL:?Set MEMOBASE_API_URL}"

# Step 1: add history into each memory system
# Note: MemoryBank's forgetting mechanism is disabled by default for reproducibility.
# To enable it, set MEMORYBANK_ENABLE_FORGETTING=1 before running the add command.

uv run "$EVAL_SCRIPT" add --memory_system mem0 --memory_key "$MEM0_API_KEY" --history_dir "$HISTORY_DIR" --file_range "1-100"

uv run "$EVAL_SCRIPT" add --memory_system memos --memory_url "$MEMOS_API_URL" --memory_key "$MEMOS_API_KEY" --history_dir "$HISTORY_DIR" --file_range "1-100"

uv run "$EVAL_SCRIPT" add --memory_system lightmem --history_dir "$HISTORY_DIR" --file_range "1-4" --memory_key "$LIGHTMEM_API_KEY" --memory_url "$LIGHTMEM_API_BASE" --model "$LIGHTMEM_MODEL" --device "cpu"

uv run "$EVAL_SCRIPT" add --memory_system supermemory --memory_key "$SUPERMEMORY_API_KEY" --history_dir "$HISTORY_DIR" --file_range "1" --max_workers 1

uv run "$EVAL_SCRIPT" add --memory_system memobase --memory_key "$MEMOBASE_API_KEY" --memory_url "$MEMOBASE_API_URL" --history_dir "$HISTORY_DIR" --file_range "6-100" --max_workers 2

MB_STORE_ROOT="${ROOT_DIR}/memory_system_log/memorybank_store_$(date +%Y%m%d_%H%M%S)_${RANDOM}"

uv run "$EVAL_SCRIPT" add --memory_system memorybank --embedding_api_base "$EMBEDDING_API_BASE" --embedding_api_key "$EMBEDDING_API_KEY" --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --history_dir "$HISTORY_DIR" --file_range "1-100" --store_root "$MB_STORE_ROOT"

# Step 2: evaluate qwen + memory-system combinations

uv run "$EVAL_SCRIPT" test --memory_system mem0 --memory_key "$MEM0_API_KEY" --benchmark_dir "$BENCHMARK_DIR" --file_range "1-100" --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --max_workers 5 --output_dir "$OUTPUT_DIR" --enable_thinking true

uv run "$EVAL_SCRIPT" test --memory_system memos --memory_url "$MEMOS_API_URL" --memory_key "$MEMOS_API_KEY" --benchmark_dir "$BENCHMARK_DIR" --file_range "1-100" --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --max_workers 5 --output_dir "$OUTPUT_DIR" --enable_thinking true --user_id_prefix memos

uv run "$EVAL_SCRIPT" test --memory_system lightmem --benchmark_dir "$BENCHMARK_DIR" --file_range "1-100" --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --max_workers 5 --output_dir "$OUTPUT_DIR" --user_id_prefix lightmem --memory_key "$LIGHTMEM_API_KEY" --memory_url "$LIGHTMEM_API_BASE" --lightmem_model "$LIGHTMEM_MODEL" --lightmem_device "cpu"

uv run "$EVAL_SCRIPT" test --memory_system supermemory --memory_key "$SUPERMEMORY_API_KEY" --benchmark_dir "$BENCHMARK_DIR" --file_range "1-100" --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --max_workers 5 --output_dir "$OUTPUT_DIR" --enable_thinking true --user_id_prefix supermemory

uv run "$EVAL_SCRIPT" test --memory_system memobase --memory_key "$MEMOBASE_API_KEY" --memory_url "$MEMOBASE_API_URL" --benchmark_dir "$BENCHMARK_DIR" --file_range "1-100" --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --max_workers 1 --output_dir "$OUTPUT_DIR" --enable_thinking true --user_id_prefix memobase

uv run "$EVAL_SCRIPT" test --memory_system memorybank --benchmark_dir "$BENCHMARK_DIR" --file_range "1-100" --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --embedding_api_base "$EMBEDDING_API_BASE" --embedding_api_key "$EMBEDDING_API_KEY" --user_id_prefix memorybank --max_workers 5 --output_dir "$OUTPUT_DIR" --enable_thinking true --store_root "$MB_STORE_ROOT"
