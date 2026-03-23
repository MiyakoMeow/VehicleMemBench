#!/usr/bin/env bash
set -euo pipefail

: "${LLM_API_BASE:?Set LLM_API_BASE}"
: "${LLM_API_KEY:?Set LLM_API_KEY}"
: "${LLM_MODEL:?Set LLM_MODEL}"

# None Mode: no memory context
python model_evaluation.py --memory_type "none" --enable_thinking false --benchmark_dir "../benchmark/qa_data" --file_range "1" --sample_size 1 --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --prefix "qwen_no_think_none_eval" --max_workers 10 --reflect_num 10

# Gold Mode: use each sample's events as gold context
python model_evaluation.py --memory_type "gold" --enable_thinking false --benchmark_dir "../benchmark/qa_data" --file_range "1" --sample_size 1 --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --prefix "qwen_no_think_gold_eval" --max_workers 10 --reflect_num 10

# Summary Mode
python model_evaluation.py --enable_thinking true --memory_type "summary" --benchmark_dir "../benchmark/qa_data" --file_range "1" --sample_size 1 --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --prefix "qwen_thinking_summary_memory" --reflect_num 10 --max_workers 8

# Key-Value Mode
python model_evaluation.py --enable_thinking true --memory_type "key_value" --benchmark_dir "../benchmark/qa_data" --file_range "1" --sample_size 1 --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --prefix "qwen_thinking_key_value_memory" --reflect_num 20 --max_workers 8

# API Quota Check
python check_api_quota.py --enable_thinking true --api_base "$LLM_API_BASE" --api_key "$LLM_API_KEY" --model "$LLM_MODEL" --check_function_call --require_tool_call
