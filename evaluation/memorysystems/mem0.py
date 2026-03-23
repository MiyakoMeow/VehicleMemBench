import os
import time
from typing import Any, Optional, Tuple

from .common import collect_history_files, load_hourly_history, require_value, resolve_memory_key, run_add_jobs


TAG = "MEM0"
USER_ID_PREFIX = "mem0"


class Mem0Client:
    def __init__(self, api_key: str, enable_graph: bool = False):
        from mem0 import MemoryClient

        max_retries = 5
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                self.client = MemoryClient(api_key=api_key)
                break
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    raise
                time.sleep(2 ** (attempt - 1))

        if last_error and not hasattr(self, "client"):
            raise last_error

        self.enable_graph = enable_graph

    def add(self, messages, user_id, timestamp, batch_size: int = 1):
        max_retries = 5
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    kwargs = {
                        "messages": batch_messages,
                        "timestamp": timestamp,
                        "user_id": user_id,
                    }
                    if self.enable_graph:
                        kwargs["enable_graph"] = True
                    self.client.add(**kwargs)
                    break
                except Exception as exc:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise exc

    def search(self, query, user_id, top_k):
        return self.client.search(
            query=query,
            top_k=top_k,
            user_id=user_id,
            enable_graph=self.enable_graph,
            filters={"AND": [{"user_id": f"{user_id}"}]},
        )


def _resolve_api_key(args) -> Optional[str]:
    return resolve_memory_key(args, "MEM0_API_KEY")


def validate_add_args(args) -> None:
    require_value(
        _resolve_api_key(args),
        "MEM0 API key is required: pass --memory_key or set MEMORY_KEY/MEM0_API_KEY",
    )


def validate_test_args(args) -> None:
    validate_add_args(args)


def run_add(args) -> None:
    validate_add_args(args)
    api_key = require_value(
        _resolve_api_key(args),
        "MEM0 API key is required: pass --memory_key or set MEMORY_KEY/MEM0_API_KEY",
    )
    history_dir = os.path.abspath(args.history_dir)
    if not os.path.isdir(history_dir):
        raise FileNotFoundError(f"history directory not found: {history_dir}")

    history_files = collect_history_files(history_dir, args.file_range)
    print(
        f"[{TAG} ADD] history_dir={history_dir} files={len(history_files)} max_workers={args.max_workers}"
    )

    def processor(idx: int, history_path: str) -> Tuple[int, int, Optional[str]]:
        client = Mem0Client(api_key=api_key, enable_graph=args.enable_graph)
        user_id = f"{USER_ID_PREFIX}_{idx}"
        try:
            message_count = 0
            for bucket in load_hourly_history(history_path):
                messages = [{"role": "user", "content": "\n".join(bucket.lines)}]
                timestamp = int(bucket.dt.timestamp()) if bucket.dt else int(time.time())
                client.add(messages=messages, user_id=user_id, timestamp=timestamp)
                message_count += len(messages)
            return idx, message_count, None
        except Exception as exc:
            return idx, 0, str(exc)

    run_add_jobs(
        history_files=history_files,
        tag=TAG,
        max_workers=args.max_workers,
        processor=processor,
    )


def build_test_client(args, file_num: int, user_id_prefix: str, shared_state: Any):
    del file_num, user_id_prefix, shared_state
    api_key = require_value(
        _resolve_api_key(args),
        "MEM0 API key is required: pass --memory_key or set MEMORY_KEY/MEM0_API_KEY",
    )
    return Mem0Client(api_key=api_key, enable_graph=args.enable_graph)


def init_test_state(args, file_numbers, user_id_prefix):
    del args, file_numbers, user_id_prefix
    return None


def close_test_state(shared_state: Any) -> None:
    del shared_state


def is_test_sequential() -> bool:
    return False


def format_search_results(search_result: Any) -> Tuple[str, int]:
    if not isinstance(search_result, dict):
        return "", 0

    results = search_result.get("results", [])
    if not results:
        return "", 0

    lines = []
    for idx, item in enumerate(results, 1):
        memory = item.get("memory", "")
        categories = item.get("categories")
        created_at = item.get("created_at", "")
        score = item.get("score", 0)

        if isinstance(categories, list):
            categories_text = ", ".join(categories)
        else:
            categories_text = str(categories) if categories is not None else ""

        try:
            score_text = f"{float(score):.4f}"
        except Exception:
            score_text = str(score)

        lines.append(
            f"{idx}. memory: {memory}\n"
            f"   categories: {categories_text}\n"
            f"   created_at: {created_at}\n"
            f"   score: {score_text}"
        )

    return "\n\n".join(lines), len(results)
