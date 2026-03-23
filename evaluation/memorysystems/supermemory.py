import os
import time
from datetime import datetime
from typing import Any, Optional, Tuple

from .common import collect_history_files, load_hourly_history, require_value, resolve_memory_key, run_add_jobs


TAG = "SUPERMEMORY"
USER_ID_PREFIX = "supermemory"


class SupermemoryClient:
    def __init__(self, api_key: str):
        from supermemory import Supermemory

        self.client = Supermemory(api_key=api_key)

    def add(self, messages, user_id):
        content = "\n".join(
            f"{msg['chat_time']} {msg['role']}: {msg['content']}" for msg in messages
        )
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.client.add(content=content, container_tag=user_id)
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise exc

    def search(self, query, user_id, top_k):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                results = self.client.search.documents(
                    q=query,
                    container_tags=[user_id],
                    chunk_threshold=0,
                    rerank=True,
                    rewrite_query=True,
                    limit=top_k,
                )
                chunks = []
                for doc in results.results:
                    for chunk in doc.chunks:
                        chunks.append(chunk.content)
                return "\n\n".join(chunks)
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise exc


def _resolve_api_key(args) -> Optional[str]:
    return resolve_memory_key(args, "SUPERMEMORY_API_KEY")


def validate_add_args(args) -> None:
    require_value(
        _resolve_api_key(args),
        "Supermemory API key is required: pass --memory_key or set MEMORY_KEY/SUPERMEMORY_API_KEY",
    )


def validate_test_args(args) -> None:
    validate_add_args(args)


def run_add(args) -> None:
    validate_add_args(args)
    api_key = require_value(
        _resolve_api_key(args),
        "Supermemory API key is required: pass --memory_key or set MEMORY_KEY/SUPERMEMORY_API_KEY",
    )
    history_dir = os.path.abspath(args.history_dir)
    if not os.path.isdir(history_dir):
        raise FileNotFoundError(f"history directory not found: {history_dir}")

    history_files = collect_history_files(history_dir, args.file_range)
    print(
        f"[{TAG} ADD] history_dir={history_dir} files={len(history_files)} max_workers={args.max_workers}"
    )

    def processor(idx: int, history_path: str) -> Tuple[int, int, Optional[str]]:
        client = SupermemoryClient(api_key=api_key)
        user_id = f"{USER_ID_PREFIX}_{idx}"
        try:
            message_count = 0
            for bucket in load_hourly_history(history_path):
                dt = bucket.dt or datetime.now()
                messages = [
                    {
                        "role": "user",
                        "content": "\n".join(bucket.lines),
                        "chat_time": dt.isoformat(),
                    }
                ]
                client.add(messages=messages, user_id=user_id)
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
        "Supermemory API key is required: pass --memory_key or set MEMORY_KEY/SUPERMEMORY_API_KEY",
    )
    return SupermemoryClient(api_key=api_key)


def init_test_state(args, file_numbers, user_id_prefix):
    del args, file_numbers, user_id_prefix
    return None


def close_test_state(shared_state: Any) -> None:
    del shared_state


def is_test_sequential() -> bool:
    return False


def format_search_results(search_result: Any) -> Tuple[str, int]:
    if not isinstance(search_result, str):
        return "", 0
    chunks = [part for part in search_result.split("\n\n") if part.strip()]
    return search_result, len(chunks)
