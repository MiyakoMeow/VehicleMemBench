import os
import time
import uuid
from datetime import datetime
from typing import Any, Optional, Tuple

from .common import collect_history_files, load_hourly_history, require_value, resolve_memory_key, resolve_memory_url, run_add_jobs


TAG = "MEMOBASE"
USER_ID_PREFIX = "memobase"


class MemobaseClient:
    def __init__(self, project_url: Optional[str] = None, api_key: Optional[str] = None):
        from memobase import MemoBaseClient

        self.project_url = project_url or os.getenv(
            "MEMOBASE_PROJECT_URL",
            "https://api.memobase.dev",
        )
        self.api_key = api_key or os.getenv("MEMOBASE_API_KEY")
        self.client = MemoBaseClient(project_url=self.project_url, api_key=self.api_key)

    def string_to_uuid(self, value: str, salt: str = "memobase_client") -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, value + salt))

    def _get_or_create_user(self, user_id: str):
        real_uid = self.string_to_uuid(user_id)
        try:
            return self.client.get_user(real_uid)
        except Exception:
            self.client.add_user(id=real_uid)
            return self.client.get_user(real_uid)

    def insert_messages(self, user, messages, batch_size: int = 1):
        from memobase import ChatBlob

        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            blob_messages = []
            for msg in batch_messages:
                blob_msg = {"role": msg["role"], "content": msg["content"]}
                if "chat_time" in msg:
                    blob_msg["created_at"] = msg["chat_time"]
                elif "created_at" in msg:
                    blob_msg["created_at"] = msg["created_at"]
                blob_messages.append(blob_msg)

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    user.insert(ChatBlob(messages=blob_messages))
                    break
                except Exception as exc:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise exc

    def flush_user(self, user):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                user.flush(sync=True)
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise exc

    def search(self, query, user_id, top_k):
        real_uid = self.string_to_uuid(user_id)
        try:
            user = self.client.get_user(real_uid)
        except Exception:
            return ""

        max_retries = 5
        for attempt in range(max_retries):
            try:
                return user.context(
                    max_token_size=top_k * 200,
                    chats=[{"role": "user", "content": query}],
                )
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise exc


def _resolve_project_url(args) -> str:
    return resolve_memory_url(args, "MEMOBASE_PROJECT_URL", default="https://api.memobase.dev") or "https://api.memobase.dev"


def _resolve_api_key(args) -> Optional[str]:
    return resolve_memory_key(args, "MEMOBASE_API_KEY")


def validate_add_args(args) -> None:
    require_value(
        _resolve_api_key(args),
        "Memobase API key is required: pass --memory_key or set MEMORY_KEY/MEMOBASE_API_KEY",
    )


def validate_test_args(args) -> None:
    validate_add_args(args)


def run_add(args) -> None:
    validate_add_args(args)
    project_url = _resolve_project_url(args)
    api_key = require_value(
        _resolve_api_key(args),
        "Memobase API key is required: pass --memory_key or set MEMORY_KEY/MEMOBASE_API_KEY",
    )
    history_dir = os.path.abspath(args.history_dir)
    if not os.path.isdir(history_dir):
        raise FileNotFoundError(f"history directory not found: {history_dir}")

    history_files = collect_history_files(history_dir, args.file_range)
    print(
        f"[{TAG} ADD] history_dir={history_dir} files={len(history_files)} max_workers={args.max_workers}"
    )

    def processor(idx: int, history_path: str) -> Tuple[int, int, Optional[str]]:
        client = MemobaseClient(project_url=project_url, api_key=api_key)
        user_id = f"{USER_ID_PREFIX}_{idx}"
        try:
            user = client._get_or_create_user(user_id)
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
                client.insert_messages(user, messages)
                message_count += len(messages)

            client.flush_user(user)
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
        "Memobase API key is required: pass --memory_key or set MEMORY_KEY/MEMOBASE_API_KEY",
    )
    return MemobaseClient(project_url=_resolve_project_url(args), api_key=api_key)


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
    lines = [line for line in search_result.splitlines() if line.strip()]
    return search_result, len(lines)
