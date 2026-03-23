import json
import os
import time
from datetime import datetime
from typing import Any, Optional, Tuple

import requests

from .common import collect_history_files, load_hourly_history, require_value, resolve_memory_key, resolve_memory_url, run_add_jobs


TAG = "MEMOS"
USER_ID_PREFIX = "memos"


class MemosApiOnlineClient:
    def __init__(self, memos_url: Optional[str] = None, memos_key: Optional[str] = None):
        self.memos_url = memos_url or os.getenv("MEMOS_ONLINE_URL")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": memos_key or os.getenv("MEMOS_KEY"),
        }

    def add(self, messages, user_id, conv_id=None, batch_size: int = 9999):
        url = f"{self.memos_url}/add/message"
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i : i + batch_size]
            payload = json.dumps(
                {
                    "messages": batch_messages,
                    "user_id": user_id,
                    "conversation_id": conv_id,
                }
            )
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.post(url, data=payload, headers=self.headers)
                    assert response.status_code == 200, response.text
                    assert json.loads(response.text)["message"] == "ok", response.text
                    break
                except Exception as exc:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                    else:
                        raise exc

    def search(self, query, user_id, top_k):
        url = f"{self.memos_url}/search/memory"
        payload = json.dumps(
            {
                "query": query,
                "user_id": user_id,
                "memory_limit_number": top_k,
                "mode": os.getenv("SEARCH_MODE", "fast"),
                "include_preference": True,
                "pref_top_k": 6,
            }
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=payload, headers=self.headers)
                assert response.status_code == 200, response.text
                assert json.loads(response.text)["message"] == "ok", response.text
                data = json.loads(response.text)["data"]
                text_mem_res = data["memory_detail_list"]
                pref_mem_res = data["preference_detail_list"]
                preference_note = data["preference_note"]
                for item in text_mem_res:
                    item.update({"memory": item.pop("memory_value")})

                explicit_prefs = [
                    item["preference"]
                    for item in pref_mem_res
                    if item.get("preference_type", "") == "explicit_preference"
                ]
                implicit_prefs = [
                    item["preference"]
                    for item in pref_mem_res
                    if item.get("preference_type", "") == "implicit_preference"
                ]

                pref_parts = []
                if explicit_prefs:
                    pref_parts.append(
                        "Explicit Preference:\n"
                        + "\n".join(f"{idx + 1}. {pref}" for idx, pref in enumerate(explicit_prefs))
                    )
                if implicit_prefs:
                    pref_parts.append(
                        "Implicit Preference:\n"
                        + "\n".join(f"{idx + 1}. {pref}" for idx, pref in enumerate(implicit_prefs))
                    )

                pref_string = "\n".join(pref_parts) + preference_note
                return {"text_mem": [{"memories": text_mem_res}], "pref_string": pref_string}
            except Exception as exc:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise exc


def _resolve_memos_url(args) -> Optional[str]:
    return resolve_memory_url(args, "MEMOS_ONLINE_URL")


def _resolve_memos_key(args) -> Optional[str]:
    return resolve_memory_key(args, "MEMOS_KEY")


def validate_add_args(args) -> None:
    require_value(
        _resolve_memos_url(args),
        "Memos URL is required: pass --memory_url or set MEMORY_URL/MEMOS_ONLINE_URL",
    )
    require_value(
        _resolve_memos_key(args),
        "Memos key is required: pass --memory_key or set MEMORY_KEY/MEMOS_KEY",
    )


def validate_test_args(args) -> None:
    validate_add_args(args)


def run_add(args) -> None:
    validate_add_args(args)
    memos_url = require_value(
        _resolve_memos_url(args),
        "Memos URL is required: pass --memory_url or set MEMORY_URL/MEMOS_ONLINE_URL",
    )
    memos_key = require_value(
        _resolve_memos_key(args),
        "Memos key is required: pass --memory_key or set MEMORY_KEY/MEMOS_KEY",
    )
    history_dir = os.path.abspath(args.history_dir)
    if not os.path.isdir(history_dir):
        raise FileNotFoundError(f"history directory not found: {history_dir}")

    history_files = collect_history_files(history_dir, args.file_range)
    print(
        f"[{TAG} ADD] history_dir={history_dir} files={len(history_files)} max_workers={args.max_workers}"
    )

    def processor(idx: int, history_path: str) -> Tuple[int, int, Optional[str]]:
        client = MemosApiOnlineClient(memos_url=memos_url, memos_key=memos_key)
        user_id = f"{USER_ID_PREFIX}_{idx}"
        try:
            message_count = 0
            for bucket in load_hourly_history(history_path):
                dt = bucket.dt or datetime.now()
                messages = [
                    {
                        "role": "user",
                        "content": "\n".join(bucket.lines),
                        "chat_time": int(dt.timestamp()),
                    }
                ]
                client.add(messages=messages, user_id=user_id, conv_id=dt.isoformat())
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
    memos_url = require_value(
        _resolve_memos_url(args),
        "Memos URL is required: pass --memory_url or set MEMORY_URL/MEMOS_ONLINE_URL",
    )
    memos_key = require_value(
        _resolve_memos_key(args),
        "Memos key is required: pass --memory_key or set MEMORY_KEY/MEMOS_KEY",
    )
    return MemosApiOnlineClient(memos_url=memos_url, memos_key=memos_key)


def init_test_state(args, file_numbers, user_id_prefix):
    del args, file_numbers, user_id_prefix
    return None


def close_test_state(shared_state: Any) -> None:
    del shared_state


def is_test_sequential() -> bool:
    return False


def _conv_id_to_date_str(conv_id: str) -> str:
    try:
        if "T" in conv_id:
            dt = datetime.fromisoformat(conv_id)
            return dt.strftime("%Y-%m-%d %H:%M")
        if "_" in conv_id:
            dt = datetime.strptime(conv_id, "%Y%m%d_%H")
            return dt.strftime("%Y-%m-%d %H:00")
        dt = datetime.strptime(conv_id, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return conv_id


def format_search_results(search_result: Any) -> Tuple[str, int]:
    if not isinstance(search_result, dict):
        return "", 0

    parts = []
    memory_lines = []
    count = 0
    idx = 1
    for group in search_result.get("text_mem", []):
        for item in group.get("memories", []):
            memory = item.get("memory", "")
            conv_id = item.get("conversation_id", "")
            date_str = _conv_id_to_date_str(conv_id)
            confidence = item.get("confidence", 0)
            relativity = item.get("relativity", 0)
            tags = item.get("tags", [])
            tags_text = ", ".join(tags) if isinstance(tags, list) else str(tags)

            memory_lines.append(
                f"{idx}. memory: {memory}\n"
                f"   date: {date_str}\n"
                f"   tags: {tags_text}\n"
                f"   confidence: {confidence}\n"
                f"   relativity: {relativity}"
            )
            idx += 1
            count += 1

    if memory_lines:
        parts.append("Fact Memories:\n" + "\n\n".join(memory_lines))

    pref_string = search_result.get("pref_string", "")
    if pref_string.strip():
        parts.append(pref_string.strip())

    return "\n\n".join(parts), count
