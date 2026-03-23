import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    tqdm.write = print

from .common import collect_history_files, load_hourly_history, print_add_summary, require_value, resolve_memory_key, resolve_memory_url


TAG = "LIGHTMEM"
USER_ID_PREFIX = "lightmem"
logger = logging.getLogger(__name__)


def _get_lightmem_client_class():
    evaluation_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lightmem_dir = os.path.join(evaluation_dir, "lightmem")
    if os.path.isdir(lightmem_dir):
        if lightmem_dir not in sys.path:
            sys.path.insert(0, lightmem_dir)
        from client import LightMemoryClient

        return LightMemoryClient

    try:
        from lightmem.memory.lightmem import LightMem
    except Exception as exc:
        raise ImportError(
            "LightMem client is unavailable. Expected either local module "
            f"`{lightmem_dir}` with `client.py` or a working `lightmem` package. Original error: {exc}"
        ) from exc

    return LightMem


def _format_timestamp(dt: Optional[datetime]) -> str:
    current_dt = dt or datetime.now()
    return f"{current_dt.strftime('%Y/%m/%d')} ({current_dt.strftime('%a')}) {current_dt.strftime('%H')}:00"


def validate_add_args(args) -> None:
    require_value(
        _resolve_add_api_key(args),
        "LightMem add requires an API key: pass --memory_key or set MEMORY_KEY",
    )


def validate_test_args(args) -> None:
    require_value(
        _resolve_test_api_key(args),
        "LightMem test requires an API key: pass --memory_key or set MEMORY_KEY",
    )


def _resolve_add_api_key(args) -> Optional[str]:
    return resolve_memory_key(args)


def _resolve_add_api_base(args) -> Optional[str]:
    return resolve_memory_url(args)


def _resolve_test_api_key(args) -> Optional[str]:
    return resolve_memory_key(args)


def _resolve_test_api_base(args) -> Optional[str]:
    return resolve_memory_url(args)


def run_add(args) -> None:
    validate_add_args(args)
    history_dir = os.path.abspath(args.history_dir)
    if not os.path.isdir(history_dir):
        raise FileNotFoundError(f"history directory not found: {history_dir}")

    history_files = collect_history_files(history_dir, args.file_range)
    print(f"[{TAG} ADD] history_dir={history_dir} files={len(history_files)}")
    if not history_files:
        print_add_summary(TAG, 0, 0, [])
        return

    LightMemoryClient = _get_lightmem_client_class()
    first_idx = history_files[0][0]
    client = LightMemoryClient(
        user_id=f"{USER_ID_PREFIX}_{first_idx}",
        llm_model=args.model,
        api_key=require_value(
            _resolve_add_api_key(args),
            "LightMem add requires an API key: pass --memory_key or set MEMORY_KEY",
        ),
        api_base_url=_resolve_add_api_base(args),
        device=args.device,
    )

    total_files_added = 0
    total_messages_added = 0
    failed_files = []

    try:
        for pos, (idx, history_path) in enumerate(tqdm(history_files, desc=f"[{TAG} ADD]")):
            try:
                if pos > 0:
                    client.switch_user(f"{USER_ID_PREFIX}_{idx}")

                message_count = 0
                for bucket in load_hourly_history(history_path):
                    messages = [
                        {
                            "role": "user",
                            "content": "\n".join(bucket.lines),
                            "time_stamp": _format_timestamp(bucket.dt),
                        }
                    ]
                    client.add(messages)
                    message_count += len(messages)

                total_files_added += 1
                total_messages_added += message_count
            except Exception as exc:
                failed_files.append((idx, str(exc)))
    finally:
        client.close()

    print_add_summary(TAG, total_files_added, total_messages_added, failed_files)


def init_test_state(args, file_numbers, user_id_prefix):
    validate_test_args(args)
    LightMemoryClient = _get_lightmem_client_class()
    first_num = file_numbers[0] if file_numbers else 1
    logger.info(
        "[LIGHTMEM TEST] Loading models (LLMLingua-2 + all-MiniLM-L6-v2) once for shared use."
    )
    client = LightMemoryClient(
        user_id=f"{user_id_prefix}_{first_num}",
        llm_model=args.lightmem_model,
        api_key=require_value(
            _resolve_test_api_key(args),
            "LightMem test requires an API key: pass --memory_key or set MEMORY_KEY",
        ),
        api_base_url=_resolve_test_api_base(args),
        device=args.lightmem_device,
    )
    logger.info("[LIGHTMEM TEST] Models loaded successfully.")
    return {"client": client}


def build_test_client(args, file_num: int, user_id_prefix: str, shared_state: Any):
    del args
    client = shared_state["client"]
    client.switch_user(f"{user_id_prefix}_{file_num}")
    return client


def close_test_state(shared_state: Any) -> None:
    client = shared_state.get("client") if isinstance(shared_state, dict) else None
    if client is not None:
        client.close()


def is_test_sequential() -> bool:
    return True


def format_search_results(search_result: Any) -> Tuple[str, int]:
    if not isinstance(search_result, str):
        return "", 0
    lines = [line for line in search_result.splitlines() if line.strip()]
    return search_result, len(lines)
