import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    tqdm.write = print


HISTORY_LINE_PATTERN = re.compile(
    r"^\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})\]\s*(.*)$"
)


@dataclass
class HistoryBucket:
    dt: Optional[datetime]
    lines: List[str]


def parse_file_range(file_range: Optional[str]) -> List[int]:
    if not file_range:
        return []

    selected = set()
    for part in file_range.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    return sorted(selected)


def collect_history_files(
    history_dir: str, file_range: Optional[str]
) -> List[Tuple[int, str]]:
    history_dir = os.path.abspath(history_dir)
    history_files: List[Tuple[int, str]] = []
    for name in os.listdir(history_dir):
        match = re.match(r"history_(\d+)\.txt$", name)
        if match:
            history_files.append((int(match.group(1)), os.path.join(history_dir, name)))

    history_files.sort(key=lambda item: item[0])
    selected_ids = set(parse_file_range(file_range))
    if selected_ids:
        history_files = [item for item in history_files if item[0] in selected_ids]
    return history_files


def load_hourly_history(history_path: str) -> List[HistoryBucket]:
    known_buckets = {}
    unknown_lines: List[str] = []

    with open(history_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            match = HISTORY_LINE_PATTERN.match(line)
            if not match:
                unknown_lines.append(line)
                continue

            year, month, day, hour, _minute, content = match.groups()
            content = content.strip()
            if not content:
                continue

            dt = datetime.strptime(f"{year}-{month}-{day} {hour}", "%Y-%m-%d %H")
            bucket = known_buckets.setdefault(
                dt,
                HistoryBucket(dt=dt, lines=[]),
            )
            bucket.lines.append(content)

    buckets = [known_buckets[dt] for dt in sorted(known_buckets.keys())]
    if unknown_lines:
        buckets.append(HistoryBucket(dt=None, lines=unknown_lines))
    return buckets


def print_add_summary(
    tag: str,
    total_files_added: int,
    total_messages_added: int,
    failed_files: List[Tuple[int, str]],
) -> None:
    print(
        f"[{tag} ADD] completed files={total_files_added} total_messages={total_messages_added}"
    )
    if failed_files:
        print(f"[{tag} ADD] failed files ({len(failed_files)}):")
        for file_idx, error in failed_files:
            print(f"  history_{file_idx}.txt: {error}")


def _first_non_empty(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None


def resolve_memory_key(args: Any, *env_names: str) -> Optional[str]:
    return _first_non_empty([getattr(args, "memory_key", None), *(os.getenv(name) for name in env_names)])


def resolve_memory_url(args: Any, *env_names: str, default: Optional[str] = None) -> Optional[str]:
    return _first_non_empty(
        [getattr(args, "memory_url", None), *(os.getenv(name) for name in env_names), default]
    )


def require_value(value: Optional[str], message: str) -> str:
    if not value:
        raise ValueError(message)
    return value


def run_add_jobs(
    *,
    history_files: List[Tuple[int, str]],
    tag: str,
    max_workers: int,
    processor: Callable[[int, str], Tuple[int, int, Optional[str]]],
    force_sequential: bool = False,
) -> None:
    total_files_added = 0
    total_messages_added = 0
    failed_files: List[Tuple[int, str]] = []

    actual_workers = 1 if force_sequential else max(1, min(max_workers, len(history_files)))
    if not history_files:
        print_add_summary(tag, 0, 0, [])
        return

    if actual_workers <= 1:
        for idx, history_path in tqdm(history_files, desc=f"[{tag} ADD]"):
            file_idx, msg_count, error = processor(idx, history_path)
            if error:
                failed_files.append((file_idx, error))
            else:
                total_files_added += 1
                total_messages_added += msg_count
        print_add_summary(tag, total_files_added, total_messages_added, failed_files)
        return

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = {
            executor.submit(processor, idx, path): idx for idx, path in history_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{tag} ADD]"):
            submitted_idx = futures[future]
            try:
                file_idx, msg_count, error = future.result()
                if error:
                    failed_files.append((file_idx, error))
                else:
                    total_files_added += 1
                    total_messages_added += msg_count
            except Exception as exc:
                failed_files.append((submitted_idx, str(exc)))

    print_add_summary(tag, total_files_added, total_messages_added, failed_files)
