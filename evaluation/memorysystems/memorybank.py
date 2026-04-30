# ruff: noqa: RUF002, RUF003
"""
MemoryBank: 基于 FAISS 向量检索的本地记忆系统，专为 VehicleMemBench 测评场景设计。

原项目: https://github.com/zhongwanjun/MemoryBank-SiliconFriend
论文: https://arxiv.org/abs/2305.10250

主要特性:
- FAISS IndexFlatIP + L2 归一化实现余弦相似度检索
- OpenAI 兼容的 Embedding API 支持
- 动态说话人解析与多用户感知检索
- 艾宾浩斯遗忘曲线（修正运算符优先级bug）
- 自适应 CHUNK_SIZE（基于 P90 文本长度校准）
- 四阶段检索管道：FAISS粗排 → 邻居合并 → 去重 → 截断
- LLM 驱动的摘要/性格分析（车载场景适配）
- 记忆强度与 spacing effect 保护
"""

import json
import logging
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable

import faiss
import numpy as np
import openai as _openai

from .common import (
    collect_history_files,
    load_hourly_history,
    require_value,
    resolve_memory_key,
    resolve_memory_url,
    run_add_jobs,
)

logger = logging.getLogger(__name__)

TAG = "MEMORYBANK"
USER_ID_PREFIX = "memorybank"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 1500
CHUNK_SIZE_MIN = 200  # 自适应下界
CHUNK_SIZE_MAX = 8192  # 自适应上限，避免将整个日期塞入单条记忆
MEMORY_SKIP_TYPES = frozenset({"daily_summary"})
_MERGED_TEXT_DELIMITER = "\x00"
_GENERATION_EMPTY = "GENERATION_EMPTY"  # LLM 生成空结果时的哨兵值

# ── 数值 / 行为常量 ────────────────────────────────────────────────────────────
# 嵌入 API 配置
DEFAULT_EMBEDDING_DIM = 1536  # text-embedding-3-small 默认维度
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_BACKOFF_BASE = 2  # 指数退避基础秒数
EMBEDDING_BATCH_SIZE = 100  # 单次 API 调用最大文本数（OpenAI 上限 2048）

# LLM / 摘要生成
LLM_MAX_RETRIES = 3
LLM_MAX_TOKENS = 400
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 1.0
LLM_FREQUENCY_PENALTY = 0.4
LLM_PRESENCE_PENALTY = 0.2
LLM_CTX_TRIM_START = 1800  # 首轮截断字符数
LLM_CTX_TRIM_STEP = 200  # 每次重试缩减字符数
LLM_CTX_TRIM_MIN = 500  # 截断下限字符数

# 日期 / 时间戳
DATE_PREFIX_LEN = 10  # "YYYY-MM-DD" 前缀长度
DEFAULT_TIME_SUFFIX = "T00:00:00"

# 记忆生命周期
INITIAL_MEMORY_STRENGTH = 1
MEMORY_STRENGTH_INCREMENT = 1
FORGETTING_TIME_SCALE = 1  # 对齐原论文遗忘公式 R = e^{-t/S}

# 检索
DEFAULT_TOP_K = 5
COARSE_SEARCH_FACTOR = 4  # top_k 倍率，FAISS 粗排窗口
REFERENCE_DATE_OFFSET = 1  # 最大历史时间戳后追加的天数

def _safe_memory_strength(value: Any) -> float:
    """将 memory_strength 转换为 float，无效值回退到 INITIAL_MEMORY_STRENGTH。

    原实现中此函数对 NaN/Inf/非正值抛 ValueError，会导致 _merge_neighbors、
    search 等核心路径在 metadata 损坏时崩溃。改为日志警告 + 回退默认值，
    保证单条目损坏不中断整个评测流程。
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "MemoryBank: memory_strength=%r is not a number, "
            "falling back to %d", value, INITIAL_MEMORY_STRENGTH,
        )
        return float(INITIAL_MEMORY_STRENGTH)
    if math.isnan(f) or math.isinf(f) or f <= 0:
        logger.warning(
            "MemoryBank: memory_strength=%r is invalid (NaN/Inf/非正), "
            "falling back to %d", value, INITIAL_MEMORY_STRENGTH,
        )
        return float(INITIAL_MEMORY_STRENGTH)
    return f

def _resolve_chunk_size() -> int:
    """从环境变量 MEMORYBANK_CHUNK_SIZE 解析分块大小。

    未设置时使用 DEFAULT_CHUNK_SIZE=1500。
    """
    raw = os.getenv("MEMORYBANK_CHUNK_SIZE")
    if raw is not None:
        try:
            parsed = int(raw)
        except ValueError:
            logger.warning(
                "MemoryBank: MEMORYBANK_CHUNK_SIZE=%r is not a valid int, "
                "falling back to %d",
                raw,
                DEFAULT_CHUNK_SIZE,
            )
            return DEFAULT_CHUNK_SIZE
        if parsed <= 0:
            logger.warning(
                "MemoryBank: MEMORYBANK_CHUNK_SIZE=%d is not positive, "
                "falling back to %d",
                parsed,
                DEFAULT_CHUNK_SIZE,
            )
            return DEFAULT_CHUNK_SIZE
        return parsed
    return DEFAULT_CHUNK_SIZE

def _resolve_embedding_dim() -> int | None:
    """从环境变量 EMBEDDING_DIM 读取嵌入维度，不支持时返回 None。"""
    raw = os.getenv("EMBEDDING_DIM")
    if raw is not None:
        try:
            parsed = int(raw)
        except ValueError:
            logger.warning(
                "MemoryBank: EMBEDDING_DIM=%r is not a valid int, "
                "falling back to auto-detect",
                raw,
            )
            return None
        if parsed <= 0:
            logger.warning(
                "MemoryBank: EMBEDDING_DIM=%d is not positive, "
                "falling back to auto-detect",
                parsed,
            )
            return None
        return parsed
    return None

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STORE_ROOT = os.environ.get(
    "MEMORYBANK_STORE_ROOT",
    os.path.join(_ROOT_DIR, "log", "memorybank"),
)

def _resolve_embedding_api_key(args) -> str | None:
    """获取 Embedding API 密钥，优先使用命令行参数，回退到环境变量。"""
    return getattr(args, "embedding_api_key", None) or resolve_memory_key(
        args, "MEMORY_KEY", "EMBEDDING_API_KEY"
    )

def _resolve_embedding_api_base(args) -> str | None:
    """获取 Embedding API 基础 URL，优先使用命令行参数，回退到环境变量。"""
    return getattr(args, "embedding_api_base", None) or resolve_memory_url(
        args, "MEMORY_URL", "EMBEDDING_API_BASE"
    )

def _resolve_reference_date() -> str | None:
    """从环境变量 MEMORYBANK_REFERENCE_DATE 读取参考日期。"""
    return os.getenv("MEMORYBANK_REFERENCE_DATE")

def _word_in_text(word: str, text: str) -> bool:
    """检测 word 是否作为独立词出现在 text 中（\b 词边界）。"""
    if not word or not word.strip():
        return False
    return bool(re.search(r"\b" + re.escape(word.strip()) + r"\b", text))

_TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on", "y"})
_FALSY_TOKENS = frozenset({"0", "false", "no", "off", "n"})

def _parse_bool_token(raw: str) -> bool | None:
    """将非空字符串解析为布尔值，无法识别时返回 None。"""
    normalized = raw.strip().lower()
    if normalized in _TRUTHY_TOKENS:
        return True
    if normalized in _FALSY_TOKENS:
        return False
    return None

def _resolve_bool_env(name: str, default: bool) -> bool:
    """从环境变量解析布尔值，支持常见 truthy/falsy 词元。"""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = _parse_bool_token(value)
    if parsed is not None:
        return parsed
    logger.warning(
        "MemoryBank: env %s=%r not recognized as boolean "
        "(truthy: %s, falsy: %s); falling back to default %s",
        name,
        value,
        sorted(_TRUTHY_TOKENS),
        sorted(_FALSY_TOKENS),
        default,
    )
    return default

def _resolve_enable_summary() -> bool:
    """从环境变量 MEMORYBANK_ENABLE_SUMMARY 读取是否启用摘要生成。"""
    return _resolve_bool_env("MEMORYBANK_ENABLE_SUMMARY", True)

def _resolve_enable_forgetting() -> bool:
    """从环境变量 MEMORYBANK_ENABLE_FORGETTING 读取是否启用遗忘机制。
    
    需要启用时设置 MEMORYBANK_ENABLE_FORGETTING=1。
    """
    return _resolve_bool_env("MEMORYBANK_ENABLE_FORGETTING", False)

def _resolve_seed() -> int | None:
    """从环境变量 MEMORYBANK_SEED 读取随机种子。"""
    raw = os.getenv("MEMORYBANK_SEED")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "MemoryBank: MEMORYBANK_SEED=%r is not a valid int, "
                "falling back to None",
                raw,
            )
            return None
    return None

def _resolve_store_root(args) -> str:
    """获取存储根目录，优先使用命令行参数，回退到环境变量或默认值。"""
    return getattr(args, "store_root", None) or STORE_ROOT

def _user_store_dir(user_id: str, store_root: str = STORE_ROOT) -> str:
    """返回指定用户的存储目录路径。"""
    return os.path.join(store_root, f"user_{user_id}")

def _strip_source_prefix(text: str, date_part: str) -> str:
    """去除对话内容或摘要的英文前缀标记。

    Note: 当前仅支持英文前缀；VehicleMemBench 测试集为纯英文内容，
    无需中文前缀处理。若迁移到中文场景，需添加 `时间{date}的对话内容：` 
    和 `时间{date}的对话总结为：` 对应的剥离逻辑。
    """
    for pfx in (
        f"Conversation content on {date_part}:",
        f"The summary of the conversation on {date_part} is:",
    ):
        if text.startswith(pfx):
            return text[len(pfx) :]
    return text

def _merge_overlapping_results(results: list[dict]) -> list[dict]:
    """合并结果中共享 index 或互为子集/超集的条目，消除内容重复。
    
    本实现每结果独立构建合并条目并保留各自 score，然后通过子集过滤
    消除跨结果重叠，修复原版分数 bug。
    """
    non_merging = [
        r for r in results
        if not isinstance(r.get("_merged_indices"), list)
        or len(r["_merged_indices"]) <= 1
    ]
    merging = [
        r for r in results
        if isinstance(r.get("_merged_indices"), list)
        and len(r["_merged_indices"]) > 1
    ]
    if len(merging) <= 1:
        return results

    # 反向映射：每个 index → 包含它的结果下标
    idx_owners: dict[int, list[int]] = defaultdict(list)
    for ri, r in enumerate(merging):
        for idx in r["_merged_indices"]:
            idx_owners[idx].append(ri)

    # dict 版并查集（key 为结果下标）
    parent = {i: i for i in range(len(merging))}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        px, py = _find(x), _find(y)
        if px != py:
            parent[py] = px

    for owners in idx_owners.values():
        for i in range(1, len(owners)):
            _union(owners[0], owners[i])

    # 按 root 分组
    groups = defaultdict(list)
    for i in range(len(merging)):
        groups[_find(i)].append(i)

    merged: list[dict] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(merging[members[0]])
        else:
            all_indices: set[int] = set()
            best_idx = max(members, key=lambda mi: merging[mi].get("score", 0.0))
            for mi in members:
                all_indices.update(merging[mi]["_merged_indices"])
            r = dict(merging[best_idx])
            r["_merged_indices"] = sorted(all_indices)
            
            # 本实现每个 FAISS 命中独立保留其 _meta_idx；当多个命中因索引重叠
            # 被合并时，所有原始命中的 _meta_idx 都需获得 memory_strength 提升——
            # 它们均被 FAISS 独立召回，非被动邻居扩展。
            r["_all_meta_indices"] = sorted({
                merging[mi].get("_meta_idx") for mi in members
                if merging[mi].get("_meta_idx") is not None
            })
            r["memory_strength"] = max(
                _safe_memory_strength(
                    merging[mi].get("memory_strength", INITIAL_MEMORY_STRENGTH)
                )
                for mi in members
            )
            r["speakers"] = sorted({
                s for mi in members
                for s in (merging[mi].get("speakers") or [])
            })
            index_to_part: dict[int, str] = {}
            for mi in members:
                parts = merging[mi].get("text", "").split(_MERGED_TEXT_DELIMITER)
                indices = merging[mi].get("_merged_indices", [])
                if len(indices) != len(parts):
                    logger.warning(
                        "MemoryBank: _merge_overlapping_results text/indices "
                        "length mismatch (%d vs %d) for result %d, skipping",
                        len(indices),
                        len(parts),
                        mi,
                    )
                    continue
                for idx, part in zip(indices, parts, strict=True):
                    index_to_part.setdefault(idx, part)
            deduped_parts = [
                index_to_part[idx]
                for idx in r["_merged_indices"]
                if idx in index_to_part
            ]
            if deduped_parts:
                r["text"] = _MERGED_TEXT_DELIMITER.join(deduped_parts)
            else:
                # 回退：所有成员均存在 text/indices 长度不匹配（元数据损坏），
                # 优先保留最佳成员的完整文本，其次使用任意可恢复的 part。
                if not r.get("text", ""):
                    r["text"] = next(iter(index_to_part.values()), "")
                if not r.get("text", ""):
                    logger.warning(
                        "MemoryBank: _merge_overlapping_results produced empty text "
                        "for merged result (best_idx=%s, _meta_idx=%s, "
                        "%d members, %d parts recovered). "
                        "Metadata corruption — skipping this result.",
                        best_idx,
                        merging[best_idx].get("_meta_idx"),
                        len(members),
                        len(index_to_part),
                    )
                    continue
            merged.append(r)

    if non_merging:
        merged.extend(non_merging)
    merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    return merged

class MemoryBankClient:
    """MemoryBank：基于 FAISS 向量检索的本地长期记忆系统。

    从 MemoryBank-SiliconFriend
    (https://github.com/zhongwanjun/MemoryBank-SiliconFriend) 移植并
    适配至 VehicleMemBench 多用户测评场景。使用 OpenAI 兼容嵌入 API，
    支持 LLM 驱动的每日摘要/性格分析、艾宾浩斯遗忘曲线衰减、以及多用户
    说话人感知检索过滤。
    """

    def __init__(
        self,
        *,
        embedding_api_base: str,
        embedding_api_key: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_forgetting: bool = False,
        enable_summary: bool = True,
        seed: int | None = None,
        reference_date: str | None = None,
        llm_api_base: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        store_root: str = STORE_ROOT,
    ):
        self._store_root = store_root

        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.enable_forgetting = enable_forgetting
        self.enable_summary = enable_summary
        self.reference_date = reference_date

        self._embedding_dim: int | None = None
        self._indices: dict[str, faiss.IndexIDMap] = {}
        self._metadata: dict[str, list[dict]] = {}
        self._next_id: dict[str, int] = {}
        self._rng = random.Random(seed)
        self._warned_no_ref_date = False
        self._chunk_fallback_warned: set[str] = set()

        self._extra_metadata: dict[str, dict] = {}
        self._id_to_meta_cache: dict[str, dict[int, int]] = {}
        
        # 序列化：Python set 不可 JSON 序列化，会导致崩溃。用 sorted list
        # 替代 set 以保证 JSON 兼容性。
        self._speakers_cache: dict[str, list[str]] = {}

        self._embedding_client = _openai.OpenAI(
            base_url=embedding_api_base,
            api_key=embedding_api_key,
        )

        self._llm_client = None
        if enable_summary and llm_api_base and llm_api_key:
            self._llm_client = _openai.OpenAI(
                base_url=llm_api_base,
                api_key=llm_api_key,
            )
            self._llm_model = llm_model or "gpt-4o-mini"

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """调用 Embedding API 将文本列表转为向量，支持分批以适配各提供商上限。        本实现通过 _get_embeddings_single 做带重试的单批 API 调用，
        外层 _get_embeddings 按 EMBEDDING_BATCH_SIZE 分批聚合结果。

        Returns:
            嵌入向量列表；任何不可恢复的错误返回空列表（调用方需做空值守卫）。
        """
        try:
            results: list[list[float]] = []
            for batch_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                batch_result = self._get_embeddings_single(batch)
                results.extend(batch_result)

            if len(results) != len(texts):
                logger.error(
                "MemoryBank _get_embeddings: count mismatch — requested %d "
                "but got %d from API (partial data discarded). "
                "Returning empty. Check your embedding model.",
                len(texts), len(results),
                )
                return []

            if self._embedding_dim is None and results:
                self._embedding_dim = len(results[0])

            return results
        except Exception:
            logger.error(
                "MemoryBank _get_embeddings: unhandled API error "
                "(%d texts requested) — returning empty.",
                len(texts),
                exc_info=True,
            )
            return []

    def _get_embeddings_single(self, texts: list[str]) -> list[list[float]]:
        """单批 Embedding API 调用，带可恢复错误的指数退避重试。

        本实现改用 OpenAI Embedding API，需处理网络错误、限速、批次上限和维度一致性。
        
        可恢复错误：连接/超时/限速 → 重试；
        不可恢复：4xx（凭证错误/模型不存在等）→ 直接抛出。
        """
        max_retries = EMBEDDING_MAX_RETRIES
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self._embedding_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model,
                )
                break
            except Exception as exc:
                retryable = isinstance(
                    exc,
                    (
                        _openai.APIConnectionError,
                        _openai.APITimeoutError,
                        _openai.RateLimitError,
                    ),
                )
                if not retryable and isinstance(exc, _openai.APIStatusError):
                    retryable = exc.status_code >= 500

                if retryable and attempt < max_retries - 1:
                    jitter = self._rng.random()
                    time.sleep(EMBEDDING_BACKOFF_BASE**attempt + jitter)
                    continue

                if not retryable:
                    raise

                logger.warning(
                    "MemoryBank _get_embeddings_single failed after %d retries: %s",
                    max_retries,
                    exc,
                )
                raise

        results: list[list[float]] = []
        dims_seen: set[int] = set()
        for item in resp.data:
            dims_seen.add(len(item.embedding))
            results.append(item.embedding)
        if len(dims_seen) > 1:
            logger.error(
                "MemoryBank _get_embeddings_single: inconsistent dimensions %s. "
                "Returning empty to avoid downstream crash.",
                dims_seen,
            )
            return []
        return results

    def _get_or_create_index(self, user_id: str) -> tuple[faiss.IndexIDMap, list[dict]]:
        """获取或创建用户的 FAISS 索引和元数据列表，支持从磁盘加载已有索引。"""
        if user_id in self._indices:
            return self._indices[user_id], self._metadata[user_id]

        store_dir = _user_store_dir(user_id, self._store_root)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.json")
        extra_path = os.path.join(store_dir, "extra_metadata.json")

        if os.path.isfile(index_path) and os.path.isfile(meta_path):
            index = faiss.read_index(index_path)
            
            # 本实现使用原生 FAISS + IndexFlatIP（内积）配合 L2 归一化 =
            # 余弦相似度。若检测到旧格式 L2 索引，自动重建空索引并警告——
            # run_add 阶段始终先清除存储目录后重建，此路径仅在 test 阶段
            # 加载了手动放置的旧格式索引时触发。
            try:
                _inner = index.index if isinstance(index, faiss.IndexIDMap) else index
            except AttributeError:
                _inner = index
            needs_rebuild = False
            metadata: list[dict] = []  # 防御性默认值；L2/损坏路径不加载 JSON
            if isinstance(_inner, faiss.IndexFlatL2):
                logger.warning(
                    "MemoryBank: L2 index detected for user=%s (store_dir=%s). "
                    "IndexFlatL2 is not supported — rebuilding empty index. "
                    "Re-run the 'add' stage to rebuild from scratch.",
                    user_id, store_dir,
                )
                needs_rebuild = True
                # Rebuild deferred to the unified block below
            if not needs_rebuild:
                # 加载并验证 metadata；JSON 解析错误、类型错误、字段缺失
                # 均视为损坏，统一触发重建而非崩溃。
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    if not isinstance(metadata, list):
                        raise TypeError(
                            f"expected list, got {type(metadata).__name__}"
                        )
                    for i, meta in enumerate(metadata):
                        if not isinstance(meta, dict):
                            raise TypeError(
                                f"entry {i}: expected dict, got "
                                f"{type(meta).__name__}"
                            )
                        if "faiss_id" not in meta:
                            raise ValueError(
                                f"entry {i}: missing faiss_id"
                            )
                except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
                    logger.warning(
                        "MemoryBank: metadata corrupted for user=%s "
                        "(store_dir=%s): %s. Rebuilding empty index.",
                        user_id, store_dir, exc,
                    )
                    metadata = []
                    needs_rebuild = True
                else:
                    n_loaded = index.ntotal
                    if n_loaded != len(metadata):
                        logger.warning(
                            "MemoryBank: index-metadata count mismatch for %s "
                            "(ntotal=%d, metadata=%d, store_dir=%s). "
                            "Rebuilding empty index.",
                            user_id, n_loaded, len(metadata), store_dir,
                        )
                        needs_rebuild = True

            if needs_rebuild:
                dim = (
                    self._embedding_dim
                    or _resolve_embedding_dim()
                    or DEFAULT_EMBEDDING_DIM
                )
                index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
                metadata = []
                self._next_id[user_id] = 0
            else:
                self._next_id[user_id] = (
                    max((m["faiss_id"] for m in metadata), default=-1) + 1
                )
            if os.path.isfile(extra_path):
                with open(extra_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self._extra_metadata[user_id] = (
                    loaded if isinstance(loaded, dict) else {}
                )
        else:
            
            # 本实现默认为 1536（text-embedding-3-small），首次调用 _get_embeddings
            # 后动态校正为实际维度。若使用非 1536 维模型（如 text-embedding-3-large=3072），
            # 需确保 add 阶段先于 test 执行（add 时首次 embedding 调用会更新 _embedding_dim）。
            dim = (
                self._embedding_dim or _resolve_embedding_dim() or DEFAULT_EMBEDDING_DIM
            )
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            metadata = []
            self._next_id[user_id] = 0

        self._indices[user_id] = index
        self._metadata[user_id] = metadata
        self._id_to_meta_cache[user_id] = {
            m["faiss_id"]: i for i, m in enumerate(metadata)
        }
        return index, metadata

    def _allocate_id(self, user_id: str) -> int:
        """为指定用户分配一个递增的 FAISS 向量 ID。"""
        vector_id = self._next_id.get(user_id, 0)
        self._next_id[user_id] = vector_id + 1
        return vector_id

    def _add_vector(
        self,
        user_id: str,
        text: str,
        embedding: list[float],
        timestamp: str,
        extra_meta: dict | None = None,
    ) -> None:
        """向用户索引中添加一条向量记录及对应元数据。"""
        index, metadata = self._get_or_create_index(user_id)
        emb_dim = len(embedding)
        if index.d != emb_dim:
            logger.warning(
                "MemoryBank _add_vector: dimension mismatch for user=%s "
                "(got %d-dim vector but index expects %d-dim). "
                "Skipping this vector. Rebuild the index with a consistent model.",
                user_id, emb_dim, index.d,
            )
            return
        vector_id = self._allocate_id(user_id)
        vec = np.array([embedding], dtype=np.float32)
        
        # 否则内积不等价于余弦相似度，未归一化的向量模长会偏置检索结果。
        faiss.normalize_L2(vec)
        index.add_with_ids(vec, np.array([vector_id], dtype=np.int64))
        meta_entry = {
            "text": text,
            "timestamp": timestamp,
            "memory_strength": INITIAL_MEMORY_STRENGTH,
            "last_recall_date": timestamp[:DATE_PREFIX_LEN]
            if len(timestamp) >= DATE_PREFIX_LEN
            else timestamp,
            "faiss_id": vector_id,
        }
        if extra_meta:
            meta_entry.update(extra_meta)
        metadata.append(meta_entry)
        cache = self._id_to_meta_cache.get(user_id)
        if cache is not None:
            cache[meta_entry["faiss_id"]] = len(metadata) - 1
        self._speakers_cache.pop(user_id, None)  # 使缓存失效；可能有新说话人

    def _get_effective_chunk_size(self, user_id: str) -> int:
        """返回指定用户的合并分块大小，支持自适应校准。
        
        若环境变量 MEMORYBANK_CHUNK_SIZE 已显式设置，直接使用该值
        （跳过自适应逻辑）。
        """
        if os.getenv("MEMORYBANK_CHUNK_SIZE") is not None:
            return _resolve_chunk_size()
        metadata = self._metadata.get(user_id, [])
        if not metadata:
            return DEFAULT_CHUNK_SIZE
        lengths = sorted(len(m.get("text", "")) for m in metadata)
        n = len(lengths)
        # P90 至少需要 10 条才能有意义；n ≤ 9 时，ceil(n*0.9) 坍塌至 n
        # （即取最大值 = P100），离群条目会过度膨胀 chunk_size。回退至默认值。
        if n < 10:
            if user_id not in self._chunk_fallback_warned:
                self._chunk_fallback_warned.add(user_id)
                logger.info(
                    "MemoryBank: only %d entries for user=%s, "
                    "insufficient for P90 adaptive chunk calibration. "
                    "Falling back to DEFAULT_CHUNK_SIZE=%d.",
                    n, user_id, DEFAULT_CHUNK_SIZE,
                )
            return DEFAULT_CHUNK_SIZE
        p90_idx = math.ceil(n * 0.9) - 1  # 如 n=10 → 索引 8（近似 P90）
        p90 = lengths[p90_idx]
        candidate = max(1, p90) * 3
        return max(CHUNK_SIZE_MIN, min(CHUNK_SIZE_MAX, candidate))

    def _merge_neighbors(self, results: list[dict], user_id: str) -> list[dict]:
        """合并检索结果中来自同一来源的相邻条目，减少碎片化。

        results 中的每个条目必须携带 _meta_idx（metadata 列表索引），
        由 search() 保证。公开调用此方法的外部代码需自行确保此不变式；
        未设置 _meta_idx 的条目将被透传（不参与合并）。
        """
        
    # 2. 邻居遍历中 break 只跳出内层循环，导致另一方向被跳过；
        #    且共享 total_length 导致方向序偏置。本实现改为独立双向收集
        #    + deque 从外向内裁剪，结果与迭代顺序无关。
        # 3. 合并后的文本先剥离前缀再用 _MERGED_TEXT_DELIMITER 连接。
        if not results:
            return results

        metadata = self._metadata.get(user_id, [])
        if not metadata:
            return results

        indexed = [
            (r, r["_meta_idx"]) for r in results if r.get("_meta_idx") is not None
        ]
        if not indexed:
            return results
        non_indexed = [r for r in results if r.get("_meta_idx") is None]

        merged_results: list[dict] = []

        for r, meta_idx in indexed:
            score = float(r.get("score", 0.0))
            source = r.get("source", "")

            
            # backward 可用容量减少）。改为先收集所有同源邻居（不限 chunk_size），
            # 再从外向内裁剪至有效 chunk_size 以内，结果与迭代顺序无关。
            neighbor_indices: list[int] = [meta_idx]

            # 向前收集所有同源条目
            pos = meta_idx + 1
            while pos < len(metadata) and metadata[pos].get("source") == source:
                neighbor_indices.append(pos)
                pos += 1

            # 向后收集所有同源条目
            pos = meta_idx - 1
            while pos >= 0 and metadata[pos].get("source") == source:
                neighbor_indices.append(pos)
                pos -= 1

            neighbor_indices.sort()

            # 从外向内裁剪至有效 chunk_size 以内
            effective_chunk = self._get_effective_chunk_size(user_id)
            trim_queue = deque(neighbor_indices)
            total = sum(len(metadata[i].get("text", "")) for i in trim_queue)
            while len(trim_queue) > 1:
                if total <= effective_chunk:
                    break
                left_dist = meta_idx - trim_queue[0]
                right_dist = trim_queue[-1] - meta_idx
                # 等距时优先移除左侧（低索引）条目
                if left_dist >= right_dist:
                    removed = trim_queue.popleft()
                else:
                    removed = trim_queue.pop()
                total -= len(metadata[removed].get("text", ""))
            neighbor_indices = list(trim_queue)

            # neighbor_indices 经双向扩展后始终为单一连续块，
            # 无需 _group_consecutive/hit_seen 分组。
            parts: list[str] = []
            for idx in neighbor_indices:
                t = metadata[idx].get("text", "")
                src = metadata[idx].get("source", "")
                date_part = src.removeprefix("summary_")
                t = _strip_source_prefix(t, date_part)
                parts.append(t.strip())
            
            # 导致合并后出现重复前缀和混乱格式。此处用 _MERGED_TEXT_DELIMITER 连接并剥离前缀。
            combined_text = _MERGED_TEXT_DELIMITER.join(parts)
            base_meta = dict(metadata[neighbor_indices[0]])
            base_meta["text"] = combined_text
            base_meta["_meta_idx"] = meta_idx  # 保留原始命中索引供 search() 更新 strength

            base_meta["score"] = float(score)

            if len(neighbor_indices) > 1:
                base_meta["_merged_indices"] = sorted(neighbor_indices)
                base_meta["speakers"] = sorted(
                    {s for i in neighbor_indices
                     for s in (metadata[i].get("speakers") or [])}
                )
                base_meta["memory_strength"] = max(
                    _safe_memory_strength(
                        metadata[i].get("memory_strength", INITIAL_MEMORY_STRENGTH)
                    )
                    for i in neighbor_indices
                )
            merged_results.append(base_meta)

        
        # 去重后统一分割为连续组并产出合并文档（但 scores[0][j] 使用循环结束后的
        # 最后一个 j，所有合并文档共享同一分数——分数 bug）。本实现每结果独立
        # 构建合并条目并保留各自 score，然后通过子集过滤消除跨结果重叠，
        # 同时修复原版的分数 bug。
        if len(merged_results) > 1:
            merged_results = _merge_overlapping_results(merged_results)

        merged_results.extend(non_indexed)
        merged_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return merged_results

    def _get_or_init_extra(self, user_id: str) -> dict:
        """获取用户额外元数据字典，防御 JSON 反序列化产生的 null→None。

        JSON 中 user_id 条目若为 null，json.load 反序列化为 Python None，
        后续 `.get("key")` 会因 AttributeError 崩溃。此方法统一处理该场景。
        """
        extra = self._extra_metadata.setdefault(user_id, {})
        if not isinstance(extra, dict):
            extra = {}
            self._extra_metadata[user_id] = extra
        return extra

    def save_index(self, user_id: str) -> None:
        """将用户的 FAISS 索引和元数据持久化到磁盘。        元数据散落在内存中的 memory_loader.memory_bank 字典内（未持久化到
        索引文件旁）。本实现使用原生 FAISS write_index + metadata.json +
        extra_metadata.json 三文件模式，格式统一、可独立迁移。
        """
        if user_id not in self._indices:
            return
        store_dir = _user_store_dir(user_id, self._store_root)
        os.makedirs(store_dir, mode=0o700, exist_ok=True)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.json")
        extra_path = os.path.join(store_dir, "extra_metadata.json")
        faiss.write_index(self._indices[user_id], index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata[user_id], f, ensure_ascii=False, indent=2)
        # 注：此处使用裸 .get() 而非 _get_or_init_extra——save_index 为只读+持久化
        # 操作，不应产生初始化空 dict 的副作用。若 _extra_metadata 中无此用户条目，
        # 跳过 extra_metadata.json 写入即可（无内容需持久化）。
        extra = self._extra_metadata.get(user_id, {})
        if extra:
            with open(extra_path, "w", encoding="utf-8") as f:
                json.dump(extra, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_speaker(line: str) -> tuple[str | None, str]:
        """从对话行中解析说话人和内容，格式为 "Speaker: content"。        本测试集为多用户车载场景（如 Gary、Justin、Patricia 等），
        需要动态解析说话人名称以保留身份信息。

        Returns:
            (speaker_name, content) — speaker_name 为 None 表示无法解析。
        """
        colon_pos = line.find(": ")
        if colon_pos > 0:
            return line[:colon_pos].strip(), line[colon_pos + 2 :].strip()
        logger.warning(
            "MemoryBank: unparseable speaker line: %r", line[:80]
        )
        return None, line.strip()

    def add(self, messages: list[dict], user_id: str, timestamp: str) -> None:
        """将对话消息分对编码为向量并存入用户索引。"""
        date_key = (
            timestamp[:DATE_PREFIX_LEN]
            if len(timestamp) >= DATE_PREFIX_LEN
            else timestamp
        )
        all_entries: list[tuple[str | None, str]] = []
        for msg in messages:
            content = msg.get("content", "")
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped:
                    speaker, text = self._parse_speaker(stripped)
                    if speaker is None:
                        # 无法解析的说话人行（格式不匹配），跳过
                        continue
                    text = text.replace("\x00", "")
                    all_entries.append((speaker, text))

        if not all_entries:
            return

        pair_texts: list[str] = []
        pair_speakers: list[list[str]] = []
        for i in range(0, len(all_entries), 2):
            speaker_a, text_a = all_entries[i]
            speakers = [speaker_a]
            if i + 1 < len(all_entries):
                speaker_b, text_b = all_entries[i + 1]
                speakers.append(speaker_b)
                formatted = (
                    f"Conversation content on {date_key}:"
                    f"[|{speaker_a}|]: {text_a}; "
                    f"[|{speaker_b}|]: {text_b}"
                )
            else:
                formatted = (
                    f"Conversation content on {date_key}:[|{speaker_a}|]: {text_a}"
                )
            pair_texts.append(formatted)
            pair_speakers.append(speakers)

        embeddings = self._get_embeddings(pair_texts)

        if len(embeddings) != len(pair_texts):
            logger.warning(
                "MemoryBank add: embedding count mismatch (%d embeddings for "
                "%d text pairs) for user=%s — skipping this batch. "
                "Check embedding API availability.",
                len(embeddings), len(pair_texts), user_id,
            )
            return

        for text, emb, spks in zip(pair_texts, embeddings, pair_speakers, strict=True):
            self._add_vector(
                user_id,
                text,
                emb,
                timestamp,
                
                # 导致合并逻辑实际无效。本实现 source=date_key（同日期共享），使同一日期的
                # 连续条目可在 _merge_neighbors 中合并，检索结果更连贯。
                extra_meta={
                    "source": date_key,
                    "speakers": sorted(set(spks)),
                },
            )

    def _call_llm(self, prompt_text: str) -> str | None:
        """调用 LLM 生成回复，带重试逻辑处理可恢复的 API 错误和上下文长度回退。

        Returns:
            LLM 返回的文本内容（去除首尾空白）；
            None: 重试耗尽，无法获取有效响应（网络错误/限速等）；
            "": LLM API 成功调用但返回了空内容。
        """
        if not self._llm_client:
            return None
        max_retries = LLM_MAX_RETRIES
        for attempt in range(max_retries):
            try:
                resp = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        
                        # "Below is a transcript of a conversation between a human and an AI
                        #  assistant that is intelligent and knowledgeable in psychology."
                        # 本实现改为车载助手场景，聚焦车辆偏好、驾驶习惯和多用户交互。
                        # 此 prompt 直接影响摘要/性格生成的质量和焦点。
                        {
                            "role": "system",
                            "content": (
                                "You are an in-car AI assistant with expertise in remembering "
                                "vehicle preferences, driving habits, and in-car conversation "
                                "context. Your task is to summarize and analyze multi-user "
                                "conversations occurring inside a vehicle, focusing on user "
                                "preferences related to vehicle settings (seat, climate, "
                                "lighting, media, navigation, etc.)."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Hello! Please help me summarize the content of the conversation.",
                        },
                        
                        # 属于 assistant 角色（"Sure, I will do my best to assist you."）。
                        # 修正为 role="assistant" 使消息序列符合 system/assistant 角色分工。
                        # 注意：原序列为 system→user→system→user（非连续 system，
                        # 但第三条 system 承载 assistant 语义属角色误用）。
                        {
                            "role": "assistant",
                            "content": "Sure, I will do my best to assist you.",
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    frequency_penalty=LLM_FREQUENCY_PENALTY,
                    presence_penalty=LLM_PRESENCE_PENALTY,
                    
                    # BELLE 模型和中文场景专用。英文 OpenAI 兼容 API 无需设置。
                )
                response_text = resp.choices[0].message.content
                return response_text.strip() if response_text else ""
            except Exception as exc:
                context_exceeded = isinstance(exc, _openai.BadRequestError) or any(
                    pattern in str(exc).lower()
                    for pattern in (
                        "maximum context",
                        "context length",
                        "too long",
                        "reduce the length",
                        "prompt is too long",
                        "input length",
                    )
                )
                if context_exceeded and attempt < max_retries - 1:
                    cut_length = max(
                        LLM_CTX_TRIM_START - LLM_CTX_TRIM_STEP * attempt,
                        LLM_CTX_TRIM_MIN,
                    )
                    logger.warning(
                        "MemoryBank _call_llm context length exceeded, "
                        "trimming to last %d chars (attempt %d/%d)",
                        cut_length,
                        attempt + 1,
                        max_retries,
                    )
                    # 从开头截断，保留最后 cut_length 个字符。
                    # 对话内容场景下，此操作保留最近消息（更相关），
                    # 丢弃较旧上下文——与原代码 summarize_memory.py:44
                    # 的行为一致。
                    prompt_text = prompt_text[-cut_length:]
                    continue

                retryable = isinstance(
                    exc,
                    (
                        _openai.APIConnectionError,
                        _openai.APITimeoutError,
                        _openai.RateLimitError,
                    ),
                )
                if not retryable and isinstance(exc, _openai.APIStatusError):
                    retryable = exc.status_code >= 500

                if retryable and attempt < max_retries - 1:
                    time.sleep(2**attempt + self._rng.random())
                    continue

                if not retryable:
                    raise

                logger.warning(
                    "MemoryBank _call_llm exhausted %d retries: %s",
                    max_retries,
                    exc,
                )
                return None

    def _summarize(self, text: str) -> str | None:
        """调用 LLM 对对话文本生成摘要，聚焦车辆偏好和用户身份。"""
        
        # "Please summarize the following dialogue as concisely as possible,
        #  extracting the main themes and key information."
        # 本实现改为聚焦车辆偏好（座椅/空调/灯光/导航等）、多用户冲突和条件约束，
        # 忽略与车辆无关的通用对话内容。
        return self._call_llm(
            "Please summarize the following in-car dialogue concisely, "
            "focusing specifically on:\n"
            "1. Vehicle settings or preferences mentioned (seat position, "
            "climate temperature/ventilation, ambient light color, navigation "
            "mode, music/radio settings, HUD brightness, etc.)\n"
            "2. Which person (by name) expressed or changed each preference\n"
            "3. Any conflicts or differences between users' vehicle preferences\n"
            "4. Conditional constraints (e.g. preference depends on time of day, "
            "weather, or passenger presence)\n"
            "Ignore general conversation topics unrelated to the vehicle.\n"
            f"Dialogue content:\n{text}\n"
            "Summarization："  # noqa: RUF001
        )

    @staticmethod
    def _get_date_key(meta: dict) -> str:
        """从元数据条目中提取日期键（YYYY-MM-DD 格式）。"""
        return meta.get("source") or meta.get("timestamp", "")[:DATE_PREFIX_LEN]

    def _collect_daily_texts(
        self, user_id: str, *, skip_type: str | None = None,
        existing_dates: set | None = None, context: str = "processing",
    ) -> dict[str, list[str]]:
        """按日期聚合 metadata 中的对话文本。

        Args:
            user_id: 用户标识（用于日志）
            skip_type: 需要跳过的 metadata type 字段值（例如 "daily_summary"）
            existing_dates: 已有条目的日期集合，已有者不重复收集
            context: 日志标识字符串（例如 "summarization"、"personality"）

        Returns:
            {date_key: [text1, text2, ...]} 映射
        """
        metadata = self._metadata.get(user_id, [])
        daily_texts: dict[str, list[str]] = {}
        for meta in metadata:
            if skip_type is not None and meta.get("type") == skip_type:
                continue
            date_key = self._get_date_key(meta)
            if not date_key:
                logger.warning(
                    "MemoryBank: skipping metadata entry faiss_id=%s "
                    "(user=%s, context=%s) — empty date_key. "
                    "Check metadata for missing source/timestamp fields.",
                    meta.get("faiss_id", -1),
                    user_id,
                    context,
                )
                continue
            if existing_dates and date_key in existing_dates:
                continue
            daily_texts.setdefault(date_key, []).append(meta.get("text", ""))
        return daily_texts

    def _process_daily_llm_task(
        self,
        user_id: str,
        *,
        llm_func: Callable[[str], str | None],
        result_handler: Callable[[str, str, str], None],
        skip_type: str | None = None,
        existing_dates: set | None = None,
        context: str = "processing",
    ) -> None:
        """通用每日文本LLM处理辅助函数。
        
        Args:
            user_id: 用户标识
            llm_func: LLM调用函数，接受合并文本，返回生成结果或None
            result_handler: 结果处理函数，接受(user_id, date_key, result)
            skip_type: 需要跳过的metadata type
            existing_dates: 已有条目的日期集合
            context: 日志标识字符串
        """
        if not self._llm_client:
            return
        
        daily_texts = self._collect_daily_texts(
            user_id, skip_type=skip_type, existing_dates=existing_dates, context=context
        )
        
        for date_key, texts in sorted(daily_texts.items()):
            cleaned = [_strip_source_prefix(t, date_key).strip() for t in texts]
            combined = "\n".join(cleaned)
            logger.info(
                "MemoryBank: processing daily text for user=%s date=%s (%d lines)",
                user_id, date_key, len(texts),
            )
            try:
                result = llm_func(combined)
            except Exception:
                logger.warning(
                    "MemoryBank: LLM call raised for %s user=%s date=%s — skipping",
                    context, user_id, date_key, exc_info=True,
                )
                continue
            if result is None:
                logger.warning(
                    "MemoryBank: LLM call failed for %s user=%s date=%s — skipping",
                    context, user_id, date_key,
                )
                continue
            if result:
                result_handler(user_id, date_key, result)

    def _generate_daily_summaries(self, user_id: str) -> None:
        """按日期聚合对话内容并生成每日摘要向量。"""
        def _handle_summary(user_id: str, date_key: str, summary: str) -> None:
            summary_text = f"The summary of the conversation on {date_key} is: {summary}"
            ts = f"{date_key}{DEFAULT_TIME_SUFFIX}"
            try:
                summary_embs = self._get_embeddings([summary_text])
                if not summary_embs:
                    logger.warning(
                        "MemoryBank: embedding API returned empty for daily summary "
                        "user=%s date=%s — skipping this date",
                        user_id, date_key,
                    )
                    return
                summary_emb = summary_embs[0]
                self._add_vector(
                    user_id, summary_text, summary_emb, ts,
                    {"type": "daily_summary", "source": f"summary_{date_key}"},
                )
            except Exception:
                logger.warning(
                    "MemoryBank: embedding or index write failed for "
                    "daily summary user=%s date=%s — skipping this date",
                    user_id, date_key, exc_info=True,
                )
        
        existing_summary_dates = set()
        metadata = self._metadata.get(user_id, [])
        for m in metadata:
            if m.get("type") == "daily_summary":
                date = (m.get("source") or "").removeprefix("summary_")
                existing_summary_dates.add(date)
        
        self._process_daily_llm_task(
            user_id,
            llm_func=self._summarize,
            result_handler=_handle_summary,
            existing_dates=existing_summary_dates,
            context="summarization",
        )

    def _generate_overall_summary(self, user_id: str) -> None:
        """基于所有每日摘要生成整体摘要，存入额外元数据。"""
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        extra = self._get_or_init_extra(user_id)
        # 若已有整体摘要则跳过（避免增量 add 时重复消耗 LLM token）
        if extra.get("overall_summary"):
            return

        daily_summaries = [m for m in metadata if m.get("type") == "daily_summary"]
        if not daily_summaries:
            return

        summary_parts = []
        for m in daily_summaries:
            raw_source = m.get("source")
            date = (
                raw_source if raw_source else m.get("timestamp", "")[:DATE_PREFIX_LEN]
            ).removeprefix("summary_")
            text = m["text"]
            prefix = f"The summary of the conversation on {date} is: "
            if text.startswith(prefix):
                text = text[len(prefix) :]
            summary_parts.append((date, text))

        prompt_parts = [
            "Please provide a highly concise summary of the following event, "
            "capturing the essential key information as succinctly as possible. "
            "Focus on vehicle preferences, user habits, and in-car interactions. "
            "Summarize the event:\n",
        ]
        for date, text in summary_parts:
            prompt_parts.append(f"\nAt {date}, the events are {text.strip()}")
        prompt_parts.append("\nSummarization：")  # noqa: RUF001
        prompt = "".join(prompt_parts)

        logger.info(
            "MemoryBank: generating overall summary for user=%s (%d dates)",
            user_id, len(summary_parts),
        )
        try:
            summary = self._call_llm(prompt)
        except Exception:
            logger.warning(
                "MemoryBank: LLM call raised for overall summary user=%s",
                user_id,
                exc_info=True,
            )
            return
        if summary is None:
            logger.warning(
                "MemoryBank: LLM call failed for overall summary user=%s",
                user_id,
            )
            return
        if summary:
            extra["overall_summary"] = summary
        else:
            # 空结果（"" 或 None）：记录"已尝试"旗标避免下次 add 重复消耗 token
            extra["overall_summary"] = _GENERATION_EMPTY

    def _analyze_personality(self, text: str) -> str | None:
        """调用 LLM 分析对话中体现的用户驾驶习惯和车辆偏好。"""
        
        # prompt 中明确包含 `{user_name}` 和 `{boot_name}`（"AI lover"）。
        # 本测试集为多用户车载场景，每日对话可能涉及多个用户（Gary/Justin/Patricia），
        # 无法建立单一用户↔AI 对应关系。改为按日期聚合所有参与者后做多用户偏好汇总分析，
        # prompt 改为通用 "users" / "AI assistant" 表述。
        # 分析粒度从 per-user personality portrait 变为 multi-user preference aggregation。
        return self._call_llm(
            "Based on the following in-car dialogue, analyze the users' "
            "vehicle-related preferences and habits:\n"
            "1. What vehicle settings does each user prefer (seat, climate, "
            "lighting, media, navigation, etc.)?\n"
            "2. How do their preferences vary by context (time of day, "
            "weather, passengers)?\n"
            "3. What driving or comfort habits are exhibited?\n"
            "4. What response strategy should the AI use to anticipate "
            "each user's needs?\n"
            f"Dialogue content:\n{text}\n"
            "Analysis:"
        )

    def _generate_daily_personalities(self, user_id: str) -> None:
        """按日期聚合对话并分析每日用户性格，存入额外元数据。"""
        extra = self._get_or_init_extra(user_id)
        existing_personalities = extra.setdefault("daily_personalities", {})
        if not isinstance(existing_personalities, dict):
            existing_personalities = {}
            extra["daily_personalities"] = existing_personalities
        
        def _handle_personality(user_id: str, date_key: str, personality: str) -> None:
            existing_personalities[date_key] = personality
        
        self._process_daily_llm_task(
            user_id,
            llm_func=self._analyze_personality,
            result_handler=_handle_personality,
            skip_type="daily_summary",
            existing_dates=set(existing_personalities.keys()),
            context="personality",
        )

    def _generate_overall_personality(self, user_id: str) -> None:
        """基于每日性格分析生成整体性格画像，存入额外元数据。"""
        if not self._llm_client:
            return

        extra = self._get_or_init_extra(user_id)
        # 若已有整体性格画像则跳过
        if extra.get("overall_personality"):
            return

        daily_personalities = extra.get("daily_personalities", {})
        if not daily_personalities:
            return

        prompt_parts = [
            "The following are analyses of users' vehicle-related preferences "
            "and habits across multiple driving sessions:\n",
        ]
        for date, text in sorted(daily_personalities.items()):
            prompt_parts.append(f"\nAt {date}, the analysis shows {text.strip()}")
        prompt_parts.append(
            
            "\nPlease provide a highly concise summary of the users' vehicle "
            "preferences and driving habits, organized by user, and the most "
            "appropriate in-car response strategy for the AI assistant, "
            "summarized as:"
        )
        prompt = "".join(prompt_parts)

        logger.info(
            "MemoryBank: generating overall personality for user=%s (%d dates)",
            user_id, len(daily_personalities),
        )
        try:
            personality = self._call_llm(prompt)
        except Exception:
            logger.warning(
                "MemoryBank: LLM call raised for overall personality user=%s",
                user_id,
                exc_info=True,
            )
            return
        if personality is None:
            logger.warning(
                "MemoryBank: LLM call failed for overall personality user=%s",
                user_id,
            )
            return
        if personality:
            extra["overall_personality"] = personality
        else:
            # 空结果（"" 或 None）：记录已尝试旗标
            extra["overall_personality"] = _GENERATION_EMPTY

    def _forgetting_retention(self, days_elapsed: float, memory_strength: float) -> float:
        """基于艾宾浩斯遗忘曲线计算记忆保留概率。

        论文公式 R = e^{-t/S}，S 为 memory_strength，越大保留率越高。        （`/` 与 `*` 同级、左结合）实际计算为 `math.exp(-(t/5)*S)`，
        导致 strength 越大遗忘越多，与艾宾浩斯曲线定义矛盾。
        本实现修正为 `math.exp(-t / S)`，对齐原论文。
        """
        effective_s = _safe_memory_strength(memory_strength)
        return math.exp(-max(0.0, days_elapsed) / (FORGETTING_TIME_SCALE * effective_s))

    def _forget_at_ingestion(self, user_id: str) -> None:
        """在数据摄入阶段根据遗忘曲线概率性地丢弃部分记忆。"""
        if not self.enable_forgetting or not self.reference_date:
            return

        index, metadata = self._get_or_create_index(user_id)

        try:
            ref_dt = datetime.strptime(self.reference_date[:DATE_PREFIX_LEN], "%Y-%m-%d")
        except (ValueError, TypeError):
            logger.error(
                "MemoryBank: invalid reference_date=%r for user=%s. "
                "Skipping forgetting entirely. "
                "Expected format YYYY-MM-DD (e.g. 2024-06-15). "
                "Set MEMORYBANK_REFERENCE_DATE or pass --history_dir.",
                self.reference_date, user_id,
            )
            return

        ids_to_remove: list[int] = []
        kept_ids: set[int] = set()

        for meta in metadata:
            if meta.get("type") in MEMORY_SKIP_TYPES:
                kept_ids.add(meta["faiss_id"])
                continue

            ts_str = meta.get("last_recall_date", meta.get("timestamp", ""))[
                :DATE_PREFIX_LEN
            ]
            try:
                mem_dt = datetime.strptime(ts_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                logger.warning(
                    "MemoryBank: unparseable date %r for entry "
                    "faiss_id=%s (user=%s) — keeping this entry. "
                    "Metadata corruption suspected.",
                    ts_str, meta.get("faiss_id", -1), user_id,
                )
                kept_ids.add(meta["faiss_id"])
                continue
            days_elapsed = (ref_dt - mem_dt).days
            strength = meta.get("memory_strength", INITIAL_MEMORY_STRENGTH)
            retention = self._forgetting_retention(days_elapsed, strength)
            if self._rng.random() > retention:
                ids_to_remove.append(meta["faiss_id"])
            else:
                kept_ids.add(meta["faiss_id"])

        if ids_to_remove:
            index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            self._metadata[user_id] = [
                m for m in metadata if m["faiss_id"] in kept_ids
            ]
            self._id_to_meta_cache[user_id] = {
                m["faiss_id"]: i for i, m in enumerate(self._metadata[user_id])
            }
            
            # 原项目无此场景（ingestion 后只读），本实现为防御性一致性修正。
            self._next_id[user_id] = max(
                (m["faiss_id"] for m in self._metadata[user_id]), default=-1
            ) + 1
            self._speakers_cache.pop(user_id, None)  # 使缓存失效；说话人可能已移除

    # 检索结果内部字段——移除仅用于管道中间阶段的元数据，
    # 保留消费端所需的 text / source / speakers / memory_strength / score。
    _INTERNAL_KEYS: frozenset[str] = frozenset({
        "_merged_indices", "_all_meta_indices", "_meta_idx", "faiss_id",
    })

    @staticmethod
    def _clean_search_result(result: dict) -> None:
        """移除检索结果中的内部字段，解码合并分隔符。"""
        for key in MemoryBankClient._INTERNAL_KEYS:
            result.pop(key, None)
        text = result.get("text")
        if text:
            result["text"] = text.replace(_MERGED_TEXT_DELIMITER, "; ")

    
    # forget_memory.py=6，ChatGPT/LlamaIndex 路径=2（cli_llamaindex.py:36）。
    # 本实现取 5 以适配多事件车载场景（每个文件约 10 个事件跨越多天）。
    def search(
        self, query: str, user_id: str, top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """基于向量相似度检索与查询最相关的记忆，并合并相邻条目。"""
        index, metadata = self._get_or_create_index(user_id)

        if index.ntotal == 0:
            return []

        # 仅当 reference_date 未设置时发出一次告警（replay 场景正常缺失）
        if not self.reference_date and not self._warned_no_ref_date:
            self._warned_no_ref_date = True
            logger.warning("MemoryBank: reference_date not set; recency decay disabled")

        query_embs = self._get_embeddings([query])
        if not query_embs:
            logger.warning(
                "MemoryBank search: failed to embed query for user=%s — "
                "returning empty results.", user_id,
            )
            return []
        query_emb = query_embs[0]
        query_vec = np.array([query_emb], dtype=np.float32)
        
        faiss.normalize_L2(query_vec)

        
        # 本实现取 top_k*4 倍率扩大粗排窗口，为后续邻居合并预留空间。
        k = min(top_k * COARSE_SEARCH_FACTOR, index.ntotal)
        scores, indices = index.search(query_vec, k)

        id_to_meta = self._id_to_meta_cache.get(user_id, {})

        results: list[dict] = []
        for score, faiss_id in zip(scores[0], indices[0]):
            meta_idx = id_to_meta.get(int(faiss_id))
            if meta_idx is None:
                continue
            meta = dict(metadata[meta_idx])
            meta["score"] = float(score)
            meta["_meta_idx"] = meta_idx
            results.append(meta)

        
        # 对不涉及该用户的记忆条目施加 0.75× 降权因子（软过滤），减少跨用户噪声。
        # 例如 query 中提到 "Gary" 时，不涉及 Gary 的 Patricia/Justin 记忆会被降权，
        # 但仍保留（避免因 query 中省略用户名而误杀相关记忆）。
        all_speakers = self._speakers_cache.get(user_id)
        if all_speakers is None:
            all_speakers_set: set[str] = set()
            for m in metadata:
                spks = m.get("speakers")
                if isinstance(spks, list):
                    all_speakers_set.update(spks)
            all_speakers = sorted(all_speakers_set)  # 转 JSON 兼容列表
            self._speakers_cache[user_id] = all_speakers
        _mentioned_speakers: set[str] = set()
        query_lower = query.lower()
        for spk_full in all_speakers:
            spk_first = spk_full.split(" ", 1)[0] if " " in spk_full else spk_full
            if (_word_in_text(spk_full.lower(), query_lower)
                    or _word_in_text(spk_first.lower(), query_lower)):
                _mentioned_speakers.add(spk_full.lower())

        merged = self._merge_neighbors(results, user_id)
        # 合并后再应用说话人软过滤，因为合并后的条目可能继承自多个邻居的 speakers
        if _mentioned_speakers:
            for r in merged:
                spks = r.get("speakers")
                if isinstance(spks, list):
                    if not spks:
                        continue  # 旧格式条目无 speaker 数据，跳过惩罚
                    if not any(s.lower() in _mentioned_speakers for s in spks):
                        score = r.get("score", 0.0)
                        # 正分 *0.75 向零靠近（惩罚）；负分 *1.25
                        # 远离零（惩罚——越大负数排名越低）。统一 *0.75
                        # 会缩小负数幅度，反而提升排名。
                        r["score"] = score * 0.75 if score >= 0 else score * 1.25
            # 惩罚后重新排序
            merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        merged = merged[:top_k]

        
        # 被合并的邻居条目不获得 strength 提升——它们虽作为上下文返回但未被实际 recall，
        # 不应受到 spacing effect 保护（否则合并噪声将被错误强化）。
        # 本实现更新所有被 FAISS 原始召回（_meta_idx / _all_meta_indices）的条目强度；
        # _all_meta_indices 由 _merge_overlapping_results 产生，记录因索引重叠被合并的
        # 多个独立 FAISS 命中的元数据索引——它们均需 spacing effect 保护。
        for r in merged:
            meta_indices: list[int] = []
            all_indices = r.get("_all_meta_indices")
            if isinstance(all_indices, list):
                # _all_meta_indices 来自 _merge_overlapping_results，已包含所有
                # 被 FAISS 独立召回的成员索引（含 best_idx 的 _meta_idx）。
                meta_indices.extend(all_indices)
            else:
                # 无跨结果合并：仅 _merge_neighbors 产生的单条命中，
                # _meta_idx 为该原始命中的元数据索引。
                meta_idx = r.get("_meta_idx")
                if meta_idx is not None:
                    meta_indices.append(meta_idx)
            for mi in meta_indices:
                if 0 <= mi < len(metadata):
                    metadata[mi]["memory_strength"] = (
                        _safe_memory_strength(
                            metadata[mi].get("memory_strength", INITIAL_MEMORY_STRENGTH)
                        )
                        + MEMORY_STRENGTH_INCREMENT
                    )
                    if self.reference_date:
                        metadata[mi]["last_recall_date"] = self.reference_date[
                            :DATE_PREFIX_LEN
                        ]

        for r in merged:
            self._clean_search_result(r)

        
        # 持久化 memory_strength 和 last_recall_date。缺少此步会导致遗忘机制跨会话失效。
        if merged:
            self.save_index(user_id)
        return merged

    def get_extra_metadata(self, user_id: str) -> dict:
        """获取用户的额外元数据（整体摘要、性格画像等）。"""
        extra = self._extra_metadata.get(user_id, {})
        return extra if isinstance(extra, dict) else {}

def validate_add_args(args) -> None:
    """验证 add 操作所需的 Embedding API 凭据是否已提供。"""
    require_value(
        _resolve_embedding_api_key(args),
        "Embedding API key is required: pass --memory_key or set MEMORY_KEY/EMBEDDING_API_KEY",
    )
    require_value(
        _resolve_embedding_api_base(args),
        "Embedding API base URL is required: pass --memory_url or set MEMORY_URL/EMBEDDING_API_BASE",
    )

def validate_test_args(args) -> None:
    """验证测试操作所需参数，委托给 validate_add_args。"""
    validate_add_args(args)

def _build_client(args: Any, seed_override: int | None = None) -> MemoryBankClient:
    """根据命令行参数和环境变量构建 MemoryBankClient 实例。"""
    api_key = require_value(
        _resolve_embedding_api_key(args),
        "Embedding API key is required: pass --memory_key or set MEMORY_KEY/EMBEDDING_API_KEY",
    )
    api_base = require_value(
        _resolve_embedding_api_base(args),
        "Embedding API base URL is required: pass --memory_url or set MEMORY_URL/EMBEDDING_API_BASE",
    )

    enable_summary = _resolve_enable_summary()
    enable_forgetting = _resolve_enable_forgetting()
    seed = seed_override if seed_override is not None else _resolve_seed()
    reference_date = _resolve_reference_date()

    llm_api_base = (
        getattr(args, "llm_api_base", None)
        or os.getenv("MEMORYBANK_LLM_API_BASE")
        or os.getenv("LLM_API_BASE")
        or getattr(args, "api_base", None)
    )
    llm_api_key = (
        getattr(args, "llm_api_key", None)
        or os.getenv("MEMORYBANK_LLM_API_KEY")
        or os.getenv("LLM_API_KEY")
        or getattr(args, "api_key", None)
    )
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

def _compute_reference_date(history_dir: str, file_range: str | None) -> str:
    """扫描历史文件中的时间戳，计算最新日期的下一天作为参考日期。"""
    
    # 数周/数月（遗忘更激进）。本实现使用历史文件最新日期的下一天，使遗忘量
    # 保持合理且结果可复现，适合测评场景。
    history_files = collect_history_files(history_dir, file_range)
    max_ts: datetime | None = None
    for _, path in history_files:
        for bucket in load_hourly_history(path):
            if bucket.dt is not None:
                if max_ts is None or bucket.dt > max_ts:
                    max_ts = bucket.dt
    if max_ts is None:
        fallback = datetime.now().strftime("%Y-%m-%d")
        logger.warning(
            "MemoryBank: no valid timestamps found in history files "
            "under %s. Falling back to current date %s. "
            "Forgetting results may differ from expected.",
            history_dir, fallback,
        )
        return fallback
    ref_date = max_ts + timedelta(days=REFERENCE_DATE_OFFSET)
    return ref_date.strftime("%Y-%m-%d")

def run_add(args) -> None:
    """将对话历史摄入到 MemoryBank，构建向量索引并可选生成摘要和遗忘。"""
    validate_add_args(args)
    history_dir = os.path.abspath(args.history_dir)
    if not os.path.isdir(history_dir):
        raise FileNotFoundError(f"history directory not found: {history_dir}")

    history_files = collect_history_files(history_dir, args.file_range)
    print(
        f"[{TAG} ADD] history_dir={history_dir} files={len(history_files)} max_workers={args.max_workers} store_root={_resolve_store_root(args)}"
    )

    reference_date = _resolve_reference_date()
    if not reference_date:
        reference_date = _compute_reference_date(history_dir, args.file_range)

    base_seed = _resolve_seed()

    def processor(idx: int, history_path: str) -> tuple[int, int, str | None]:
        
        # jitter/retry 完全同步，在 rate limit 场景加剧故障。
        # 此处对 base_seed 按 worker index 偏移，保证各 worker 的随机序列独立。
        worker_seed = None if base_seed is None else base_seed + idx
        client = _build_client(args, seed_override=worker_seed)
        client.reference_date = reference_date
        user_id = f"{USER_ID_PREFIX}_{idx}"
        store_dir = _user_store_dir(user_id, client._store_root)
        if os.path.isdir(store_dir):
            shutil.rmtree(store_dir)
        try:
            message_count = 0
            daily_lines: dict[str, list[str]] = {}
            for bucket in load_hourly_history(history_path):
                if bucket.dt:
                    day_key = bucket.dt.strftime("%Y-%m-%d")
                else:
                    day_key = datetime.now().strftime("%Y-%m-%d")
                daily_lines.setdefault(day_key, []).extend(bucket.lines)

            for day_key, lines in sorted(daily_lines.items()):
                ts = f"{day_key}{DEFAULT_TIME_SUFFIX}"
                messages = [{"role": "user", "content": "\n".join(lines)}]
                client.add(messages=messages, user_id=user_id, timestamp=ts)
                message_count += len(lines)

            if client.enable_summary:
                client._generate_daily_summaries(user_id)
                client._generate_overall_summary(user_id)
                client._generate_daily_personalities(user_id)
                client._generate_overall_personality(user_id)

            
            # 摘要/性格在遗忘之后生成。本实现先摄入全部对话、生成摘要/性格，
            # 最后执行遗忘——摘要/性格属于 MEMORY_SKIP_TYPES 不受遗忘影响，
            # 结果等价且逻辑更清晰。
            client._forget_at_ingestion(user_id)

            client.save_index(user_id)
            return idx, message_count, None
        except Exception as exc:
            return idx, 0, str(exc)

    run_add_jobs(
        history_files=history_files,
        tag=TAG,
        max_workers=args.max_workers,
        processor=processor,
    )

def init_test_state(args, file_numbers, user_id_prefix):
    """初始化测试状态（MemoryBank 不需要共享状态）。"""
    del file_numbers, user_id_prefix
    validate_test_args(args)
    return None

def build_test_client(args, file_num: int, user_id_prefix: str, shared_state: Any):
    """构建用于测试的 MemoryBank 客户端包装器。"""
    del shared_state
    client = _build_client(args)
    if not client.reference_date:
        history_dir = os.path.abspath(args.history_dir)
        client.reference_date = _compute_reference_date(history_dir, args.file_range)
    uid = f"{user_id_prefix}_{file_num}"
    index, _ = client._get_or_create_index(uid)
    if index.ntotal == 0:
        logger.warning(
            "MemoryBank: FAISS index for %s is empty (ntotal=0). "
            "Did you forget to run the 'add' stage first? "
            "Evaluation will run but search results will be empty.",
            uid,
        )
    return _MemoryBankTestWrapper(client, uid)

class _MemoryBankTestWrapper:
    """评测流水线的 MemoryBankClient 薄包装。

    将 overall_summary 和 overall_personality 作为合成第一条结果
    插入，使 agent 的 LLM 在排序后的逐查询命中之前先看见全局上下文。
    """

    def __init__(self, client: MemoryBankClient, user_id: str):
        self._client = client
        self._user_id = user_id

    @staticmethod
    def _is_valid_context(value: str | None) -> bool:
        """检查 LLM 生成的上下文是否有效（非空且非哨兵值）。"""
        return bool(value) and value != _GENERATION_EMPTY

    def search(
        self, query: str, user_id: str | None = None, top_k: int = DEFAULT_TOP_K
    ) -> list[dict]:
        """检索记忆并附带整体摘要和性格画像。        注入整体上下文，不纳入检索结果。本测评流程中 agent 通过 tool call 获取
        记忆，将 overall_summary 和 overall_personality 作为额外条目插入到
        检索结果头部——因为 LLM 读取搜索结果时倾向于关注前几条高相关度条目，
        将全局上下文放在头部确保其被优先消费。
        """
        uid = user_id if user_id is not None else self._user_id
        results = self._client.search(query=query, user_id=uid, top_k=top_k)

        extra = self._client.get_extra_metadata(uid)
        overall_summary = extra.get("overall_summary", "")
        overall_personality = extra.get("overall_personality", "")

        if self._is_valid_context(overall_summary) or self._is_valid_context(
            overall_personality
        ):
            parts = []
            if self._is_valid_context(overall_summary):
                parts.append(f"Overall summary of past memories: {overall_summary}")
            if self._is_valid_context(overall_personality):
                parts.append(
                    f"User vehicle preferences and habits: {overall_personality}"
                )
            # 全局摘要/性格画像插入列表头部而非尾部：LLM 读取工具调用结果时
            # 倾向于优先关注前几条高相关度条目，将全局上下文前置确保其被优先消费。
            # score 使用 float('inf') 哨兵值以保证在任何按 score 排序的处理逻辑中
            # 都位于首位（format_search_results 通过 _type 识别，不依赖 score）。
            results.insert(
                0,
                {
                    "_type": "overall_context",
                    "text": "\n".join(parts),
                    "source": "overall",
                    "memory_strength": INITIAL_MEMORY_STRENGTH,
                    "score": float("inf"),
                },
            )

        return results

def close_test_state(shared_state: Any) -> None:
    """清理测试状态（MemoryBank 无需清理）。"""
    del shared_state

def is_test_sequential() -> bool:
    """MemoryBank 测试支持并行执行。"""
    return False

def format_search_results(search_result: Any) -> tuple[str, int]:
    """将检索结果格式化为带编号的文本，按相关性顺序分组并标注记忆强度。"""
    if not isinstance(search_result, list):
        return "", 0
    if not search_result:
        return "", 0

    overall_items = [r for r in search_result if r.get("_type") == "overall_context"]
    non_overall = [r for r in search_result if r.get("_type") != "overall_context"]

    # 键聚合：FAISS 结果按 score 排序，同 source 的条目可能被异源条目
    # 穿插。若用邻接合并（groups[-1][0] != group_key），同源非邻接条目
    # 会被拆分为多个独立分组。改用 dict 做全集聚合，以首次出现序保序。
    group_order: list[str] = []
    group_map: dict[str, tuple[list[str], list[dict]]] = {}
    for item in non_overall:
        text = item.get("text", "")
        raw_source = item.get("source") or ""
        date_part = raw_source.removeprefix("summary_")
        # 前缀已在 _merge_neighbors 中剥离（_strip_source_prefix），
        # 此处仅做 strip 以清理空白字符。
        text = text.strip()

        # 每日摘要和原始对话作为独立分组展示：二者虽有相同日期，
        # 但语义层次不同（摘要=提炼，对话=原始记录），混合输出会造成 LLM 困惑。
        # 使用 source 原始值（含 summary_ 前缀）作为分组键确保两者分离。
        group_key = raw_source if raw_source else date_part

        if group_key not in group_map:
            group_map[group_key] = ([], [])
            group_order.append(group_key)
        group_map[group_key][0].append(text)
        group_map[group_key][1].append(item)

    groups: list[tuple[str, str, list[dict]]] = []
    for gk in group_order:
        texts, items = group_map[gk]
        groups.append((gk, "\n".join(texts), items))

    lines: list[str] = []
    
    for idx, item in enumerate(overall_items, 1):
        lines.append(
            f"{idx}. [memory_strength={item.get('memory_strength', 1)}] {item.get('text', '')}"
        )

    group_start = len(overall_items) + 1
    for idx, (group_key, combined_text, items) in enumerate(groups, group_start):
        max_strength = max(
            it.get("memory_strength", INITIAL_MEMORY_STRENGTH) for it in items
        )
        # 对于摘要分组，显示 "2024-06-15 (summary)"；对话分组仅显示 "2024-06-15"
        if group_key.startswith("summary_"):
            display_date = f"{group_key.removeprefix('summary_')} (summary)"
        else:
            display_date = group_key
        date_info = f" [date={display_date}]" if display_date else ""
        lines.append(
            f"{idx}. [memory_strength={max_strength}]{date_info} {combined_text}"
        )

    return "\n\n".join(lines), len(non_overall)
