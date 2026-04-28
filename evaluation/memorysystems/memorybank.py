# ruff: noqa: RUF002, RUF003
"""
MemoryBank: 基于 FAISS 向量检索的本地记忆系统，复刻自原项目并适配 VehicleMemBench 测评场景。

原项目: https://github.com/zhongwanjun/MemoryBank-SiliconFriend
论文: https://arxiv.org/abs/2305.10250

相较于原项目的主要变更（搜索 `[DIFF]` 可定位所有差异点）:
- 向量索引: LangChain FAISS (IndexFlatL2) → 原生 FAISS IndexFlatIP + L2 归一化
- 嵌入模型: 本地 HuggingFace → OpenAI Embedding API (或兼容接口)
- 说话人格式: 固定 `[|User|]`/`[|AI|]` → 动态解析历史行中的说话人名称
- 遗忘公式: 修正原项目运算符优先级 bug (`-t/5*S` → `-t/(5*S)`)；
  引入 FORGETTING_TIME_SCALE=5 因子使衰减速率较论文公式 R=e^{-t/S} 温和 5 倍
- CHUNK_SIZE: 200 → 1500（原项目中文短对话 200，本测试集英文长对话
  单日可达 2000+ 字符，1500 保留充分同日上下文）；首个 add 后基于 P90
  文本长度做自适应校准（原项目固定值，无自适应性）
- 移除 ChatGLM/BELLE 专用的 stop 序列
- 搜索后持久化 memory_strength（原项目 local_doc_qa 路径缺失，forget_memory 路径有；
  本实现统一持久化，并在 _forget_at_ingestion 后 save_index 保证跨会话一致性）；
  记忆强度更新仅作用于原始命中条目（_meta_idx），非合并邻居——后者未被实际 recall
  不应享受 spacing effect 保护
- 合并逻辑: 修正原项目两处 bug：
  (a) `break` 只跳出内层 `for l` 循环导致另一方向被跳过；
  (b) `docs_len` 在正向/反向间共享，先探索的方向消耗 `chunk_size` 容量
      导致另一方向无可用空间（方向序偏置）。
  改为先收集所有同源邻居再从外向内裁剪（deque + precomputed total）
- 合并文本: 原项目直接拼接且英文模式下前缀未被剥离（混乱格式），
  改为先剥离前缀再用 _MERGED_TEXT_DELIMITER ("\x00") 分隔，检索输出时解码为 "; "
- 合并 source 字段: 原项目 `forget_memory.py` 用 memory_id（唯一 ID）导致
  相邻合并完全失效，改为 date_key（同日期共享）使合并逻辑可用
- 原项目 `similarity_search_with_score_by_vector` 中 `len(docs)` 误用了
    结果累积器变量的大小（始终为 0）而非索引总条目数。对于靠近索引前端
    的命中条目（FAISS position i 较小时），搜索范围被限制为 [0, 2i]，
    严重限制了邻居搜索。当前实现无此限制，通过 metadata 列表直接按
    source 收集邻居，覆盖整个日期的所有条目
- 分割器: 原项目使用 ChineseTextSplitter 对文档做二次切割后再入库，
  本测试集为英文长对话，ChineseTextSplitter 不适用，省略该步骤
- L2→IP 迁移: 加载旧格式 L2 索引时，自动重构向量并 L2 归一化，
  迁移至 IndexFlatIP，原项目无此逻辑（始终新建索引）
- 遗忘时机: 原项目中摘要/性格由 `summarize_memory.py` 预生成存入 JSON，
  `initial_load_forget_and_save` 在对对话条目执行遗忘的同时将已有摘要
  直接作为文档加载（摘要条目无独立的遗忘丢弃分支）；本实现先摄入全部对话、
  调用 LLM 在线生成摘要/性格，最后执行遗忘。摘要/性格属 MEMORY_SKIP_TYPES
  不受遗忘影响，两种实现功能等价：摘要内容均基于遗忘前的完整对话数据
- 合并去重: 原项目所有 top-k 结果共享全局 id_set 做跨结果去重，
  但 scores[0][j] 因 Python 循环变量残留导致所有合并文档共享同一分数；
  本实现每结果独立构建合并条目并保留各自 score，再通过子集过滤
  (_dedup_subset_results) 消除跨结果重叠，修复原版分数 bug
- 摘要存储: 原项目 `forget_memory.py` 摘要 source=`{user}_{date}_summary`（已与
  对话的 source=memory_id 不同）；但 `local_doc_qa.py` 中摘要与对话共享
  source=date 可意外合并。本实现统一使用 source=summary_{date}，与对话的
  source=date_key 明确分离，检索结果更清晰
- 检索粗排窗口: 原项目固定 VECTOR_SEARCH_TOP_K={2,3,6}；本实现使用
  top_k*4 倍率扩大粗排窗口，为邻居合并预留空间
- memory_strength 加权: 原项目无 strength 对检索分数的加权机制；本实现
  在粗排后用 memory_strength 对分数做对数级提升，使高频调用记忆获得更高排位
- 摘要向量加权: 对 daily_summary 类型条目在检索排序时给予 1.2x boost，
  利用 LLM 提取的高质量偏好信息优于原始对话片段
- 时间衰减: 原项目检索排序无时间维度；本实现基于 last_recall_date 应用
  指数衰减（exp(-days/120)，半衰期 ≈ 83 天），使近期偏好获得更高排位，
  可复现测评场景使用 reference_date 而非真实当前时间
- 嵌入维度: 原项目使用 SentenceTransformer 运行时确定维度；
  本实现默认 1536（text-embedding-3-small），首次 embedding 调用
  后动态校正，若使用其他维度模型需确保 add 先于 test 执行
- 合并密度惩罚: 若合并后文本远长于原始命中条目（噪声占主导），
  对分数做惩罚并在密度低于 0.25 时放弃合并；原项目无此机制
- 检索架构: 原实现通过 monkey-patched `similarity_search_with_score_by_vector`
  在 FAISS 内部执行邻居合并（存在上述多个 bug），再按 source 排序分组；
  本实现将检索拆分为独立的五阶段管道：FAISS 粗排 → 分数加权（strength/
  时间衰减/summary boost）→ metadata 邻居合并 → 去重过滤 → 截断 top_k。
  每阶段职责清晰、可独立调优、易于测试
- 说话人感知过滤: 原项目为固定用户↔AI 双人对话，不区分多用户身份。
  本实现为每条记忆新增 `speakers` 字段（对话参与者集合），检索时若 query 中
  提及已知用户名，对不涉及该用户的记忆施加 0.75× 降权因子，减少跨用户噪声
- 全局上下文注入: 检索时将 overall_summary 和 overall_personality 作为
  上下文前置；原项目通过 prompt 模板变量注入，此实现适配测评流程
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
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

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
# [DIFF] 原项目 CHUNK_SIZE=200，适配中文短对话（~80字符/对）。
# 本测试集为英文长对话（平均 ~272 字符/对，单日可达 2000+ 字符），
# 200 会导致合并逻辑完全失效。默认 1500 保留充分的同日上下文（约 5-6 对），
# 同时在首个 add 完成后基于实际文本长度做自适应校准。
# 可通过环境变量 MEMORYBANK_CHUNK_SIZE 覆盖。
DEFAULT_CHUNK_SIZE = 1500
CHUNK_SIZE_MIN = 200  # 原项目值，自适应下界
CHUNK_SIZE_MAX = 8192  # 自适应上限，避免将整个日期塞入单条记忆
MEMORY_SKIP_TYPES = frozenset({"daily_summary"})
_MERGED_TEXT_DELIMITER = "\x00"

# ── 数值 / 行为常量 ────────────────────────────────────────────────────────────
# 嵌入 API 配置
DEFAULT_EMBEDDING_DIM = 1536  # text-embedding-3-small 默认维度
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_BACKOFF_BASE = 2  # 指数退避基础秒数

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
FORGETTING_TIME_SCALE = 5  # exp(-t / (SCALE * S)) 中的除数

# 检索
DEFAULT_TOP_K = 5
COARSE_SEARCH_FACTOR = 4  # top_k 倍率，FAISS 粗排窗口
MEMORY_STRENGTH_BOOST = 1.0  # log1p 乘子，strength 加权
SUMMARY_VECTOR_BOOST = 1.2  # daily_summary 额外加权
NEIGHBOR_BONUS = 0.1  # 合并邻居时每个邻居的分数增量
MIN_MERGE_DENSITY = 0.25  # 低于此值放弃合并（噪声占主导）
TIME_DECAY_DAYS = 120  # exp(-天数/DAYS) 检索排序衰减，半衰期≈83天
REFERENCE_DATE_OFFSET = 1  # 最大历史时间戳后追加的天数


def _resolve_chunk_size() -> int:
    """从环境变量 MEMORYBANK_CHUNK_SIZE 解析分块大小。"""
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


CHUNK_SIZE = _resolve_chunk_size()


def _resolve_embedding_dim() -> Optional[int]:
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


def _resolve_embedding_api_key(args) -> Optional[str]:
    """获取 Embedding API 密钥，优先使用命令行参数，回退到环境变量。"""
    return getattr(args, "embedding_api_key", None) or resolve_memory_key(
        args, "EMBEDDING_API_KEY"
    )


def _resolve_embedding_api_base(args) -> Optional[str]:
    """获取 Embedding API 基础 URL，优先使用命令行参数，回退到环境变量。"""
    return getattr(args, "embedding_api_base", None) or resolve_memory_url(
        args, "EMBEDDING_API_BASE"
    )


def _resolve_reference_date() -> Optional[str]:
    """从环境变量 MEMORYBANK_REFERENCE_DATE 读取参考日期。"""
    return os.getenv("MEMORYBANK_REFERENCE_DATE")


def _word_in_text(word: str, text: str) -> bool:
    """检测 word 是否作为独立词出现在 text 中（\b 词边界）。"""
    if not word or not word.strip():
        return False
    return bool(re.search(r"\b" + re.escape(word.strip()) + r"\b", text))


_TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on", "y"})
_FALSY_TOKENS = frozenset({"0", "false", "no", "off", "n"})


def _parse_bool_token(raw: str) -> Optional[bool]:
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
        "(truthy: %s, falsy: %s); treating as False",
        name,
        value,
        sorted(_TRUTHY_TOKENS),
        sorted(_FALSY_TOKENS),
    )
    return False


def _resolve_enable_summary() -> bool:
    """从环境变量 MEMORYBANK_ENABLE_SUMMARY 读取是否启用摘要生成。"""
    return _resolve_bool_env("MEMORYBANK_ENABLE_SUMMARY", True)


def _resolve_enable_forgetting() -> bool:
    """从环境变量 MEMORYBANK_ENABLE_FORGETTING 读取是否启用遗忘机制。

    [DIFF] 原项目遗忘机制始终启用。本测评场景默认禁用，以保证结果可复现性。
    需要启用时设置 MEMORYBANK_ENABLE_FORGETTING=1。
    """
    new_val = os.getenv("MEMORYBANK_ENABLE_FORGETTING")
    if new_val is not None and new_val.strip():
        return _resolve_bool_env("MEMORYBANK_ENABLE_FORGETTING", False)
    # 兼容已弃用的 MEMORYBANK_DISABLE_FORGETTING 变量
    # (MEMORYBANK_DISABLE_FORGETTING=1 means enable_forgetting=False)
    old_val = os.getenv("MEMORYBANK_DISABLE_FORGETTING")
    if old_val is not None and old_val.strip():
        logger.warning(
            "MemoryBank: MEMORYBANK_DISABLE_FORGETTING=%r is deprecated; "
            "use MEMORYBANK_ENABLE_FORGETTING instead",
            old_val,
        )
        parsed = _parse_bool_token(old_val)
        if parsed is None:
            logger.warning(
                "MemoryBank: MEMORYBANK_DISABLE_FORGETTING=%r not recognized, "
                "defaulting to enable_forgetting=False",
                old_val,
            )
            return False
        return not parsed
    return False


def _resolve_seed() -> Optional[int]:
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

    [DIFF] 原项目 search_memory 仅去除中文前缀 `时间{date}的对话内容：`，
    英文模式下前缀不会被去除（bug）。此处正确处理英文前缀。

    Note: 当前仅支持英文前缀；VehicleMemBench 测试集为纯英文内容，
    无需中文前缀处理。
    """
    for pfx in (
        f"Conversation content on {date_part}:",
        f"The summary of the conversation on {date_part} is:",
    ):
        if text.startswith(pfx):
            return text[len(pfx) :]
    return text


def _dedup_subset_results(results: List[dict]) -> List[dict]:
    """合并结果中共享 index 或互为子集/超集的条目，消除内容重复。

    [DIFF] 原项目所有 top-k 结果在 similarity_search_with_score_by_vector
    中共享全局 id_set，跨结果去重后统一产出合并文档，但 scores[0][j]
    使用循环结束后的最后一个 j 导致所有合并文档共享同一分数。
    本实现每结果独立构建合并条目并保留各自 score，再通过本函数
    基于反向映射和并查集检测跨结果重叠，消除内容重复。
    典型场景：top-2 结果分别命中 {0,1} 和 {1,2}，合并为 {0,1,2}。
    """
    non_merging = [r for r in results if not r.get("_merged_indices")]
    merging = [r for r in results if r.get("_merged_indices")]
    if len(merging) <= 1:
        return results

    # 反向映射：每个 index → 包含它的结果下标
    idx_owners: Dict[int, List[int]] = defaultdict(list)
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

    merged: List[dict] = []
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
            r["memory_strength"] = max(
                merging[mi].get("memory_strength", 0.0) for mi in members
            )
            r["speakers"] = sorted({
                s for mi in members
                for s in (merging[mi].get("speakers") or [])
            })
            index_to_part: Dict[int, str] = {}
            for mi in members:
                parts = merging[mi].get("text", "").split(_MERGED_TEXT_DELIMITER)
                indices = merging[mi].get("_merged_indices", [])
                if len(indices) != len(parts):
                    logger.warning(
                        "MemoryBank: _dedup_subset_results text/indices "
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
            elif r.get("text", ""):
                pass  # 保留 best_idx 结果中已有的 text
            else:
                # 兜底：使用任意成员中第一个可用的 part
                r["text"] = next(iter(index_to_part.values()), "")
            merged.append(r)

    if non_merging:
        merged.extend(non_merging)
    merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    return merged


class MemoryBankClient:
    """MemoryBank: A local long-term memory system based on FAISS vector retrieval.

    Ported and adapted from MemoryBank-SiliconFriend
    (https://github.com/zhongwanjun/MemoryBank-SiliconFriend) for the
    VehicleMemBench multi-user evaluation scenario.  Uses OpenAI-compatible
    embedding APIs, supports LLM-driven daily summarisation / personality
    analysis, Ebbinghaus forgetting-curve memory decay, and multi-user
    speaker-aware retrieval filtering.
    """

    def __init__(
        self,
        *,
        embedding_api_base: str,
        embedding_api_key: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        enable_forgetting: bool = False,
        enable_summary: bool = False,
        seed: Optional[int] = None,
        reference_date: Optional[str] = None,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        store_root: str = STORE_ROOT,
    ):
        import openai as _openai

        self._store_root = store_root

        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.enable_forgetting = enable_forgetting
        self.enable_summary = enable_summary
        self.seed = seed
        self.reference_date = reference_date

        self._embedding_dim: Optional[int] = None
        self._indices: Dict[str, faiss.IndexIDMap] = {}
        self._metadata: Dict[str, List[dict]] = {}
        self._next_id: Dict[str, int] = {}
        self._rng = random.Random(seed)

        self._extra_metadata: Dict[str, dict] = {}
        self._id_to_meta_cache: Dict[str, Dict[int, int]] = {}
        # [DIFF] 说话人缓存独立于 _extra_metadata（后者由 save_index 做 JSON
        # 序列化；Python set 不可 JSON 序列化，会导致崩溃）。用 sorted list
        # 替代 set 以保证 JSON 兼容性。
        self._speakers_cache: Dict[str, List[str]] = {}

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

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用 Embedding API 将文本列表转为向量，带指数退避重试。"""
        max_retries = EMBEDDING_MAX_RETRIES
        resp = None
        for attempt in range(max_retries):
            try:
                resp = self._embedding_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model,
                )
                break
            except Exception:
                if attempt < max_retries - 1:
                    jitter = self._rng.random()
                    time.sleep(EMBEDDING_BACKOFF_BASE**attempt + jitter)
                else:
                    raise

        results: List[List[float]] = []
        for item in resp.data:
            vec = np.array(item.embedding, dtype=np.float32)
            results.append(vec.tolist())

        if len(results) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: requested {len(texts)} "
                f"but got {len(results)} from API. Check your embedding model."
            )

        if self._embedding_dim is None and results:
            self._embedding_dim = len(results[0])

        return results

    def _get_or_create_index(self, user_id: str) -> Tuple[faiss.IndexIDMap, List[dict]]:
        """获取或创建用户的 FAISS 索引和元数据列表，支持从磁盘加载已有索引。"""
        if user_id in self._indices:
            return self._indices[user_id], self._metadata[user_id]

        store_dir = _user_store_dir(user_id, self._store_root)
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "metadata.json")
        extra_path = os.path.join(store_dir, "extra_metadata.json")

        if os.path.isfile(index_path) and os.path.isfile(meta_path):
            index = faiss.read_index(index_path)
            index_rebuilt = False
            if not isinstance(index, faiss.IndexIDMap):
                dim = index.d
                # [DIFF] 原项目使用 LangChain FAISS 封装（默认 IndexFlatL2，欧氏距离）。
                # 本实现改用原生 FAISS + IndexFlatIP（内积），配合 L2 归一化
                # 等价于余弦相似度。原项目所用 SentenceTransformer 同样针对余弦相似度
                # 优化（原版选用 L2 属 suboptimal），OpenAI Embedding API 亦是如此，
                # 因此使用 IP 在所有嵌入模型下均为更正确选择。
                # [DIFF] 原项目无此 L2→IP 迁移逻辑（始终新建索引）。此处加载到旧格式
                # L2 索引时，自动将所有向量重构、L2 归一化后迁移至 IndexFlatIP。
                new_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
                n = index.ntotal
                if n > 0:
                    all_vecs = index.reconstruct_n(0, n)
                    faiss.normalize_L2(all_vecs)
                    ids = np.arange(n, dtype=np.int64)
                    new_index.add_with_ids(all_vecs, ids)
                index = new_index
                index_rebuilt = True
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for i, meta in enumerate(metadata):
                if "faiss_id" not in meta:
                    meta["faiss_id"] = i
                # [DIFF] 旧格式元数据可能缺少 source 字段，按 type 回退。
                # 原项目 source 由 memory_id 或 date 硬编码，不存在此问题。
                # 使用空串会导致不同日期的旧记录在邻居合并中被误判为同源；
                # daily_summary 必须带 summary_ 前缀以避免与同日期对话合并。
                if "source" not in meta or not meta["source"]:
                    ts = meta.get("timestamp", "")
                    date_part = (
                        ts[:DATE_PREFIX_LEN]
                        if len(ts) >= DATE_PREFIX_LEN
                        else f"_legacy_{i}"
                    )
                    if meta.get("type") == "daily_summary":
                        meta["source"] = f"summary_{date_part}"
                    else:
                        meta["source"] = date_part
            if index_rebuilt:
                if len(metadata) != n:
                    logger.warning(
                        "MemoryBank: metadata length (%d) != index size (%d) "
                        "after L2→IP rebuild for %s",
                        len(metadata),
                        n,
                        user_id,
                    )
                for i, meta in enumerate(metadata):
                    meta["faiss_id"] = i
                self._next_id[user_id] = max(n, len(metadata))
            else:
                self._next_id[user_id] = (
                    max((m["faiss_id"] for m in metadata), default=-1) + 1
                )
            if os.path.isfile(extra_path):
                with open(extra_path, "r", encoding="utf-8") as f:
                    self._extra_metadata[user_id] = json.load(f)
        else:
            # [DIFF] 原项目使用 SentenceTransformer，运行时自动确定维度。
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
        embedding: List[float],
        timestamp: str,
        extra_meta: Optional[dict] = None,
    ) -> None:
        """向用户索引中添加一条向量记录及对应元数据。"""
        index, metadata = self._get_or_create_index(user_id)
        emb_dim = len(embedding)
        if index.d != emb_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {emb_dim}-dim vector "
                f"but index expects {index.d}-dim. "
                f"Check EMBEDDING_MODEL / EMBEDDING_DIM settings, "
                f"or rebuild the index with a consistent model."
            )
        vector_id = self._allocate_id(user_id)
        vec = np.array([embedding], dtype=np.float32)
        # [DIFF] 原项目使用 L2 距离无需归一化。改用 IndexFlatIP 后必须 L2 归一化，
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
        self._speakers_cache.pop(user_id, None)  # invalidate; new speakers may exist

    def _get_effective_chunk_size(self, user_id: str) -> int:
        """返回指定用户的合并分块大小，支持自适应校准。

        [DIFF] 原项目固定 CHUNK_SIZE=200（model_config.py:51），所有对话类型
        共用同一固定阈值。本实现基于该用户 metadata 中文本长度的第 90 百分位数
        × 3 动态计算，保证一个 chunk 约容纳 3 个典型对话对（± 离群值），
        范围 [CHUNK_SIZE_MIN, CHUNK_SIZE_MAX]。英文长对话与中文短对话的文本长度
        差异显著，固定值会同时伤害两类场景（中文过度合并/英文碎片化），
        自适应校准消除了语言/数据集相关的调参负担。

        若环境变量 MEMORYBANK_CHUNK_SIZE 已显式设置，直接使用全局 CHUNK_SIZE
        （跳过自适应逻辑）。
        """
        if os.getenv("MEMORYBANK_CHUNK_SIZE") is not None:
            return CHUNK_SIZE
        metadata = self._metadata.get(user_id, [])
        if not metadata:
            return DEFAULT_CHUNK_SIZE
        lengths = sorted(len(m.get("text", "")) for m in metadata)
        n = len(lengths)
        # ceil 取整 P90：N=10 时索引 8（真 P90），int() 向下取整则会取到索引 9（最大值）
        p90_idx = max(0, math.ceil(n * 0.9) - 1)
        p90_idx = min(p90_idx, n - 1)
        p90 = lengths[p90_idx]
        candidate = max(1, p90) * 3
        return max(CHUNK_SIZE_MIN, min(CHUNK_SIZE_MAX, candidate))

    def _merge_neighbors(self, results: List[dict], user_id: str) -> List[dict]:
        """合并检索结果中来自同一来源的相邻条目，减少碎片化。"""
        # [DIFF] 原项目有三处关键 bug：
        # 1. source=memory_id（唯一 ID）→ 相邻条目永远不会共享 source，
        #    合并逻辑完全失效。本实现 source=date_key（同日期共享）。
        # 2. 邻居遍历中 `break` 只跳出内层循环，导致另一方向被跳过；
        #    且共享 total_length 导致方向序偏置。本实现改为独立双向收集
        #    + deque 从外向内裁剪，结果与迭代顺序无关。
        # 3. 合并后的文本直接拼接（无分隔符），英文模式下前缀未被剥离，
        #    格式混乱。本实现先剥离前缀再用 _MERGED_TEXT_DELIMITER 连接。
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

        merged_results: List[dict] = []

        for r, meta_idx in indexed:
            score = float(r.get("score", 0.0))
            source = r.get("source", "")

            # [DIFF] 原项目共享 total_length 导致方向序偏置（forward 先消耗空间，
            # backward 可用容量减少）。改为先收集所有同源邻居（不限 chunk_size），
            # 再从外向内裁剪至有效 chunk_size 以内，结果与迭代顺序无关。
            neighbor_indices: List[int] = [meta_idx]

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
            parts: List[str] = []
            for idx in neighbor_indices:
                t = metadata[idx].get("text", "")
                src = metadata[idx].get("source", "")
                date_part = src.removeprefix("summary_")
                t = _strip_source_prefix(t, date_part)
                parts.append(t.strip())
            # [DIFF] 原项目将合并文本直接拼接（无分隔符），英文模式下前缀未被剥离，
            # 导致合并后出现重复前缀和混乱格式。此处用 _MERGED_TEXT_DELIMITER 连接并剥离前缀。
            combined_text = _MERGED_TEXT_DELIMITER.join(parts)
            base_meta = dict(metadata[neighbor_indices[0]])
            base_meta["text"] = combined_text
            base_meta["_meta_idx"] = meta_idx  # 保留原始命中索引供 search() 更新 strength

            # [DIFF] 密度惩罚：若合并后文本远长于原始命中条目，说明邻居
            # 包含大量可能无关的内容，对分数做适度惩罚。当噪声占比过高
            # （密度 < MIN_MERGE_DENSITY）时直接放弃合并，仅保留原始命中条目，
            # 避免高噪声合并结果挤占更聚焦的命中。
            orig_text = metadata[meta_idx].get("text", "")
            orig_src = metadata[meta_idx].get("source", "")
            orig_date = orig_src.removeprefix("summary_")
            orig_stripped = _strip_source_prefix(orig_text, orig_date).strip()
            merged_clean_len = sum(len(p) for p in parts)
            if merged_clean_len > len(orig_stripped) and len(orig_stripped) > 0:
                density = len(orig_stripped) / merged_clean_len
                if density < MIN_MERGE_DENSITY:
                    # 噪声占主导，放弃合并，仅保留原始命中
                    base_meta = dict(metadata[meta_idx])
                    base_meta["_meta_idx"] = meta_idx  # 保留原始命中索引
                    base_meta["text"] = orig_stripped
                    base_meta["score"] = float(score)
                    base_meta["_raw_score"] = r.get("_raw_score", float(score))
                    base_meta["memory_strength"] = metadata[meta_idx].get(
                        "memory_strength", INITIAL_MEMORY_STRENGTH
                    )
                    base_meta["_merged_indices"] = [meta_idx]
                    merged_results.append(base_meta)
                    continue
            else:
                density = 1.0
            neighbor_bonus = 1 + NEIGHBOR_BONUS * (len(neighbor_indices) - 1)
            if float(score) > 0:
                base_meta["score"] = float(score) * density * neighbor_bonus
            else:
                base_meta["score"] = float(score)
            base_meta["_raw_score"] = r.get("_raw_score", float(score))

            base_meta["memory_strength"] = max(
                metadata[i].get("memory_strength", INITIAL_MEMORY_STRENGTH)
                for i in neighbor_indices
            )
            base_meta["_merged_indices"] = sorted(neighbor_indices)
            base_meta["speakers"] = sorted(
                {s for i in neighbor_indices
                 for s in (metadata[i].get("speakers") or [])}
            )
            merged_results.append(base_meta)

        # [DIFF] 原项目所有 top-k 结果共享一个全局 id_set，邻居索引跨结果
        # 去重后统一分割为连续组并产出合并文档（但 scores[0][j] 使用循环结束后的
        # 最后一个 j，所有合并文档共享同一分数——分数 bug）。本实现每结果独立
        # 构建合并条目并保留各自 score，然后通过子集过滤消除跨结果重叠，
        # 同时修复原版的分数 bug。
        if len(merged_results) > 1:
            merged_results = _dedup_subset_results(merged_results)

        merged_results.extend(non_indexed)
        merged_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return merged_results

    def save_index(self, user_id: str) -> None:
        """将用户的 FAISS 索引和元数据持久化到磁盘。"""
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
        extra = self._extra_metadata.get(user_id, {})
        if extra:
            with open(extra_path, "w", encoding="utf-8") as f:
                json.dump(extra, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_speaker(line: str) -> Tuple[str, str]:
        """从对话行中解析说话人和内容，格式为 "Speaker: content"。

        [DIFF] 原项目使用固定标签 `[|User|]` / `[|AI|]`（用户↔AI 双人对话）。
        本测试集为多用户车载场景（如 Gary、Justin、Patricia 等），
        需要动态解析说话人名称以保留身份信息。
        """
        colon_pos = line.find(": ")
        if colon_pos > 0:
            return line[:colon_pos].strip(), line[colon_pos + 2 :].strip()
        return "Speaker", line.strip()

    def add(self, messages: List[dict], user_id: str, timestamp: str) -> None:
        """将对话消息分对编码为向量并存入用户索引。"""
        date_key = (
            timestamp[:DATE_PREFIX_LEN]
            if len(timestamp) >= DATE_PREFIX_LEN
            else timestamp
        )
        all_entries: List[Tuple[str, str]] = []
        for msg in messages:
            content = msg.get("content", "")
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped:
                    speaker, text = self._parse_speaker(stripped)
                    text = text.replace("\x00", "")
                    all_entries.append((speaker, text))

        if not all_entries:
            return

        pair_texts: List[str] = []
        pair_speakers: List[List[str]] = []
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

        for text, emb, spks in zip(pair_texts, embeddings, pair_speakers, strict=True):
            self._add_vector(
                user_id,
                text,
                emb,
                timestamp,
                # [DIFF] 原项目 source=memory_id（每个对话独立，如 f'{user}_{date}_{i}'），
                # 导致合并逻辑实际无效。本实现 source=date_key（同日期共享），使同一日期的
                # 连续条目可在 _merge_neighbors 中合并，检索结果更连贯。
                extra_meta={
                    "source": date_key,
                    "speakers": sorted(set(spks)),
                },
            )

    def _call_llm(self, last_user_content: str) -> str:
        """调用 LLM 生成回复，带重试逻辑处理可恢复的 API 错误和上下文长度回退。"""
        if not self._llm_client:
            return ""
        max_retries = LLM_MAX_RETRIES
        content = last_user_content
        for attempt in range(max_retries):
            try:
                resp = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
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
                        {
                            "role": "system",
                            "content": "Sure, I will do my best to assist you.",
                        },
                        {"role": "user", "content": content},
                    ],
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE,
                    top_p=LLM_TOP_P,
                    frequency_penalty=LLM_FREQUENCY_PENALTY,
                    presence_penalty=LLM_PRESENCE_PENALTY,
                    # [DIFF] 原项目含 stop=["<|im_end|>", "¬人类¬"]，为 ChatGLM/
                    # BELLE 模型和中文场景专用。英文 OpenAI 兼容 API 无需设置。
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                import openai as _openai_exc

                context_exceeded = any(
                    pattern in str(exc).lower()
                    for pattern in (
                        "maximum context",
                        "context length",
                        "too long",
                        "reduce the length",
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
                    content = content[-cut_length:]
                    continue

                retryable = isinstance(
                    exc,
                    (
                        _openai_exc.APIConnectionError,
                        _openai_exc.APITimeoutError,
                        _openai_exc.RateLimitError,
                    ),
                )
                if not retryable and isinstance(exc, _openai_exc.APIStatusError):
                    retryable = exc.status_code >= 500

                if retryable and attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue

                if not retryable:
                    raise

                logger.warning(
                    "MemoryBank _call_llm failed after %d retries: %s",
                    max_retries,
                    exc,
                )
                return ""

    def _summarize(self, text: str) -> str:
        """调用 LLM 对对话文本生成摘要，聚焦车辆偏好和用户身份。"""
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

    def _generate_daily_summaries(self, user_id: str) -> None:
        """按日期聚合对话内容并生成每日摘要向量。"""
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        existing_summary_dates = {
            (m.get("source") or "").removeprefix("summary_")
            for m in metadata
            if m.get("type") == "daily_summary"
        }
        daily_texts: Dict[str, List[str]] = {}
        for meta in metadata:
            if meta.get("type") == "daily_summary":
                continue
            date_key = meta.get("source", meta.get("timestamp", "")[:DATE_PREFIX_LEN])
            if not date_key or date_key in existing_summary_dates:
                continue
            daily_texts.setdefault(date_key, []).append(meta["text"])

        for date_key, texts in sorted(daily_texts.items()):
            cleaned = [_strip_source_prefix(t, date_key).strip() for t in texts]
            combined = "\n".join(cleaned)
            summary = self._summarize(combined)
            if summary:
                summary_text = (
                    f"The summary of the conversation on {date_key} is: {summary}"
                )
                ts = f"{date_key}{DEFAULT_TIME_SUFFIX}"
                summary_emb = self._get_embeddings([summary_text])[0]
                # [DIFF] 原项目 forget_memory.py 摘要 source={user}_{date}_summary（已与
                # 对话的 source=memory_id 分离），但 local_doc_qa.py 中摘要与对话共享
                # source=date 可意外合并。本实现统一用 source=summary_{date_key}，
                # 与对话的 source=date_key 明确分离，合并/检索互不干扰。
                self._add_vector(
                    user_id,
                    summary_text,
                    summary_emb,
                    ts,
                    {"type": "daily_summary", "source": f"summary_{date_key}"},
                )

    def _generate_overall_summary(self, user_id: str) -> None:
        """基于所有每日摘要生成整体摘要，存入额外元数据。"""
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
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

        summary = self._call_llm(prompt)
        if summary:
            extra = self._extra_metadata.setdefault(user_id, {})
            extra["overall_summary"] = summary

    def _analyze_personality(self, text: str) -> str:
        """调用 LLM 分析对话中体现的用户驾驶习惯和车辆偏好。"""
        # [DIFF] 原项目 personality 分析按单个用户进行（summarize_memory.py:94-105），
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
        if not self._llm_client:
            return

        metadata = self._metadata.get(user_id, [])
        extra = self._extra_metadata.setdefault(user_id, {})
        existing_personalities = extra.setdefault("daily_personalities", {})
        daily_texts: Dict[str, List[str]] = {}
        for meta in metadata:
            if meta.get("type") == "daily_summary":
                continue
            date_key = meta.get("source", meta.get("timestamp", "")[:DATE_PREFIX_LEN])
            if not date_key or date_key in existing_personalities:
                continue
            daily_texts.setdefault(date_key, []).append(meta["text"])

        for date_key, texts in sorted(daily_texts.items()):
            cleaned = [_strip_source_prefix(t, date_key).strip() for t in texts]
            combined = "\n".join(cleaned)
            personality = self._analyze_personality(combined)
            if personality:
                existing_personalities[date_key] = personality

    def _generate_overall_personality(self, user_id: str) -> None:
        """基于每日性格分析生成整体性格画像，存入额外元数据。"""
        if not self._llm_client:
            return

        extra = self._extra_metadata.get(user_id, {})
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
            # [DIFF] 原项目为 "AI lover"（AI 伴侣场景），改为 "AI"（车载助手场景）。
            "\nPlease provide a highly concise summary of the users' vehicle "
            "preferences and driving habits, organized by user, and the most "
            "appropriate in-car response strategy for the AI assistant, "
            "summarized as:"
        )
        prompt = "".join(prompt_parts)

        personality = self._call_llm(prompt)
        if personality:
            extra = self._extra_metadata.setdefault(user_id, {})
            extra["overall_personality"] = personality

    def _forgetting_retention(self, days_elapsed: float, memory_strength: int) -> float:
        """基于艾宾浩斯遗忘曲线计算记忆保留概率。

        [DIFF] 原项目公式为 `math.exp(-t / 5*S)`（forget_memory.py:36），
        因 Python 运算符优先级实际计算为 `math.exp((-t/5) * S)`，
        导致 memory_strength 越大遗忘越多——与该方法 docstring
        "The higher the memory strength, the slower the rate of forgetting"
        以及艾宾浩斯曲线定义矛盾。此处修正为正确公式
        `math.exp(-t / (5*S))`，使 strength 越大保留率越高。

        [DIFF] 原论文公式为 R = e^{-t/S}（无系数 5）。原始代码中因子 5 已隐含
        存在（`-t / 5*S`，但因运算符优先级 bug 被 scaling 扭曲为乘法因子），
        本实现保留因子 5 作为实用调参，使遗忘曲线较论文纯公式温和 5 倍。
        这是实用性调整——论文纯公式 e^{-1/1}≈37%/天后过于激进，
        5 倍时间缩放后首日保留率约 82%，更符合实际对话场景的记忆衰减节奏。
        """
        return math.exp(
            -max(0.0, days_elapsed) / (FORGETTING_TIME_SCALE * memory_strength)
        )

    def _forget_at_ingestion(self, user_id: str) -> None:
        """在数据摄入阶段根据遗忘曲线概率性地丢弃部分记忆。"""
        if not self.enable_forgetting or not self.reference_date:
            return

        index, metadata = self._get_or_create_index(user_id)

        ref_dt = datetime.strptime(self.reference_date[:DATE_PREFIX_LEN], "%Y-%m-%d")
        ids_to_remove: List[int] = []
        indices_to_keep: List[int] = []

        for i, meta in enumerate(metadata):
            if meta.get("type") in MEMORY_SKIP_TYPES:
                indices_to_keep.append(i)
                continue

            ts_str = meta.get("last_recall_date", meta.get("timestamp", ""))[
                :DATE_PREFIX_LEN
            ]
            try:
                mem_dt = datetime.strptime(ts_str, "%Y-%m-%d")
            except ValueError:
                indices_to_keep.append(i)
                continue
            days_elapsed = (ref_dt - mem_dt).days
            strength = meta.get("memory_strength", INITIAL_MEMORY_STRENGTH)
            retention = self._forgetting_retention(days_elapsed, strength)
            if self._rng.random() > retention:
                ids_to_remove.append(meta["faiss_id"])
            else:
                indices_to_keep.append(i)

        if ids_to_remove:
            index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            self._metadata[user_id] = [metadata[i] for i in indices_to_keep]
            self._indices[user_id] = index
            self._id_to_meta_cache[user_id] = {
                m["faiss_id"]: i for i, m in enumerate(self._metadata[user_id])
            }
            # [DIFF] 遗忘后需同步 _next_id，否则后续 add() 可能分配已被回收的 ID。
            # 原项目无此场景（ingestion 后只读），本实现为防御性一致性修正。
            self._next_id[user_id] = max(
                (m["faiss_id"] for m in self._metadata[user_id]), default=-1
            ) + 1
            self._speakers_cache.pop(user_id, None)  # invalidate; speakers may be removed

    # [DIFF] 原项目 VECTOR_SEARCH_TOP_K：ChatGLM/BELLE 路径=3，
    # ChatGPT/LlamaIndex 路径=2（cli_llamaindex.py:36）。
    # 本实现取 5 以适配多事件车载场景（每个文件约 10 个事件跨越多天）。
    def search(
        self, query: str, user_id: str, top_k: int = DEFAULT_TOP_K
    ) -> List[dict]:
        """基于向量相似度检索与查询最相关的记忆，并合并相邻条目。"""
        global _warned_no_ref_date
        index, metadata = self._get_or_create_index(user_id)

        if index.ntotal == 0:
            return []

        # 仅当 reference_date 未设置时发出一次告警（replay 场景正常缺失）
        if not self.reference_date and not _warned_no_ref_date:
            _warned_no_ref_date = True
            logger.warning("MemoryBank: reference_date not set; recency decay disabled")

        query_emb = self._get_embeddings([query])[0]
        query_vec = np.array([query_emb], dtype=np.float32)
        # [DIFF] 同 _add_vector，查询向量也需 L2 归一化以保证 IP ≈ 余弦相似度。
        faiss.normalize_L2(query_vec)

        # [DIFF] 原项目固定 VECTOR_SEARCH_TOP_K={2,3,6}（取决于代码路径），
        # 本实现取 top_k*4 倍率扩大粗排窗口，为后续邻居合并预留空间。
        k = min(top_k * COARSE_SEARCH_FACTOR, index.ntotal)
        scores, indices = index.search(query_vec, k)

        id_to_meta = self._id_to_meta_cache.get(user_id, {})

        results: List[dict] = []
        for score, faiss_id in zip(scores[0], indices[0]):
            meta_idx = id_to_meta.get(int(faiss_id))
            if meta_idx is None:
                continue
            meta = dict(metadata[meta_idx])
            meta["score"] = float(score)
            meta["_raw_score"] = float(score)
            meta["_meta_idx"] = meta_idx
            results.append(meta)

        # [DIFF] 原项目无 memory_strength 加权和时间衰减机制。本实现：
        # 1. 用 memory_strength 对分数做对数级提升
        # 2. 对 daily_summary 类型条目额外给予 1.2x boost（LLM 摘要质量更高）
        # 3. 基于 last_recall_date 应用指数衰减，使近期偏好获得更高排位
        for r in results:
            if r["_raw_score"] > 0:
                ms = float(r.get("memory_strength", INITIAL_MEMORY_STRENGTH))
                boost = 1 + math.log1p(ms) * MEMORY_STRENGTH_BOOST
                if r.get("type") == "daily_summary":
                    boost *= SUMMARY_VECTOR_BOOST
                r["score"] = r["_raw_score"] * boost

                # [DIFF] 原项目无时间衰减。基于 last_recall_date 应用指数衰减，
                # 衰减常数为 120 天（半衰期 ≈ 83 天）。注：此时间衰减用于检索排序
                # （连续），与遗忘机制的 _forgetting_retention（二元保留/丢弃）参数
                # 独立——衰减常数 120 vs 5*S 服务于不同目的，非矛盾。
                if self.reference_date:
                    ts_str = r.get("last_recall_date", r.get("timestamp", ""))[
                        :DATE_PREFIX_LEN
                    ]
                    try:
                        mem_dt = datetime.strptime(ts_str, "%Y-%m-%d")
                        ref_dt = datetime.strptime(
                            self.reference_date[:DATE_PREFIX_LEN], "%Y-%m-%d"
                        )
                        days_diff = max(0.0, (ref_dt - mem_dt).days)
                        r["score"] *= math.exp(-days_diff / TIME_DECAY_DAYS)
                    except (ValueError, TypeError):
                        # Date parsing failed (malformed timestamp); skip decay for this item.
                        pass

        # [DIFF] 原项目无说话人感知过滤。本实现：提取 query 中提及的已知用户名，
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
            all_speakers = sorted(all_speakers_set)  # JSON-compatible list
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
                        r["score"] = score * 0.75 if score >= 0 else score * 1.25
            # 惩罚后重新排序
            merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        merged = merged[:top_k]

        # [DIFF] 原项目仅更新精确匹配条目的 memory_strength（forget_memory.py:63-71），
        # 被合并的邻居条目不获得 strength 提升——它们虽作为上下文返回但未被实际 recall，
        # 不应受到 spacing effect 保护（否则合并噪声将被错误强化）。
        # 本实现对所有类型条目均更新强度，但仅更新原始命中（_meta_idx），不波及合并邻居。
        for r in merged:
            mi = r.get("_meta_idx")
            if mi is not None and 0 <= mi < len(metadata):
                metadata[mi]["memory_strength"] = (
                    metadata[mi].get("memory_strength", INITIAL_MEMORY_STRENGTH)
                    + MEMORY_STRENGTH_INCREMENT
                )
                if self.reference_date:
                    metadata[mi]["last_recall_date"] = self.reference_date[
                        :DATE_PREFIX_LEN
                    ]

        for r in merged:
            r.pop("_merged_indices", None)
            r.pop("_meta_idx", None)
            r.pop("_raw_score", None)
            if "text" in r:
                r["text"] = r["text"].replace(_MERGED_TEXT_DELIMITER, "; ")

        # [DIFF] 原项目在 search 后通过 update_memory_when_searched → write_memories
        # 持久化 memory_strength 和 last_recall_date。缺少此步会导致遗忘机制跨会话失效。
        if merged:
            self.save_index(user_id)
        return merged

    def get_extra_metadata(self, user_id: str) -> dict:
        """获取用户的额外元数据（整体摘要、性格画像等）。"""
        return self._extra_metadata.get(user_id, {})


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


_warned_no_ref_date = False


def _build_client(args, user_id: str = "") -> MemoryBankClient:
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
    seed = _resolve_seed()
    reference_date = _resolve_reference_date()

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


def _compute_reference_date(history_dir: str, file_range: Optional[str]) -> str:
    """扫描历史文件中的时间戳，计算最新日期的下一天作为参考日期。"""
    # [DIFF] 原项目使用 `datetime.date.today()` 作为参考日期，可能距最后对话
    # 数周/数月（遗忘更激进）。本实现使用历史文件最新日期的下一天，使遗忘量
    # 保持合理且结果可复现，适合测评场景。
    history_files = collect_history_files(history_dir, file_range)
    max_ts: Optional[datetime] = None
    for _, path in history_files:
        for bucket in load_hourly_history(path):
            if bucket.dt is not None:
                if max_ts is None or bucket.dt > max_ts:
                    max_ts = bucket.dt
    if max_ts is None:
        max_ts = datetime.now()
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

    def processor(idx: int, history_path: str) -> Tuple[int, int, Optional[str]]:
        client = _build_client(args)
        client.reference_date = reference_date
        user_id = f"{USER_ID_PREFIX}_{idx}"
        store_dir = _user_store_dir(user_id, client._store_root)
        if os.path.isdir(store_dir):
            shutil.rmtree(store_dir)
        try:
            message_count = 0
            daily_lines: Dict[str, List[str]] = {}
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

            if _resolve_enable_summary():
                client._generate_daily_summaries(user_id)
                client._generate_overall_summary(user_id)
                client._generate_daily_personalities(user_id)
                client._generate_overall_personality(user_id)

            # [DIFF] 原项目遗忘在文档构建前执行（initial_load_forget_and_save），
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
    """Thin wrapper around MemoryBankClient for the evaluation pipeline.

    Adds overall_summary and overall_personality context as a synthetic
    first result so the agent's LLM sees global context before ranked
    per-query hits.
    """

    def __init__(self, client: MemoryBankClient, user_id: str):
        self._client = client
        self._user_id = user_id

    def search(
        self, query: str, user_id: Optional[str] = None, top_k: int = DEFAULT_TOP_K
    ) -> List[dict]:
        """检索记忆并附带整体摘要和性格画像。

        [DIFF] 原项目通过 prompt 模板变量 {history_summary} 和 {personality}
        注入整体上下文，不纳入检索结果。本测评流程中 agent 通过 tool call 获取
        记忆，将 overall_summary 和 overall_personality 作为额外条目插入到
        检索结果头部——因为 LLM 读取搜索结果时倾向于关注前几条高相关度条目，
        将全局上下文放在头部确保其被优先消费。
        """
        uid = user_id if user_id is not None else self._user_id
        results = list(self._client.search(query=query, user_id=uid, top_k=top_k))

        extra = self._client.get_extra_metadata(uid)
        overall_summary = extra.get("overall_summary", "")
        overall_personality = extra.get("overall_personality", "")

        if overall_summary or overall_personality:
            parts = []
            if overall_summary:
                parts.append(f"Overall summary of past memories: {overall_summary}")
            if overall_personality:
                parts.append(
                    f"User vehicle preferences and habits: {overall_personality}"
                )
            # 全局摘要/性格画像插入列表头部而非尾部：LLM 读取工具调用结果时
            # 倾向于优先关注前几条高相关度条目，将全局上下文前置确保其被优先消费
            results.insert(
                0,
                {
                    "_type": "overall_context",
                    "text": "\n".join(parts),
                    "source": "overall",
                    "memory_strength": INITIAL_MEMORY_STRENGTH,
                    "score": 0.0,
                },
            )

        return results


def close_test_state(shared_state: Any) -> None:
    """清理测试状态（MemoryBank 无需清理）。"""
    del shared_state


def is_test_sequential() -> bool:
    """MemoryBank 测试支持并行执行。"""
    return False


def format_search_results(search_result: Any) -> Tuple[str, int]:
    """将检索结果格式化为带编号的文本，按相关性顺序分组并标注记忆强度。"""
    if not isinstance(search_result, list):
        return "", 0
    if not search_result:
        return "", 0

    overall_items = [r for r in search_result if r.get("_type") == "overall_context"]
    non_overall = [r for r in search_result if r.get("_type") != "overall_context"]

    groups: List[Tuple[str, str, List[dict]]] = []
    for item in non_overall:
        text = item.get("text", "")
        date_part = (item.get("source") or "").removeprefix("summary_")
        text = _strip_source_prefix(text, date_part).strip()

        if not groups or groups[-1][0] != date_part:
            groups.append((date_part, text, [item]))
        else:
            groups[-1] = (
                groups[-1][0],
                groups[-1][1] + "\n" + text,
                groups[-1][2] + [item],
            )

    lines: List[str] = []
    # [DIFF] 整体摘要/性格画像作为全局上下文前置，再按日期列出检索到的记忆片段。
    for idx, item in enumerate(overall_items, 1):
        lines.append(
            f"{idx}. [memory_strength={item.get('memory_strength', 1)}] {item.get('text', '')}"
        )

    group_start = len(overall_items) + 1
    for idx, (date_part, combined_text, items) in enumerate(groups, group_start):
        max_strength = max(
            it.get("memory_strength", INITIAL_MEMORY_STRENGTH) for it in items
        )
        date_info = f" [date={date_part}]" if date_part else ""
        lines.append(
            f"{idx}. [memory_strength={max_strength}]{date_info} {combined_text}"
        )

    return "\n\n".join(lines), len(non_overall)
