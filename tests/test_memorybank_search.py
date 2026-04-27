"""Unit tests for MemoryBank search score optimizations."""
import tempfile
from unittest.mock import patch

import faiss
import numpy as np
import pytest

from evaluation.memorysystems import memorybank
from evaluation.memorysystems.memorybank import (
    DEFAULT_EMBEDDING_MODEL,
    MemoryBankClient,
)


class TestSearchScoreOptimizations:
    """Verify boost/decay logic in MemoryBankClient.search()."""

    _FAKE_DIM = 8
    _USER_ID = "test_user"

    @pytest.fixture(autouse=True)
    def _reset_warning_flag(self):
        """Reset module-level warning flag before each test for isolation."""
        memorybank._warned_no_ref_date = False
        yield
        memorybank._warned_no_ref_date = False

    @pytest.fixture
    def client(self):
        """Create a MemoryBankClient with a pre-populated index (3 entries)."""
        store_root = tempfile.mkdtemp()
        client = MemoryBankClient(
            embedding_api_base="http://localhost/fake",
            embedding_api_key="sk-fake",
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            store_root=store_root,
            reference_date="2026-04-27",
        )
        client._embedding_dim = self._FAKE_DIM

        user_id = self._USER_ID
        index, metadata = client._get_or_create_index(user_id)

        vec = np.ones((1, self._FAKE_DIM), dtype=np.float32)
        faiss.normalize_L2(vec)

        entries = [
            {
                "text": "entry_0",
                "timestamp": "2026-01-01T00:00:00",
                "last_recall_date": "2026-01-01",
                "memory_strength": 1,
                "source": "2026-01-01",
                "type": "daily_summary",
            },
            {
                "text": "entry_1",
                "timestamp": "2026-03-05T00:00:00",
                "last_recall_date": "2026-03-05",
                "memory_strength": 1,
                "source": "2026-03-05",
            },
            {
                "text": "entry_2",
                "timestamp": "2026-04-15T00:00:00",
                "last_recall_date": "2026-04-15",
                "memory_strength": 1,
                "source": "2026-04-15",
            },
        ]

        for i, entry in enumerate(entries):
            vid = client._allocate_id(user_id)
            index.add_with_ids(vec, np.array([vid], dtype=np.int64))
            entry["faiss_id"] = vid
            metadata.append(entry)
            cache = client._id_to_meta_cache.setdefault(user_id, {})
            cache[vid] = len(metadata) - 1

        yield client, user_id

    def test_summary_boost_ranks_first_without_decay(self, client):
        """With recency disabled, daily_summary should outrank equal-similarity
        non-summary entries."""
        client_obj, user_id = client
        client_obj.reference_date = None  # disable recency decay

        with patch.object(client_obj, "_get_embeddings") as mock_emb:
            mock_emb.return_value = [[1.0] * self._FAKE_DIM]
            results = client_obj.search("test query", user_id, top_k=3)

        assert len(results) >= 2, f"Expected >=2 results, got {len(results)}"
        assert results[0].get("type") == "daily_summary", (
            "summary entry (1.2x boost) should rank first when recency is disabled"
        )

    def test_recency_decay_prefers_recent(self, client):
        """More recent entries should outrank older entries with equal similarity."""
        client_obj, user_id = client

        with patch.object(client_obj, "_get_embeddings") as mock_emb:
            mock_emb.return_value = [[1.0] * self._FAKE_DIM]
            results = client_obj.search("test query", user_id, top_k=3)

        recent_pos = next(
            i
            for i, r in enumerate(results)
            if r.get("source") == "2026-04-15"
        )
        older_pos = next(
            i
            for i, r in enumerate(results)
            if r.get("source") == "2026-03-05"
        )
        assert recent_pos < older_pos, (
            f"recent (pos {recent_pos}) should rank before older (pos {older_pos})"
        )

    def test_no_crash_without_reference_date(self, client, caplog):
        """search() should not crash when reference_date is None, and emit one warning."""
        client_obj, user_id = client
        client_obj.reference_date = None

        with patch.object(client_obj, "_get_embeddings") as mock_emb:
            mock_emb.return_value = [[1.0] * self._FAKE_DIM]
            results = client_obj.search("test query", user_id, top_k=3)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert "reference_date not set" in caplog.text
