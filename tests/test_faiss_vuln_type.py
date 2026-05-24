"""Tests for the vuln_type extension in FAISSIndexManager."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from faiss_index import FAISSIndexManager


class TestVulnTypeClassification:
    def setup_method(self):
        self.manager = FAISSIndexManager(embed_dim=64)
        metadata = [
            {"idx": 0, "label": 1, "vuln_type": "buffer_overflow", "func_snippet": "strcpy(buf, input)"},
            {"idx": 1, "label": 1, "vuln_type": "buffer_overflow", "func_snippet": "memcpy(dst, src, len)"},
            {"idx": 2, "label": 1, "vuln_type": "buffer_overflow", "func_snippet": "sprintf(buf, fmt)"},
            {"idx": 3, "label": 1, "vuln_type": "use_after_free", "func_snippet": "free(ptr); *ptr"},
            {"idx": 4, "label": 0, "func_snippet": "safe_code()"},
            {"idx": 5, "label": 1, "vuln_type": "integer_overflow", "func_snippet": "a + b"},
        ]
        rng = np.random.RandomState(42)
        embeddings = rng.randn(6, 64).astype(np.float32)
        # Make first 3 embeddings similar to each other (buffer overflow cluster)
        embeddings[0:3] = embeddings[0] + rng.randn(3, 64).astype(np.float32) * 0.1
        self.manager.build_index(embeddings, metadata)
        self.query = embeddings[0].copy()

    def test_compute_vuln_type(self):
        vuln_type, confidence = self.manager.compute_vuln_type(self.query, top_k=5)
        assert vuln_type == "buffer_overflow"
        assert confidence > 0.0

    def test_compute_vuln_type_empty_index(self):
        empty = FAISSIndexManager(embed_dim=64)
        vuln_type, conf = empty.compute_vuln_type(np.random.randn(64).astype(np.float32))
        assert vuln_type == "unknown"
        assert conf == 0.0

    def test_compute_vuln_type_no_vuln_neighbors(self):
        manager = FAISSIndexManager(embed_dim=64)
        metadata = [
            {"idx": 0, "label": 0, "func_snippet": "safe1"},
            {"idx": 1, "label": 0, "func_snippet": "safe2"},
        ]
        embeddings = np.random.randn(2, 64).astype(np.float32)
        manager.build_index(embeddings, metadata)
        vuln_type, conf = manager.compute_vuln_type(embeddings[0], top_k=2)
        assert vuln_type == "unknown"
        assert conf == 0.0

    def test_compute_vuln_type_with_confidence(self):
        manager = FAISSIndexManager(embed_dim=64)
        rng = np.random.RandomState(123)
        metadata = [
            {"idx": i, "label": 1, "vuln_type": "buffer_overflow"} for i in range(5)
        ]
        embeddings = rng.randn(5, 64).astype(np.float32)
        manager.build_index(embeddings, metadata)

        vuln_type, conf = manager.compute_vuln_type(embeddings[0], top_k=5)
        assert vuln_type == "buffer_overflow"
        assert conf == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
