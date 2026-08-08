"""
Unittests cho Cross-Encoder Multilingual Reranker (Buổi 08).
Thực thi 100% offline sử dụng Mock / Fake Reranker Injection, không tải model từ Internet.
"""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Thêm buoi_08 vào sys.path để import advanced_rag
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import advanced_rag
from advanced_rag import check_advanced_status, rerank_candidates


class TestCrossEncoderReranker(unittest.TestCase):

    def setUp(self):
        """Khởi tạo sample candidates đã qua RRF fusion cho testing."""
        self.sample_fused_candidates = [
            {
                "chunk_id": "chunk-01",
                "text": "Nội dung văn bản 1 về L/C không hủy ngang.",
                "source": "QuyDinh.pdf",
                "page_start": 1,
                "page_end": 1,
                "fused_rank": 1,
                "rrf_score": 0.032,
                "matched_by": ["bm25", "semantic"]
            },
            {
                "chunk_id": "chunk-02",
                "text": "Nội dung văn bản 2 về bảo hiểm hàng hóa CIF.",
                "source": "QuyDinh.pdf",
                "page_start": 2,
                "page_end": 2,
                "fused_rank": 2,
                "rrf_score": 0.016,
                "matched_by": ["bm25"]
            },
            {
                "chunk_id": "chunk-03",
                "text": "Nội dung văn bản 3 về vận đơn đường biển B/L.",
                "source": "QuyDinh.pdf",
                "page_start": 3,
                "page_end": 3,
                "fused_rank": 3,
                "rrf_score": 0.015,
                "matched_by": ["semantic"]
            }
        ]

    def test_01_lazy_loading(self):
        """Test 1: Model không bị load khi import module hoặc chạy status check."""
        with patch("advanced_rag.load_reranker_model") as mock_load:
            st = check_advanced_status(strategy="hierarchical")
            mock_load.assert_not_called()
            self.assertIn("reranker_model", st)

    def test_02_one_pair_per_candidate(self):
        """Test 2: Mỗi candidate chunk tạo ra đúng một cặp (query, text) để rerank."""
        received_pairs = []

        def mock_fn(query, texts, cfg):
            for t in texts:
                received_pairs.append((query, t))
            return [(2.0, 1.0 / (1.0 + math.exp(-2.0)))] * len(texts)

        results = rerank_candidates("Câu hỏi", self.sample_fused_candidates, top_k=3, reranker_fn=mock_fn)
        self.assertEqual(len(received_pairs), len(self.sample_fused_candidates))
        self.assertEqual(received_pairs[0][0], "Câu hỏi")
        self.assertEqual(received_pairs[0][1], self.sample_fused_candidates[0]["text"])

    def test_03_batch_preserves_count(self):
        """Test 3: Xử lý theo batch giữ nguyên chính xác số lượng candidate."""
        def mock_fn(query, texts, cfg):
            return [(float(i), 1.0 / (1.0 + math.exp(-i))) for i in range(len(texts))]

        results = rerank_candidates("Query", self.sample_fused_candidates, top_k=3, reranker_fn=mock_fn)
        self.assertEqual(len(results), 3)

    def test_04_sigmoid_score_calculation(self):
        """Test 4: Tính toán Sigmoid Score chính xác 1 / (1 + exp(-logit))."""
        raw_logit = 2.0
        expected_sig = 1.0 / (1.0 + math.exp(-raw_logit))

        def mock_fn(query, texts, cfg):
            return [(raw_logit, expected_sig)]

        results = rerank_candidates("Query", [self.sample_fused_candidates[0]], top_k=1, reranker_fn=mock_fn)
        self.assertAlmostEqual(results[0]["rerank_raw_score"], raw_logit)
        self.assertAlmostEqual(results[0]["rerank_score"], expected_sig, places=6)

    def test_05_sort_and_tie_breaking(self):
        """Test 5: Sắp xếp theo rerank_score giảm dần, sau đó fused_rank tăng dần."""
        def mock_fn(query, texts, cfg):
            # c1: 0.5, c2: 0.9, c3: 0.5
            return [(0.5, 0.622), (2.0, 0.880), (0.5, 0.622)]

        results = rerank_candidates("Query", self.sample_fused_candidates, top_k=3, reranker_fn=mock_fn)
        # c2 có score cao nhất (0.880) => rank 1
        # c1 và c3 hòa score (0.622), c1 có fused_rank=1 < c3 có fused_rank=3 => c1 xếp trước c3
        self.assertEqual(results[0]["chunk_id"], "chunk-02")
        self.assertEqual(results[1]["chunk_id"], "chunk-01")
        self.assertEqual(results[2]["chunk_id"], "chunk-03")

    def test_06_rank_change_calculation(self):
        """Test 6: Tính toán rank_change = fused_rank - rerank_rank chính xác."""
        def mock_fn(query, texts, cfg):
            # Giả lập c3 (fused #3) vọt lên đứng đầu rerank #1 => rank_change = 3 - 1 = +2
            return [(0.1, 0.52), (0.2, 0.55), (3.0, 0.95)]

        results = rerank_candidates("Query", self.sample_fused_candidates, top_k=3, reranker_fn=mock_fn)
        c3 = results[0] # chunk-03 vọt lên top 1
        self.assertEqual(c3["chunk_id"], "chunk-03")
        self.assertEqual(c3["fused_rank"], 3)
        self.assertEqual(c3["rerank_rank"], 1)
        self.assertEqual(c3["rank_change"], 2) # 3 - 1 = +2

    def test_07_rerank_candidates_limit(self):
        """Test 7: Chỉ rerank tối đa RERANK_CANDIDATES phần tử đầu từ danh sách fused."""
        many_candidates = [
            {"chunk_id": f"c-{i}", "text": f"Text {i}", "fused_rank": i + 1}
            for i in range(10)
        ]
        base_cfg = advanced_rag.get_advanced_config()
        base_cfg["rerank_candidates"] = 3
        base_cfg["final_top_k"] = 3

        def mock_fn(query, texts, cfg):
            self.assertEqual(len(texts), 3) # Chỉ nhận 3 candidate đầu
            return [(1.0, 0.73)] * len(texts)

        results = rerank_candidates("Query", many_candidates, top_k=3, custom_config=base_cfg, reranker_fn=mock_fn)
        self.assertEqual(len(results), 3)

    def test_08_returns_final_top_k_only(self):
        """Test 8: Chỉ trả về tối đa FINAL_TOP_K kết quả sau khi rerank."""
        base_cfg = advanced_rag.get_advanced_config()
        base_cfg["rerank_candidates"] = 5
        base_cfg["final_top_k"] = 2

        def mock_fn(query, texts, cfg):
            return [(float(i), 0.5) for i in range(len(texts))]

        results = rerank_candidates("Query", self.sample_fused_candidates, top_k=2, custom_config=base_cfg, reranker_fn=mock_fn)
        self.assertEqual(len(results), 2)

    def test_09_model_loading_failure_raises_exception(self):
        """Test 9: Lỗi nạp model không được tự động silent fallback sang RRF."""
        with patch("advanced_rag.load_reranker_model", side_effect=RuntimeError("reranker_unavailable: Cannot load model")):
            with self.assertRaises(RuntimeError) as ctx:
                rerank_candidates("Query", self.sample_fused_candidates, top_k=2)
            self.assertIn("reranker_unavailable", str(ctx.exception))

    def test_10_offline_execution(self):
        """Test 10: Toàn bộ suite test chạy hoàn toàn offline không tải model hay mở kết nối mạng."""
        def mock_fn(query, texts, cfg):
            return [(1.0, 0.73)] * len(texts)

        results = rerank_candidates("Query offline", self.sample_fused_candidates, top_k=2, reranker_fn=mock_fn)
        self.assertEqual(len(results), 2)
        self.assertIn("rerank_rank", results[0])
        self.assertIn("rank_change", results[0])


if __name__ == "__main__":
    unittest.main()
