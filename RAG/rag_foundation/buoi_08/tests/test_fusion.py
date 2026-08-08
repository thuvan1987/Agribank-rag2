"""
Unittests cho Reciprocal Rank Fusion (RRF) và Hybrid Search (Buổi 08).
Thực thi offline không tải model Reranker và không gọi LLM Generation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Thêm buoi_08 vào sys.path để import advanced_rag
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from advanced_rag import hybrid_retrieval, rrf_fusion


class TestReciprocalRankFusion(unittest.TestCase):

    def setUp(self):
        """Khởi tạo sample BM25 và Semantic candidates cho testing."""
        self.bm25_sample = [
            {
                "chunk_id": "chunk-01",
                "text": "Nội dung văn bản 1",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "bm25_rank": 1,
                "bm25_score": 5.0,
                "strategy": "hierarchical"
            },
            {
                "chunk_id": "chunk-02",
                "text": "Nội dung văn bản 2",
                "source": "doc.pdf",
                "page_start": 2,
                "page_end": 2,
                "bm25_rank": 2,
                "bm25_score": 3.0,
                "strategy": "hierarchical"
            }
        ]

        self.semantic_sample = [
            {
                "chunk_id": "chunk-02",
                "text": "Nội dung văn bản 2",
                "source": "doc.pdf",
                "page_start": 2,
                "page_end": 2,
                "semantic_rank": 1,
                "semantic_distance": 0.1,
                "strategy": "hierarchical"
            },
            {
                "chunk_id": "chunk-03",
                "text": "Nội dung văn bản 3",
                "source": "doc.pdf",
                "page_start": 3,
                "page_end": 3,
                "semantic_rank": 2,
                "semantic_distance": 0.3,
                "strategy": "hierarchical"
            }
        ]

    def test_01_rrf_formula_arithmetic(self):
        """Test 1: Công thức RRF tính chính xác giá trị số học."""
        # k = 60, w_bm25 = 1.0, w_sem = 1.0
        # chunk-02 xuất hiện ở BM25 rank 2 và Semantic rank 1
        # RRF_Score(chunk-02) = 1.0 / (60 + 2) + 1.0 / (60 + 1) = 1/62 + 1/61
        expected_score = (1.0 / 62.0) + (1.0 / 61.0)
        fused = rrf_fusion(self.bm25_sample, self.semantic_sample, k=60, bm25_weight=1.0, semantic_weight=1.0)
        
        # Find chunk-02
        cand_02 = next(c for c in fused if c["chunk_id"] == "chunk-02")
        self.assertAlmostEqual(cand_02["rrf_score"], expected_score, places=6)
        self.assertEqual(cand_02["fused_rank"], 1)

    def test_02_overlap_no_duplicates(self):
        """Test 2: Candidate trùng khớp giữa 2 nhánh (overlap) không bị nhân bản (duplicate)."""
        fused = rrf_fusion(self.bm25_sample, self.semantic_sample, k=60)
        chunk_ids = [c["chunk_id"] for c in fused]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertEqual(len(fused), 3) # chunk-01, chunk-02, chunk-03

    def test_03_bm25_only_candidate_preserved(self):
        """Test 3: Candidate chỉ xuất hiện ở nhánh BM25 vẫn được bảo toàn."""
        fused = rrf_fusion(self.bm25_sample, self.semantic_sample, k=60)
        cand_01 = next(c for c in fused if c["chunk_id"] == "chunk-01")
        self.assertEqual(cand_01["matched_by"], ["bm25"])
        self.assertIsNotNone(cand_01["bm25_rank"])
        self.assertIsNone(cand_01["semantic_rank"])

    def test_04_semantic_only_candidate_preserved(self):
        """Test 4: Candidate chỉ xuất hiện ở nhánh Semantic vẫn được bảo toàn."""
        fused = rrf_fusion(self.bm25_sample, self.semantic_sample, k=60)
        cand_03 = next(c for c in fused if c["chunk_id"] == "chunk-03")
        self.assertEqual(cand_03["matched_by"], ["semantic"])
        self.assertIsNone(cand_03["bm25_rank"])
        self.assertIsNotNone(cand_03["semantic_rank"])

    def test_05_weight_zero_excludes_branch(self):
        """Test 5: Trọng số bằng 0 loại bỏ đóng góp của nhánh tương ứng."""
        # Đặt bm25_weight = 0.0, chỉ dùng semantic
        fused = rrf_fusion(self.bm25_sample, self.semantic_sample, k=60, bm25_weight=0.0, semantic_weight=1.0)
        cand_01 = next(c for c in fused if c["chunk_id"] == "chunk-01") # chỉ có BM25
        self.assertEqual(cand_01["rrf_score"], 0.0)

    def test_06_deterministic_tie_breaking(self):
        """Test 6: Quy tắc Tie-breaking hoạt động ổn định và nhất quán."""
        # Tạo 2 candidate có cùng RRF score
        bm25_t = [
            {"chunk_id": "chunk-b", "text": "Text B", "source": "d.pdf", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 2.0},
            {"chunk_id": "chunk-a", "text": "Text A", "source": "d.pdf", "page_start": 1, "page_end": 1, "bm25_rank": 1, "bm25_score": 2.0}
        ]
        fused = rrf_fusion(bm25_t, [], k=60)
        # Vì cả 2 có cùng RRF score và rank, tie-break theo chunk_id tăng dần ('chunk-a' trước 'chunk-b')
        self.assertEqual(fused[0]["chunk_id"], "chunk-a")
        self.assertEqual(fused[1]["chunk_id"], "chunk-b")

    def test_07_metadata_mismatch_fails(self):
        """Test 7: Bắt lỗi ValueError khi trùng chunk_id nhưng metadata (text/source) không khớp."""
        bad_semantic = [
            {
                "chunk_id": "chunk-01",
                "text": "Nội dung bị sai lệch khác hoàn toàn BM25",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "semantic_rank": 1,
                "semantic_distance": 0.1
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            rrf_fusion(self.bm25_sample, bad_semantic, k=60)
        self.assertIn("mismatch", str(ctx.exception).lower())

    @patch("advanced_rag.semantic_retrieval")
    @patch("advanced_rag.bm25_retrieval")
    @patch("rag.load_chunks")
    def test_08_trace_counts_correct(self, mock_load_chunks, mock_bm25, mock_sem):
        """Test 8: Kiểm tra các chỉ số đếm trong pipeline trace hoàn toàn chính xác."""
        mock_load_chunks.return_value = ([], {"files_read": 1})
        mock_bm25.return_value = self.bm25_sample # len = 2
        mock_sem.return_value = self.semantic_sample # len = 2

        res = hybrid_retrieval(question="Hỏi đáp test", strategy="hierarchical")
        trace = res["trace"]

        self.assertEqual(trace["bm25_candidate_count"], 2)
        self.assertEqual(trace["semantic_candidate_count"], 2)
        self.assertEqual(trace["overlap_count"], 1) # chunk-02
        self.assertEqual(trace["union_count"], 3) # chunk-01, 02, 03
        self.assertEqual(trace["fused_count"], 3)

    @patch("advanced_rag.semantic_retrieval")
    @patch("advanced_rag.bm25_retrieval")
    @patch("rag.load_chunks")
    def test_09_hybrid_calls_each_retriever_once(self, mock_load_chunks, mock_bm25, mock_sem):
        """Test 9: Hybrid pipeline chỉ gọi BM25 retriever và Semantic retriever đúng một lần."""
        mock_load_chunks.return_value = ([], {"files_read": 1})
        mock_bm25.return_value = self.bm25_sample
        mock_sem.return_value = self.semantic_sample

        hybrid_retrieval(question="Kiểm tra số lần gọi", strategy="hierarchical")

        mock_bm25.assert_called_once()
        mock_sem.assert_called_once()

    @patch("advanced_rag.semantic_retrieval")
    @patch("advanced_rag.bm25_retrieval")
    @patch("rag.load_chunks")
    def test_10_no_reranker_and_no_generation(self, mock_load_chunks, mock_bm25, mock_sem):
        """Test 10: Đảm bảo Hybrid stage không nạp Reranker model và không gọi LLM Answer Generation."""
        mock_load_chunks.return_value = ([], {"files_read": 1})
        mock_bm25.return_value = self.bm25_sample
        mock_sem.return_value = self.semantic_sample

        with patch("advanced_rag.check_reranker_cache_exists") as mock_reranker_cache:
            with patch("google.genai.Client") as mock_genai:
                res = hybrid_retrieval(question="Hỏi đáp", strategy="hierarchical")
                
                mock_reranker_cache.assert_not_called()
                mock_genai.assert_not_called()
                self.assertIn("candidates", res)
                self.assertIn("trace", res)


if __name__ == "__main__":
    unittest.main()
