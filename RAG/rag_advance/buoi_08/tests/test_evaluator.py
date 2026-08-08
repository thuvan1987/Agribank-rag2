"""
Unittests cho Metric Calculator & Evaluator (Buổi 08).
Thực thi 100% offline với ví dụ số học tính tay được.
"""

import json
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Thêm buoi_08 vào sys.path để import evaluate
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluate import (
    calculate_mrr_at_k,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    run_evaluation_benchmark,
)


class TestEvaluatorMetrics(unittest.TestCase):

    def test_01_recall_at_k_hand_calculated(self):
        """Test 1: Recall@K với ví dụ tính tay (Retrieved 1/2 gold items => Recall = 0.5)."""
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        gold = ["c2", "c6"]
        # Top 5 retrieved chứa 'c2' (1 hit out of 2 gold) -> Recall@5 = 1 / 2 = 0.5
        score = calculate_recall_at_k(retrieved, gold, k=5)
        self.assertEqual(score, 0.5)

    def test_02_mrr_at_k_hand_calculated(self):
        """Test 2: MRR@K với ví dụ tính tay (First hit ở rank 2 => MRR = 1/2 = 0.5)."""
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        gold = ["c2", "c6"]
        # Match đầu tiên 'c2' ở vị trí index 1 (rank 2) -> MRR@5 = 1 / 2 = 0.5
        score = calculate_mrr_at_k(retrieved, gold, k=5)
        self.assertEqual(score, 0.5)

    def test_03_ndcg_at_k_hand_calculated(self):
        """Test 3: nDCG@K với ví dụ số học tính tay."""
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        gold = ["c2", "c6"]
        # Match ở rank 2 (c2): rel = 1 -> DCG@5 = 1 / log2(2 + 1) = 1 / log2(3) = 0.63092975
        # Ideal hits: rank 1 (1 / log2(2) = 1.0) và rank 2 (1 / log2(3) = 0.63092975) -> IDCG = 1.63092975
        # nDCG = 0.63092975 / 1.63092975 = 0.3868528
        expected_ndcg = (1.0 / math.log2(3)) / (1.0 + (1.0 / math.log2(3)))
        score = calculate_ndcg_at_k(retrieved, gold, k=5)
        self.assertAlmostEqual(score, expected_ndcg, places=6)

    def test_04_empty_gold_or_retrieved_returns_zero(self):
        """Test 4: Tập gold hoặc retrieved rỗng trả về điểm 0.0."""
        self.assertEqual(calculate_recall_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(calculate_recall_at_k(["c1"], [], k=5), 0.0)
        self.assertEqual(calculate_mrr_at_k([], ["c1"], k=5), 0.0)
        self.assertEqual(calculate_ndcg_at_k([], ["c1"], k=5), 0.0)

    def test_05_evaluation_warns_human_review(self):
        """Test 5: Benchmark ghi cảnh báo khi tập dataset gold chứa `needs_human_review = True`."""
        sample_q = [
            {
                "query_id": "Q01",
                "question": "Thử nghiệm L/C?",
                "relevant_chunk_ids": ["c1"],
                "needs_human_review": True
            }
        ]

        def mock_bm25(query, chunks, top_k):
            return [{"chunk_id": "c1"}]

        with patch("evaluate.json.load", return_value=sample_q):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("rag.load_chunks", return_value=([], {})):
                    report = run_evaluation_benchmark(
                        questions_path=str(BASE_DIR / "eval" / "questions.json"),
                        modes=["bm25"],
                        k=5
                    )

        self.assertTrue(report["needs_human_review_warning"])
        self.assertIn("bm25", report["metrics_by_mode"])

    def test_06_report_file_saved(self):
        """Test 6: Báo cáo được xuất thành file JSON trong thư mục reports/."""
        sample_q = [
            {
                "query_id": "Q01",
                "question": "Query test?",
                "relevant_chunk_ids": ["c1"],
                "needs_human_review": False
            }
        ]

        def mock_bm25(query, chunks, top_k):
            return [{"chunk_id": "c1"}]

        with patch("evaluate.json.load", return_value=sample_q):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("rag.load_chunks", return_value=([], {})):
                    report = run_evaluation_benchmark(
                        questions_path=str(BASE_DIR / "eval" / "questions.json"),
                        modes=["bm25"],
                        k=5
                    )

        out_path = BASE_DIR / "reports" / f"eval_report_{report['strategy']}_k5.json"
        self.assertTrue(out_path.exists())

    def test_07_evaluator_never_calls_generation(self):
        """Test 7: Benchmark KHÔNG GỌI LLM Generation bất kỳ lần nào."""
        sample_q = [
            {
                "query_id": "Q01",
                "question": "No generation test?",
                "relevant_chunk_ids": ["c1"],
                "needs_human_review": False
            }
        ]

        def mock_bm25(query, chunks, top_k):
            return [{"chunk_id": "c1"}]

        with patch("evaluate.json.load", return_value=sample_q):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("rag.load_chunks", return_value=([], {})):
                    with patch("google.genai.Client") as mock_genai:
                        report = run_evaluation_benchmark(
                            questions_path=str(BASE_DIR / "eval" / "questions.json"),
                            modes=["bm25"],
                            k=5
                        )
                        mock_genai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
