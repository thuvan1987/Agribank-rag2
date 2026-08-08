"""
Unittests cho Answer Pipeline, Confidence Gating, Citations & Multi-mode Comparison (Buổi 08).
Thực thi 100% offline sử dụng Mock / Fake LLM & Reranker Injection.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Thêm buoi_08 vào sys.path để import advanced_rag
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import advanced_rag
from advanced_rag import compare_retrieval_modes, query_advanced_rag


class TestAnswerPipelineAndComparison(unittest.TestCase):

    def setUp(self):
        """Khởi tạo sample evidence cho testing."""
        self.sample_evidence = [
            {
                "chunk_id": "chunk-01",
                "text": "Ngân hàng mở L/C có nghĩa vụ thanh toán không hủy ngang.",
                "source": "LuatNganHang.pdf",
                "page_start": 5,
                "page_end": 5,
                "fused_rank": 1,
                "rerank_score": 0.85,
                "semantic_distance": 0.20,
                "bm25_score": 4.5
            },
            {
                "chunk_id": "chunk-02",
                "text": "Điều khoản bảo hiểm CIP Incoterms 2020.",
                "source": "Incoterms.pdf",
                "page_start": 10,
                "page_end": 10,
                "fused_rank": 2,
                "rerank_score": 0.30, # Bị loại ở hybrid_rerank gate (0.30 < 0.50)
                "semantic_distance": 0.50, # Bị loại ở semantic gate (0.50 > 0.45)
                "bm25_score": 2.0
            }
        ]

    def test_01_gating_by_mode(self):
        """Test 1: Gating đúng theo từng mode (rerank_score >= 0.50 cho hybrid_rerank, distance <= 0.45 cho semantic)."""
        def mock_sem(query, strategy, top_k, custom_config):
            return [dict(self.sample_evidence[0]), dict(self.sample_evidence[1])]

        def mock_bm25(query, chunks, top_k):
            return [dict(self.sample_evidence[0])]

        def mock_rerank(query, candidates, top_k, custom_config, reranker_fn):
            return [dict(self.sample_evidence[0]), dict(self.sample_evidence[1])]

        with patch("advanced_rag.semantic_retrieval", side_effect=mock_sem):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("advanced_rag.rerank_candidates", side_effect=mock_rerank):
                    with patch("rag.load_chunks", return_value=([], {})):
                        res_rr = query_advanced_rag(
                            question="L/C thanh toán?",
                            mode="hybrid_rerank",
                            llm_fn=lambda prompt, cfg: "Theo [E1], ngân hàng mở L/C có nghĩa vụ thanh toán."
                        )
                        self.assertEqual(res_rr["status"], "answered")
                        self.assertEqual(res_rr["trace"]["accepted"], 1)
                        self.assertTrue(res_rr["evidence"][0]["accepted"])
                        self.assertFalse(res_rr["evidence"][1]["accepted"])

    def test_02_rejected_evidence_excluded_from_prompt(self):
        """Test 2: Evidence bị loại (rejected) không được đưa vào LLM prompt."""
        passed_prompts = []

        def mock_llm(prompt, cfg):
            passed_prompts.append(prompt)
            return "Theo [E1], thanh toán L/C."

        def mock_sem(query, strategy, top_k, custom_config):
            return [dict(self.sample_evidence[0]), dict(self.sample_evidence[1])]

        def mock_bm25(query, chunks, top_k):
            return [dict(self.sample_evidence[0])]

        def mock_rerank(query, candidates, top_k, custom_config, reranker_fn):
            return [dict(self.sample_evidence[0]), dict(self.sample_evidence[1])]

        with patch("advanced_rag.semantic_retrieval", side_effect=mock_sem):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("advanced_rag.rerank_candidates", side_effect=mock_rerank):
                    with patch("rag.load_chunks", return_value=([], {})):
                        res = query_advanced_rag(
                            question="L/C?",
                            mode="hybrid_rerank",
                            llm_fn=mock_llm
                        )

        self.assertEqual(len(passed_prompts), 1)
        prompt_text = passed_prompts[0]
        # Text của chunk-01 được nhận (accepted), nhưng text của chunk-02 bị loại (rejected) nên không xuất hiện
        self.assertIn("Ngân hàng mở L/C", prompt_text)
        self.assertNotIn("Incoterms.pdf", prompt_text)
        self.assertNotIn("CIP Incoterms", prompt_text)

    def test_03_trace_counts_and_timings_complete(self):
        """Test 3: Trace chứa đầy đủ các count keys và latency_ms cho tất cả các tầng."""
        def mock_sem(query, strategy, top_k, custom_config):
            return [dict(self.sample_evidence[0])]

        def mock_bm25(query, chunks, top_k):
            return [dict(self.sample_evidence[0])]

        def mock_rerank(query, candidates, top_k, custom_config, reranker_fn):
            return [dict(self.sample_evidence[0])]

        with patch("advanced_rag.semantic_retrieval", side_effect=mock_sem):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("advanced_rag.rerank_candidates", side_effect=mock_rerank):
                    with patch("rag.load_chunks", return_value=([], {})):
                        res = query_advanced_rag(
                            question="Test trace",
                            mode="hybrid_rerank",
                            llm_fn=lambda p, c: "Trả lời [E1]"
                        )

        trace = res["trace"]
        self.assertIn("bm25_candidates", trace)
        self.assertIn("semantic_candidates", trace)
        self.assertIn("overlap", trace)
        self.assertIn("union", trace)
        self.assertIn("reranked", trace)
        self.assertIn("accepted", trace)
        self.assertIn("generation_called", trace)
        self.assertTrue(trace["generation_called"])

        lat = trace["latency_ms"]
        for key in ["bm25", "semantic", "fusion", "rerank", "generation", "total"]:
            self.assertIn(key, lat)

    def test_04_citation_maps_real_metadata(self):
        """Test 4: Citation mapping ánh xạ chính xác [E1] sang metadata thật (chunk_id, source, page)."""
        def mock_sem(query, strategy, top_k, custom_config):
            return [dict(self.sample_evidence[0])]

        def mock_bm25(query, chunks, top_k):
            return [dict(self.sample_evidence[0])]

        def mock_rerank(query, candidates, top_k, custom_config, reranker_fn):
            return [dict(self.sample_evidence[0])]

        with patch("advanced_rag.semantic_retrieval", side_effect=mock_sem):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("advanced_rag.rerank_candidates", side_effect=mock_rerank):
                    with patch("rag.load_chunks", return_value=([], {})):
                        res = query_advanced_rag(
                            question="Citations?",
                            mode="hybrid_rerank",
                            llm_fn=lambda p, c: "Theo quy định tại [E1], ngân hàng phải thanh toán."
                        )

        self.assertEqual(len(res["citations"]), 1)
        cit = res["citations"][0]
        self.assertEqual(cit["evidence_id"], "[E1]")
        self.assertEqual(cit["chunk_id"], "chunk-01")
        self.assertEqual(cit["source"], "LuatNganHang.pdf")
        self.assertEqual(cit["page_start"], 5)

    def test_05_generation_called_at_most_once(self):
        """Test 5: LLM Generation chỉ được gọi tối đa 1 lần duy nhất khi query."""
        mock_llm = MagicMock(return_value="Trả lời mẫu [E1]")

        def mock_sem(query, strategy, top_k, custom_config):
            return [dict(self.sample_evidence[0])]

        def mock_bm25(query, chunks, top_k):
            return [dict(self.sample_evidence[0])]

        def mock_rerank(query, candidates, top_k, custom_config, reranker_fn):
            return [dict(self.sample_evidence[0])]

        with patch("advanced_rag.semantic_retrieval", side_effect=mock_sem):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("advanced_rag.rerank_candidates", side_effect=mock_rerank):
                    with patch("rag.load_chunks", return_value=([], {})):
                        query_advanced_rag(question="Once?", mode="hybrid_rerank", llm_fn=mock_llm)

        mock_llm.assert_called_once()

    def test_06_compare_does_not_call_generation(self):
        """Test 6: Hàm compare_retrieval_modes KHÔNG GỌI LLM Generation bất kỳ lần nào."""
        mock_llm = MagicMock()

        def mock_sem(query, strategy, top_k, custom_config):
            return [dict(self.sample_evidence[0])]

        def mock_bm25(query, chunks, top_k):
            return [dict(self.sample_evidence[0])]

        def mock_rerank(query, candidates, top_k, custom_config, reranker_fn):
            return [dict(self.sample_evidence[0])]

        with patch("advanced_rag.semantic_retrieval", side_effect=mock_sem):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("advanced_rag.rerank_candidates", side_effect=mock_rerank):
                    with patch("rag.load_chunks", return_value=([], {})):
                        cmp_res = compare_retrieval_modes(question="Compare test?", strategy="hierarchical")

        mock_llm.assert_not_called()
        self.assertIn("comparison_table", cmp_res)
        self.assertIn("mode_latencies", cmp_res)

    def test_07_reranker_unavailable_status(self):
        """Test 7: Trả về status 'reranker_unavailable' khi Reranker bị lỗi nạp/thực thi."""
        def mock_sem(query, strategy, top_k, custom_config):
            return [dict(self.sample_evidence[0])]

        def mock_bm25(query, chunks, top_k):
            return [dict(self.sample_evidence[0])]

        def mock_failing_rerank(query, candidates, top_k, custom_config, reranker_fn):
            raise RuntimeError("reranker_unavailable: Failed to load model weights.")

        with patch("advanced_rag.semantic_retrieval", side_effect=mock_sem):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("advanced_rag.rerank_candidates", side_effect=mock_failing_rerank):
                    with patch("rag.load_chunks", return_value=([], {})):
                        res = query_advanced_rag(question="Reranker error?", mode="hybrid_rerank")

        self.assertEqual(res["status"], "reranker_unavailable")
        self.assertFalse(res["trace"]["generation_called"])
        self.assertTrue(len(res["warnings"]) > 0)

    def test_08_schema_completeness_for_all_statuses(self):
        """Test 8: Tất cả các status (answered, insufficient_evidence, retrieval_only, reranker_unavailable) đều trả đúng schema."""
        required_keys = {"status", "mode", "question", "answer", "evidence", "citations", "warnings", "trace"}

        def mock_sem_empty(query, strategy, top_k, custom_config):
            cand = dict(self.sample_evidence[0])
            cand["semantic_distance"] = 0.99
            cand["rerank_score"] = 0.10
            return [cand]

        def mock_bm25(query, chunks, top_k):
            return []

        with patch("advanced_rag.semantic_retrieval", side_effect=mock_sem_empty):
            with patch("advanced_rag.bm25_retrieval", side_effect=mock_bm25):
                with patch("rag.load_chunks", return_value=([], {})):
                    res_insufficient = query_advanced_rag(question="No info?", mode="semantic")

        self.assertEqual(res_insufficient["status"], "insufficient_evidence")
        self.assertTrue(required_keys.issubset(res_insufficient.keys()))


if __name__ == "__main__":
    unittest.main()
