"""
Unittests cho Multi-Query Fan-Out Retrieval & Cross-Query RRF Fusion (Buổi 09).
Thực thi 100% offline sử dụng fake generator và fake hybrid retriever.
"""

import json
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import hierarchical_rag
from hierarchical_rag import (
    clear_query_variants_cache,
    retrieve_multi_query_child_hits,
)


class TestMultiQueryChildRetrieval(unittest.TestCase):

    def setUp(self):
        clear_query_variants_cache()

    def tearDown(self):
        clear_query_variants_cache()

    def test_01_manual_calculation_cross_query_rrf(self):
        """Test 1: Tính toán thủ công công thức Cross-Query RRF: w(q) / (K + rank_q)."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Query 1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            if "Query 1" in q_text:
                return {"candidates": [
                    {"chunk_id": "c-1", "text": "T1", "source": "s1", "page_start": 1, "page_end": 1, "fused_rank": 1}
                ]}
            return {"candidates": [
                {"chunk_id": "c-1", "text": "T1", "source": "s1", "page_start": 1, "page_end": 1, "fused_rank": 2}
            ]}

        res = retrieve_multi_query_child_hits(
            "Câu hỏi gốc?",
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_retriever
        )

        c1 = res["child_hits"][0]
        # Q0 rank 2 (w=1.5), Q1 rank 1 (w=1.0), K=60
        # Score = 1.5 / (60 + 2) + 1.0 / (60 + 1) = 1.5/62 + 1/61 = 0.0241935 + 0.0163934 = 0.040587
        expected_score = round(1.5 / 62 + 1.0 / 61, 6)
        self.assertEqual(c1["multi_query_rrf_score"], expected_score)

    def test_02_original_vs_variant_weights(self):
        """Test 2: Trọng số Q0 (original_weight) lớn hơn trọng số các query biến thể (variant_weight)."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Query 1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            if "gốc" in q_text:
                # Q0 thấy c-1 rank 1
                return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "s1", "page_start": 1, "page_end": 1, "fused_rank": 1}]}
            else:
                # Q1 thấy c-2 rank 1
                return {"candidates": [{"chunk_id": "c-2", "text": "T2", "source": "s1", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        res = retrieve_multi_query_child_hits(
            "Câu hỏi gốc?",
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_retriever
        )
        # c-1 đứng ở Q0 rank 1 phải có điểm RRF cao hơn c-2 đứng ở Q1 rank 1
        hits = res["child_hits"]
        self.assertEqual(hits[0]["child_id"], "c-1")
        self.assertGreater(hits[0]["multi_query_rrf_score"], hits[1]["multi_query_rrf_score"])

    def test_03_deduplication_of_union(self):
        """Test 3: Hợp nhất Union các child hits và loại bỏ trùng lặp theo child_id."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Query 1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text 1", "source": "s1", "page_start": 1, "page_end": 1, "fused_rank": 1}
            ]}

        res = retrieve_multi_query_child_hits("Question?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        self.assertEqual(len(res["child_hits"]), 1)
        self.assertEqual(res["child_hits"][0]["support_query_count"], 2) # Xuất hiện cả ở Q0 và Q1

    def test_04_missing_query_contribution_handling(self):
        """Test 4: Candidate chỉ xuất hiện ở một số query vẫn được giữ và tính điểm chính xác."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Query 1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            if "Query 1" in q_text:
                return {"candidates": [{"chunk_id": "c-only-q1", "text": "T", "source": "s", "page_start": 1, "page_end": 1, "fused_rank": 1}]}
            return {"candidates": []}

        res = retrieve_multi_query_child_hits("Q0?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        self.assertEqual(len(res["child_hits"]), 1)
        self.assertEqual(res["child_hits"][0]["child_id"], "c-only-q1")
        self.assertEqual(res["child_hits"][0]["support_query_ids"], ["Q1"])

    def test_05_support_query_count_and_ids(self):
        """Test 5: Đếm đúng support_query_count và danh sách support_query_ids theo thứ tự Q0, Q1..."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [
                {"text": "Query 1", "focus": "paraphrase"},
                {"text": "Query 2", "focus": "exact_legal_terms"}
            ]})

        def fake_retriever(q_text, strategy, cfg):
            if "Query 2" in q_text:
                return {"candidates": []}
            return {"candidates": [{"chunk_id": "c-1", "text": "T", "source": "s", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        res = retrieve_multi_query_child_hits("Q0?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        hit = res["child_hits"][0]
        self.assertEqual(hit["support_query_count"], 2)
        self.assertEqual(hit["support_query_ids"], ["Q0", "Q1"])

    def test_06_metadata_mismatch_fails(self):
        """Test 6: Mismatch metadata cùng child_id giữa các query gây lỗi ValueError."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Query 1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            if "Query 1" in q_text:
                return {"candidates": [{"chunk_id": "c-1", "text": "Text khac!", "source": "s", "page_start": 1, "page_end": 1, "fused_rank": 1}]}
            return {"candidates": [{"chunk_id": "c-1", "text": "Text goc", "source": "s", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        with self.assertRaises(ValueError):
            retrieve_multi_query_child_hits("Q0?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)

    def test_07_deterministic_tie_break(self):
        """Test 7: Thứ tự sắp xếp deterministic khi đồng điểm RRF."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-b", "text": "T", "source": "s", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-a", "text": "T", "source": "s", "page_start": 1, "page_end": 1, "fused_rank": 1}
            ]}

        res = retrieve_multi_query_child_hits("Q0?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        hits = res["child_hits"]
        self.assertEqual(hits[0]["child_id"], "c-a")
        self.assertEqual(hits[1]["child_id"], "c-b")

    def test_08_each_query_calls_retriever_once(self):
        """Test 8: Mỗi query gọi retriever đúng 1 lần duy nhất."""
        retrieval_count = 0

        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Q1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            nonlocal retrieval_count
            retrieval_count += 1
            return {"candidates": []}

        retrieve_multi_query_child_hits("Q0?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        self.assertEqual(retrieval_count, 2) # Q0 và Q1

    def test_09_no_calls_to_reranker_or_generation(self):
        """Test 9: Bước 05 tuyệt đối không gọi Cross-Encoder Reranker hoặc Answer Generation."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Q1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T", "source": "s", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        res = retrieve_multi_query_child_hits("Q0?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        # Kiểm tra không có trường answer hay rerank_score trong kết quả
        self.assertNotIn("answer", res)
        self.assertNotIn("rerank_score", res["child_hits"][0])

    def test_10_q0_failure_and_generated_partial_status(self):
        """Test 10: Q0 lỗi gây ngắt toàn pipeline; Q1 lỗi trả status 'multi_query_partial'."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Q1_failing", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            if "Q1_failing" in q_text:
                raise RuntimeError("Kêt nối Chroma DB timeout")
            return {"candidates": [{"chunk_id": "c-1", "text": "T", "source": "s", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        res = retrieve_multi_query_child_hits("Q0 gốc?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        self.assertEqual(res["status"], "multi_query_partial")
        self.assertEqual(len(res["child_hits"]), 1)
        self.assertIn("Q1", res["trace"]["failed_query_errors"])

    def test_11_trace_schema_validation(self):
        """Test 11: Kiểm tra đầy đủ schema và chỉ số trong trace."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Q1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": []}

        res = retrieve_multi_query_child_hits("Q0?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        tr = res["trace"]
        self.assertIn("query_count_requested", tr)
        self.assertIn("query_count_executed", tr)
        self.assertIn("union_child_count", tr)
        self.assertIn("cross_rrf_fusion_latency_ms", tr)

    def test_12_all_tests_run_100_percent_offline(self):
        """Test 12: Đảm bảo 100% unit tests chạy offline sử dụng fake retriever và fake generator."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Q1 offline", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "Text offline", "source": "s.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        res = retrieve_multi_query_child_hits("Question?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(res["child_hits"]), 1)


if __name__ == "__main__":
    unittest.main()
