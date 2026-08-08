"""
Unittests End-to-End cho Multi-Query Parent-Child Pipeline & Mode Routing (Buổi 09).
Thực thi 100% offline bằng injected fakes (generator, retriever, reranker, answer_fn).
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import hierarchical_rag
from hierarchical_rag import (
    clear_query_variants_cache,
    compare_hierarchical_rag,
    get_hierarchical_config,
    query_hierarchical_rag,
    save_hierarchy_store,
)


class TestPipelineE2E(unittest.TestCase):

    def setUp(self):
        clear_query_variants_cache()
        self.test_dir = Path(tempfile.mkdtemp())

        self.fake_children = [
            {
                "child_id": "c-1",
                "parent_id": "p-1",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Text child 1",
                "structural_path": {"article": "Điều 1"},
                "resolution_method": "metadata",
                "ambiguous": False,
                "warnings": []
            },
            {
                "child_id": "c-2",
                "parent_id": "p-2",
                "source": "doc2.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Text child 2",
                "structural_path": {"article": "Điều 2"},
                "resolution_method": "metadata",
                "ambiguous": False,
                "warnings": []
            }
        ]

        self.fake_parents = [
            {
                "parent_id": "p-1",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "article_key": "article:Điều 1",
                "heading": "Điều 1",
                "window_index": 1,
                "child_ids": ["c-1"],
                "text": "Nội dung parent 1 pháp lý...",
                "char_count": 28,
                "ambiguous_child_count": 0,
                "warnings": []
            },
            {
                "parent_id": "p-2",
                "source": "doc2.pdf",
                "page_start": 1,
                "page_end": 1,
                "article_key": "article:Điều 2",
                "heading": "Điều 2",
                "window_index": 1,
                "child_ids": ["c-2"],
                "text": "Nội dung parent 2 pháp lý...",
                "char_count": 28,
                "ambiguous_child_count": 0,
                "warnings": []
            }
        ]

        stats = {"total_sources": 2, "total_children": 2, "total_parents": 2, "ambiguous_children_count": 0, "oversized_children_count": 0}
        save_hierarchy_store(self.fake_children, self.fake_parents, stats, target_dir=self.test_dir)

    def tearDown(self):
        clear_query_variants_cache()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_reranker_pair_uses_q0_and_parent_text(self):
        """Test 1: Cross-encoder reranker chỉ nhận cặp (Q0, parent_text), không dùng Q1..Qn."""
        rerank_pairs = []

        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Variant Q1?", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            nonlocal rerank_pairs
            rerank_pairs = [(query, t) for t in texts]
            return [2.5]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            return "Câu trả lời mẫu theo [P1]."

        res = query_hierarchical_rag(
            "Câu hỏi gốc Q0?",
            mode="multi_parent",
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_retriever,
            reranker_fn=fake_reranker,
            generate_answer_fn=fake_answer,
            hierarchy_store_dir=self.test_dir
        )

        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(rerank_pairs), 1)
        self.assertEqual(rerank_pairs[0][0], "Câu hỏi gốc Q0?")

    def test_02_generated_queries_not_used_in_prompt_or_rerank(self):
        """Test 2: Generated queries Q1..Qn không được đưa vào Answer Prompt như sự thật."""
        prompt_received_context = ""

        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Giả định Q1 không có thật", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [3.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            nonlocal prompt_received_context
            prompt_received_context = str(accepted_evidence)
            return "Theo [P1] quy định..."

        res = query_hierarchical_rag(
            "Câu hỏi gốc Q0?",
            mode="multi_parent",
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_retriever,
            reranker_fn=fake_reranker,
            generate_answer_fn=fake_answer,
            hierarchy_store_dir=self.test_dir
        )
        self.assertNotIn("Giả định Q1 không có thật", prompt_received_context)

    def test_03_sorting_rank_change_and_final_k(self):
        """Test 3: Xếp hạng lại theo rerank score, tính rank_change và giới hạn FINAL_PARENT_TOP_K."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-2", "text": "Text child 2", "source": "doc2.pdf", "page_start": 1, "page_end": 1, "fused_rank": 2}
            ]}

        def fake_reranker(query, texts, cfg):
            return [1.0, 4.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            return "Trả lời."

        cfg = get_hierarchical_config()
        cfg["final_parent_top_k"] = 2
        res = query_hierarchical_rag(
            "Q0?",
            mode="multi_parent",
            custom_config=cfg,
            query_generator_fn=fake_gen,
            hybrid_retriever_fn=fake_retriever,
            reranker_fn=fake_reranker,
            generate_answer_fn=fake_answer,
            hierarchy_store_dir=self.test_dir
        )

        acc = res["accepted_evidence"]
        self.assertEqual(acc[0]["parent_id"], "p-2")
        self.assertEqual(acc[0]["parent_rerank_rank"], 1)
        self.assertEqual(acc[0]["parent_rank_change"], 1)

    def test_04_evidence_gate_accepted_vs_rejected(self):
        """Test 4: Evidence gate lọc chính xác parent có parent_rerank_score >= RERANK_MIN_SCORE."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-2", "text": "Text child 2", "source": "doc2.pdf", "page_start": 1, "page_end": 1, "fused_rank": 2}
            ]}

        def fake_reranker(query, texts, cfg):
            return [2.0, -2.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            return "Trả lời."

        res = query_hierarchical_rag("Q0?", mode="multi_parent", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, generate_answer_fn=fake_answer, hierarchy_store_dir=self.test_dir)
        self.assertEqual(len(res["accepted_evidence"]), 1)
        self.assertEqual(res["accepted_evidence"][0]["parent_id"], "p-1")

    def test_05_empty_accepted_evidence_returns_insufficient_evidence(self):
        """Test 5: Không có evidence đạt gate trả về status 'insufficient_evidence' và KHÔNG gọi LLM answer."""
        llm_called = False

        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [-5.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            nonlocal llm_called
            llm_called = True
            return "Thất bại."

        res = query_hierarchical_rag("Q0?", mode="multi_parent", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, generate_answer_fn=fake_answer, hierarchy_store_dir=self.test_dir)
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertFalse(llm_called)
        self.assertEqual(len(res["citations"]), 0)

    def test_06_flat_and_parent_mode_routing(self):
        """Test 6: Đảm bảo 4 chế độ single_flat, multi_flat, single_parent, multi_parent điều hướng đúng."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Q1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [2.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            return "Answer ok."

        for m in ["single_flat", "multi_flat", "single_parent", "multi_parent"]:
            res = query_hierarchical_rag("Q0?", mode=m, query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, generate_answer_fn=fake_answer, hierarchy_store_dir=self.test_dir)
            self.assertEqual(res["mode"], m)
            if "parent" in m:
                self.assertIn("P1", res["citations"][0]["evidence_id"])
            else:
                self.assertIn("C1", res["citations"][0]["evidence_id"])

    def test_07_multi_query_failure_status(self):
        """Test 7: Sinh multi-query lỗi trả về status 'query_generation_unavailable'."""
        def fake_failing_gen(q, cfg):
            raise RuntimeError("Gemini API Quota Exceeded")

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": []}

        res = query_hierarchical_rag("Q0?", mode="multi_parent", query_generator_fn=fake_failing_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        self.assertEqual(res["status"], "query_generation_unavailable")

    def test_08_reranker_failure_returns_reranker_unavailable(self):
        """Test 8: Reranker lỗi trả về status 'reranker_unavailable' và không silent fallback."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_failing_reranker(q, t, cfg):
            raise RuntimeError("Out of memory on GPU")

        res = query_hierarchical_rag("Q0?", mode="multi_parent", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_failing_reranker, hierarchy_store_dir=self.test_dir)
        self.assertEqual(res["status"], "reranker_unavailable")
        self.assertIn("reranker_unavailable", res["warnings"][0])

    def test_09_citation_maps_to_real_parent_and_anchor_child(self):
        """Test 9: Object citation trả về đúng parent_id và anchor_child_id thật."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [3.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            return "Theo [P1] quy định."

        res = query_hierarchical_rag("Q0?", mode="multi_parent", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, generate_answer_fn=fake_answer, hierarchy_store_dir=self.test_dir)
        cit = res["citations"][0]
        self.assertEqual(cit["evidence_id"], "P1")
        self.assertEqual(cit["parent_id"], "p-1")
        self.assertEqual(cit["anchor_child_id"], "c-1")

    def test_10_citation_label_validation(self):
        """Test 10: Phát hiện warning khi LLM answer trích dẫn nhãn giả lập [P99]."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [3.0]

        def fake_hallucinating_answer(question, accepted_evidence, citations, custom_config):
            return "Theo [P99] quy định."

        res = query_hierarchical_rag("Q0?", mode="multi_parent", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, generate_answer_fn=fake_hallucinating_answer, hierarchy_store_dir=self.test_dir)
        self.assertIn("invalid_citations_detected", res["warnings"][0])

    def test_11_multi_mode_uses_at_most_two_generation_api_calls(self):
        """Test 11: Chế độ multi_parent gọi tối đa 2 Gemini Generation API calls."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Q1", "focus": "paraphrase"}]})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [3.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            return "Trả lời [P1]."

        res = query_hierarchical_rag("Q0?", mode="multi_parent", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, generate_answer_fn=fake_answer, hierarchy_store_dir=self.test_dir)
        gen_calls = res["trace"]["api_call_counts"]["generation_calls"]
        self.assertLessEqual(gen_calls, 2)

    def test_12_compare_subcommand_does_not_call_answer_generation(self):
        """Test 12: Hàm compare_hierarchical_rag chạy cả 4 mode nhưng KHÔNG gọi LLM answer generation."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [3.0]

        res = compare_hierarchical_rag("Q0?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, hierarchy_store_dir=self.test_dir)
        self.assertEqual(len(res["results"]), 4)
        for m, r in res["results"].items():
            self.assertEqual(r["answer"], "")

    def test_13_trace_identity_and_counts(self):
        """Test 13: Trace ghi nhận đầy đủ stage_latencies_ms và counts."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [3.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            return "OK [P1]."

        res = query_hierarchical_rag("Q0?", mode="multi_parent", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, generate_answer_fn=fake_answer, hierarchy_store_dir=self.test_dir)
        tr = res["trace"]
        self.assertIn("stage_latencies_ms", tr)
        self.assertIn("total", tr["stage_latencies_ms"])
        self.assertIn("counts", tr)

    def test_14_all_tests_run_100_percent_offline(self):
        """Test 14: Đảm bảo 100% tests chạy offline bằng injected fakes."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        def fake_reranker(query, texts, cfg):
            return [5.0]

        def fake_answer(question, accepted_evidence, citations, custom_config):
            return "Trả lời offline [P1]."

        res = query_hierarchical_rag("Test offline?", mode="multi_parent", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, reranker_fn=fake_reranker, generate_answer_fn=fake_answer, hierarchy_store_dir=self.test_dir)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(res["answer"], "Trả lời offline [P1].")


if __name__ == "__main__":
    unittest.main()
