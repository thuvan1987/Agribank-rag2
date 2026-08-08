"""
Unittests cho Parent Candidate Retrieval & Context Budgeting System (Buổi 09).
Thực thi 100% offline bằng fake hierarchy store, fake generator và fake hybrid retriever.
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
    get_hierarchical_config,
    retrieve_parent_candidates,
    save_hierarchy_store,
)


class TestParentCandidateRetrieval(unittest.TestCase):

    def setUp(self):
        clear_query_variants_cache()
        self.test_dir = Path(tempfile.mkdtemp())

        # Tạo fake hierarchy store cho testing
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
                "parent_id": "p-1",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "text": "Text child 2",
                "structural_path": {"article": "Điều 1"},
                "resolution_method": "carried_forward",
                "ambiguous": False,
                "warnings": []
            },
            {
                "child_id": "c-3",
                "parent_id": "p-2",
                "source": "doc2.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Text child 3",
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
                "page_end": 2,
                "article_key": "article:Điều 1",
                "heading": "Điều 1. Phạm vi",
                "window_index": 1,
                "child_ids": ["c-1", "c-2"],
                "text": "A" * 6000,
                "char_count": 6000,
                "ambiguous_child_count": 0,
                "warnings": []
            },
            {
                "parent_id": "p-2",
                "source": "doc2.pdf",
                "page_start": 1,
                "page_end": 1,
                "article_key": "article:Điều 2",
                "heading": "Điều 2. Đối tượng",
                "window_index": 1,
                "child_ids": ["c-3"],
                "text": "B" * 4000,
                "char_count": 4000,
                "ambiguous_child_count": 0,
                "warnings": []
            }
        ]

        stats = {"total_sources": 2, "total_children": 3, "total_parents": 2, "ambiguous_children_count": 0, "oversized_children_count": 0}
        save_hierarchy_store(self.fake_children, self.fake_parents, stats, target_dir=self.test_dir)

    def tearDown(self):
        clear_query_variants_cache()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_child_maps_to_correct_parent(self):
        """Test 1: Child hit ánh xạ chính xác sang parent_id tương ứng trong Hierarchy Store."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        res = retrieve_parent_candidates("Test?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(res["parent_candidates"]), 1)
        p = res["parent_candidates"][0]
        self.assertEqual(p["parent_id"], "p-1")
        self.assertEqual(p["anchor_child_id"], "c-1")

    def test_02_missing_hierarchy_store_returns_hierarchy_not_ready(self):
        """Test 2: Thư mục Hierarchy Store không tồn tại trả về status 'hierarchy_not_ready'."""
        empty_dir = self.test_dir / "non_existent"
        res = retrieve_parent_candidates("Test?", hierarchy_store_dir=empty_dir)
        self.assertEqual(res["status"], "hierarchy_not_ready")
        self.assertEqual(len(res["parent_candidates"]), 0)

    def test_03_manual_calculation_parent_rrf_score(self):
        """Test 3: Tính toán thủ công công thức Parent RRF Score: sum 1 / (PARENT_RRF_K + multi_query_rank)."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            # Q0 trả về c-1 (mq_rank 1) và c-2 (mq_rank 2) thuộc cùng parent p-1
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-2", "text": "Text child 2", "source": "doc1.pdf", "page_start": 1, "page_end": 2, "fused_rank": 2}
            ]}

        res = retrieve_parent_candidates("Test?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        p = res["parent_candidates"][0]
        # PARENT_RRF_K = 60. Score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.032522
        expected_score = round(1.0 / 61 + 1.0 / 62, 6)
        self.assertEqual(p["parent_rrf_score"], expected_score)

    def test_04_child_score_cap(self):
        """Test 4: PARENT_SCORE_CHILD_LIMIT giới hạn số child hits tối đa dùng để tính điểm parent."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-2", "text": "Text child 2", "source": "doc1.pdf", "page_start": 1, "page_end": 2, "fused_rank": 2}
            ]}

        cfg = get_hierarchical_config()
        cfg["parent_score_child_limit"] = 1
        res = retrieve_parent_candidates("Test?", custom_config=cfg, query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        p = res["parent_candidates"][0]
        self.assertEqual(len(p["scoring_child_ids"]), 1)
        self.assertEqual(len(p["supporting_child_ids"]), 2)
        expected_score = round(1.0 / 61, 6)
        self.assertEqual(p["parent_rrf_score"], expected_score)

    def test_05_separation_of_scoring_and_supporting_children(self):
        """Test 5: Phân biệt rõ scoring_child_ids và supporting_child_ids trong Parent Candidate."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-2", "text": "Text child 2", "source": "doc1.pdf", "page_start": 1, "page_end": 2, "fused_rank": 2}
            ]}

        res = retrieve_parent_candidates("Test?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        p = res["parent_candidates"][0]
        self.assertEqual(p["scoring_child_ids"], ["c-1", "c-2"])
        self.assertEqual(p["supporting_child_ids"], ["c-1", "c-2"])

    def test_06_parent_deduplication(self):
        """Test 6: Nhiều child hits thuộc cùng một parent được tổng hợp vào 1 Parent Candidate duy nhất."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-2", "text": "Text child 2", "source": "doc1.pdf", "page_start": 1, "page_end": 2, "fused_rank": 2}
            ]}

        res = retrieve_parent_candidates("Test?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        self.assertEqual(len(res["parent_candidates"]), 1)
        self.assertEqual(res["parent_candidates"][0]["parent_id"], "p-1")

    def test_07_deterministic_tie_break_parent(self):
        """Test 7: Thứ tự sắp xếp Parent Candidate deterministic khi đồng điểm."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-3", "text": "Text child 3", "source": "doc2.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}
            ]}

        res = retrieve_parent_candidates("Test?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        parents = res["parent_candidates"]
        self.assertEqual(parents[0]["parent_id"], "p-1")
        self.assertEqual(parents[1]["parent_id"], "p-2")

    def test_08_parent_candidate_limit(self):
        """Test 8: Giới hạn PARENT_CANDIDATES giữ đúng số lượng parent candidates tối đa."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-3", "text": "Text child 3", "source": "doc2.pdf", "page_start": 1, "page_end": 1, "fused_rank": 2}
            ]}

        cfg = get_hierarchical_config()
        cfg["parent_candidates"] = 1
        cfg["final_parent_top_k"] = 1
        res = retrieve_parent_candidates("Test?", custom_config=cfg, query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        self.assertEqual(len(res["parent_candidates"]), 1)
        self.assertEqual(res["parent_candidates"][0]["parent_id"], "p-1")

    def test_09_context_budget_cuts_at_parent_boundary(self):
        """Test 9: Ngân sách context (TOTAL_CONTEXT_MAX_CHARS) cắt chính xác tại ranh giới Parent Document."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [
                {"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1},
                {"chunk_id": "c-3", "text": "Text child 3", "source": "doc2.pdf", "page_start": 1, "page_end": 1, "fused_rank": 2}
            ]}

        cfg = get_hierarchical_config()
        cfg["parent_max_chars"] = 2000
        cfg["total_context_max_chars"] = 7000
        res = retrieve_parent_candidates("Test?", custom_config=cfg, query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        self.assertEqual(len(res["parent_candidates"]), 1)
        self.assertEqual(res["trace"]["parents_dropped_by_context_budget"], 1)

    def test_10_oversized_first_parent_warning(self):
        """Test 10: Parent đầu tiên vượt TOTAL_CONTEXT_MAX_CHARS vẫn được giữ nguyên và đánh warning."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "Text child 1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        cfg = get_hierarchical_config()
        cfg["parent_max_chars"] = 2000
        cfg["total_context_max_chars"] = 3000
        res = retrieve_parent_candidates("Test?", custom_config=cfg, query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        self.assertEqual(len(res["parent_candidates"]), 1)
        self.assertIn("oversized_first_parent", res["parent_candidates"][0]["warnings"][0])

    def test_11_expansion_factor_and_char_counts_in_trace(self):
        """Test 11: Chỉ số context_expansion_factor và tổng ký tự child/parent chính xác trong trace."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "1234567890", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        res = retrieve_parent_candidates("Test?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        tr = res["trace"]
        self.assertEqual(tr["child_chars_total"], 10)
        self.assertEqual(tr["expanded_parent_chars_total"], 6000)
        self.assertEqual(tr["context_expansion_factor"], 600.0)

    def test_12_no_calls_to_reranker_or_generation(self):
        """Test 12: Bước 06 không gọi Reranker hoặc Answer Generation."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": []})

        def fake_retriever(q_text, strategy, cfg):
            return {"candidates": [{"chunk_id": "c-1", "text": "T1", "source": "doc1.pdf", "page_start": 1, "page_end": 1, "fused_rank": 1}]}

        res = retrieve_parent_candidates("Test?", query_generator_fn=fake_gen, hybrid_retriever_fn=fake_retriever, hierarchy_store_dir=self.test_dir)
        self.assertNotIn("answer", res)
        self.assertNotIn("rerank_score", res["parent_candidates"][0])


if __name__ == "__main__":
    unittest.main()
