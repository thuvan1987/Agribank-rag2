"""
Unittests cho Streamlit UI Helper Functions (Buổi 09).
Thực thi 100% thuần Python offline, không cần trình duyệt, mạng hay live model.
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ui_helpers import (
    build_mode_comparison_row,
    build_parent_tree_data,
    build_query_child_matrix,
    format_citation_display,
    map_status_warning_badge,
)


class TestUIHelpers(unittest.TestCase):

    def test_01_map_status_warning_badge(self):
        """Test 1: Ánh xạ chính xác status sang Emoji badge và message."""
        badge, msg, alert_type = map_status_warning_badge("ready")
        self.assertIn("THÀNH CÔNG", badge)
        self.assertEqual(alert_type, "success")

        badge, msg, alert_type = map_status_warning_badge("hierarchy_not_ready")
        self.assertIn("HIERARCHY CHƯA SẴN SÀNG", badge)
        self.assertEqual(alert_type, "error")

    def test_02_build_query_child_matrix(self):
        """Test 2: Tạo ma trận DataFrame Query-Child chính xác."""
        query_set = {
            "queries": [
                {"query_id": "Q0", "text": "Q gốc"},
                {"query_id": "Q1", "text": "Q biến thể 1"}
            ]
        }
        child_hits = [
            {
                "child_id": "c-1",
                "support_query_count": 2,
                "multi_query_rrf_score": 0.040587,
                "per_query_ranks": {"Q0": 2, "Q1": 1}
            },
            {
                "child_id": "c-2",
                "support_query_count": 1,
                "multi_query_rrf_score": 0.016393,
                "per_query_ranks": {"Q0": 1}
            }
        ]
        df = build_query_child_matrix(query_set, child_hits)
        self.assertEqual(len(df), 2)
        self.assertIn("Child ID", df.columns)
        self.assertIn("Q0", df.columns)
        self.assertIn("Q1", df.columns)
        self.assertEqual(df.iloc[0]["Q0"], "Rank 2")
        self.assertEqual(df.iloc[0]["Q1"], "Rank 1")
        self.assertEqual(df.iloc[1]["Q1"], "—")

    def test_03_build_parent_tree_data(self):
        """Test 3: Cấu trúc dữ liệu cây Parent-Child."""
        parents = [
            {
                "parent_id": "p-1",
                "heading": "Điều 1. Phạm vi",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "parent_rank": 2,
                "parent_rerank_rank": 1,
                "parent_rank_change": 1,
                "parent_rrf_score": 0.0322,
                "parent_rerank_score": 0.892,
                "anchor_child_id": "c-1",
                "supporting_child_ids": ["c-1", "c-2"],
                "text": "Full text p1..."
            }
        ]
        nodes = build_parent_tree_data(parents)
        self.assertEqual(len(nodes), 1)
        n = nodes[0]
        self.assertEqual(n["parent_id"], "p-1")
        self.assertIn("Rank 2 ➔ Rank 1", n["rank_movement_badge"])
        self.assertIn("▲+1", n["rank_movement_badge"])

    def test_04_build_mode_comparison_row(self):
        """Test 4: Chuyển đổi dữ liệu kết quả RAG mode sang hàng so sánh phẳng."""
        res = {
            "status": "ready",
            "child_hits": [{"text": "12345"}],
            "parent_candidates": [{"parent_id": "p-1", "char_count": 50}],
            "accepted_evidence": [{"source": "doc1.pdf", "parent_id": "p-1", "char_count": 50}],
            "citations": [{"evidence_id": "P1"}],
            "trace": {
                "stage_latencies_ms": {"total": 12.5},
                "api_call_counts": {"generation_calls": 2, "embedding_calls": 3}
            }
        }
        row = build_mode_comparison_row("multi_parent", res)
        self.assertEqual(row["Mode"], "multi_parent")
        self.assertEqual(row["Status"], "READY")
        self.assertEqual(row["Unit Type"], "Parent Document")
        self.assertEqual(row["Evidence IDs"], "P1")
        self.assertEqual(row["Gen API Calls"], 2)
        self.assertEqual(row["Emb API Calls"], 3)
        self.assertEqual(row["Expansion Factor"], "x10.00") # 50 / 5 = 10.0

    def test_05_format_citation_display(self):
        """Test 5: Định dạng hiển thị trích dẫn Markdown."""
        citations = [
            {
                "evidence_id": "P1",
                "parent_id": "p-1",
                "anchor_child_id": "c-1",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "parent_rerank_score": 0.892
            }
        ]
        md_text = format_citation_display(citations)
        self.assertIn("**[P1]** Parent Document: `p-1`", md_text)
        self.assertIn("Anchor Child: `c-1`", md_text)
        self.assertIn("0.8920", md_text)


if __name__ == "__main__":
    unittest.main()
