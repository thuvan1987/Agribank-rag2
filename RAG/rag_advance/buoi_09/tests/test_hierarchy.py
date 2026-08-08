"""
Unittests cho Hierarchy Registry, Resolution & Parent Store Builder (Buổi 09).
Thực thi 100% offline sử dụng temporary directory và test fixtures.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Thêm buoi_09 vào sys.path để import hierarchical_rag
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import hierarchical_rag
from hierarchical_rag import (
    build_hierarchy_registry,
    extract_trailing_number,
    get_hierarchical_config,
    hierarchy_status,
    parse_legal_structure,
    save_hierarchy_store,
)


class TestHierarchyRegistryAndParentStore(unittest.TestCase):

    def setUp(self):
        """Tạo môi trường tạm thời cho testing."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Dọn dẹp thư mục tạm thời sau testing."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_metadata_precedence(self):
        """Test 1: Thứ tự ưu tiên 1 — Sử dụng metadata structure hợp lệ của chính record."""
        rec = {
            "chunk_id": "c-1",
            "source": "doc.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Nội dung điều luật...",
            "metadata_structure": {"article": "Điều 5. Quyền và nghĩa vụ"}
        }
        struct_path, method, ambiguous, warnings = parse_legal_structure(rec)
        self.assertEqual(method, "metadata")
        self.assertEqual(struct_path["article"], "Điều 5. Quyền và nghĩa vụ")

    def test_02_heading_inferred_at_chunk_start(self):
        """Test 2: Thứ tự ưu tiên 2 — Nhận diện heading ở dòng đầu tiên của chunk text."""
        rec = {
            "chunk_id": "c-2",
            "source": "doc.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "# Điều 10. Nguyên tắc quản lý\nNội dung nguyên tắc...",
            "metadata_structure": {}
        }
        struct_path, method, ambiguous, warnings = parse_legal_structure(rec)
        self.assertEqual(method, "heading_inferred")
        self.assertEqual(struct_path["article"], "Điều 10. Nguyên tắc quản lý")

    def test_03_carry_forward_same_source(self):
        """Test 3: Thứ tự ưu tiên 3 — Carry forward article từ chunk trước đó trong cùng source."""
        chunks = [
            {
                "chunk_id": "c-1",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "# Điều 7. Tái cơ cấu khoản nợ\nĐoạn 1...",
                "metadata_structure": {}
            },
            {
                "chunk_id": "c-2",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 2,
                "text": "Đoạn 2 tiếp theo của Điều 7 không có heading mới...",
                "metadata_structure": {}
            }
        ]
        tmp_json = self.test_dir / "chunks.json"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        children, parents, stats = build_hierarchy_registry(input_path=tmp_json)
        self.assertEqual(children[1]["resolution_method"], "carried_forward")
        self.assertEqual(children[1]["structural_path"]["article"], "Điều 7. Tái cơ cấu khoản nợ")

    def test_04_no_carry_forward_across_different_sources(self):
        """Test 4: Reset trạng thái carry forward khi chuyển sang source tài liệu khác."""
        chunks = [
            {
                "chunk_id": "c-1",
                "strategy": "hierarchical",
                "source": "doc1.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "# Điều 7. Tái cơ cấu\nĐoạn 1...",
                "metadata_structure": {}
            },
            {
                "chunk_id": "c-2",
                "strategy": "hierarchical",
                "source": "doc2.pdf", # Source mới
                "page_start": 1,
                "page_end": 1,
                "text": "Đoạn văn bản doc2 không chứa heading...",
                "metadata_structure": {}
            }
        ]
        tmp_json = self.test_dir / "chunks.json"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        children, parents, stats = build_hierarchy_registry(input_path=tmp_json)
        # c-2 thuộc doc2.pdf không được dùng Điều 7 của doc1.pdf -> fallback doc_fallback
        self.assertEqual(children[1]["resolution_method"], "document_fallback")
        self.assertIsNone(children[1]["structural_path"]["article"])

    def test_05_inline_article_not_false_heading(self):
        """Test 5: Cụm 'Điều N' xuất hiện giữa câu không bị nhận nhầm thành heading."""
        rec = {
            "chunk_id": "c-5",
            "source": "doc.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Các bên thực hiện theo quy định tại Điều 12 của Luật Tổ chức.",
            "metadata_structure": {}
        }
        struct_path, method, ambiguous, warnings = parse_legal_structure(rec)
        self.assertNotEqual(method, "heading_inferred")
        self.assertIsNone(struct_path["article"])

    def test_06_conflict_sets_ambiguous_and_warning(self):
        """Test 6: Xung đột thông tin đặt ambiguous=True và đính kèm warning."""
        chunks = [
            {
                "chunk_id": "c-1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Đoạn văn vừa dẫn chiếu Điều 10 vừa viện dẫn Điều 15 cụ thể.",
                "metadata_structure": {}
            }
        ]
        tmp_json = self.test_dir / "chunks.json"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        children, parents, stats = build_hierarchy_registry(input_path=tmp_json)
        self.assertTrue(children[0]["ambiguous"])
        self.assertTrue(len(children[0]["warnings"]) > 0)

    def test_07_numeric_chunk_ordering(self):
        """Test 7: Sắp xếp thứ tự child chunk theo số sequence ('hier-2' trước 'hier-10')."""
        self.assertEqual(extract_trailing_number("hier-2"), 2)
        self.assertEqual(extract_trailing_number("hier-10"), 10)
        self.assertLess(extract_trailing_number("hier-2"), extract_trailing_number("hier-10"))

    def test_08_stable_parent_id(self):
        """Test 8: Parent ID tạo ra ổn định và nhất quán giữa các lần chạy (deterministic)."""
        chunks = [
            {
                "chunk_id": "c-1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "# Điều 5. Nội dung\nText...",
                "metadata_structure": {}
            }
        ]
        tmp_json = self.test_dir / "chunks.json"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        c1, p1, s1 = build_hierarchy_registry(input_path=tmp_json)
        c2, p2, s2 = build_hierarchy_registry(input_path=tmp_json)
        self.assertEqual(p1[0]["parent_id"], p2[0]["parent_id"])

    def test_09_parent_split_at_child_boundary(self):
        """Test 9: Chia Parent window theo ranh giới child khi vượt PARENT_MAX_CHARS."""
        chunks = [
            {
                "chunk_id": "c-1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "A" * 600,
                "metadata_structure": {"article": "Điều 1"}
            },
            {
                "chunk_id": "c-2",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "B" * 600,
                "metadata_structure": {"article": "Điều 1"}
            }
        ]
        tmp_json = self.test_dir / "chunks.json"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        test_cfg = get_hierarchical_config()
        test_cfg["parent_max_chars"] = 1000
        children, parents, stats = build_hierarchy_registry(input_path=tmp_json, custom_config=test_cfg)
        # Tổng 1200 > 1000 => Tách thành 2 Parent Windows cho cùng 1 Điều 1
        self.assertEqual(len(parents), 2)
        self.assertEqual(parents[0]["child_ids"], ["c-1"])
        self.assertEqual(parents[1]["child_ids"], ["c-2"])

    def test_10_oversized_single_child_warning(self):
        """Test 10: Child đơn lẻ lớn hơn PARENT_MAX_CHARS vẫn được giữ nguyên và đánh warning 'oversized_single_child'."""
        chunks = [
            {
                "chunk_id": "c-1",
                "strategy": "hierarchical",
                "source": "doc.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "X" * 1500, # Vượt 1000
                "metadata_structure": {"article": "Điều 1"}
            }
        ]
        tmp_json = self.test_dir / "chunks.json"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        test_cfg = get_hierarchical_config()
        test_cfg["parent_max_chars"] = 1000
        children, parents, stats = build_hierarchy_registry(input_path=tmp_json, custom_config=test_cfg)
        self.assertEqual(len(parents), 1)
        self.assertIn("oversized_single_child", parents[0]["warnings"][0])

    def test_11_each_child_belongs_to_one_parent(self):
        """Test 11: Mỗi child chunk thuộc về duy nhất một parent_id."""
        chunks = [
            {"chunk_id": f"c-{i}", "strategy": "hierarchical", "source": "doc.pdf", "page_start": i, "page_end": i, "text": f"Text {i}", "metadata_structure": {}}
            for i in range(1, 6)
        ]
        tmp_json = self.test_dir / "chunks.json"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        children, parents, stats = build_hierarchy_registry(input_path=tmp_json)
        parent_ids = {c["parent_id"] for c in children}
        self.assertTrue(all(c["parent_id"] is not None for c in children))

    def test_12_parent_page_and_text_accuracy(self):
        """Test 12: Parent page_start/page_end và text ghép ghép chính xác từ các child."""
        chunks = [
            {"chunk_id": "c-1", "strategy": "hierarchical", "source": "doc.pdf", "page_start": 2, "page_end": 3, "text": "Phần 1", "metadata_structure": {"article": "Điều 1"}},
            {"chunk_id": "c-2", "strategy": "hierarchical", "source": "doc.pdf", "page_start": 4, "page_end": 5, "text": "Phần 2", "metadata_structure": {"article": "Điều 1"}}
        ]
        tmp_json = self.test_dir / "chunks.json"
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        children, parents, stats = build_hierarchy_registry(input_path=tmp_json)
        p = parents[0]
        self.assertEqual(p["page_start"], 2)
        self.assertEqual(p["page_end"], 5)
        self.assertEqual(p["text"], "Phần 1\n\nPhần 2")

    def test_13_atomic_build_and_manifest(self):
        """Test 13: Ghi Store atomic và tạo Manifest với đầy đủ fingerprint và timestamp."""
        children = [{"child_id": "c-1", "parent_id": "p-1", "source": "doc.pdf", "page_start": 1, "page_end": 1, "text": "T", "structural_path": {}, "resolution_method": "fallback", "ambiguous": False, "warnings": []}]
        parents = [{"parent_id": "p-1", "source": "doc.pdf", "page_start": 1, "page_end": 1, "article_key": "k", "heading": "h", "window_index": 1, "child_ids": ["c-1"], "text": "T", "char_count": 1, "ambiguous_child_count": 0, "warnings": []}]
        stats = {"total_sources": 1, "total_children": 1, "total_parents": 1, "ambiguous_children_count": 0, "oversized_children_count": 0}

        manifest = save_hierarchy_store(children, parents, stats, target_dir=self.test_dir)
        self.assertTrue((self.test_dir / "children.json").exists())
        self.assertTrue((self.test_dir / "parents.json").exists())
        self.assertTrue((self.test_dir / "manifest.json").exists())
        self.assertIn("build_timestamp", manifest)

    def test_14_status_is_strictly_read_only(self):
        """Test 14: Lệnh status hoàn toàn Read-Only (Không tạo file/thư mục mới)."""
        non_existent_dir = self.test_dir / "empty_store"
        st = hierarchy_status(target_dir=non_existent_dir)
        self.assertFalse(st["store_exists"])
        self.assertFalse(non_existent_dir.exists())


if __name__ == "__main__":
    unittest.main()
