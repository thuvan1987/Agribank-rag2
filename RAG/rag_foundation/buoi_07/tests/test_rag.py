"""
Buổi 07: Automated Unit Test Suite cho RAG Pipeline (47 Mandatory Test Cases).
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Thêm đường dẫn chứa rag.py vào sys.path
TEST_DIR = Path(__file__).resolve().parent
BUOI_07_DIR = TEST_DIR.parent
if str(BUOI_07_DIR) not in sys.path:
    sys.path.insert(0, str(BUOI_07_DIR))

import rag

# Fake Config dành riêng cho Test Suite (dimension = 128)
TEST_CONFIG = {
    "api_key": "test_mock_api_key_12345",
    "has_api_key": True,
    "embedding_model": "gemini-embedding-2",
    "embedding_dim": 128,
    "generation_model": "gemini-3.5-flash-lite",
    "top_k": 5,
    "max_distance": 0.45
}


def make_deterministic_embedder(dim: int = 128):
    """
    Tạo deterministic fake embedder cho test.
    Mặc định trả về vector đơn vị đồng nhất để query và chunk trùng hướng (distance = 0.0 <= 0.45).
    """
    val = 1.0 / math.sqrt(dim)
    def mock_embedder(chunk, config):
        return [val] * dim
    return mock_embedder


class Test1_LoaderAndValidation(unittest.TestCase):
    """Nhóm 1: Kiểm thử Data Loader & Validation (Cases 1-9, 38)"""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_01_loader_reads_json_list(self):
        """Case 1: Loader đọc JSON list"""
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s1.pdf", "page_start": 1, "page_end": 1, "text": "Text 1"}
        ]
        fpath = self.tmp_path / "data.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        chunks, stats = rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(stats["valid_chunks"], 1)
        self.assertEqual(chunks[0]["chunk_id"], "c1")

    def test_02_loader_reads_object_with_chunks_field(self):
        """Case 2: Loader đọc object có field 'chunks'"""
        data = {
            "chunks": [
                {"chunk_id": "c2", "strategy": "hierarchical", "source": "s2.pdf", "page_start": 2, "page_end": 2, "text": "Text 2"}
            ]
        }
        fpath = self.tmp_path / "data_obj.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        chunks, stats = rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "c2")

    def test_03_loader_filters_correct_strategy(self):
        """Case 3: Chỉ lấy đúng strategy"""
        data = [
            {"chunk_id": "c_h", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "Hier text"},
            {"chunk_id": "c_s", "strategy": "semantic", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "Sem text"}
        ]
        fpath = self.tmp_path / "mixed.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        chunks_h, _ = rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertEqual(len(chunks_h), 1)
        self.assertEqual(chunks_h[0]["chunk_id"], "c_h")

        chunks_s, _ = rag.load_chunks(input_path=fpath, strategy="semantic")
        self.assertEqual(len(chunks_s), 1)
        self.assertEqual(chunks_s[0]["chunk_id"], "c_s")

    def test_04_missing_required_field_fails(self):
        """Case 4: Thiếu field bắt buộc phải fail"""
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1}
        ]
        fpath = self.tmp_path / "missing.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertIn("Thiếu trường bắt buộc 'text'", str(ctx.exception))

    def test_05_field_wrong_type_fails(self):
        """Case 5: Field sai kiểu phải fail"""
        data = [
            {"chunk_id": 12345, "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "Valid text"}
        ]
        fpath = self.tmp_path / "wrong_type.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertIn("Trường 'chunk_id' phải là string", str(ctx.exception))

    def test_06_boolean_page_number_fails(self):
        """Case 6: Boolean không được chấp nhận làm page number"""
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": True, "page_end": 1, "text": "Text"}
        ]
        fpath = self.tmp_path / "bool_page.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertIn("không chấp nhận boolean", str(ctx.exception))

    def test_07_page_start_greater_than_page_end_fails(self):
        """Case 7: page_start > page_end phải fail"""
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 5, "page_end": 2, "text": "Text"}
        ]
        fpath = self.tmp_path / "invalid_range.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertIn("page_start (5) lớn hơn page_end (2)", str(ctx.exception))

    def test_08_empty_text_skipped_and_counted(self):
        """Case 8: Text rỗng bị bỏ qua và thống kê đúng"""
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "   "},
            {"chunk_id": "c2", "strategy": "hierarchical", "source": "s.pdf", "page_start": 2, "page_end": 2, "text": "Valid Text"}
        ]
        fpath = self.tmp_path / "empty_text.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        chunks, stats = rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(stats["empty_text_skipped"], 1)
        self.assertEqual(stats["valid_chunks"], 1)

    def test_09_duplicate_chunk_id_fails(self):
        """Case 9: Duplicate chunk_id phải fail"""
        data = [
            {"chunk_id": "dup-id", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "Text 1"},
            {"chunk_id": "dup-id", "strategy": "hierarchical", "source": "s.pdf", "page_start": 2, "page_end": 2, "text": "Text 2"}
        ]
        fpath = self.tmp_path / "duplicate.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertIn("Trùng chunk_id 'dup-id'", str(ctx.exception))

    def test_38_loader_blocks_non_object_record(self):
        """Case 38: Loader chặn record không phải JSON object (dict)"""
        data = ["string_record_instead_of_dict"]
        fpath = self.tmp_path / "non_dict.json"
        fpath.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            rag.load_chunks(input_path=fpath, strategy="hierarchical")
        self.assertIn("không phải JSON object", str(ctx.exception))


class Test2_EmbeddingAndVectorValidation(unittest.TestCase):
    """Nhóm 2: Kiểm thử Embedding & Vector Validation (Cases 15-18, 20, 39)"""

    def test_15_embedding_wrong_count_fails(self):
        """Case 15: Embedding trả sai số vector phải fail"""
        vecs = [[0.1] * 128]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vecs, expected_dim=128, expected_count=2)
        self.assertIn("Kỳ vọng 2 vector, nhận được 1", str(ctx.exception))

    def test_16_embedding_empty_vector_fails(self):
        """Case 16: Embedding trả vector rỗng phải fail"""
        vecs = [[]]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vecs, expected_dim=128, expected_count=1)
        self.assertIn("chiều 0 không khớp với expected_dim = 128", str(ctx.exception))

    def test_17_embedding_wrong_dimension_fails(self):
        """Case 17: Embedding trả sai dimension phải fail"""
        vecs = [[0.1] * 64]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vecs, expected_dim=128, expected_count=1)
        self.assertIn("chiều 64 không khớp với expected_dim = 128", str(ctx.exception))

    def test_18_embedding_nan_or_inf_fails(self):
        """Case 18: Embedding có NaN hoặc Infinity phải fail"""
        vec_nan = [[0.1] * 127 + [float("nan")]]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vec_nan, expected_dim=128, expected_count=1)
        self.assertIn("có giá trị NaN", str(ctx.exception))

        vec_inf = [[0.1] * 127 + [float("inf")]]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vec_inf, expected_dim=128, expected_count=1)
        self.assertIn("có giá trị Infinity", str(ctx.exception))

    def test_20_missing_api_key_fails_safely_no_fake_vectors(self):
        """Case 20: Thiếu API key phải fail rõ và không upsert vector giả"""
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)
        fpath = tmp_path / "data.json"
        fpath.write_text(json.dumps([{"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "Text"}]), encoding="utf-8")

        no_key_config = dict(TEST_CONFIG)
        no_key_config["api_key"] = ""
        no_key_config["has_api_key"] = False

        with self.assertRaises(ValueError) as ctx:
            rag.index_chunks(input_path=fpath, strategy="hierarchical", storage_dir=tmp_path / "storage", custom_config=no_key_config)
        self.assertIn("Thiếu GEMINI_API_KEY", str(ctx.exception))
        tmp_dir.cleanup()

    def test_39_embedding_blocks_boolean_and_zero_vector(self):
        """Case 39: Embedding chặn boolean và zero vector"""
        vec_bool = [[True] + [0.1] * 127]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vec_bool, expected_dim=128, expected_count=1)
        self.assertIn("không được là boolean", str(ctx.exception))

        vec_zero = [[0.0] * 128]
        with self.assertRaises(ValueError) as ctx:
            rag.validate_embeddings(vec_zero, expected_dim=128, expected_count=1)
        self.assertIn("là zero vector", str(ctx.exception))


class Test3_IndexingAndCollectionIdentity(unittest.TestCase):
    """Nhóm 3: Kiểm thử Indexing & Collection Identity (Cases 10-13, 19, 40-42)"""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.storage_dir = self.tmp_path / "storage"
        self.mock_embedder = make_deterministic_embedder(dim=128)

        self.fixture_file = self.tmp_path / "fixture.json"
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "doc.pdf", "page_start": 1, "page_end": 2, "text": "Sample content"}
        ]
        self.fixture_file.write_text(json.dumps(data), encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_10_indexing_twice_does_not_increase_count(self):
        """Case 10: Index hai lần không tăng record count (Idempotency)"""
        res1 = rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)
        self.assertEqual(res1["total_collection_records"], 1)

        res2 = rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)
        self.assertEqual(res2["total_collection_records"], 1)

    def test_11_metadata_citation_stored_completely(self):
        """Case 11: Metadata citation được lưu đầy đủ"""
        rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)
        client = rag.get_chroma_client(self.storage_dir)
        coll_name = rag.get_collection_name("hierarchical", TEST_CONFIG["embedding_model"], TEST_CONFIG["embedding_dim"])
        coll = client.get_collection(coll_name)
        rec = coll.get(ids=["c1"], include=["metadatas"])
        meta = rec["metadatas"][0]
        self.assertEqual(meta["source"], "doc.pdf")
        self.assertEqual(meta["page_start"], 1)
        self.assertEqual(meta["page_end"], 2)
        self.assertEqual(meta["chunk_id"], "c1")

    def test_12_collection_identity_changes_with_strategy(self):
        """Case 12: Collection identity thay đổi khi strategy thay đổi"""
        name_h = rag.get_collection_name("hierarchical", "model_a", 128)
        name_s = rag.get_collection_name("semantic", "model_a", 128)
        self.assertNotEqual(name_h, name_s)

    def test_13_collection_identity_changes_with_model_or_dimension(self):
        """Case 13: Collection identity thay đổi khi model hoặc dimension thay đổi"""
        name1 = rag.get_collection_name("hierarchical", "model_a", 128)
        name2 = rag.get_collection_name("hierarchical", "model_b", 128)
        name3 = rag.get_collection_name("hierarchical", "model_a", 256)
        self.assertNotEqual(name1, name2)
        self.assertNotEqual(name1, name3)

    def test_19_embedding_error_before_upsert_adds_no_records(self):
        """Case 19: Embedding lỗi trước upsert không thêm record mới"""
        def faulty_embedder(chunk, config):
            raise RuntimeError("API Fault")

        with self.assertRaises(RuntimeError):
            rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=faulty_embedder, custom_config=TEST_CONFIG)

        st = rag.check_status(strategy="hierarchical", storage_dir=self.storage_dir, custom_config=TEST_CONFIG)
        self.assertFalse(st["collection_exists"])

    def test_40_status_on_empty_storage_creates_no_collection(self):
        """Case 40: status trên storage trống không tạo collection"""
        st = rag.check_status(strategy="hierarchical", storage_dir=self.storage_dir, custom_config=TEST_CONFIG)
        self.assertFalse(st["collection_exists"])
        self.assertEqual(st["record_count"], 0)
        client = rag.get_chroma_client(self.storage_dir)
        self.assertEqual(len(client.list_collections()), 0)

    def test_41_reset_with_embedding_error_preserves_old_valid_collection(self):
        """Case 41: --reset gặp embedding lỗi vẫn giữ nguyên collection hợp lệ cũ"""
        rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)

        def faulty_embedder(chunk, config):
            raise RuntimeError("Embed error during reset")

        with self.assertRaises(RuntimeError):
            rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", reset=True, storage_dir=self.storage_dir, embedder_fn=faulty_embedder, custom_config=TEST_CONFIG)

        st = rag.check_status(strategy="hierarchical", storage_dir=self.storage_dir, custom_config=TEST_CONFIG)
        self.assertTrue(st["collection_exists"])
        self.assertEqual(st["record_count"], 1)

    def test_42_existing_collection_mismatch_blocked_before_upsert(self):
        """Case 42: Existing collection có metadata/configuration mismatch bị chặn trước upsert"""
        rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)

        mismatched_config = dict(TEST_CONFIG)
        mismatched_config["embedding_model"] = "different-model-name"

        coll_name = rag.get_collection_name("hierarchical", TEST_CONFIG["embedding_model"], TEST_CONFIG["embedding_dim"])
        client = rag.get_chroma_client(self.storage_dir)
        coll = client.get_collection(coll_name)

        with self.assertRaises(ValueError) as ctx:
            rag.verify_collection_metadata(coll, "hierarchical", mismatched_config)
        self.assertIn("Mismatch Embedding Model", str(ctx.exception))


class Test4_RetrievalAndGating(unittest.TestCase):
    """Nhóm 4: Kiểm thử Retrieval & Confidence Gate (Cases 14, 21-23, 25-27, 43)"""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.storage_dir = self.tmp_path / "storage"

        self.fixture_file = self.tmp_path / "multi_chunks.json"
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "doc.pdf", "page_start": 1, "page_end": 1, "text": "Content 1"},
            {"chunk_id": "c2", "strategy": "hierarchical", "source": "doc.pdf", "page_start": 2, "page_end": 3, "text": "Content 2"},
            {"chunk_id": "c3", "strategy": "hierarchical", "source": "doc.pdf", "page_start": 4, "page_end": 4, "text": "Content 3"}
        ]
        self.fixture_file.write_text(json.dumps(data), encoding="utf-8")
        self.mock_embedder = make_deterministic_embedder(dim=128)

        rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_14_query_blocks_mismatched_collection(self):
        """Case 14: Query chặn collection có metadata không khớp"""
        client = rag.get_chroma_client(self.storage_dir)
        coll_name = rag.get_collection_name("hierarchical", TEST_CONFIG["embedding_model"], TEST_CONFIG["embedding_dim"])
        coll = client.get_collection(coll_name)

        with self.assertRaises(ValueError):
            rag.verify_collection_metadata(coll, "semantic", TEST_CONFIG)

    def test_21_retrieval_returns_correct_top_k(self):
        """Case 21: Retrieval trả đúng top-k"""
        res = rag.ask_question("Test query", top_k=2, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)
        self.assertEqual(len(res["evidence"]), 2)

    def test_22_retrieval_maintains_order(self):
        """Case 22: Retrieval giữ đúng thứ tự (E1, E2, E3...)"""
        res = rag.ask_question("Test query", top_k=3, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)
        ids = [e["evidence_id"] for e in res["evidence"]]
        self.assertEqual(ids, ["E1", "E2", "E3"])

    def test_23_top_k_greater_than_count_runs_correctly(self):
        """Case 23: top_k > collection.count() vẫn chạy đúng"""
        res = rag.ask_question("Test query", top_k=10, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)
        self.assertEqual(len(res["evidence"]), 3)

    def test_25_top_k_out_of_range_fails(self):
        """Case 25: Top-k ngoài khoảng (1-20) phải fail"""
        with self.assertRaises(ValueError):
            rag.ask_question("Query", top_k=0, strategy="hierarchical", storage_dir=self.storage_dir, custom_config=TEST_CONFIG)
        with self.assertRaises(ValueError):
            rag.ask_question("Query", top_k=25, strategy="hierarchical", storage_dir=self.storage_dir, custom_config=TEST_CONFIG)

    def test_26_empty_collection_fails_clearly(self):
        """Case 26: Collection rỗng phải fail rõ"""
        empty_storage = self.tmp_path / "empty_storage"
        empty_storage.mkdir()
        client = rag.get_chroma_client(empty_storage)
        coll_name = rag.get_collection_name("hierarchical", TEST_CONFIG["embedding_model"], TEST_CONFIG["embedding_dim"])
        client.create_collection(coll_name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)

        with self.assertRaises(ValueError) as ctx:
            rag.ask_question("Query", top_k=5, strategy="hierarchical", storage_dir=empty_storage, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)
        self.assertIn("rỗng (0 record)", str(ctx.exception))

    def test_27_all_evidence_exceeds_threshold_returns_insufficient_evidence(self):
        """Case 27: Evidence tốt nhất vượt threshold -> status insufficient_evidence & generator_fn không được gọi"""
        generator_called = []

        def mock_generator(prompt, accepted, config):
            generator_called.append(True)
            return "Answer [E1]"

        def high_distance_embedder(chunk, config):
            if chunk.get("source") == "query":
                return [-1.0] * 128
            return [1.0] * 128

        res = rag.ask_question("Query", top_k=3, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=high_distance_embedder, generator_fn=mock_generator, custom_config=TEST_CONFIG)
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["answer"], "Không tìm thấy đủ thông tin liên quan trong tài liệu đã cung cấp.")
        self.assertEqual(len(generator_called), 0)

    def test_43_one_accepted_one_rejected_evidence_behavior(self):
        """Case 43: Một evidence đạt và một evidence vượt threshold: result giữ cả hai, prompt chỉ chứa evidence đạt threshold"""
        captured_prompts = []

        def mock_generator(prompt, accepted, config):
            captured_prompts.append((prompt, accepted))
            return "Answer based on accepted evidence [E1]."

        # Vector custom: c1 trùng query (dist=0.0 <= 0.45), c2 vuông góc query (dist=1.0 > 0.45)
        def custom_embedder(chunk, config):
            cid = chunk.get("chunk_id", "")
            if chunk.get("source") == "query" or cid == "c1":
                return [1.0] + [0.0] * 127
            return [0.0, 1.0] + [0.0] * 126

        # Re-index with custom embedder for this test
        sep_storage = self.tmp_path / "sep_storage"
        rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=sep_storage, embedder_fn=custom_embedder, custom_config=TEST_CONFIG)

        res = rag.ask_question("Query", top_k=2, strategy="hierarchical", storage_dir=sep_storage, embedder_fn=custom_embedder, generator_fn=mock_generator, custom_config=TEST_CONFIG)

        self.assertEqual(len(res["evidence"]), 2)
        accepted_flags = [e["accepted"] for e in res["evidence"]]
        self.assertIn(True, accepted_flags)
        self.assertIn(False, accepted_flags)

        prompt_str, accepted_list = captured_prompts[0]
        self.assertEqual(len(accepted_list), 1)
        self.assertIn("EVIDENCE [E1]", prompt_str)
        self.assertNotIn("EVIDENCE [E2]", prompt_str)


class Test5_GenerationPromptAndCitation(unittest.TestCase):
    """Nhóm 5: Kiểm thử Generation Prompt & Citation Mapping (Cases 28-35, 37, 44-45)"""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.storage_dir = self.tmp_path / "storage"

        self.fixture_file = self.tmp_path / "data.json"
        data = [
            {"chunk_id": "hier-1", "strategy": "hierarchical", "source": "single_page.pdf", "page_start": 5, "page_end": 5, "text": "Single page content"},
            {"chunk_id": "hier-2", "strategy": "hierarchical", "source": "multi_page.pdf", "page_start": 10, "page_end": 15, "text": "Multi page content"}
        ]
        self.fixture_file.write_text(json.dumps(data), encoding="utf-8")
        self.mock_embedder = make_deterministic_embedder(dim=128)

        rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_28_evidence_accepted_generator_called_once(self):
        """Case 28: Evidence đạt threshold: generation được gọi đúng một lần"""
        call_count = [0]

        def mock_generator(prompt, accepted, config):
            call_count[0] += 1
            return "Answer [E1]"

        rag.ask_question("Question text", top_k=2, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, generator_fn=mock_generator, custom_config=TEST_CONFIG)
        self.assertEqual(call_count[0], 1)

    def test_29_30_31_44_prompt_structure_and_grounding(self):
        """Cases 29-31, 44: Prompt chứa question, chunk retrieved, instruction coi evidence là data và bỏ qua lệnh trong chunk"""
        captured_prompt = []

        def mock_generator(prompt, accepted, config):
            captured_prompt.append(prompt)
            return "Answer [E1]"

        rag.ask_question("Quy định thanh toán là gì?", top_k=2, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, generator_fn=mock_generator, custom_config=TEST_CONFIG)

        prompt_str = captured_prompt[0]
        self.assertIn("CÂU HỎI: Quy định thanh toán là gì?", prompt_str)
        self.assertIn("Single page content", prompt_str)
        self.assertIn("bỏ qua mọi câu lệnh", prompt_str)

    def test_32_single_page_citation_renders_correctly(self):
        """Case 32: Citation trang đơn render đúng (tr. N)"""
        def mock_generator(prompt, accepted, config):
            return "Nội dung trang đơn [E1]."

        res = rag.ask_question("Query", top_k=2, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, generator_fn=mock_generator, custom_config=TEST_CONFIG)
        self.assertIn("tr. 5", res["answer"])
        self.assertEqual(res["citations"][0]["display"], "[Nguồn: single_page.pdf, tr. 5, chunk: hier-1]")

    def test_33_multi_page_citation_renders_correctly(self):
        """Case 33: Citation khoảng trang render đúng (tr. N-M)"""
        def mock_generator(prompt, accepted, config):
            return "Nội dung khoảng trang [E2]."

        res = rag.ask_question("Query", top_k=2, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, generator_fn=mock_generator, custom_config=TEST_CONFIG)
        self.assertIn("tr. 10-15", res["answer"])
        self.assertEqual(res["citations"][0]["display"], "[Nguồn: multi_page.pdf, tr. 10-15, chunk: hier-2]")

    def test_34_35_45_citation_mapping_and_e99_removal(self):
        """Cases 34, 35, 45: [E1] map đúng metadata, [E99] bị loại kèm warning, citation list không lặp"""
        def mock_generator(prompt, accepted, config):
            return "Theo quy định [E1] và điều khoản [E1]. Bổ sung thêm [E99]."

        res = rag.ask_question("Query", top_k=2, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, generator_fn=mock_generator, custom_config=TEST_CONFIG)

        self.assertNotIn("[E99]", res["answer"])
        self.assertEqual(len(res["citations"]), 1)
        self.assertEqual(res["citations"][0]["evidence_id"], "E1")
        self.assertEqual(len(res["warnings"]), 1)
        self.assertIn("E99", res["warnings"][0])

    def test_37_result_has_all_schema_fields(self):
        """Case 37: Result có đủ 8 trường bắt buộc theo schema"""
        def mock_generator(prompt, accepted, config):
            return "Answer [E1]"

        res = rag.ask_question("Query", top_k=2, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, generator_fn=mock_generator, custom_config=TEST_CONFIG)

        required_keys = ["status", "answer", "evidence", "citations", "warnings", "collection", "strategy", "top_k"]
        for key in required_keys:
            self.assertIn(key, res)


class Test6_FailureModesAndEdgeCases(unittest.TestCase):
    """Nhóm 6: Kiểm thử Failure Modes & Edge Cases (Cases 24, 36, 46, 47)"""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.storage_dir = self.tmp_path / "storage"

        self.fixture_file = self.tmp_path / "data.json"
        data = [
            {"chunk_id": "c1", "strategy": "hierarchical", "source": "s.pdf", "page_start": 1, "page_end": 1, "text": "Content"}
        ]
        self.fixture_file.write_text(json.dumps(data), encoding="utf-8")
        self.mock_embedder = make_deterministic_embedder(dim=128)

        rag.index_chunks(input_path=self.fixture_file, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, custom_config=TEST_CONFIG)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_24_empty_question_fails(self):
        """Case 24: Question rỗng phải fail"""
        with self.assertRaises(ValueError):
            rag.ask_question("", top_k=5, strategy="hierarchical", storage_dir=self.storage_dir, custom_config=TEST_CONFIG)
        with self.assertRaises(ValueError):
            rag.ask_question("   ", top_k=5, strategy="hierarchical", storage_dir=self.storage_dir, custom_config=TEST_CONFIG)

    def test_36_generation_error_returns_retrieval_only(self):
        """Case 36: Generation lỗi -> status retrieval_only, evidence vẫn còn"""
        def faulty_generator(prompt, accepted, config):
            raise RuntimeError("API Timeout Error")

        res = rag.ask_question("Query", top_k=1, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, generator_fn=faulty_generator, custom_config=TEST_CONFIG)

        self.assertEqual(res["status"], "retrieval_only")
        self.assertEqual(res["answer"], "Đã truy xuất được nguồn nhưng chưa thể tạo câu trả lời tổng hợp.")
        self.assertEqual(len(res["evidence"]), 1)
        self.assertEqual(len(res["citations"]), 0)
        self.assertEqual(len(res["warnings"]), 1)

    def test_46_generation_returns_empty_text_converts_to_retrieval_only(self):
        """Case 46: Generation trả text rỗng chuyển thành retrieval_only và vẫn giữ evidence"""
        def empty_generator(prompt, accepted, config):
            return "   "

        res = rag.ask_question("Query", top_k=1, strategy="hierarchical", storage_dir=self.storage_dir, embedder_fn=self.mock_embedder, generator_fn=empty_generator, custom_config=TEST_CONFIG)

        self.assertEqual(res["status"], "retrieval_only")
        self.assertEqual(res["answer"], "Đã truy xuất được nguồn nhưng chưa thể tạo câu trả lời tổng hợp.")
        self.assertEqual(len(res["evidence"]), 1)
        self.assertEqual(len(res["citations"]), 0)

    def test_47_config_and_cli_work_when_cwd_is_not_buoi_07(self):
        """Case 47: Config và CLI hoạt động khi current working directory không phải buoi_07/"""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.tmp_path)
            cfg = rag.get_config()
            self.assertIsNotNone(cfg)

            st = rag.check_status(strategy="hierarchical", storage_dir=self.storage_dir, custom_config=TEST_CONFIG)
            self.assertTrue(st["collection_exists"])

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
