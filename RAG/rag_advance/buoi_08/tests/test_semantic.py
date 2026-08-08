"""
Unittests cho Semantic Candidate Retrieval và Status Check (Buổi 08).
Sử dụng Mock Embeddings và Temporary Chroma Storage để kiểm thử offline.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb

# Thêm buoi_08 vào sys.path để import advanced_rag và rag
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import advanced_rag
import rag


class TestSemanticCandidateRetrieval(unittest.TestCase):

    def setUp(self):
        """Tạo môi trường giả lập với temporary Chroma storage."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)

        # Cấu hình giả lập chuẩn
        self.test_config = {
            "api_key": "test_mock_key_xyz",
            "has_api_key": True,
            "embedding_model": "gemini-embedding-2",
            "embedding_dim": 768,
            "generation_model": "gemini-3.5-flash-lite",
            "rag_max_distance": 0.45,
            "bm25_candidates": 20,
            "semantic_candidates": 20,
            "rrf_k": 60,
            "rrf_bm25_weight": 1.0,
            "rrf_semantic_weight": 1.0,
            "rerank_candidates": 20,
            "final_top_k": 5,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_max_length": 512,
            "rerank_batch_size": 4,
            "rerank_min_score": 0.50,
            "rerank_device": "auto"
        }

    def tearDown(self):
        """Dọn dẹp tài nguyên tạm thời."""
        self.temp_dir.cleanup()

    @patch("rag.STORAGE_DIR")
    @patch("rag.generate_embeddings")
    def test_01_semantic_top_k_count_order(self, mock_generate_embeddings, mock_storage_dir):
        """Test 1: Semantic retrieval trả về đúng top-k, count và sắp xếp distance tăng dần."""
        mock_storage_dir.resolve.return_value = self.storage_dir
        client = chromadb.PersistentClient(path=str(self.storage_dir))

        col_name = rag.get_collection_name("hierarchical", "gemini-embedding-2", 768)
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "model": "gemini-embedding-2", "dimension": 768}
        )

        # Thêm 3 record mẫu
        vec1 = [0.1] * 768
        vec2 = [0.5] * 768
        vec3 = [0.9] * 768

        col.add(
            ids=["chunk-01", "chunk-02", "chunk-03"],
            embeddings=[vec1, vec2, vec3],
            documents=["Văn bản 1", "Văn bản 2", "Văn bản 3"],
            metadatas=[
                {"source": "doc1.pdf", "page_start": 1, "page_end": 1},
                {"source": "doc2.pdf", "page_start": 2, "page_end": 2},
                {"source": "doc3.pdf", "page_start": 3, "page_end": 3}
            ]
        )

        # Query vector gần vec1 nhất
        mock_generate_embeddings.return_value = [[0.11] * 768]

        with patch("advanced_rag.get_advanced_config", return_value=self.test_config):
            with patch("rag.get_chroma_client", return_value=client):
                results = advanced_rag.semantic_retrieval(
                    query="Điều khoản L/C",
                    strategy="hierarchical",
                    top_k=2,
                    custom_config=self.test_config
                )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chunk-01")
        self.assertEqual(results[0]["semantic_rank"], 1)
        self.assertLessEqual(results[0]["semantic_distance"], results[1]["semantic_distance"])

    @patch("rag.generate_embeddings")
    def test_02_metadata_completeness(self, mock_generate_embeddings):
        """Test 2: Candidate kết quả bảo toàn đầy đủ metadata (chunk_id, source, page_start, page_end)."""
        client = chromadb.PersistentClient(path=str(self.storage_dir))
        col_name = rag.get_collection_name("hierarchical", "gemini-embedding-2", 768)
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "model": "gemini-embedding-2", "dimension": 768}
        )

        col.add(
            ids=["meta-chunk-1"],
            embeddings=[[0.2] * 768],
            documents=["Nội dung pháp lý"],
            metadatas=[{"source": "QuyDinh.pdf", "page_start": 10, "page_end": 12}]
        )

        mock_generate_embeddings.return_value = [[0.2] * 768]

        with patch("rag.get_chroma_client", return_value=client):
            results = advanced_rag.semantic_retrieval(
                query="Quy định",
                strategy="hierarchical",
                top_k=1,
                custom_config=self.test_config
            )

        res = results[0]
        self.assertEqual(res["chunk_id"], "meta-chunk-1")
        self.assertEqual(res["source"], "QuyDinh.pdf")
        self.assertEqual(res["page_start"], 10)
        self.assertEqual(res["page_end"], 12)
        self.assertIn("semantic_rank", res)
        self.assertIn("semantic_distance", res)

    def test_03_collection_mismatch_blocked(self):
        """Test 3: Lỗi metadata/dimension không khớp cấu hình phải bị chặn."""
        client = chromadb.PersistentClient(path=str(self.storage_dir))
        col_name = rag.get_collection_name("hierarchical", "gemini-embedding-2", 768)
        # Tạo collection với dimension sai (512 thay vì 768)
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "model": "gemini-embedding-2", "dimension": 512}
        )

        col.add(
            ids=["mismatch-chunk"],
            embeddings=[[0.1] * 512],
            documents=["Văn bản sai dimension"],
            metadatas=[{"source": "bad.pdf", "page_start": 1, "page_end": 1}]
        )

        with patch("rag.get_chroma_client", return_value=client):
            with patch("rag.generate_embeddings", return_value=[[0.1] * 768]):
                with self.assertRaises(Exception) as ctx:
                    advanced_rag.semantic_retrieval(
                        query="Thử nghiệm",
                        strategy="hierarchical",
                        top_k=1,
                        custom_config=self.test_config
                    )
                err_msg = str(ctx.exception).lower()
                self.assertTrue("dimension" in err_msg or "512" in err_msg or "mismatch" in err_msg)

    def test_04_status_does_not_create_collection(self):
        """Test 4: Gọi status không được tự động tạo collection mới trong storage."""
        client = chromadb.PersistentClient(path=str(self.storage_dir))
        initial_cols = len(client.list_collections())

        with patch("rag.get_chroma_client", return_value=client):
            st_res = advanced_rag.check_advanced_status(
                strategy="hierarchical",
                custom_config=self.test_config
            )

        after_cols = len(client.list_collections())
        self.assertEqual(initial_cols, 0)
        self.assertEqual(after_cols, 0)
        self.assertFalse(st_res["collection_exists"])
        self.assertEqual(st_res["record_count"], 0)

    def test_05_missing_key_fails_no_fake_vector(self):
        """Test 5: Thiếu GEMINI_API_KEY phải ném ValueError, không sử dụng vector giả."""
        no_key_config = dict(self.test_config)
        no_key_config["api_key"] = ""
        no_key_config["has_api_key"] = False

        with self.assertRaises(ValueError) as ctx:
            advanced_rag.semantic_retrieval(
                query="Câu hỏi",
                strategy="hierarchical",
                top_k=5,
                custom_config=no_key_config
            )
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))
        self.assertIn("không được để rỗng", str(ctx.exception))

    def test_06_no_llm_generation_called(self):
        """Test 6: Đảm bảo tầng semantic candidate chỉ truy xuất vector, không gọi LLM Generation."""
        client = chromadb.PersistentClient(path=str(self.storage_dir))
        col_name = rag.get_collection_name("hierarchical", "gemini-embedding-2", 768)
        col = client.create_collection(
            name=col_name,
            metadata={"strategy": "hierarchical", "model": "gemini-embedding-2", "dimension": 768}
        )
        col.add(
            ids=["gen-test-1"],
            embeddings=[[0.1] * 768],
            documents=["Nội dung câu trả lời mẫu"],
            metadatas=[{"source": "test.pdf", "page_start": 1, "page_end": 1}]
        )

        with patch("rag.get_chroma_client", return_value=client):
            with patch("rag.generate_embeddings", return_value=[[0.1] * 768]):
                with patch("google.genai.Client") as mock_genai_client:
                    results = advanced_rag.semantic_retrieval(
                        query="Hỏi đáp",
                        strategy="hierarchical",
                        top_k=1,
                        custom_config=self.test_config
                    )
                    # Khẳng định Gemini LLM Client không bao giờ được khởi tạo hay gọi trong semantic stage
                    mock_genai_client.assert_not_called()
                    self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
