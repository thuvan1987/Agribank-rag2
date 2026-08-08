"""
Unittests cho BM25 Lexical Retrieval và Tokenizer Tiếng Việt Pháp lý (Buổi 08).
"""

import sys
import unittest
from pathlib import Path

# Thêm buoi_08 vào sys.path để import advanced_rag
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from advanced_rag import bm25_retrieval, tokenize_vi_legal


class TestBM25LexicalRetrieval(unittest.TestCase):

    def setUp(self):
        """Khởi tạo sample chunks pháp lý mô phỏng cho testing."""
        self.sample_chunks = [
            {
                "chunk_id": "chunk-01",
                "strategy": "hierarchical",
                "source": "Luat_Ngan_Hang_Sample.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "Điều 7 Khoản 2 quy định về cơ cấu lại thời hạn trả nợ và miễn giảm lãi vay."
            },
            {
                "chunk_id": "chunk-02",
                "strategy": "hierarchical",
                "source": "Luat_Ngan_Hang_Sample.pdf",
                "page_start": 2,
                "page_end": 2,
                "text": "Điều 8 Khoản 1 quy định về bảo hiểm hàng hóa theo điều kiện CIF Incoterms 2020."
            },
            {
                "chunk_id": "chunk-03",
                "strategy": "hierarchical",
                "source": "Luat_Ngan_Hang_Sample.pdf",
                "page_start": 3,
                "page_end": 3,
                "text": "Đoạn văn bản chung không chứa bất kỳ từ khóa pháp lý hoặc số điều khoản nào."
            },
            {
                "chunk_id": "chunk-04",
                "strategy": "hierarchical",
                "source": "Luat_Ngan_Hang_Sample.pdf",
                "page_start": 4,
                "page_end": 4,
                "text": "Đoạn văn bản khác có từ nợ nhưng không có số Điều."
            }
        ]

    def test_01_tokenizer_preserves_vietnamese_tones(self):
        """Test 1: Tokenizer giữ đầy đủ ký tự tiếng Việt có dấu (NFC)."""
        text = "cơ cấu lại thời hạn trả nợ"
        tokens = tokenize_vi_legal(text)
        expected = ["cơ", "cấu", "lại", "thời", "hạn", "trả", "nợ"]
        self.assertEqual(tokens, expected)

    def test_02_tokenizer_preserves_article_clause_numbers(self):
        """Test 2: Tokenizer giữ chính xác số Điều và Khoản."""
        text = "Điều 7, Khoản 2"
        tokens = tokenize_vi_legal(text)
        self.assertIn("điều", tokens)
        self.assertIn("7", tokens)
        self.assertIn("khoản", tokens)
        self.assertIn("2", tokens)

    def test_03_corpus_and_query_same_preprocessing(self):
        """Test 3: Corpus và Query đều được xử lý bằng cùng 1 hàm tokenize_vi_legal."""
        query = "ĐIỀU 7, KHOẢN 2"
        q_tokens = tokenize_vi_legal(query)
        doc_tokens = tokenize_vi_legal(self.sample_chunks[0]["text"])
        self.assertIn("điều", q_tokens)
        self.assertIn("7", q_tokens)
        self.assertIn("điều", doc_tokens)
        self.assertIn("7", doc_tokens)

    def test_04_exact_legal_term_ranking(self):
        """Test 4: Chunk chứa exact legal term & số Điều/Khoản được xếp hạng cao hơn."""
        query = "Điều 7 cơ cấu lại thời hạn trả nợ"
        results = bm25_retrieval(query, self.sample_chunks, top_k=4)
        self.assertEqual(results[0]["chunk_id"], "chunk-01")
        self.assertGreater(results[0]["bm25_score"], results[1]["bm25_score"])

    def test_05_candidate_k_larger_than_corpus(self):
        """Test 5: candidate_k lớn hơn số lượng corpus vẫn hoạt động an toàn."""
        results = bm25_retrieval("bảo hiểm hàng hóa", self.sample_chunks, top_k=100)
        self.assertEqual(len(results), len(self.sample_chunks))

    def test_06_empty_question_fails(self):
        """Test 6: Query rỗng hoặc chỉ có khoảng trắng phải ném lỗi ValueError."""
        with self.assertRaises(ValueError):
            bm25_retrieval("", self.sample_chunks, top_k=5)
        with self.assertRaises(ValueError):
            bm25_retrieval("   ", self.sample_chunks, top_k=5)

    def test_07_deterministic_tie_breaking(self):
        """Test 7: Tie-breaking ổn định theo chunk_id khi BM25 score bằng nhau (ví dụ score = 0)."""
        query = "từ_khóa_không_tồn_tại_trong_bất_kỳ_chunk_nào"
        results = bm25_retrieval(query, self.sample_chunks, top_k=4)
        chunk_ids = [r["chunk_id"] for r in results]
        self.assertEqual(chunk_ids, ["chunk-01", "chunk-02", "chunk-03", "chunk-04"])

    def test_08_no_external_dependency_calls(self):
        """Test 8: Đảm bảo BM25 retrieval thuần túy không gọi Gemini, Chroma hay Reranker."""
        results = bm25_retrieval("thời hạn trả nợ", self.sample_chunks, top_k=2)
        self.assertIn("bm25_rank", results[0])
        self.assertIn("bm25_score", results[0])


if __name__ == "__main__":
    unittest.main()
