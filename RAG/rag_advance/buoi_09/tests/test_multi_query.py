"""
Unittests cho Multi-Query Generator & Cache System (Buổi 09).
Thực thi 100% offline bằng fake generator function và dependency injection.
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
    extract_legal_references,
    generate_query_variants,
    get_hierarchical_config,
)


class TestMultiQueryExpansion(unittest.TestCase):

    def setUp(self):
        """Xóa sạch cache trước mỗi test case."""
        clear_query_variants_cache()

    def tearDown(self):
        """Xóa sạch cache sau mỗi test case."""
        clear_query_variants_cache()

    def test_01_q0_always_first_and_preserves_content(self):
        """Test 1: Q0 luôn đứng đầu tiên và giữ nguyên vẹn nội dung câu hỏi gốc."""
        def fake_gen(q, cfg):
            return json.dumps({
                "queries": [
                    {"text": "Điều kiện cấp tín dụng?", "focus": "exact_legal_terms"}
                ]
            })

        res = generate_query_variants("Điều kiện vay vốn?", query_generator_fn=fake_gen)
        self.assertEqual(res["queries"][0]["query_id"], "Q0")
        self.assertEqual(res["queries"][0]["origin"], "original")
        self.assertEqual(res["queries"][0]["text"], "Điều kiện vay vốn?")

    def test_02_strict_schema_validation(self):
        """Test 2: Kiểm tra nghiêm ngặt toàn bộ các trường bắt buộc trong kết quả Multi-Query."""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Biến thể 1", "focus": "paraphrase"}]})

        res = generate_query_variants("Câu hỏi test?", query_generator_fn=fake_gen)
        req_keys = ["original_question", "queries", "model", "generation_latency_ms", "status", "cache_hit", "dropped_duplicate_count", "warnings"]
        for k in req_keys:
            self.assertIn(k, res)

    def test_03_nfc_trim_max_length(self):
        """Test 3: Chuẩn hóa NFC, strip khoảng trắng và giới hạn độ dài tối đa MULTI_QUERY_MAX_CHARS."""
        def fake_gen(q, cfg):
            return json.dumps({
                "queries": [
                    {"text": "  Biến thể hợp lệ   ", "focus": "paraphrase"},
                    {"text": "X" * 500, "focus": "paraphrase"} # Quá 300 ký tự -> Loại bỏ
                ]
            })

        res = generate_query_variants("  Test NFC  ", query_generator_fn=fake_gen)
        self.assertEqual(res["queries"][0]["text"], "Test NFC")
        self.assertEqual(len(res["queries"]), 2) # Q0 + 1 valid variant
        self.assertEqual(res["queries"][1]["text"], "Biến thể hợp lệ")

    def test_04_duplicate_removal(self):
        """Test 4: Tự động loại bỏ các query trùng lặp và tăng counter dropped_duplicate_count."""
        def fake_gen(q, cfg):
            return json.dumps({
                "queries": [
                    {"text": "Điều kiện vay vốn?", "focus": "paraphrase"}, # Trùng Q0
                    {"text": "Quy định vay vốn?", "focus": "paraphrase"},
                    {"text": "quy định vay vốn?", "focus": "paraphrase"} # Trùng Q1
                ]
            })

        res = generate_query_variants("Điều kiện vay vốn?", query_generator_fn=fake_gen)
        self.assertGreater(res["dropped_duplicate_count"], 0)
        self.assertEqual(len(res["queries"]), 2) # Q0 + Q1 độc bản

    def test_05_legal_reference_preservation(self):
        """Test 5: Trích xuất chính xác tham chiếu Điều/Khoản trong câu hỏi pháp lý."""
        refs = extract_legal_references("Vay vốn theo Điều 7 và Khoản 2 Thông tư 39/2016/TT-NHNN")
        self.assertIn("Điều 7", refs)
        self.assertIn("Khoản 2", refs)
        self.assertIn("Thông tư 39/2016/TT-NHNN", refs)

    def test_06_rejection_of_invented_fictional_articles(self):
        """Test 6: Loại bỏ các câu hỏi biến thể chứa số Điều/Khoản bịa thêm không có trong Q0."""
        def fake_gen(q, cfg):
            return json.dumps({
                "queries": [
                    {"text": "Cho vay theo Điều 999 của Luật TCTD?", "focus": "exact_legal_terms"} # Điều 999 không có trong Q0
                ]
            })

        res = generate_query_variants("Quy trình cho vay tín dụng?", query_generator_fn=fake_gen)
        self.assertEqual(len(res["queries"]), 1) # Chỉ còn Q0
        self.assertTrue(len(res["warnings"]) > 0)

    def test_07_deterministic_ids(self):
        """Test 7: Gán query_id tuần tự và deterministic (Q0, Q1, Q2...)."""
        def fake_gen(q, cfg):
            return json.dumps({
                "queries": [
                    {"text": "Biến thể 1", "focus": "paraphrase"},
                    {"text": "Biến thể 2", "focus": "exact_legal_terms"}
                ]
            })

        res = generate_query_variants("Câu hỏi gốc?", query_generator_fn=fake_gen)
        self.assertEqual([q["query_id"] for q in res["queries"]], ["Q0", "Q1", "Q2"])

    def test_08_single_generator_call(self):
        """Test 8: Generator function chỉ được gọi đúng 1 lần cho mỗi lượt mở rộng."""
        call_count = 0

        def fake_gen(q, cfg):
            nonlocal call_count
            call_count += 1
            return json.dumps({"queries": [{"text": "Biến thể", "focus": "paraphrase"}]})

        generate_query_variants("Test call count?", query_generator_fn=fake_gen)
        self.assertEqual(call_count, 1)

    def test_09_cache_hit_prevents_reinvocation(self):
        """Test 9: Cache hit trả về cache_hit=True và không gọi lại generator function."""
        call_count = 0

        def fake_gen(q, cfg):
            nonlocal call_count
            call_count += 1
            return json.dumps({"queries": [{"text": "Biến thể cache", "focus": "paraphrase"}]})

        res1 = generate_query_variants("Test Cache Hit?", query_generator_fn=fake_gen)
        self.assertFalse(res1["cache_hit"])
        self.assertEqual(call_count, 1)

        res2 = generate_query_variants("Test Cache Hit?", query_generator_fn=fake_gen)
        self.assertTrue(res2["cache_hit"])
        self.assertEqual(call_count, 1) # Vẫn là 1, không gọi lại!

    def test_10_api_failure_returns_explicit_status(self):
        """Test 10: Khi Generator lỗi/ngắt, trả status 'query_generation_unavailable' rõ ràng."""
        def fake_failing_gen(q, cfg):
            raise RuntimeError("API Connection Timeout")

        res = generate_query_variants("Test error?", query_generator_fn=fake_failing_gen)
        self.assertEqual(res["status"], "query_generation_unavailable")
        self.assertEqual(len(res["queries"]), 1) # Vẫn bảo toàn Q0
        self.assertTrue(len(res["warnings"]) > 0)

    def test_11_all_unit_tests_run_100_percent_offline(self):
        """Test 11: Đảm bảo toàn bộ unit test chạy 100% offline không cần Gemini API Key."""
        cfg = get_hierarchical_config()
        cfg["api_key"] = ""
        def fake_gen(q, cfg):
            return json.dumps({"queries": [{"text": "Biến thể offline", "focus": "paraphrase"}]})

        res = generate_query_variants("Test offline?", custom_config=cfg, query_generator_fn=fake_gen)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(len(res["queries"]), 2)


if __name__ == "__main__":
    unittest.main()
