# Buổi 05 — Demo OCR & Chunking

Files:
- `ocr_chunk_demo.py`: script demo xử lý OCR (PyMuPDF fallback) và 3 chiến lược chunking.
- `check_ocr_env.py`: kiểm tra môi trường (đã có).
- `.env`: placeholder cho `LLAMA_CLOUD_API_KEY` (không lưu giá trị secret vào repo).

Quick commands (từ thư mục `src`):

Dry-run (không ghi output):

```bash
python ocr_chunk_demo.py --dry-run
```

Full run (ghi output) — KHÔNG bật llamaparse mặc định:

```bash
python ocr_chunk_demo.py --no-dry-run --write
```

Bật llamaparse (chỉ khi bạn hiểu rủi ro LLM):

```bash
python ocr_chunk_demo.py --no-dry-run --write --use-llamaparse
```

Ghi chú:
- Không in hoặc lưu giá trị API key. API key phải tồn tại trong `src/.env` hoặc biến môi trường `LLAMA_CLOUD_API_KEY` nếu bạn bật `--use-llamaparse`.
- Theo SPEC, Buổi 5 không được tạo embedding, không lưu vector DB, không gọi LLM trực tiếp. LlamaParse có thể sử dụng mô hình phía server — xem đoạn cảnh báo trong SPEC nếu bạn bật tính năng này.
