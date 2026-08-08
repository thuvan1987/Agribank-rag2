# SPECIFICATION BUỔI 08 — ADVANCED RAG CHO TÀI LIỆU PHÁP LÝ
## Hybrid Search (BM25 + Semantic), Reciprocal Rank Fusion và Cross-Encoder Reranking

---

## 1. Workspace và Security Contract

- **Workspace Scope**: Tất cả code, config, test fixture và báo cáo của Buổi 08 được lưu trữ duy nhất trong thư mục `rag_foundation/buoi_08/`.
- **Chỉ đọc**: `rag_foundation/buoi_05/output/chunks/`, `rag_foundation/buoi_05/.venv/`, `rag_foundation/buoi_07/`.
- **Bảo mật**:
  - Không hardcode `GEMINI_API_KEY` trong mã nguồn.
  - Sử dụng biến môi trường từ `.env` local của Buổi 08 (`rag_foundation/buoi_08/.env`).
  - `.env` và thư mục `storage/chroma/` bị `.gitignore` chặn commit.

---

## 2. Quan hệ với Buổi 05 và Buổi 07

- **Buổi 05**: Cung cấp dữ liệu chunks thực tế thuộc 3 chiến lược (`fixed-size`, `hierarchical`, `semantic`) và virtual environment Python (`.venv`).
- **Buổi 07**: Cung cấp Semantic baseline RAG (`rag.py`) làm thành phần tạo Semantic Candidates.
- **Tính độc lập**: Buổi 08 sao chép `rag.py` từ Buổi 07 để hoạt động độc lập, không import trực tiếp tại runtime từ Buổi 07, không ghi đè storage hay `.env` của Buổi 07.

---

## 3. Data Contract

Mỗi chunk được load vào pipeline Advanced RAG phải tuân thủ chuẩn JSON schema Buổi 07/08:
- `chunk_id` (str, non-empty, unique)
- `strategy` (str, thuộc `fixed-size`, `semantic`, `hierarchical`)
- `source` (str, tên file tài liệu gốc)
- `page_start` (int, >= 1)
- `page_end` (int, >= page_start)
- `text` (str, non-empty)
- `metadata_structure` (dict, tùy chọn)

---

## 4. BM25 Tokenizer & Lexical Retrieval Contract

- **Thuật toán**: BM25Okapi (`rank-bm25`).
- **Tokenizer**: Tiền xử lý tiếng Việt bằng lowercase, loại bỏ dấu câu, tách từ theo khoảng trắng (hoặc regex `\w+`).
- **Input**: Query (câu hỏi) và Danh sách tất cả valid chunks của chiến lược đã chọn.
- **Output**: Top $K_{bm25}$ candidates (mặc định $K_{bm25} = 20$) kèm `bm25_score` và `bm25_rank`.

---

## 5. Semantic Candidate Retrieval Contract

- **Thuật toán**: Cosine distance trên ChromaDB Persistent Collection với Google Gemini Embeddings (`gemini-embedding-2`, dim 768).
- **Input**: Query và strategy đã chọn.
- **Output**: Top $K_{semantic}$ candidates (mặc định $K_{semantic} = 20$) kèm `semantic_distance`, `semantic_score` và `semantic_rank`.

---

## 6. Reciprocal Rank Fusion (RRF) Contract

- **Công thức RRF**:
  $$RRF\_Score(d) = w_{bm25} \cdot \frac{1}{k + r_{bm25}(d)} + w_{semantic} \cdot \frac{1}{k + r_{semantic}(d)}$$
  với $k = 60$, $w_{bm25} = 1.0$, $w_{semantic} = 1.0$.
- **Hợp nhất**: Kết hợp ứng viên xuất hiện ở một hoặc cả hai danh sách $BM25$ và $Semantic$. Nếu ứng viên không nằm trong top $K$ của một danh sách, rank của nó ở danh sách đó không được tính vào RRF score.
- **Output**: Danh sách candidates đã được sắp xếp giảm dần theo `rrf_score` và gán `rrf_rank`.

---

## 7. Cross-Encoder Reranker Contract

- **Model**: Cross-Encoder Multilingual (mặc định `BAAI/bge-reranker-v2-m3`).
- **Cặp chấm điểm**: Chấm điểm từng cặp `(query, candidate.text)` bằng Cross-Encoder logits.
- **Normalization**: Sử dụng hàm Sigmoid để chuyển logits về miền giá trị $[0, 1]$:
  $$Score_{rerank} = \frac{1}{1 + e^{-logit}}$$
- **Confidence Gate**: Loại bỏ các candidate có $Score_{rerank} < RERANK\_THRESHOLD$ (mặc định $0.35$).
- **Lazy Loading**: Model chỉ được nạp khi gọi hàm rerank hoặc chạy pipeline thật, không nạp khi `import` module hay chạy `status`.

---

## 8. Final Evidence & Citation Contract

- **Top-K Cuối**: Lấy tối đa $TopK_{final}$ evidence vượt qua gate (mặc định 5 evidence).
- **Citation**: Đánh số `[1]`, `[2]`, ... tương ứng với các evidence đạt ngưỡng và truyền cho Gemini LLM để tổng hợp câu trả lời có trích dẫn chuẩn xác.

---

## 9. Pipeline Trace Contract

Mọi câu truy vấn trong Advanced RAG phải trả về object `trace` chi tiết để phục vụ giải trình và audit:
- `bm25_candidates`: Danh sách top candidate từ BM25 kèm rank & score.
- `semantic_candidates`: Danh sách top candidate từ Semantic kèm rank & distance.
- `rrf_candidates`: Danh sách candidate sau hợp nhất RRF kèm rrf_rank & rrf_score.
- `rerank_candidates`: Danh sách candidate sau khi rerank kèm rerank_rank, rerank_score và trạng thái pass/fail gate.
- `latency_ms`: Thời gian thực thi từng tầng (lexical, semantic, fusion, rerank, generation).

---

## 10. Evaluation Metrics Contract

Bộ đánh giá offline tính toán 3 chỉ số retrieval cơ bản trên tập `eval/questions.json`:
1. **Recall@K**: Tỷ lệ chunk liên quan chuẩn (gold labels) được truy xuất trong Top K.
2. **MRR@K (Mean Reciprocal Rank)**: Giá trị nghịch đảo của vị trí xuất hiện đầu tiên của chunk liên quan chuẩn.
3. **nDCG@K**: Discounted Cumulative Gain được chuẩn hóa theo vị trí xếp hạng chuẩn.

---

## 11. Offline Testing Contract

- **Môi trường Test**: Toàn bộ unit test trong `tests/` phải chạy thành công offline mà không cần API key Gemini thật, không tải model HuggingFace nặng và không ghi dữ liệu thật vào storage.
- **Mocking**: Sử dụng Mock Gemini Embedding/Generation client và Mock Reranker Model cho test tự động CI/CD.

---

## 12. UI Comparison Contract

- Dashboard Streamlit (`app.py`) cung cấp tính năng so sánh trực quan 4 mode:
  1. `bm25`: Chỉ tìm kiếm theo từ khóa BM25.
  2. `semantic`: Chỉ tìm kiếm theo vector embedding Gemini.
  3. `hybrid`: BM25 + Semantic hợp nhất bằng RRF (không Reranker).
  4. `hybrid_rerank`: Hybrid RRF + Cross-Encoder Reranker.
- Hiển thị bảng dịch chuyển thứ hạng (Rank Movement Table) và biểu đồ thời gian xử lý (Latency Breakdown).
