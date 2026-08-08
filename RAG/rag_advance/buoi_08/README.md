# Buổi 08 — Advanced RAG: Hybrid Search & Cross-Encoder Reranking Cho Tài Liệu Pháp Lý

Báo cáo chi tiết và tài liệu hướng dẫn thực thi cho **Buổi 08** thuộc chuỗi Workshop RAG Nâng Cao.

---

## 🎯 1. Mục Tiêu & Khác Biệt Giữa Buổi 07 và Buổi 08

- **Buổi 07 (Semantic Baseline)**: Chỉ sử dụng đơn lẻ Semantic Retrieval dựa trên Gemini Embeddings kết hợp ChromaDB vector database. Gặp hạn chế lớn khi truy xuất từ khóa chính xác (Exact Term), các ký hiệu viết tắt chuyên ngành (như *L/C, CIP, FOB, B/L*) hoặc số Điều/Khoản cụ thể trong văn bản pháp lý.
- **Buổi 08 (Advanced RAG)**: Nâng cấp lên kiến trúc **Tối ưu Đa tầng (Multi-stage Retrieval Pipeline)**:
  1. **Lexical Branch (BM25Okapi)**: Tokenize tiếng Việt NFC chuẩn hóa, bảo toàn số Điều/Khoản và cụm từ pháp lý.
  2. **Semantic Branch (Gemini Embeddings)**: Truy xuất các đoạn văn tương quan về mặt ý nghĩa/ngữ cảnh.
  3. **Reciprocal Rank Fusion (RRF)**: Hợp nhất 2 thứ hạng độc lập không phụ thuộc thang đo score/distance.
  4. **Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`)**: Chấm điểm mối quan hệ trực tiếp từng cặp `(Câu hỏi, Trích đoạn)` để xếp hạng lại thứ hạng chính xác cao nhất trước khi đưa vào LLM.

---

## 📐 2. Sơ Đồ Kiến Trúc Advanced RAG Pipeline

```mermaid
flowchart TD
    Q[User Question / Query] --> B1[BM25 Lexical Search]
    Q --> B2[Gemini Semantic Search]
    
    B1 -->|BM25 Candidates Top-K| RRF[Reciprocal Rank Fusion - RRF]
    B2 -->|Semantic Candidates Top-K| RRF
    
    RRF -->|Union & Fused Candidates| RR[Cross-Encoder Reranker\nBAAI/bge-reranker-v2-m3]
    
    RR -->|Reranked Candidates| GT[Confidence Gating\nrerank_score >= 0.50]
    
    GT -->|Accepted Evidence| LLM[Gemini LLM Generation\nGrounding & Citations]
    LLM --> ANS[Final Answer + [E1] Citations + Pipeline Trace]
```

---

## 📂 3. Cấu Trúc Dự Án (Project Structure)

```text
rag_advance/buoi_08/
├── SPEC_buoi_08.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── rag.py                  # Semantic Baseline helper & Chroma storage loader
├── advanced_rag.py         # Advanced RAG Pipeline (BM25, Semantic, RRF, Reranker, Query, Compare)
├── evaluate.py             # Evaluator Benchmark (Recall@K, MRR@K, nDCG@K, Latencies)
├── app.py                  # Streamlit Multi-tab Comparison Dashboard
├── eval/
│   └── questions.json      # Gold benchmark test dataset
├── tests/
│   ├── __init__.py
│   ├── test_bm25.py        # Unittests cho Tokenizer & BM25
│   ├── test_semantic.py    # Unittests cho Chroma Semantic Candidate Stage
│   ├── test_fusion.py      # Unittests cho RRF Fusion logic
│   ├── test_reranker.py    # Unittests cho Cross-Encoder Reranker
│   ├── test_pipeline.py    # Unittests cho Answer Pipeline & Gating
│   └── test_evaluator.py   # Unittests cho Evaluation Metrics
├── reports/
│   └── .gitkeep            # Lưu các báo cáo eval JSON
└── storage/
    ├── chroma/             # Storage ChromaDB persistent collection
    └── huggingface/        # Storage local cache cho model Reranker
```

---

## 🛠️ 4. Hướng Dẫn Cài Đặt & Môi Trường

### A. Sử dụng Môi trường ảo Python
Sử dụng chung Python Interpreter của Buổi 05/07:
```bash
source rag_foundation/buoi_05/.venv/bin/activate
```

### B. Cài đặt Dependencies
```bash
pip install -r rag_advance/buoi_08/requirements.txt
```

### C. Cấu hình File `.env`
Sao chép `.env.example` thành `.env` tại `rag_advance/buoi_08/.env`:
```ini
GEMINI_API_KEY=AIzaSy...
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIM=768
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite
RAG_MAX_DISTANCE=0.45
BM25_CANDIDATES=20
SEMANTIC_CANDIDATES=20
RRF_K=60
RRF_BM25_WEIGHT=1.0
RRF_SEMANTIC_WEIGHT=1.0
RERANK_CANDIDATES=20
FINAL_TOP_K=5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_MAX_LENGTH=512
RERANK_BATCH_SIZE=4
RERANK_MIN_SCORE=0.50
RERANK_DEVICE=auto
```

---

## ⚠️ 5. Cảnh Báo Tài Nguyên & Kích Thước Model Reranker

- Model `BAAI/bge-reranker-v2-m3` có dung lượng khoảng **2.2 GB**.
- **Yêu cầu khi nạp model**: Cần RAM trống tối thiểu 4GB và kết nối Internet ổn định trong lần khởi tạo đầu tiên.
- **Lazy Loading**: Model **chỉ được tải khi thực sự gọi** lệnh `rerank` hoặc mode `hybrid_rerank`. Quá trình kiểm tra `status` hay chạy `unittest` hoàn toàn không tải model.

---

## 🖥️ 6. Hướng Dẫn Sử Dụng Các Lệnh CLI

### 1. Kiểm tra trạng thái hệ thống
```bash
python rag_advance/buoi_08/advanced_rag.py status --strategy hierarchical
```

### 2. Chuẩn bị Index Semantic trong ChromaDB
```bash
python rag_advance/buoi_08/advanced_rag.py prepare-semantic --strategy hierarchical
```

### 3. Chạy truy xuất Lexical BM25
```bash
python rag_advance/buoi_08/advanced_rag.py bm25 --strategy hierarchical --question "Điều 7 quy định gì?"
```

### 4. Chạy truy xuất Hybrid Search RRF (BM25 + Semantic)
```bash
python rag_advance/buoi_08/advanced_rag.py hybrid --strategy hierarchical --question "Điều 7 quy định gì?"
```

### 5. Chạy Cross-Encoder Reranker
```bash
python rag_advance/buoi_08/advanced_rag.py rerank --strategy hierarchical --question "Điều 7 quy định gì?"
```

### 6. Hỏi đáp Pipeline hoàn chỉnh (Query)
```bash
python rag_advance/buoi_08/advanced_rag.py query --mode hybrid_rerank --strategy hierarchical --question "Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?"
```

### 7. So sánh song song 4 chế độ Retrieval (Không gọi LLM)
```bash
python rag_advance/buoi_08/advanced_rag.py compare --strategy hierarchical --question "Điều 7 quy định gì?"
```

---

## 🧪 7. Lệnh Chạy Unittests, Evaluator Benchmark & Streamlit Dashboard

### A. Thực thi Toàn Bộ 49 Offline Unittests (100% Pass)
```bash
python -m unittest discover -s rag_advance/buoi_08/tests
```

### B. Chạy Benchmark Đánh Giá Định Lượng
```bash
python rag_advance/buoi_08/evaluate.py --strategy hierarchical --k 5
```

### C. Khởi chạy Streamlit Dashboard
```bash
python -m streamlit run rag_advance/buoi_08/app.py
```

---

## 📊 8. Giải Thích Ý Nghĩa Các Điểm Số (Score Legend)

1. **BM25 Score**: Điểm số trùng khớp từ vựng dựa trên tần suất từ (TF-IDF cải tiến). Điểm **càng cao càng tương quan tốt**.
2. **Cosine Distance**: Khoảng cách hình học giữa 2 vector embedding ($1 - \text{cosine\_similarity}$). Điểm **càng nhỏ càng gần nghĩa**.
3. **RRF Score**: Điểm số kết hợp vị trí thứ hạng theo công thức Reciprocal Rank Fusion: $\frac{w}{k + rank}$. Điểm **càng cao thứ hạng tổng hợp càng tốt**.
4. **Rerank Score**: Điểm số phân loại trực tiếp câu hỏi & trích đoạn từ Cross-Encoder qua hàm Sigmoid ($[0, 1]$). Điểm **càng cao mức độ liên quan ngữ cảnh càng lớn**. *Lưu ý: Rerank score là điểm phân loại của model, không phải xác suất đúng tuyệt đối.*

---

## ⚖️ 9.Candidate K vs Final K

- **Candidate K (`BM25_CANDIDATES`, `SEMANTIC_CANDIDATES`, `RERANK_CANDIDATES`)**: Số lượng ứng viên được giữ lại ở các tầng trung gian (mặc định 20) để đảm bảo không bỏ sót thông tin tiềm năng.
- **Final K (`FINAL_TOP_K`)**: Số lượng trích đoạn chất lượng cao nhất sau cùng (mặc định 5) được chọn lọc để đưa vào LLM làm căn cứ trả lời.

---

## 📈 10. Evaluation Metrics & Giới Hạn Gold Dataset

- **Recall@K**: Tỷ lệ tìm thấy các trích đoạn chuẩn (Gold Chunk IDs) trong Top K.
- **MRR@K (Mean Reciprocal Rank)**: Thứ hạng nghịch đảo của trích đoạn chuẩn xuất hiện đầu tiên.
- **nDCG@K (Normalized Discounted Cumulative Gain)**: Đánh giá vị trí hiển thị của trích đoạn đúng (thứ hạng càng cao càng được chấm điểm cao).
- **Giới hạn Gold Dataset**: Do tập dữ liệu benchmark chứa các câu hỏi có nhãn `needs_human_review = True`, hệ thống xuất cảnh báo và **không tự động tuyên bố mode chiến thắng chính thức** mà cần thêm kiểm chứng của chuyên gia.

---

## 🔍 11. Các Câu Hỏi So Sánh Thực Tế (Manual Comparison Questions)

Dưới đây là 4 mẫu câu hỏi kiểm thử đại diện cho các kịch bản thực tế:

### A. Exact Legal Reference (Truy xuất số Điều/Khoản chính xác)
- **Câu hỏi**: `Điều 7 quy định như thế nào về cơ cấu lại thời hạn trả nợ?`
- **Đặc điểm**: Nhánh **BM25** vượt trội nhờ giữ nguyên chính xác token số `7` và từ khóa `cơ cấu lại thời hạn trả nợ`.

### B. Paraphrase Semantic (Diễn đạt lại ngữ nghĩa)
- **Câu hỏi**: `Khách hàng gặp khó khăn có thể được điều chỉnh kỳ hạn trả nợ ra sao?`
- **Đặc điểm**: Nhánh **Semantic** và **Reranker** vượt trội nhờ hiểu được ngữ nghĩa "điều chỉnh kỳ hạn trả nợ" đồng nghĩa với "cơ cấu lại thời hạn trả nợ".

### C. Multi-concept (Kết hợp nhiều khái niệm)
- **Câu hỏi**: `Phân loại nợ và trích lập dự phòng được thực hiện như thế nào?`
- **Đặc điểm**: Nhánh **Hybrid RRF + Reranker** phát huy hiệu quả tổng hợp thông tin từ cả 2 nhánh từ vựng và ngữ nghĩa.

### D. Out-of-scope (Ngoài phạm vi tài liệu)
- **Câu hỏi**: `Ngân hàng nào có lãi suất tiết kiệm cao nhất hôm nay?`
- **Đặc điểm**: Không có evidence nào đạt ngưỡng Confidence Gate (`rerank_score >= 0.50` hoặc `distance <= 0.45`). Pipeline lập tức trả về trạng thái `insufficient_evidence` và dừng gọi LLM.

---

## 🚨 12. Troubleshooting (Xử Lý Lỗi Thường Gặp)

- **Lỗi tải Reranker Model (Network / Timeout)**: Kiểm tra kết nối mạng và đảm bảo thư mục `rag_advance/buoi_08/storage/huggingface/` có quyền ghi.
- **CPU hoạt động chậm / Thiếu RAM**: Giảm `RERANK_BATCH_SIZE` xuống `2` hoặc `1` trong `.env`.
- **Lỗi thiếu GEMINI_API_KEY**: Đảm bảo file `.env` đã được điền API key hợp lệ trước khi thực thi `prepare-semantic` hoặc `query`.

---

## ⚖️ 13. Tuyên Bố Miễn Trừ Trách Nhiệm (Disclaimer)

*Ứng dụng RAG này được xây dựng cho mục đích thử nghiệm kỹ thuật và đào tạo kiến thức xử lý dữ liệu nâng cao. Thông tin phản hồi từ ứng dụng không được coi là tư vấn pháp lý hoặc khuyến nghị chính thức của ngân hàng.*
