# Buổi 14 — Hybrid Search + Reranking + Mini Knowledge Graph

Dự án triển khai đường ống RAG nâng cao kết hợp **Hybrid Search (BM25 + Dense Retrieval)**, **Cross-Encoder Reranking**, và **Mini Knowledge Graph Traversal** cho văn bản pháp lý ngân hàng.

---

## 1. Cấu trúc Dự án (`buoi_14/`)

```text
buoi_14/
├── cache/
│   └── dense_embeddings.pkl           # Cache vector embeddings (772 chunks)
├── data/
│   └── processed/
│       └── chunks_normalized.csv      # Corpus 772 chunks chuẩn hóa
├── outputs/
│   ├── inspection_report.md           # Báo cáo Pre-Check dữ liệu & môi trường
│   └── retrieval_examples.md          # Báo cáo so sánh BM25-only vs Dense-only
├── scripts/
│   ├── prepare_corpus.py              # Script chuẩn hóa Corpus
│   ├── baseline_retrieval.py          # CLI truy vấn 2 baseline BM25 & Dense
│   └── evaluate_baselines.py          # Script tự động đánh giá và xuất báo cáo
├── src/
│   ├── __init__.py
│   ├── tokenizer.py                   # Tokenizer pháp lý Việt Nam cho BM25
│   ├── bm25_retriever.py              # Thư viện BM25-only retrieval
│   └── dense_retriever.py             # Thư viện Dense-only retrieval + Caching
├── .venv/                             # Python 3.14.6 Virtual Environment
├── buoi14.md                          # Đề bài thực hành Buổi 14
└── README.md                          # Hướng dẫn chạy dự án
```

---

## 2. Hướng dẫn Thực thi Lệnh (Execution Commands)

### Bước 0: Kiểm tra Môi trường (Pre-Check)
```bash
cd "/Users/thuvan/Agribank/DaoTaoTapHuan/Ứng dụng AI phân tích dữ liệu nâng cao/Agribank-rag2/RAG/graph_rag_labs/buoi_14"
.venv/bin/python3 -c "import pandas, rank_bm25, sentence_transformers, neo4j, dotenv; print('ENVIRONMENT READY!')"
```

## 1. Môi trường

Thư mục đã có sẵn `.venv`, vui lòng kích hoạt:

```bash
source .venv/bin/activate
```

## 2. Giao diện Streamlit (App Tra Cứu Trực Quan)

Để chạy giao diện trực quan cho hệ thống:

```bash
streamlit run app.py
```

App sẽ hiển thị tại đường dẫn: `http://localhost:8502`

Trong giao diện, bạn có thể:
- Chọn 4 phương pháp (`bm25`, `dense`, `hybrid`, `hybrid_rerank`).
- Xem kết quả truy xuất (score, citation, nội dung chunk).
- Khám phá Neo4j Graph Hints (Các văn bản & điều khoản liên đới xung quanh kết quả).

**Để dừng app:** Nhấn `Ctrl + C` trên terminal đang chạy Streamlit.

---

### Bước 1: Chuẩn hóa Corpus (`prepare_corpus.py`)
```bash
.venv/bin/python3 scripts/prepare_corpus.py
```
**Kết quả tạo ra:** `data/processed/chunks_normalized.csv` (772 chunks).

### Bước 2: Chạy Truy vấn Baseline Retrieval (`baseline_retrieval.py`)
```bash
.venv/bin/python3 scripts/baseline_retrieval.py --query "Thông tư số 01/2014/TT-NHNN Điều 5 quy định về đóng gói tài sản quý" --top-k 5
```

### Bước 3: Đánh giá & Xuất Báo cáo Baseline (`evaluate_baselines.py`)
   ```bash
   python scripts/evaluate_hybrid.py
   ```
   *Kết quả đánh giá chi tiết được ghi lại tại `outputs/retrieval_examples.md`.*

### Bước 4: Chạy Pipeline Hybrid + Reranker (Cross-Encoder)
Chạy tìm kiếm Hybrid và tinh chỉnh lại (rerank) kết quả bằng mô hình Neural Reranker (Cross-Encoder). Giúp tăng độ chính xác lên mức tối đa bằng cách xếp hạng lại Top N candidates.

```bash
python scripts/rerank.py --query "Quy trình và nguyên tắc tổ chức bảo quản tiền mặt kho quỹ trong ngân hàng" --candidate-k 20 --top-k 5
```
*(Nếu tải mô hình Cross-Encoder quá lâu do giới hạn mạng, Reranker sẽ tự động fallback trả về kết quả Hybrid gốc).*
**Kết quả tạo ra:** [`outputs/retrieval_examples.md`](file:///Users/thuvan/Agribank/DaoTaoTapHuan/%E1%BB%A8ng%20d%E1%BB%A5ng%20AI%20ph%C3%A2n%20t%C3%ADch%20d%E1%BB%AF%20li%E1%BB%87u%20n%C3%A2ng%20cao/Agribank-rag2/RAG/graph_rag_labs/buoi_14/outputs/retrieval_examples.md)
