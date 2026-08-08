# SPECIFICATION — BUỔI 09: MULTI-QUERY RETRIEVAL VÀ PARENT-CHILD RETRIEVAL FOR LEGAL DOCUMENTS

Tài liệu thiết kế chi tiết kiến trúc cho **Buổi 09** (Tầng RAG Nâng Cao 2: Mở rộng câu hỏi Đa hướng & Phân cấp Cấu trúc Văn bản Pháp luật Ngân hàng).

---

## 🎯 1. Mục Tiêu & Khác Biệt Giữa Buổi 08 và Buổi 09

- **Buổi 08 (Advanced RAG baseline)**:
  - Một câu hỏi duy nhất $Q_0 \rightarrow$ BM25 + Semantic $\rightarrow$ RRF Fusion $\rightarrow$ Cross-Encoder Reranker $\rightarrow$ Child Chunk Context.
  - *Hạn chế*: Dễ bị sót thông tin nếu cách diễn đạt của user chưa đúng thuật ngữ pháp lý, và Child Chunk đơn lẻ có thể quá ngắn, thiếu bối cảnh Chương/Điều xung quanh.
- **Buổi 09 (Multi-Query Expansion & Parent-Child Retrieval)**:
  - **Multi-Query Expansion**: Sử dụng LLM sinh ra $N$ câu hỏi biến thể $Q_1, Q_2, \dots, Q_N$ tiếp cận đa chiều (ngữ nghĩa, thuật ngữ, trọng tâm pháp lý).
  - **Cross-Query RRF Fusion**: Tổng hợp danh sách Child Hits từ tất cả các nhánh query biến thể với trọng số $W_{\text{orig}} > W_{\text{variant}}$.
  - **Child-to-Parent Mapping & Parent Aggregation**: Ánh xạ Child Hits về các Parent Documents (toàn bộ nội dung Điều/Chương chứa nó).
  - **Parent Reranking & Budgeting**: Chấm điểm Cross-Encoder Reranker trên cấp Parent, khống chế ngân sách context (`TOTAL_CONTEXT_MAX_CHARS`).

---

## 📐 2. Sơ Đồ Kiến Trúc Advanced Multi-Query Parent-Child Pipeline

```mermaid
flowchart TD
    Q0[User Question Q0] --> LLM_GEN[LLM Multi-Query Expansion]
    LLM_GEN --> Q1[Query Variant Q1]
    LLM_GEN --> Q2[Query Variant Q2]
    LLM_GEN --> Q3[Query Variant Q3]

    Q0 --> H0[Per-Query Hybrid Search BM25+Sem]
    Q1 --> H1[Per-Query Hybrid Search BM25+Sem]
    Q2 --> H2[Per-Query Hybrid Search BM25+Sem]
    Q3 --> H3[Per-Query Hybrid Search BM25+Sem]

    H0 & H1 & H2 & H3 --> CQ_RRF[Cross-Query RRF Fusion on Child Hits]

    CQ_RRF --> MAP[Hierarchy Registry: Child-to-Parent Mapping]

    MAP --> AGG[Parent Score Aggregation & Deduplication]

    AGG --> RERANK[Cross-Encoder Reranker on Parent Context]

    RERANK --> GATE[Parent Confidence Gate rerank_score >= 0.50]

    GATE --> LLM_ANS[Gemini Generation + Parent Citations]
    LLM_ANS --> OUT[Final Answer + [E1] Parent Citations + Detailed Trace]
```

---

## ⚙️ 3. Bốn Chế Độ Thực Thi (Execution Modes)

1. `single_flat`: 1 câu hỏi $Q_0 \rightarrow$ Truy xuất Child Chunks phẳng (Tương đương Buổi 08).
2. `multi_flat`: Đa câu hỏi $Q_0, Q_1, \dots \rightarrow$ Cross-Query RRF $\rightarrow$ Truy xuất Child Chunks phẳng.
3. `single_parent`: 1 câu hỏi $Q_0 \rightarrow$ Truy xuất Child Chunks $\rightarrow$ Mở rộng sang Parent Documents.
4. `multi_parent` *(Default)*: Đa câu hỏi $\rightarrow$ Cross-Query RRF $\rightarrow$ Mở rộng Parent Documents $\rightarrow$ Parent Rerank.

---

## 📋 4. Schema Chi Tiết

### A. QueryVariant Schema
```json
{
  "query_id": "var-1",
  "query_text": "Khách hàng có thể xin điều chỉnh kỳ hạn trả nợ trong trường hợp nào?",
  "weight": 1.0,
  "is_original": false
}
```

### B. Hierarchy Registry Schema
```json
{
  "parent_id": "parent-doc.pdf-dieu-7",
  "source": "LuatNganHang.pdf",
  "heading": "Điều 7. Cơ cấu lại thời hạn trả nợ",
  "page_start": 5,
  "page_end": 7,
  "text": "Điều 7. Cơ cấu lại thời hạn trả nợ\n1. Tổ chức tín dụng xem xét cơ cấu lại...",
  "child_ids": ["hier-15", "hier-16"]
}
```

### C. ParentDocument Schema
```json
{
  "parent_id": "parent-doc.pdf-dieu-7",
  "source": "LuatNganHang.pdf",
  "heading": "Điều 7. Cơ cấu lại thời hạn trả nợ",
  "page_start": 5,
  "page_end": 7,
  "text": "Nội dung đầy đủ của Điều 7...",
  "child_count": 2,
  "child_ids": ["hier-15", "hier-16"]
}
```

### D. ParentCandidate Schema (Sau khi Aggregation & Rerank)
```json
{
  "parent_id": "parent-doc.pdf-dieu-7",
  "source": "LuatNganHang.pdf",
  "heading": "Điều 7. Cơ cấu lại thời hạn trả nợ",
  "page_start": 5,
  "page_end": 7,
  "text": "Nội dung parent đầy đủ...",
  "parent_rrf_score": 0.0452,
  "parent_fused_rank": 1,
  "rerank_raw_score": 2.15,
  "rerank_score": 0.8956,
  "rerank_rank": 1,
  "rank_change": 0,
  "matched_child_ids": ["hier-15"],
  "accepted": true
}
```

---

## 🧮 5. Công Thức Cross-Query RRF & Parent Aggregation

### A. Cross-Query RRF cho Child Hits
Với mỗi child chunk $c$ xuất hiện từ kết quả tìm kiếm của query $q_j$ với trọng số $W(q_j)$:
$$\text{Score}_{\text{CrossRRF}}(c) = \sum_{j \in \text{Queries}} \frac{W(q_j)}{K_{\text{cq}} + \text{Rank}(c, q_j)}$$
với $K_{\text{cq}} = 60$, $W(Q_0) = 1.5$, $W(Q_i) = 1.0$.

### B. Parent Score Aggregation
Điểm của Parent Document $P$ được tổng hợp từ Top $M$ Child Hits tốt nhất của nó ($M = \text{PARENT\_SCORE\_CHILD\_LIMIT} = 3$):
$$\text{Score}_{\text{Parent}}(P) = \sum_{c \in \text{Top } M \text{ Children of } P} \text{Score}_{\text{CrossRRF}}(c)$$

---

## 📐 6. Context Budget & Citation Contract

- **Ngân sách Context (`TOTAL_CONTEXT_MAX_CHARS`)**: Mặc định 16,000 ký tự. Các Parent Document được sắp xếp theo thứ hạng Rerank tốt nhất, lần lượt đưa vào Prompt đến khi chạm hạn ngạch.
- **Citation Contract**: Nhãn trích dẫn `[E1]`, `[E2]` của LLM được ánh xạ trực tiếp sang metadata của Parent Document (`source`, `page_start`-`page_end`, `parent_id`, `heading`).

---

## 🧪 7. Status & Failure Contract

- `answered`: Trả lời thành công có trích dẫn từ Parent Document.
- `insufficient_evidence`: Không có Parent nào đạt ngưỡng gate `rerank_score >= RERANK_MIN_SCORE`.
- `retrieval_only`: Thiếu API key hoặc LLM rỗng/lỗi.
- `reranker_unavailable`: Lỗi nạp Reranker model $\rightarrow$ Trả status riêng biệt.

---

## 🔒 8. Cam Kết Phạm Vi Ghi
Tất cả code, config, fixture, storage và báo cáo của Buổi 09 được tạo lập **duy nhất** bên trong thư mục `rag_advance/buoi_09/`. Hoàn toàn không sửa đổi Buổi 05–08.
