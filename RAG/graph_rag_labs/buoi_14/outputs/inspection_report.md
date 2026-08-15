# Báo cáo Kiểm tra Dữ liệu & Môi trường — Buổi 14
**Ngày thực hiện:** 2026-08-15  
**Dự án:** Hybrid Search + Reranking + Mini Knowledge Graph (Buổi 14)  
**Thư mục làm việc:** `/Users/thuvan/Agribank/DaoTaoTapHuan/Ứng dụng AI phân tích dữ liệu nâng cao/Agribank-rag2/RAG/graph_rag_labs/buoi_14`

---

## 1. Cấu trúc thư mục `buoi_14/`

- `buoi14.md`: Đề bài & Hướng dẫn thực hành Buổi 14.
- `.venv/`: Môi trường ảo Python 3.14.6 đã khởi tạo và hoạt động tốt.
- `outputs/`: Thư mục chứa báo cáo và kết quả (mới khởi tạo).

---

## 2. Kiểm tra Môi trường Python & Thư viện

- **Python Interpreter:** `/Users/thuvan/Agribank/DaoTaoTapHuan/Ứng dụng AI phân tích dữ liệu nâng cao/Agribank-rag2/RAG/graph_rag_labs/buoi_14/.venv/bin/python3` (v3.14.6)
- **Tình trạng Thư viện Phụ thuộc (Dependencies):**
  - `pandas`: Đã cài đặt (`2.2.3`)
  - `rank-bm25`: Đã cài đặt (`0.2.2`)
  - `sentence-transformers` & `transformers` & `torch`: Đã cài đặt
  - `neo4j`: Đã cài đặt (`5.28.1`)
  - `google-genai`: Đã cài đặt (`2.18.1`)
  - `python-dotenv`: Đã cài đặt (`1.1.1`)
  - `chromadb` & `streamlit`: Đã cài đặt
- **LangChain / LlamaIndex:** Không cài đặt (tuân thủ quy tắc giữ kiến trúc đơn giản, minh bạch).

---

## 3. Kiểm tra Trực tiếp 3 Tệp Dữ liệu Nguồn (`../kb+hops/`)

Đã tiến hành đọc trực tiếp 3 tệp CSV tại đường dẫn `../kb+hops/` (chế độ Read-Only, không sao chép hay thay đổi file gốc):

### 3.1 `metadata.csv`
- **Số dòng (Rows):** 15 dòng dữ liệu.
- **Encoding:** UTF-8.
- **Danh sách cột (17 cột):** `id`, `title`, `so_ky_hieu`, `ngay_ban_hanh`, `loai_van_ban`, `ngay_co_hieu_luc`, `ngay_het_hieu_luc`, `nguon_thu_thap`, `ngay_dang_cong_bao`, `nganh`, `linh_vuc`, `co_quan_ban_hanh`, `chuc_danh`, `nguoi_ky`, `pham_vi`, `thong_tin_ap_dung`, `tinh_trang_hieu_luc`.
- **Số lượng Null:**
  - `ngay_co_hieu_luc`: 1
  - `ngay_het_hieu_luc`: 14
  - `nguon_thu_thap`: 5
  - `ngay_dang_cong_bao`: 11
  - `nganh`: 3
  - `linh_vuc`: 2
  - `thong_tin_ap_dung`: 15
  - *CÁC CỘT CÒN LẠI (10 CỘT):* 0 null.
- **Số lượng Duplicate:** 0 dòng trùng lặp.
- **Khóa chính (Primary Key):** `id` (15 ID định danh duy nhất của văn bản pháp lý).
- **Trường Citation (Trích dẫn):** `title`, `so_ky_hieu`, `loai_van_ban`, `co_quan_ban_hanh`, `ngay_ban_hanh`, `tinh_trang_hieu_luc`.

### 3.2 `content.csv`
- **Số dòng (Rows):** 15 dòng dữ liệu.
- **Encoding:** UTF-8.
- **Danh sách cột (2 cột):** `id`, `content_html`.
- **Số lượng Null:** 0 null.
- **Số lượng Duplicate:** 0 dòng trùng lặp.
- **Khóa tham chiếu (Foreign Key):** `id` (liên kết 1-1 với `metadata.csv`).
- **Trường Retrieval (Tìm kiếm):** `content_html` (chứa toàn bộ văn bản HTML đã bóc tách cấu trúc).

### 3.3 `relationships.csv`
- **Số dòng (Rows):** 8 dòng dữ liệu quan hệ.
- **Encoding:** UTF-8.
- **Danh sách cột (4 cột):** `doc_id`, `other_doc_id`, `relationship`, `relationship_type`.
- **Số lượng Null:** 0 null.
- **Số lượng Duplicate:** 0 dòng trùng lặp.
- **Khóa liên kết:** `doc_id` và `other_doc_id` (trỏ đến `metadata.id`).
- **Loại quan hệ (relationship_type):** `SUA_DOI_BO_SUNG`, `CAN_CU`, `DAN_CHIEU`, v.v.

---

## 4. Rà soát Mã nguồn Hiện tại trong `buoi_14/`

- **Số tệp mã nguồn:** 0 tệp `.py` (chưa có code thực thi chính nào trong `buoi_14/`).
- **Cảnh báo an toàn (Safety Audit):**
  - Không tìm thấy câu lệnh phá hủy dữ liệu (`os.remove`, `shutil.rmtree`, `open(..., "w")`).
  - Không có câu lệnh truy vấn nguy hiểm Neo4j (`DELETE`, `DROP`, `DETACH DELETE`).
  - Không tìm thấy hard-code API Key hay mật khẩu lộ.

---

## 5. Kết luận

Môi trường và dữ liệu nguồn đã sẵn sàng 100% cho việc xây dựng Pipeline Hybrid Search + Reranking + Mini Knowledge Graph ở các bước tiếp theo.
