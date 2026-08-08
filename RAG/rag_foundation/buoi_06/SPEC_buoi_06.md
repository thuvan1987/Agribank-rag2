# SPECIFICATION - BUỔI 06 (AI Agent Guidelines)

Tài liệu này quy định các nguyên tắc, phạm vi và giới hạn dành cho AI Agent khi thực hiện công việc tại dự án Buổi 06.

---

## 1. Quy định về Workspace & Quyền truy cập
### Được phép truy cập:
- `RAG/rag_foundation/buoi_05/output/chunks/`
- `RAG/rag_foundation/buoi_05/.venv/`
- `RAG/rag_foundation/buoi_06/`

### Tuyệt đối KHÔNG đọc/truy cập:
- Source code của Buổi 05
- File `README` của các buổi trước
- Các file Jupyter Notebook (`.ipynb`)
- Lịch sử Git (`git history`)
- Các thư mục khác ngoài phạm vi được phép

> **Nguyên tắc:** Buổi 05 được coi là **Black Box**. Không thực hiện reverse engineering, không phân tích hay giả định cách Buổi 05 hoạt động.

---

## 2. Môi trường Python (Interpreter)
- **Bắt buộc:** Sử dụng đúng Python Interpreter trong môi trường ảo có sẵn: `RAG/rag_foundation/buoi_05/.venv/`
- **Không** tạo virtual environment (`.venv`) mới.

---

## 3. Quản lý Package / Thư viện
Chỉ sử dụng và cài đặt các thư viện sau:
- `streamlit`
- `google-genai`
- `chromadb`
- `psycopg`
- `python-dotenv`

> **Lưu ý:** Không cài đặt thêm bất kỳ framework hay thư viện phức tạp nào khác.

---

## 4. Coding Style & Thiết kế Kiến trúc
- **Ưu tiên:** Code đơn giản, trực diện, dễ đọc, ít file, ít class, ít function.
- **Tối giản thiết kế - KHÔNG áp dụng:**
  - Repository Pattern
  - Service Layer
  - Dependency Injection
  - Factory Pattern
  - Plugin Architecture

---

## 5. Phạm vi Chức năng (Scope)
Chỉ tập trung vào 4 thành phần cốt lõi:
1. **Index**: Đọc và nạp dữ liệu chunks vào vector store (ChromaDB).
2. **Retrieval**: Tìm kiếm các chunk liên quan dựa trên query.
3. **Answer**: Gửi context + query tới LLM (Google GenAI) để sinh câu trả lời.
4. **Streamlit UI**: Giao diện người dùng đơn giản.

> **Lưu ý:** Không phát triển các tính năng mở rộng nằm ngoài yêu cầu trên.

---

## 6. Xử lý Lỗi (Error Handling)
- Chỉ áp dụng `try / except` ở mức tối thiểu cho các thao tác chính.
- **Không** tích hợp cơ chế retry phức tạp, hệ thống logging chuyên sâu, hoặc giải pháp monitoring.

---

## 7. Bảo mật (Security)
- **Tương đối nghiêm ngặt:** Tuyệt đối không in (print/log) các thông tin nhạy cảm: `API Key`, `password`, `secret token` ra console hoặc giao diện.

---

## 8. Giới hạn Quy mô Code (Code Size Limit)
- **Mục tiêu tối ưu:** Tổng số dòng code Python khoảng **300 – 500 lines**.
- **Ngưỡng cảnh báo:** Nếu quy mô dự án vượt quá **700 lines**, phải rà soát và đơn giản hóa thiết kế ngay lập tức.
