# Agent Specification - Buổi 07: RAG System Production Refactoring

## 1. Workspace

- **Vùng được đọc**:
  - `rag_foundation/buoi_05/output/chunks/`
  - `rag_foundation/buoi_05/.venv/`
  - `rag_foundation/buoi_06/`
  - `rag_foundation/buoi_07/`
- **Vùng được ghi**:
  - Duy nhất `rag_foundation/buoi_07/`
- **Quy tắc tuyệt đối**: Không được sửa đổi bất kỳ file, folder hoặc cấu hình nào thuộc Buổi 05 và Buổi 06. Mọi đường dẫn trong code phải dùng `Path(__file__).resolve()` để đảm bảo tính di động.

## 2. Python

- Bắt buộc sử dụng Python virtual environment (`.venv`) của Buổi 05 tại `rag_foundation/buoi_05/.venv/`.
- Không tạo venv mới cho Buổi 07.

## 3. Input

- Dữ liệu đầu vào là các file JSON đã được chuẩn bị sẵn trong `rag_foundation/buoi_05/output/chunks/`.
- Dữ liệu Buổi 05 đã hoàn tất xử lý; tuyệt đối không thực hiện lại OCR, parse PDF hay chunk lại văn bản.

## 4. Packages

Chỉ sử dụng các thư viện chính thức trong `requirements.txt`:
- `streamlit>=1.61,<2`
- `google-genai>=2.16,<3`
- `chromadb>=1.5,<2`
- `python-dotenv>=1.2,<2`

Không tự ý cài đặt thêm các package trực tiếp khác ngoài danh sách trên.

## 5. Pipeline

Quy trình xử lý RAG gồm các bước tuần tự:
1. **Validate**: Kiểm tra tính hợp lệ của dữ liệu đầu vào.
2. **Embedding**: Tạo vector biểu diễn văn bản bằng Gemini API.
3. **Chroma Persistent**: Lưu trữ và quản lý index trong Chroma DB.
4. **Retrieval**: Truy vấn tìm kiếm chunk liên quan nhất theo khoảng cách cosine.
5. **Confidence Gate**: Lọc khoảng cách kết quả (threshold gate); nếu khoảng cách quá lớn (evidence yếu), chặn không gọi Generation.
6. **Generation**: Sinh câu trả lời từ Gemini LLM dựa trên context hợp lệ.
7. **Citation**: Gắn trích dẫn chính xác từ metadata thực tế.
8. **Streamlit App**: Giao diện người dùng cho chatbot RAG.
9. **Unittest Offline**: Kiểm thử tự động không dùng internet/API key thật.

## 6. Data Contract

Mỗi chunk dữ liệu đầu vào bắt buộc phải chứa đúng và đủ các trường sau:
- `chunk_id`: ID định danh duy nhất của chunk.
- `strategy`: Chiến lược chunking (`fixed`, `hierarchical`, `semantic`).
- `source`: Tên tài liệu nguồn.
- `page_start`: Trang bắt đầu.
- `page_end`: Trang kết thúc.
- `text`: Nội dung văn bản của chunk.

## 7. Index Contract

- **Phân tách Collection**: Mỗi chiến lược chunking (`strategy`) lưu trong một Chroma Collection riêng biệt.
- **Tính đồng nhất Vector**: Model embedding và dimension của Index và Query phải hoàn toàn trùng khớp (mặc định model `gemini-embedding-2`, dim `768`).
- **Embedding thật**: Bắt buộc tạo vector thật thông qua Gemini API (hoặc mock vector hợp lệ trong test offline); không dùng vector giả ngẫu nhiên hay sai kích thước trong production.
- **Kiểm tra chất lượng Vector**: Chặn triệt để các giá trị `NaN`, `Infinity`, kiểu `boolean` hoặc zero vector.
- **Cấu hình Chroma**: Sử dụng khoảng cách Cosine, thiết lập `embedding_function=None` khi tự quản lý embedding.
- **Tính Idempotent & Read-only**: Việc index phải đảm bảo tính idempotent (không nhân bản dữ liệu khi chạy lại); cung cấp trạng thái read-only khi kiểm tra index đã sẵn sàng.
- **Xác thực trước khi ghi**: Phải validate toàn bộ embedding thành công trước khi thực hiện reset hoặc upsert vào collection.

## 8. Retrieval Contract

- Trả về danh sách evidence thực tế lấy từ kết quả truy vấn Chroma DB.
- Mỗi kết quả phải kèm theo thông số khoảng cách (`distance`).
- Chỉ những evidence có `distance <= RAG_MAX_DISTANCE` mới được coi là đạt ngưỡng và chuyển sang bước sinh câu trả lời.
- Nếu không có evidence nào đạt ngưỡng (evidence yếu), hệ thống dừng lại và không gọi API sinh câu trả lời.

## 9. Citation Contract

- Thông tin trích dẫn (`citations`) phải trích xuất hoàn toàn từ metadata thực tế lưu trong Chroma DB.
- Tuyệt đối không tin tưởng hay sử dụng `source`, `page`, hay `chunk_id` do LLM tự suy đoán/sinh ra trong câu trả lời.
- Kết quả câu trả lời chứa danh sách `citations` và `warnings`; hệ thống tự động thay thế các nhãn trích dẫn trong văn bản bằng citation thật từ metadata.

## 10. Security

- Tuyệt đối không hard-code hoặc làm lộ API Key/Secret.
- Đọc `GEMINI_API_KEY` từ biến môi trường qua file `.env`. File `.env` phải được thêm vào `.gitignore`.

## 11. Testing

- Sử dụng `unittest` chuẩn của Python.
- Mọi test suite phải chạy hoàn toàn offline: sử dụng Mock API cho Gemini và temporary directory cho Chroma DB storage.
- Tuyệt đối không yêu cầu kết nối Internet hoặc API Key thật trong test.

## 12. Coding Style

- Thiết kế tối giản: ít file, ít class, ít hàm ngắn gọn và rõ ràng.
- Không áp dụng các mẫu kiến trúc phức tạp (Abstractions thừa, Factory/Strategy pattern không cần thiết).
