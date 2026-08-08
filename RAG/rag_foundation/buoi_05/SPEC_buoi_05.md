# SPEC — Buổi 05: OCR PDF tiếng Việt và so sánh 3 chiến lược chunking

Mục tiêu
- Thiết lập một bài tập demo độc lập trong `RAG/rag_foundation/buoi_05/` cho việc đọc PDF tiếng Việt (scan hoặc có text layer), chuẩn hoá text về Unicode NFC, gắn metadata tối thiểu, và minh hoạ so sánh ba chiến lược chunking: `fixed-size`, `semantic`, `hierarchical`.

Phạm vi và giới hạn
- Phạm vi chỉ trong thư mục `RAG/rag_foundation/buoi_05/`.
- Tuyệt đối KHÔNG tạo embedding, KHÔNG lưu vector database, KHÔNG gọi LLM trong Buổi 5.
- Không ghi đè, đổi tên, hoặc xoá file PDF gốc trong `datademo/`.
- Chỉ mô tả việc sử dụng các key trong file `.env` (ở `src/.env`) theo tên biến; TUYỆT ĐỐI KHÔNG đọc, in, log hay tiết lộ giá trị của key.

Đầu vào
- File PDF tiếng Việt nằm trong: `RAG/rag_foundation/buoi_05/datademo/`.
- Chỉ sử dụng các file PDF công khai hoặc mô phỏng (không đưa dữ liệu nội bộ, nhạy cảm).

Đầu ra mong muốn
- Text OCR cho mỗi trang, đã chuẩn hoá Unicode theo NFC.
- Dữ liệu raw (text) lưu trong thư mục output (ví dụ `output/raw/`) cùng metadata cho mỗi trang.
- Metadata tối thiểu cho mỗi trang / đoạn:
  - `source`: đường dẫn/tên file PDF nguồn (không chứa giá trị secret),
  - `page`: số trang (1-based),
  - `ocr_used`: tên công cụ OCR/chiến lược sử dụng (ví dụ: `pymupdf_text`, `llamaparse_image`),
  - `language`: `vi` (hoặc phát hiện ngôn ngữ nếu cần),
  - bổ sung: `nfc_normalized: true/false` (tùy chọn) và `notes` cho cảnh báo (nếu có).

Định dạng lưu mỗi bản ghi (ví dụ JSON per page hoặc per document):
- `document_id` (ví dụ: file name),
- `page` (số trang),
- `text` (UTF-8, Unicode NFC),
- `metadata`: {`source`, `ocr_used`, `language`, ...}

Chiến lược chunking (yêu cầu mô tả và so sánh)
1) Fixed-size
- Mô tả: cắt text theo số ký tự (hoặc token) cố định, kèm `overlap` (số ký tự/token chồng lấp giữa hai chunk liên tiếp).
- Thông số cần thử: `chunk_size` (ví dụ 1000–3000 ký tự), `overlap` (ví dụ 100–500 ký tự).
- Lưu ý: cần báo thống kê về số chunk, độ dài min/max/trung bình, và tỷ lệ mất ngữ cảnh do cắt giữa câu.

2) Semantic
- Mô tả: cắt theo ranh giới ngữ nghĩa tự nhiên — ưu tiên ngắt đoạn, kết đoạn, ngắt dòng, dấu chấm, dấu câu lớn;
- Cách tiếp cận demo: tách theo paragraph (các nhóm dòng trống), sau đó gom paragraph để đạt `max_chunk_size` nếu cần.
- Lưu ý: đánh giá mức độ chia cắt giữa câu, và so sánh số chunk/độ dài so với fixed-size.

3) Hierarchical
- Mô tả: tận dụng cấu trúc tài liệu pháp quy (nếu có) — các chỉ báo như `Chương`, `Mục`, `Điều`, `Khoản`, `Điểm` được coi là mốc bắt đầu chunk.
- Yêu cầu: nếu PDF có heading rõ ràng (ví dụ nhận dạng bằng regex cho các mẫu phổ biến như `Chương \d+`, `Điều \d+`, `Mục \d+`), sử dụng chúng để định nghĩa ranh giới chunk; nếu không có cấu trúc rõ ràng, không được bịa heading — ghi cảnh báo và fallback sang semantic hoặc fixed-size.
- Mỗi chunk hierarchical phải kèm `metadata_structure` mô tả cấp cấu trúc (ví dụ `{"chapter": "Chương 1", "section": "Mục 2", "article": "Điều 3"}`).

Yêu cầu chung cho mọi chunk
- Mỗi chunk phải có tối thiểu các trường: `chunk_id`, `strategy` (fixed/semantic/hierarchical), `source` (file), `page_start`, `page_end`, `text` (Unicode NFC), `metadata_structure` (nếu hierarchical có cấu trúc), và các trường metadata bổ trợ nếu cần.

Quy trình OCR và xử lý (mức demo)
- Bước 1: Đọc PDF bằng PyMuPDF và thử lấy text layer (`pymupdf`/`fitz`). Nếu text layer có nội dung và ký tự hợp lệ, dùng text đó.
- Bước 2: Nếu trang rỗng, encoding bị lỗi, hoặc text chứa nhiều ký tự lạ/garbage, render trang thành ảnh và cầu fallback OCR (ví dụ `llamaparse` qua llama_cloud hoặc một công cụ OCR khác) — chỉ mô tả cách gọi API, TUYỆT ĐỐI KHÔNG ghi giá trị API key trong mã nguồn.
- Bước 3: Chuẩn hoá Unicode theo NFC.
- Bước 4: Lưu text per-page vào `output/raw/` với metadata.
- Bước 5: Chạy ba chiến lược chunking trên text đã chuẩn hoá, lưu kết quả vào `output/chunks/{strategy}/`.

Yêu cầu về bảo mật và .env
- File `.env` nằm trong `RAG/rag_foundation/buoi_05/src/.env` và có thể chứa `LLAMA_CLOUD_API_KEY`.
- SPEC chỉ yêu cầu tham chiếu tên biến (ví dụ `LLAMA_CLOUD_API_KEY`); TUYỆT ĐỐI KHÔNG đọc, in, log hoặc tiết lộ giá trị của bất kỳ biến môi trường hoặc key nào trong `.env` trong báo cáo, console, hoặc file output.
- Mã demo được phép đọc biến môi trường trong runtime để truyền vào client SDK, nhưng phải tránh in ra hoặc lưu trữ giá trị key trong logs.

Tiêu chí nghiệm thu (Acceptance criteria)
- Có một file PDF tiếng Việt trong `datademo/` và được sử dụng làm input.
- Text đầu ra per-page đã được chuẩn hoá Unicode NFC.
- Metadata cho mỗi trang/bản ghi có đầy đủ 4 trường: `source`, `page`, `ocr_used`, `language`.
- Có kết quả chunking cho cả 3 chiến lược: fixed-size, semantic, hierarchical.
- Fixed-size có thông số `chunk_size` và `overlap` được áp dụng và ghi rõ trong báo cáo.
- Semantic ưu tiên ranh giới tự nhiên (ngắt đoạn, kết đoạn, cách dòng) và cố gắng không cắt giữa câu khi có thể.
- Hierarchical dùng các heading/regex phù hợp; nếu không tìm thấy cấu trúc, ghi cảnh báo và fallback hợp lý.
- Có báo cáo so sánh cơ bản: số chunk, độ dài min/max/trung bình, và hai–ba ví dụ minh hoạ cho mỗi chiến lược.
- KHÔNG tạo embedding, KHÔNG lưu vector database, KHÔNG gọi LLM.
- KHÔNG tiết lộ giá trị secret từ `.env`.

Ghi chú giáo viên / hướng dẫn cho học viên
- Mã mẫu phải rõ ràng, tập trung vào minh hoạ thuật toán chunking chứ không phải hạ tầng.
- Đưa ví dụ đầu ra nhỏ (một hai trang) để minh hoạ kết quả trước khi chạy cả bộ.
- Ghi rõ các tình huống lỗi cần kiểm tra: trang rỗng, text layer bị encoding lạ, heading không rõ ràng.

Tài liệu kèm theo
- Một file README ngắn (nếu cần) giải thích cách chạy demo ở chế độ `--dry-run` và `--write`.

Kết thúc SPEC.